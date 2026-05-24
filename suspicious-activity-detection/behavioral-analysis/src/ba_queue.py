# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
MQTT-based queue consumer for Behavioral Analysis.

Uses an in-memory visit cache to accumulate poses incrementally across
multiple ba/requests for the same visit.  Only NEW frames (those with a
timestamp newer than the last processed) are sent through YOLO-Pose;
pattern confidence is evaluated over the full growing window.

Once a visit triggers a "suspicious" alert, subsequent requests for the
same visit immediately re-publish "suspicious" without re-running inference.
"""

import asyncio
import json
import logging
import time
from typing import Optional

from vlm_metrics_logger import log_ovms_performance_metric

import paho.mqtt.client as mqtt

from config import Settings
from pose_analyzer import PatternResult
from visit_cache import EntityVisitCache
from yolo_pipeline import extract_poses

logger = logging.getLogger(__name__)


class BAQueueConsumer:
    """Consumes ``ba/requests`` messages and runs one analysis per message."""

    def __init__(
        self,
        settings: Settings,
        frame_store=None,
        pose_analyzer=None,
    ) -> None:
        self.settings = settings
        self.request_topic = settings.ba_request_topic
        self.result_topic = settings.ba_result_topic
        self.frame_store = frame_store
        self.pose_analyzer = pose_analyzer
        self.min_frames = settings.min_frames_for_detection

        # Bound concurrent VLM calls so ovms-vlm doesn’t pile up requests.
        self._vlm_sem = asyncio.Semaphore(
            max(1, int(getattr(settings, "vlm_max_concurrency", 1)))
        )

        self.client: Optional[mqtt.Client] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.connected = False
        self._shutdown = asyncio.Event()
        # In-flight analysis tasks; tracked only so shutdown can await them.
        self._inflight: set[asyncio.Task] = set()        # Entity dedup: skip requests for entities already being analyzed.
        self._inflight_entities: set[str] = set()
        # Max concurrent analysis tasks to bound memory usage.
        self._max_inflight = max(1, int(getattr(settings, "max_inflight_analyses", 3)))

        # Visit cache: accumulates poses across requests for the same visit.
        self._visit_cache = EntityVisitCache(max_age_seconds=300.0)
        self._last_cache_cleanup = time.time()

    def initialize(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self.client = mqtt.Client(client_id="ba-queue-consumer")
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

    async def start(self) -> None:
        logger.info(
            "BA queue consumer connecting to MQTT",
            extra={"host": self.settings.mqtt_host, "port": self.settings.mqtt_port},
        )
        self.client.connect_async(
            self.settings.mqtt_host, self.settings.mqtt_port, keepalive=60
        )
        self.client.loop_start()
        await self._shutdown.wait()

    async def stop(self) -> None:
        self._shutdown.set()
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
        # Let any in-flight analyses finish so their results are published.
        if self._inflight:
            logger.info(
                f"Awaiting {len(self._inflight)} in-flight BA analyses"
            )
            await asyncio.gather(*self._inflight, return_exceptions=True)
        logger.info("BA queue consumer stopped")

    def publish_result(self, result: dict) -> None:
        if self.client and self.connected:
            self.client.publish(
                self.result_topic, json.dumps(result), qos=1
            )
            logger.info(
                "Published BA result",
                extra={
                    "person_id": result.get("person_id"),
                    "status": result.get("status"),
                },
            )

    # ---- paho callbacks ------------------------------------------------------

    def _on_connect(self, client, userdata, flags, rc) -> None:
        if rc == 0:
            self.connected = True
            client.subscribe(self.request_topic, qos=1)
            logger.info(
                f"BA queue consumer connected, subscribed to {self.request_topic}"
            )
        else:
            logger.error(f"BA queue consumer MQTT connect failed, rc={rc}")

    def _on_disconnect(self, client, userdata, rc) -> None:
        self.connected = False
        logger.warning(f"BA queue consumer MQTT disconnected, rc={rc}")

    def _on_message(self, client, userdata, msg: mqtt.MQTTMessage) -> None:
        if msg.topic != self.request_topic:
            return
        try:
            payload = json.loads(msg.payload)
        except json.JSONDecodeError:
            logger.error("Invalid JSON in BA request message")
            return

        person_id = payload.get("person_id", "")
        region_id = payload.get("region_id", "")
        entry_timestamp = payload.get("entry_timestamp", "")
        scene_id = payload.get("scene_id", "")
        last_frame_ts = payload.get("last_frame_ts", "")
        if not person_id:
            logger.warning("BA message missing person_id, skipping")
            return

        if not self.loop:
            return
        # Schedule the single-shot analysis on the asyncio loop.
        self.loop.call_soon_threadsafe(
            self._spawn_analysis,
            person_id, region_id, entry_timestamp, scene_id, last_frame_ts,
        )

    # ---- analysis dispatch ---------------------------------------------------

    def _spawn_analysis(
        self, person_id: str, region_id: str, entry_timestamp: str,
        scene_id: str, last_frame_ts: str,
    ) -> None:
        # Drop request if this entity is already being analyzed (dedup).
        if person_id in self._inflight_entities:
            logger.debug(
                f"Entity {person_id}: analysis already in-flight, skipping"
            )
            return
        # Drop request if we're at capacity to bound memory usage.
        if len(self._inflight) >= self._max_inflight:
            logger.debug(
                f"Max in-flight analyses ({self._max_inflight}) reached, dropping request for {person_id}"
            )
            return

        self._inflight_entities.add(person_id)

        async def _runner() -> None:
            try:
                await self._analyze_visit(
                    person_id, region_id, entry_timestamp,
                    scene_id, last_frame_ts,
                )
            finally:
                self._inflight_entities.discard(person_id)

        task = asyncio.create_task(_runner())
        self._inflight.add(task)
        task.add_done_callback(self._inflight.discard)

    async def _analyze_visit(
        self, person_id: str, region_id: str, entry_timestamp: str,
        scene_id: str, last_frame_ts: str,
    ) -> None:
        """Incremental analysis: only process new frames, accumulate poses."""
        # Periodic stale-entry cleanup (every 60s).
        now = time.time()
        if now - self._last_cache_cleanup > 60:
            self._visit_cache.cleanup_stale()
            self._last_cache_cleanup = now

        visit = self._visit_cache.get_or_create(person_id, region_id, entry_timestamp)
        visit.request_count += 1

        # Fast path: already alerted for this visit — re-publish immediately.
        if visit.alerted:
            logger.info(
                "Entity %s: visit already alerted (conf=%.3f), re-publishing suspicious",
                person_id, visit.alert_confidence,
            )
            self.publish_result({
                "person_id": person_id, "region_id": region_id,
                "entry_timestamp": entry_timestamp, "scene_id": scene_id,
                "last_frame_ts": last_frame_ts,
                "status": "suspicious",
                "confidence": visit.alert_confidence,
                "vlm_response": visit.alert_vlm_response,
                "frames_analyzed": len(visit.poses),
            })
            return

        # Fetch frames from SeaweedFS.
        try:
            frames = await self.frame_store.get_frames(
                entity_id=person_id,
                max_frames=self.settings.max_frames_to_fetch,
                last_frame_ts=last_frame_ts,
                region_id=region_id,
                entry_timestamp=entry_timestamp,
                scene_id=scene_id,
            )
        except Exception:
            logger.exception(f"Frame fetch failed for {person_id}")
            self.publish_result({
                "person_id": person_id, "region_id": region_id,
                "entry_timestamp": entry_timestamp, "scene_id": scene_id,
                "last_frame_ts": last_frame_ts,
                "status": "no_enough_data", "confidence": 0.0,
                "vlm_response": None, "frames_analyzed": 0,
            })
            return

        # Filter to only NEW frames (timestamp > last processed).
        new_frames = [
            (img, ts) for img, ts in frames if ts > visit.last_processed_ts
        ]

        if not new_frames:
            # No new data since last analysis — respond with current state.
            self.publish_result({
                "person_id": person_id, "region_id": region_id,
                "entry_timestamp": entry_timestamp, "scene_id": scene_id,
                "last_frame_ts": last_frame_ts,
                "status": "no_match",
                "confidence": 0.0,
                "vlm_response": None,
                "frames_analyzed": len(visit.poses),
            })
            return

        # Check minimum frames (first request for this visit).
        total_available = len(visit.poses) + len(new_frames)
        if total_available < self.min_frames and visit.request_count <= 1:
            self.publish_result({
                "person_id": person_id, "region_id": region_id,
                "entry_timestamp": entry_timestamp, "scene_id": scene_id,
                "last_frame_ts": last_frame_ts,
                "status": "no_enough_data", "confidence": 0.0,
                "vlm_response": None, "frames_analyzed": total_available,
            })
            return

        # Run YOLO-Pose only on new frames (incremental).
        await self._analyze_incremental(
            person_id, region_id, entry_timestamp, scene_id,
            new_frames, visit, last_frame_ts, all_frames=frames,
        )

    # ---- incremental analysis --------------------------------------------------

    async def _analyze_incremental(
        self, person_id: str, region_id: str, entry_timestamp: str,
        scene_id: str, new_frames: list, visit, last_frame_ts: str,
        all_frames: list,
    ) -> None:
        """Run YOLO on new frames, merge with cached poses, evaluate pattern."""
        try:
            new_poses = await extract_poses(new_frames, person_id, self.settings)

            # Update cache with new poses and watermark.
            if new_poses:
                visit.poses.extend(new_poses)
            visit.last_processed_ts = max(ts for _, ts in new_frames)

            logger.info(
                "Entity %s: +%d new poses (total cached: %d, request #%d)",
                person_id, len(new_poses), len(visit.poses), visit.request_count,
            )

            # Need minimum poses to evaluate pattern.
            if len(visit.poses) < self.min_frames:
                self.publish_result({
                    "person_id": person_id, "region_id": region_id,
                    "entry_timestamp": entry_timestamp, "scene_id": scene_id,
                    "last_frame_ts": last_frame_ts,
                    "status": "no_enough_data", "confidence": 0.0,
                    "vlm_response": None, "frames_analyzed": len(visit.poses),
                })
                return

            # Evaluate patterns over the FULL accumulated pose window.
            results = self.pose_analyzer.detect_all_patterns(visit.poses)
            matched = [r for r in results if r.matched]
            result = (
                max(matched, key=lambda r: r.confidence)
                if matched
                else results[0] if results
                else PatternResult(
                    matched=False, confidence=0.0,
                    pattern_id="shelf_to_waist",
                    description="No patterns evaluated",
                )
            )

            if result.matched:
                pattern_cfg = self.pose_analyzer.pattern_config.get(result.pattern_id, {})
                min_conf_for_alert = pattern_cfg.get("pose", {}).get(
                    "min_confidence_for_alert", 0.55
                )

                # Track peak confidence across the visit (prevents dilution).
                visit.peak_confidence = max(visit.peak_confidence, result.confidence)

                logger.warning(
                    "Entity %s: pose pattern matched "
                    "(confidence=%.3f, peak=%.3f, threshold=%.3f, window=%d poses)",
                    person_id, result.confidence, visit.peak_confidence,
                    min_conf_for_alert, len(visit.poses),
                )

                if visit.peak_confidence < min_conf_for_alert:
                    logger.info(
                        "Entity %s: peak confidence %.3f < %.3f, not alerting yet",
                        person_id, visit.peak_confidence, min_conf_for_alert,
                    )
                    self.publish_result({
                        "person_id": person_id, "region_id": region_id,
                        "entry_timestamp": entry_timestamp, "scene_id": scene_id,
                        "last_frame_ts": last_frame_ts,
                        "status": "no_match",
                        "confidence": visit.peak_confidence,
                        "vlm_response": None,
                        "frames_analyzed": len(visit.poses),
                    })
                    return

                # VLM confirmation (supplementary — alert is based on pose).
                vlm_response = None
                if self.settings.vlm_enabled and self.pose_analyzer.vlm_client:
                    # Timeout on semaphore wait so one hung VLM call
                    # doesn't block all other entities from progressing.
                    sem_timeout = self.settings.vlm_timeout
                    try:
                        acquired = await asyncio.wait_for(
                            self._vlm_sem.acquire(), timeout=sem_timeout,
                        )
                    except asyncio.TimeoutError:
                        acquired = False
                        logger.warning(
                            f"Entity {person_id}: VLM semaphore acquire "
                            f"timed out after {sem_timeout}s, skipping VLM"
                        )
                    if acquired:
                        try:
                            result = await self.pose_analyzer.analyze_with_vlm(
                                frames=all_frames,
                                pose_result=result,
                                poses=visit.poses,
                        )
                        finally:
                            self._vlm_sem.release()
                    if result.vlm_metrics:
                        log_ovms_performance_metric("USECASE_1", result.vlm_metrics)
                    if result.vlm_result:
                        vlm_response = result.vlm_result.get("reasoning")

                # Mark visit as alerted — subsequent requests short-circuit.
                visit.alerted = True
                visit.alert_confidence = result.confidence
                visit.alert_vlm_response = vlm_response

                self.publish_result({
                    "person_id": person_id, "region_id": region_id,
                    "entry_timestamp": entry_timestamp, "scene_id": scene_id,
                    "last_frame_ts": last_frame_ts,
                    "status": "suspicious",
                    "confidence": result.confidence,
                    "vlm_response": vlm_response,
                    "frames_analyzed": len(visit.poses),
                })
                return

            self.publish_result({
                "person_id": person_id, "region_id": region_id,
                "entry_timestamp": entry_timestamp, "scene_id": scene_id,
                "last_frame_ts": last_frame_ts,
                "status": "no_match",
                "confidence": result.confidence,
                "vlm_response": None,
                "frames_analyzed": len(visit.poses),
            })
        except Exception:
            logger.exception(f"Error analysing incremental batch for {person_id}")
