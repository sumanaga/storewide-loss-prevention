# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Pose Analyzer

Detects suspicious activity patterns from pose sequences.
Pose extraction is handled by the YOLO pose pipeline.
When a pose pattern matches, optionally sends frames to VLM for confirmation.
The service is generic — patterns and prompts are loaded from config.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from pose_rule_engine import PoseRuleEngine
from vlm_client import VLMClient

logger = logging.getLogger(__name__)


# COCO 17-keypoint indices
class Keypoints:
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


@dataclass
class Pose:
    """Single frame pose data."""

    keypoints: np.ndarray  # Shape: [17, 2] - x, y coordinates
    confidences: np.ndarray  # Shape: [17] - confidence per keypoint
    timestamp: Optional[int] = None
    bbox: Optional[tuple[int, int, int, int]] = None  # (x1, y1, x2, y2) in original image coords

    def get_keypoint(self, idx: int) -> tuple[float, float, float]:
        """Get keypoint (x, y, confidence)."""
        return (
            self.keypoints[idx][0],
            self.keypoints[idx][1],
            self.confidences[idx],
        )

    @property
    def left_wrist(self) -> tuple[float, float, float]:
        return self.get_keypoint(Keypoints.LEFT_WRIST)

    @property
    def right_wrist(self) -> tuple[float, float, float]:
        return self.get_keypoint(Keypoints.RIGHT_WRIST)

    @property
    def left_hip(self) -> tuple[float, float, float]:
        return self.get_keypoint(Keypoints.LEFT_HIP)

    @property
    def right_hip(self) -> tuple[float, float, float]:
        return self.get_keypoint(Keypoints.RIGHT_HIP)

    @property
    def left_shoulder(self) -> tuple[float, float, float]:
        return self.get_keypoint(Keypoints.LEFT_SHOULDER)

    @property
    def right_shoulder(self) -> tuple[float, float, float]:
        return self.get_keypoint(Keypoints.RIGHT_SHOULDER)

    @property
    def waist_midpoint(self) -> tuple[float, float]:
        """Calculate waist midpoint from hips."""
        lh = self.left_hip
        rh = self.right_hip
        return ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)

    @property
    def chest_midpoint(self) -> tuple[float, float]:
        """Calculate chest midpoint from shoulders."""
        ls = self.left_shoulder
        rs = self.right_shoulder
        return ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)


@dataclass
class PatternResult:
    """Result of pattern detection."""

    matched: bool
    confidence: float
    pattern_id: str
    description: str
    key_frames: list[int] = field(default_factory=list)
    vlm_result: Optional[dict[str, Any]] = None
    vlm_confirmed: Optional[bool] = None
    vlm_metrics: Optional[dict[str, Any]] = None


class PoseAnalyzer:
    """
    Analyzes pose sequences to detect suspicious patterns.

    Pose extraction is handled externally by the GStreamer DL Streamer pipeline.
    This class provides pattern matching on pose sequences and VLM confirmation.
    """

    def __init__(
        self,
        min_frames: int = 10,
        confidence_threshold: float = 0.5,
        vlm_client: Optional[VLMClient] = None,
        pattern_config: Optional[dict[str, Any]] = None,
    ):
        self.min_frames = min_frames
        self.confidence_threshold = confidence_threshold
        self.vlm_client = vlm_client
        self.pattern_config = pattern_config or {}
        self.rule_engine = PoseRuleEngine(min_confidence=confidence_threshold)
        logger.info("PoseAnalyzer initialized (pattern detection + VLM confirmation)")

    def is_loaded(self) -> bool:
        """Check if analyzer is ready."""
        return True

    def detect_pattern(
        self,
        pose_sequence: list[Pose],
        pattern_id: str = "shelf_to_waist",
    ) -> PatternResult:
        """
        Detect suspicious activity pattern in pose sequence.

        Args:
            pose_sequence: List of Pose objects (chronological order)
            pattern_id: Pattern to detect

        Returns:
            PatternResult indicating if pattern was detected
        """
        # Check if pattern is enabled in config
        pattern_cfg = self.pattern_config.get(pattern_id, {})
        if pattern_cfg and not pattern_cfg.get("enabled", True):
            return PatternResult(
                matched=False,
                confidence=0.0,
                pattern_id=pattern_id,
                description=f"Pattern '{pattern_id}' is disabled",
            )

        # Generic declarative rule engine
        pose_cfg = pattern_cfg.get("pose", {})
        if "phases" in pose_cfg:
            engine_result = self.rule_engine.evaluate(
                pose_sequence, pattern_cfg, min_frames=self.min_frames
            )
            return PatternResult(
                matched=engine_result.matched,
                confidence=engine_result.confidence,
                pattern_id=pattern_id,
                description=engine_result.description,
                key_frames=engine_result.key_frames,
            )

        logger.warning(
            "Pattern '%s' has no phases defined and no built-in implementation — skipping",
            pattern_id,
        )
        return PatternResult(
            matched=False,
            confidence=0.0,
            pattern_id=pattern_id,
            description=f"Pattern '{pattern_id}' has no phases defined",
        )

    def detect_all_patterns(
        self,
        pose_sequence: list[Pose],
    ) -> list[PatternResult]:
        """
        Run all enabled patterns against a pose sequence.

        Args:
            pose_sequence: List of Pose objects (chronological order)

        Returns:
            List of PatternResult for each enabled pattern
        """
        pattern_ids = list(self.pattern_config.keys()) if self.pattern_config else ["shelf_to_waist"]
        results = []
        for pattern_id in pattern_ids:
            cfg = self.pattern_config.get(pattern_id, {})
            if cfg and not cfg.get("enabled", True):
                continue
            results.append(self.detect_pattern(pose_sequence, pattern_id))
        return results

    async def analyze_with_vlm(
        self,
        frames: list[tuple[np.ndarray, int]],
        pose_result: PatternResult,
        poses: Optional[list["Pose"]] = None,
        frame_key_prefix: str = "",
    ) -> PatternResult:
        """
        Send frames to VLM for visual confirmation after pose match.

        Only called when:
        1. Pose pattern matched
        2. VLM is enabled globally
        3. VLM is enabled for this pattern in config

        Args:
            frames: List of (frame_image, timestamp) tuples
            pose_result: The result from pose-based detection
            poses: List of Pose objects with bbox for person-region cropping

        Returns:
            Updated PatternResult with VLM confirmation
        """
        if not self.vlm_client:
            return pose_result

        pattern_cfg = self.pattern_config.get(pose_result.pattern_id, {})
        vlm_cfg = pattern_cfg.get("vlm", {})

        if not vlm_cfg.get("enabled", True):
            logger.debug(f"VLM disabled for pattern {pose_result.pattern_id}")
            return pose_result

        prompt = vlm_cfg.get("prompt", "")
        if not prompt:
            logger.warning(f"No VLM prompt configured for pattern {pose_result.pattern_id}")
            return pose_result

        # Select frames deterministically: first frame (shelf grab) + last (n-1) frames (concealment)
        num_frames = vlm_cfg.get("num_frames", 4)
        if pose_result.key_frames and poses:
            # key_frames are indices into the poses list, not the frames list.
            # Map pose indices → frame indices via timestamp matching.
            frame_ts_to_idx = {int(f[1]): i for i, f in enumerate(frames)}
            key_poses = [poses[i] for i in pose_result.key_frames if i < len(poses)]
            key_frame_list = []
            for p in key_poses:
                if p.timestamp and p.timestamp in frame_ts_to_idx:
                    idx = frame_ts_to_idx[p.timestamp]
                    key_frame_list.append(frames[idx])
            if not key_frame_list:
                key_frame_list = [frames[i] for i in pose_result.key_frames if i < len(frames)]
            sampled = self._select_deterministic(key_frame_list, num_frames)
        elif pose_result.key_frames:
            key_frame_list = [
                frames[i] for i in pose_result.key_frames if i < len(frames)
            ]
            sampled = self._select_deterministic(key_frame_list, num_frames)
        else:
            sampled = self._select_deterministic(frames, num_frames)

        frame_images = [f[0] for f in sampled]
        sampled_ts = [int(f[1]) for f in sampled]
        sampled_keys = (
            [f"{frame_key_prefix}{ts}.jpg" for ts in sampled_ts]
            if frame_key_prefix else []
        )

        # Crop frames to person region for higher effective resolution
        frame_images = self._crop_person_regions(frame_images, sampled_ts, poses)

        logger.info(
            "Sending %d frames to VLM for pattern '%s' "
            "(input_pool=%d, sampled_keys=%s, cropped=%s)",
            len(frame_images), pose_result.pattern_id,
            len(frames), sampled_keys or sampled_ts,
            poses is not None,
        )

        vlm_result = await self.vlm_client.analyze(frame_images, prompt)

        if not vlm_result.success:
            logger.warning(f"VLM analysis failed: {vlm_result.error}")
            # Fall back to pose-only result
            pose_result.vlm_confirmed = None
            return pose_result

        parsed = vlm_result.parsed
        pose_result.vlm_result = parsed
        pose_result.vlm_metrics = vlm_result.metrics

        # Check if VLM confirms the suspicious behavior
        vlm_suspicious = parsed.get("suspicious", False) if parsed else False
        vlm_confidence = parsed.get("confidence", 0.0) if parsed else 0.0
        vlm_reasoning = parsed.get("reasoning", "") if parsed else ""

        # Require minimum VLM confidence to confirm alert
        vlm_conf_threshold = vlm_cfg.get("confidence_threshold", 0.7)
        if vlm_suspicious and vlm_confidence < vlm_conf_threshold:
            logger.info(
                "VLM said suspicious but confidence %.2f < threshold %.2f, treating as not confirmed",
                vlm_confidence, vlm_conf_threshold,
            )
            vlm_suspicious = False

        pose_result.vlm_confirmed = vlm_suspicious

        if vlm_suspicious:
            # Combine pose and VLM confidence
            combined = (pose_result.confidence + vlm_confidence) / 2
            pose_result.confidence = combined
            pose_result.description = (
                f"{pose_result.description} | VLM confirms: {vlm_reasoning}"
            )
        else:
            # VLM disagrees — lower confidence but keep match
            pose_result.confidence = pose_result.confidence * 0.5
            pose_result.description = (
                f"{pose_result.description} | VLM disagrees: {vlm_reasoning}"
            )

        logger.info(
            f"VLM confirmation: suspicious={vlm_suspicious}, "
            f"vlm_confidence={vlm_confidence:.2f}, "
            f"combined_confidence={pose_result.confidence:.3f}, "
            f"reasoning={vlm_reasoning}"
        )

        return pose_result

    @staticmethod
    def _select_deterministic(
        frames: list[tuple[np.ndarray, int]],
        n: int,
    ) -> list[tuple[np.ndarray, int]]:
        """Deterministically select n frames: first + last (n-1).

        Ensures VLM sees both the shelf-grab moment (first key frame)
        and the concealment moment (last key frames). No randomness.
        """
        if len(frames) <= n:
            return frames
        # First frame (Phase 1: hand at shelf) + last (n-1) frames (Phase 2: concealment)
        return [frames[0]] + frames[-(n - 1):]

    @staticmethod
    def _crop_person_regions(
        frame_images: list[np.ndarray],
        timestamps: list[int],
        poses: Optional[list["Pose"]],
        padding_ratio: float = 0.25,
    ) -> list[np.ndarray]:
        """Crop each frame to the person bounding box for higher VLM resolution.

        Uses pose bbox matched by timestamp. Falls back to full frame if no
        matching pose/bbox is found.

        Args:
            frame_images: BGR numpy arrays (full camera frames).
            timestamps: Timestamp for each frame (used to match with poses).
            poses: Pose objects containing bbox from YOLO detection.
            padding_ratio: Extra padding around bbox as fraction of bbox size.

        Returns:
            List of cropped (or original) BGR numpy arrays.
        """
        if not poses:
            return frame_images

        # Build timestamp → bbox lookup from poses
        ts_to_bbox: dict[int, tuple[int, int, int, int]] = {}
        for pose in poses:
            if pose.timestamp is not None and pose.bbox is not None:
                ts_to_bbox[pose.timestamp] = pose.bbox

        cropped = []
        for img, ts in zip(frame_images, timestamps):
            bbox = ts_to_bbox.get(ts)
            if bbox is None:
                # No matching pose — try closest timestamp within 100ms
                closest_ts = min(
                    (t for t in ts_to_bbox if abs(t - ts) < 100),
                    key=lambda t: abs(t - ts),
                    default=None,
                )
                if closest_ts is not None:
                    bbox = ts_to_bbox[closest_ts]

            if bbox is None:
                cropped.append(img)
                continue

            h, w = img.shape[:2]
            x1, y1, x2, y2 = bbox
            bw, bh = x2 - x1, y2 - y1

            # Add padding around person bbox
            pad_x = int(bw * padding_ratio)
            pad_y = int(bh * padding_ratio)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)

            # Minimum crop size sanity check (avoid tiny/degenerate crops)
            if (x2 - x1) < 50 or (y2 - y1) < 50:
                logger.debug("Crop too small (%dx%d), using full frame", x2 - x1, y2 - y1)
                cropped.append(img)
                continue

            logger.debug(
                "Cropped frame ts=%d: full=%dx%d → crop=%dx%d (bbox=%s)",
                ts, w, h, x2 - x1, y2 - y1, (x1, y1, x2, y2),
            )
            cropped.append(img[y1:y2, x1:x2])

        return cropped
