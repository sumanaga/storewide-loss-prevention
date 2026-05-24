# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
In-memory visit cache for incremental pose analysis.

Accumulates pose results across multiple ba/requests for the same visit,
enabling a growing temporal window without re-running YOLO on already-
processed frames.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pose_analyzer import Pose

logger = logging.getLogger(__name__)


@dataclass
class VisitState:
    """Accumulated state for a single person visit in a region."""

    poses: list = field(default_factory=list)  # list[Pose] at runtime
    last_processed_ts: int = 0  # epoch ms of the newest frame already processed
    alerted: bool = False
    alert_confidence: float = 0.0
    alert_vlm_response: Optional[str] = None
    peak_confidence: float = 0.0  # highest confidence seen across all evaluations
    request_count: int = 0
    created_at: float = field(default_factory=time.time)


class EntityVisitCache:
    """
    Keyed by (person_id, region_id, entry_timestamp).

    Lifecycle: entry is created on first request for a visit and removed
    when stale (no requests for >max_age_seconds).  Because entry_timestamp
    uniquely identifies a visit, different visits by the same person get
    separate cache entries.
    """

    def __init__(self, max_age_seconds: float = 300.0):
        self._visits: dict[str, VisitState] = {}
        self._max_age = max_age_seconds

    @staticmethod
    def _key(person_id: str, region_id: str, entry_timestamp: str) -> str:
        return f"{person_id}:{region_id}:{entry_timestamp}"

    def get_or_create(
        self, person_id: str, region_id: str, entry_timestamp: str
    ) -> VisitState:
        key = self._key(person_id, region_id, entry_timestamp)
        if key not in self._visits:
            self._visits[key] = VisitState()
        return self._visits[key]

    def remove(self, person_id: str, region_id: str, entry_timestamp: str) -> None:
        self._visits.pop(
            self._key(person_id, region_id, entry_timestamp), None
        )

    def cleanup_stale(self) -> int:
        """Remove visits older than max_age. Returns number removed."""
        now = time.time()
        stale_keys = [
            k for k, v in self._visits.items()
            if (now - v.created_at) > self._max_age
        ]
        for k in stale_keys:
            del self._visits[k]
        if stale_keys:
            logger.debug("Cleaned up %d stale visit cache entries", len(stale_keys))
        return len(stale_keys)

    @property
    def size(self) -> int:
        return len(self._visits)
