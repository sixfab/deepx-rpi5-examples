"""
utils/tracker.py — Reusable greedy centroid tracker for DeepX demos.

Used by: people_tracking, smart_traffic, store_queue_analysis demos.
Algorithm: For each existing track, find the nearest unmatched detection
           within m_maxDistance. Unmatched detections become new tracks.
           Tracks missing for more than max_missing_frames are pruned.

This is a straight port of the greedy matching logic in
PeopleTrackingAdapter.cpp (lines 52-83). Coordinates are normalized
floats in [0.0, 1.0], so the distance threshold is frame-size agnostic
and the same tracker instance can be used across different resolutions.
"""

from __future__ import annotations

import math
from typing import Dict, List, Set, Tuple


class CentroidTracker:
    """Greedy nearest-neighbour centroid tracker.

    Each track holds a normalized (x, y) centroid, an age counter, and a
    running ``missing_frames`` count. On every ``update()`` call we:

    1. Bump ``missing_frames`` for every active track (we'll reset it
       below for any track that gets matched this frame).
    2. For each existing track, scan all detections and pick the closest
       one that hasn't been matched yet and is within ``max_distance``.
    3. Detections with no assigned track become brand-new tracks.
    4. Tracks that have been missing for longer than
       ``max_missing_frames`` are pruned.

    The tracker is deliberately stateless between frames apart from its
    internal ``_tracks`` dict — call ``reset()`` when the input source
    changes so stale IDs do not leak across videos.
    """

    def __init__(self, max_missing_frames: int = 10, max_distance: float = 0.1):
        """
        Args:
            max_missing_frames: How many frames a track can be missing before deletion.
            max_distance: Max normalized distance (0.0–1.0) for matching a detection to a track.
        """
        self._max_missing_frames = max_missing_frames
        self._max_distance = max_distance
        # Maps track_id -> {'id', 'centroid', 'missing_frames', 'age'}.
        self._tracks: Dict[int, Dict] = {}
        # Monotonic counter for new track IDs (matches C++ m_nextId).
        self._next_id: int = 0

    def update(
        self, centroids: List[Tuple[float, float]]
    ) -> Dict[int, int]:
        """
        Match current detections to existing tracks using greedy nearest-neighbor.

        Args:
            centroids: List of (x, y) normalized centroid positions from current frame.

        Returns:
            Dict mapping detection index (in input list) → assigned track ID.
            New detections get new IDs. Lost tracks are pruned internally.
        """
        # Step 1: every active track ages by one frame. Matched tracks will
        # have this decremented back to 0 below; unmatched ones keep the +1
        # which eventually trips the prune threshold.
        for track in self._tracks.values():
            track["missing_frames"] += 1
            track["age"] += 1

        # detection_index -> assigned_track_id
        assignments: Dict[int, int] = {}
        # Detections already claimed by a prior track this frame.
        matched_indices: Set[int] = set()

        # Step 2: outer loop is tracks (matches PeopleTrackingAdapter.cpp).
        # For each track, scan every detection and pick the closest unmatched
        # one within the distance threshold.
        for track_id, track in self._tracks.items():
            tx, ty = track["centroid"]

            best_idx = -1
            best_dist = self._max_distance  # Acts as both threshold and initial min.

            for i, (cx, cy) in enumerate(centroids):
                if i in matched_indices:
                    continue
                # Normalized Euclidean distance — math.sqrt keeps numpy optional.
                dist = math.sqrt((tx - cx) ** 2 + (ty - cy) ** 2)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i

            if best_idx >= 0:
                # Claim the detection, refresh the track's position, and
                # reset its missing-frame counter to 0.
                matched_indices.add(best_idx)
                assignments[best_idx] = track_id
                track["centroid"] = centroids[best_idx]
                track["missing_frames"] = 0

        # Step 3: unmatched detections become brand-new tracks.
        for i, centroid in enumerate(centroids):
            if i in matched_indices:
                continue
            new_id = self._next_id
            self._next_id += 1
            self._tracks[new_id] = {
                "id": new_id,
                "centroid": centroid,
                "missing_frames": 0,
                "age": 0,
            }
            assignments[i] = new_id

        # Step 4: prune stale tracks. We collect first, delete second, so
        # we do not mutate the dict while iterating over it.
        to_delete = [
            tid
            for tid, track in self._tracks.items()
            if track["missing_frames"] > self._max_missing_frames
        ]
        for tid in to_delete:
            del self._tracks[tid]

        return assignments

    def get_active_tracks(self) -> Dict[int, Dict]:
        """
        Returns all currently active tracks.
        Each track dict contains: 'id', 'centroid', 'missing_frames', 'age'
        """
        return self._tracks

    def reset(self):
        """Clear all tracks. Call when switching video source."""
        self._tracks.clear()
        self._next_id = 0
