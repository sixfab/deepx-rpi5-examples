"""
utils/zone_utils.py — Polygon and line utilities for zone-based demos.

Used by: trespassing_demo, store_queue_analysis_demo, smart_traffic_demo

All coordinates in public APIs are normalized floats in [0.0, 1.0]. The
drawing helpers internally scale those to pixel space using the frame's
width/height, so demo code can stay resolution-agnostic.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


def point_in_polygon(
    point: Tuple[float, float],
    polygon: List[Tuple[float, float]],
) -> bool:
    """
    Check if a normalized point is inside a normalized polygon.
    Wraps cv2.pointPolygonTest for convenience.
    Returns True if inside or on boundary.
    """
    if len(polygon) < 3:
        # A polygon needs at least three vertices to enclose any area.
        return False

    # cv2.pointPolygonTest needs a float32 contour of shape (N, 1, 2).
    # We keep everything in normalized space — no scaling required since
    # the test is purely topological.
    contour = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    # measureDist=False returns +1 inside, 0 on edge, -1 outside.
    result = cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False)
    return result >= 0


def _cross_product(
    o: Tuple[float, float],
    a: Tuple[float, float],
    b: Tuple[float, float],
) -> float:
    """Signed 2D cross product of vectors (o->a) x (o->b).

    Port of SmartTrafficAdapter::crossProduct(). The sign tells us which
    side of line o->a the point b lies on; zero means collinear.
    """
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def check_line_crossing(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    line_start: Tuple[float, float],
    line_end: Tuple[float, float],
) -> bool:
    """
    Check if segment p1->p2 crosses the line segment line_start->line_end.
    Port of SmartTrafficAdapter::checkLineCrossing() + crossProduct() helper.

    Algorithm (from C++ source SmartTrafficAdapter.cpp lines 47-70):
    1. Compute cross products of (line_start, line_end, p1) and (line_start, line_end, p2)
    2. If p1 and p2 are on opposite sides of the infinite line extension:
       3. Also compute cross products of (p1, p2, line_start) and (p1, p2, line_end)
       4. If line_start and line_end are on opposite sides of p1->p2:
          5. Return True (full segment intersection confirmed)
    Return False otherwise.
    """
    # Which side of the counting line is each track endpoint on?
    d1 = _cross_product(line_start, line_end, p1)
    d2 = _cross_product(line_start, line_end, p2)

    # The track segment only *might* cross the line if its endpoints
    # straddle it — i.e. opposite signs on the two cross products.
    if d1 * d2 < 0:
        # Mirror check: do the line's endpoints straddle the track segment?
        # Both conditions are required to confirm an actual segment-segment
        # intersection rather than just two lines that would eventually meet.
        d3 = _cross_product(p1, p2, line_start)
        d4 = _cross_product(p1, p2, line_end)
        if d3 * d4 < 0:
            return True
    return False


def draw_polygon_overlay(
    frame: np.ndarray,
    polygon: List[Tuple[float, float]],
    color: tuple = (0, 255, 255),
    alpha: float = 0.25,
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw a semi-transparent filled polygon overlay on a frame.
    polygon points are normalized (0.0–1.0) — scale to frame dimensions internally.
    Returns the modified frame.
    """
    if len(polygon) < 3:
        return frame

    h, w = frame.shape[:2]
    # Normalized -> pixel coords. int32 is what OpenCV's polyline/fillPoly expect.
    pts = np.asarray(
        [(int(px * w), int(py * h)) for px, py in polygon],
        dtype=np.int32,
    ).reshape(-1, 1, 2)

    # Fill a copy of the frame so we can blend it back in with alpha and
    # preserve the underlying pixels — cv2.fillPoly alone would be opaque.
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, dst=frame)

    # Solid outline on top so the zone edge is still crisp after blending.
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)
    return frame


def draw_line_overlay(
    frame: np.ndarray,
    line_start: Tuple[float, float],
    line_end: Tuple[float, float],
    color: tuple = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw a counting line on a frame.
    Points are normalized (0.0–1.0).
    """
    h, w = frame.shape[:2]
    p1 = (int(line_start[0] * w), int(line_start[1] * h))
    p2 = (int(line_end[0] * w), int(line_end[1] * h))
    cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)
    return frame


def draw_multi_polygon_overlay(
    frame: np.ndarray,
    polygons: List[List[Tuple[float, float]]],
    color: tuple = (255, 200, 0),
    alpha: float = 0.2,
) -> np.ndarray:
    """
    Draw multiple polygon overlays (for StoreQueueAnalysis regions).
    Each polygon in the list is drawn separately.
    """
    for polygon in polygons:
        draw_polygon_overlay(frame, polygon, color=color, alpha=alpha)
    return frame
