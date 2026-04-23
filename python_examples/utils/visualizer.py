"""Drawing helpers shared by every demo.

The functions here are deliberately stateless and accept the frame
in-place. They consume already-decoded model output (boxes in xyxy,
masks in HxW float, keypoints in Nx{2|3}) and translate them into
OpenCV draw calls.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import cv2
import numpy as np

# A fixed 80-color palette so the colour for "person" stays the same across
# demos and frames. Generated once with a deterministic seed.
_RNG = np.random.default_rng(seed=0)
_PALETTE = (_RNG.uniform(40, 230, size=(256, 3))).astype(np.uint8)

# Standard COCO 17-keypoint skeleton used by yolov*-pose models.
COCO_SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11),  (6, 12),  (5, 6),   (5, 7),   (6, 8),
    (7, 9),   (8, 10),  (1, 2),   (0, 1),   (0, 2),
    (1, 3),   (2, 4),   (3, 5),   (4, 6),
]


def color_for(class_id: int) -> tuple:
    """Stable BGR colour for a class index."""
    r, g, b = _PALETTE[class_id % len(_PALETTE)]
    return int(b), int(g), int(r)


# --------------------------------------------------------------------- boxes


def draw_detections(
    frame: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    labels: Sequence[str],
    config,
) -> None:
    """Draw axis-aligned bounding boxes with optional label/score text.

    Args:
        frame: BGR image, modified in place.
        boxes: (N, 4) array of [x1, y1, x2, y2] in pixel coords.
        scores: (N,) confidence per box.
        class_ids: (N,) integer class index per box.
        labels: List of class-name strings indexed by class id.
        config: Module/object exposing DRAW_LABELS and DRAW_CONFIDENCE.
    """
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box.astype(int)
        color = color_for(int(class_id))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        if not (config.DRAW_LABELS or config.DRAW_CONFIDENCE):
            continue

        # Build the label string from the parts the user asked for.
        parts: List[str] = []
        if config.DRAW_LABELS and 0 <= int(class_id) < len(labels):
            parts.append(labels[int(class_id)])
        if config.DRAW_CONFIDENCE:
            parts.append(f"{score:.2f}")
        text = " ".join(parts)
        if not text:
            continue

        _draw_label(frame, text, x1, y1, color)


def _draw_label(frame: np.ndarray, text: str, x: int, y: int, color) -> None:
    """Draw a filled label box with text. Used by every box-style helper."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    # If the box is near the top of the frame, draw the label inside it
    # rather than above it so the text never disappears off-screen.
    label_y = y - 6 if y - th - 6 > 0 else y + th + 6
    cv2.rectangle(
        frame, (x, label_y - th - 4), (x + tw + 4, label_y + 4), color, cv2.FILLED
    )
    cv2.putText(
        frame, text, (x + 2, label_y), font, scale, (0, 0, 0), thickness, cv2.LINE_AA
    )


# ----------------------------------------------------------------- keypoints


def draw_keypoints(
    frame: np.ndarray,
    keypoints: np.ndarray,
    skeleton_connections: Optional[Sequence[tuple]] = None,
    point_radius: int = 3,
) -> None:
    """Draw keypoints + optional skeleton lines connecting them.

    `keypoints` shape: (N_persons, K, 2) or (N_persons, K, 3) where the
    third channel, if present, is per-keypoint visibility/confidence.
    Coordinates are pixel-space already mapped back to the original frame.
    """
    if keypoints.size == 0:
        return

    skeleton = skeleton_connections if skeleton_connections is not None else COCO_SKELETON

    for person in keypoints:
        # Skeleton edges first so that joint dots sit on top of the lines.
        for j, (a, b) in enumerate(skeleton):
            if a >= len(person) or b >= len(person):
                continue
            pa, pb = person[a], person[b]
            cv2.line(
                frame,
                (int(pa[0]), int(pa[1])),
                (int(pb[0]), int(pb[1])),
                (51, 153, 255),
                2,
                cv2.LINE_AA,
            )
        for kp in person:
            cv2.circle(
                frame, (int(kp[0]), int(kp[1])), point_radius, (0, 0, 255), -1
            )


# --------------------------------------------------------------------- masks


def draw_masks(
    frame: np.ndarray,
    masks: np.ndarray,
    class_ids: np.ndarray,
    alpha: float = 0.4,
) -> None:
    """Blend per-instance segmentation masks onto `frame`.

    `masks` shape: (N, H, W) of float values in [0, 1] already resized to
    the frame's resolution. Each mask is recoloured by its class id and
    blended with weight `alpha` so the original image stays visible.
    """
    if masks.size == 0:
        return

    for mask, class_id in zip(masks, class_ids):
        color = np.array(color_for(int(class_id)), dtype=np.float32)
        binary = mask > 0.5  # Threshold the soft mask into a hard region.
        if not np.any(binary):
            continue
        # Per-pixel blend only where the mask is active. Vectorised so that
        # the overhead per frame stays small even with many instances.
        frame[binary] = (
            frame[binary].astype(np.float32) * (1.0 - alpha) + color * alpha
        ).astype(np.uint8)


# ------------------------------------------------------------ semantic mask


def draw_semantic_mask(
    frame: np.ndarray,
    class_map: np.ndarray,
    colormap: np.ndarray,
    alpha: float = 0.55,
) -> None:
    """Overlay a semantic segmentation result onto `frame`.

    `class_map` is an HxW integer grid where each pixel holds a class id.
    `colormap` is an (N_classes, 3) BGR table used to colourise it.
    """
    if class_map.size == 0:
        return
    h, w = frame.shape[:2]
    # Use nearest-neighbour because we want to preserve sharp class edges.
    resized = cv2.resize(
        class_map.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
    )
    coloured = colormap[resized]  # Fancy-indexing into the LUT.
    cv2.addWeighted(frame, 1.0 - alpha, coloured, alpha, 0.0, dst=frame)


# --------------------------------------------------------- text overlays


def draw_text_lines(
    frame: np.ndarray,
    lines: Sequence[str],
    origin: tuple = (10, 30),
    color: tuple = (255, 255, 255),
    bg_color: tuple = (0, 0, 0),
) -> None:
    """Render a stack of text lines with a translucent background.

    Used by the classification demos to print "top-K predictions" in a
    corner of the frame without obscuring too much of the image.
    """
    if not lines:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 1
    line_h = 22
    x, y = origin

    widths = [cv2.getTextSize(line, font, scale, thickness)[0][0] for line in lines]
    box_w = max(widths) + 12
    box_h = line_h * len(lines) + 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 6, y - 22), (x - 6 + box_w, y - 22 + box_h), bg_color, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0.0, dst=frame)

    for i, line in enumerate(lines):
        cv2.putText(
            frame, line, (x, y + i * line_h), font, scale, color, thickness, cv2.LINE_AA
        )


# --------------------------------------------------------- oriented boxes


def draw_obb(
    frame: np.ndarray,
    polygons: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    labels: Sequence[str],
) -> None:
    """Draw rotated bounding boxes as 4-vertex polygons.

    `polygons` shape: (N, 4, 2) holding the four corner points of every
    oriented box, in pixel coordinates of the original frame.
    """
    for poly, score, class_id in zip(polygons, scores, class_ids):
        color = color_for(int(class_id))
        pts = poly.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

        # Anchor the label at the polygon's top-left-most vertex so that
        # rotated boxes still get readable text.
        top_left = poly[np.argmin(poly.sum(axis=1))]
        text = (
            f"{labels[int(class_id)]} {score:.2f}"
            if 0 <= int(class_id) < len(labels)
            else f"{score:.2f}"
        )
        _draw_label(frame, text, int(top_left[0]), int(top_left[1]), color)
