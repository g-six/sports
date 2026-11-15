"""Overlapping person detection tiles merged with a full-frame prediction."""

from typing import Callable

import numpy as np
import supervision as sv


PERSON_CLASS_IDS = (1, 2, 3)


def normalize_person_detections(
    detections: sv.Detections, person_class_id: int = 0,
) -> sv.Detections:
    """Adapt a general detector to the existing person-processing pipeline.

    Internal class 2 denotes a person candidate here, not a confirmed player.
    Uniform classification must independently determine their team and role.
    """
    people = detections[detections.class_id == person_class_id]
    people.class_id = np.full(len(people), 2, dtype=int)
    return people


def tile_starts(length: int, size: int, overlap: int):
    """Anchor the last tile to the image edge, avoiding narrow final slivers."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Tile size must be positive and overlap smaller than size.")
    last = max(0, length - size)
    return sorted(set(list(range(0, last + 1, size - overlap)) + [last]))


def detect_people(
    frame: np.ndarray,
    infer: Callable[[np.ndarray], sv.Detections],
    tiled: bool = False,
    tile_size: int = 960,
    overlap: int = 240,
    nms_threshold: float = .5,
) -> sv.Detections:
    """Preserve full-frame context and add higher-detail overlapping crops.

    The callback must return crop-local coordinates. Suppress incomplete people
    touching an internal tile edge; neighbours and the full frame cover them.
    NMS ignores role IDs because a person can be misclassified across tiles.
    """
    starts_x = tile_starts(frame.shape[1], tile_size, overlap)
    starts_y = tile_starts(frame.shape[0], tile_size, overlap)

    def people(image):
        detections = infer(image)
        return detections[np.isin(detections.class_id, PERSON_CLASS_IDS)]

    full = people(frame)
    if not tiled or (frame.shape[0] <= tile_size and frame.shape[1] <= tile_size):
        return full
    groups = [full]
    height, width = frame.shape[:2]
    for y in starts_y:
        for x in starts_x:
            crop = frame[y:y + tile_size, x:x + tile_size]
            detections = people(crop)
            boxes = detections.xyxy
            keep = np.ones(len(detections), dtype=bool)
            if x > 0:
                keep &= boxes[:, 0] > 3
            if y > 0:
                keep &= boxes[:, 1] > 3
            if x + tile_size < width:
                keep &= boxes[:, 2] < crop.shape[1] - 3
            if y + tile_size < height:
                keep &= boxes[:, 3] < crop.shape[0] - 3
            detections = detections[keep]
            detections.xyxy += np.array([x, y, x, y], dtype=np.float32)
            groups.append(detections)
    return sv.Detections.merge(groups).with_nms(
        threshold=nms_threshold, class_agnostic=True)
