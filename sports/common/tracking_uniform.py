"""Filter shirt colours before association, preventing cross-team ID matches."""
import cv2
import numpy as np
import supervision as sv


def blue_shirt_fraction(frame: np.ndarray, box: np.ndarray) -> float:
    """Measure blue in the inner upper torso, avoiding sleeves and shorts."""
    x1, y1, x2, y2 = np.asarray(box, dtype=float)
    if not np.isfinite([x1, y1, x2, y2]).all():
        return 0.0
    left, right = max(0, int(x1)), min(frame.shape[1], int(x2))
    top, bottom = max(0, int(y1)), min(frame.shape[0], int(y2))
    crop = frame[top:bottom, left:right]
    if right <= left or bottom <= top or crop.shape[0] < 20 or crop.shape[1] < 8:
        return 0.0
    height, width = crop.shape[:2]
    shirt = crop[int(height * .18):int(height * .48),
                 int(width * .25):int(width * .75)]
    hsv = cv2.cvtColor(shirt, cv2.COLOR_BGR2HSV)
    blue = ((hsv[:, :, 0] >= 90) & (hsv[:, :, 0] <= 135)
            & (hsv[:, :, 1] >= 80) & (hsv[:, :, 2] >= 40))
    return float(blue.mean())


def filter_blue_shirts(
    frame: np.ndarray, detections: sv.Detections, minimum_fraction: float = .12,
) -> sv.Detections:
    """Keep only blue-shirted people; uncertain detections wait for a better view.

    This is a team-colour gate, not player identification. A missed blue shirt
    can break a track, and two blue teammates can still exchange IDs.
    """
    if not np.isfinite(minimum_fraction) or not 0 < minimum_fraction <= 1:
        raise ValueError('Blue shirt minimum fraction must be in (0, 1].')
    keep = np.asarray([blue_shirt_fraction(frame, box) >= minimum_fraction
                       for box in detections.xyxy], dtype=bool)
    if detections.class_id is not None:
        keep &= np.isin(detections.class_id, [1, 2, 3])
    return detections[keep]
