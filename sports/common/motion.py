"""Estimate pan/zoom from background features for short-term image stabilization."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class MotionEstimate:
    current_to_reference: np.ndarray
    valid: bool
    inlier_ratio: float = 0.0
    feature_count: int = 0
    reset: bool = False


class CameraMotionEstimator:
    """Track background corners, excluding detected people, and fit an affine map.

    This estimates image motion only. It is not metric pitch calibration, and a
    ball-following camera can contain parallax or non-affine distortion. Failed
    estimates request a new tracking epoch rather than reusing an invalid map.
    """

    def __init__(self, processing_width: int = 640):
        self.processing_width = processing_width
        self.previous_gray = None
        self.previous_mask = None
        self.reference_to_current = np.eye(3, dtype=np.float64)
        self.previous_shape = None

    def update(self, frame: np.ndarray, people_xyxy: np.ndarray) -> MotionEstimate:
        height, width = frame.shape[:2]
        scale = min(1.0, self.processing_width / width)
        gray = cv2.cvtColor(cv2.resize(frame, (
            max(1, round(width * scale)), max(1, round(height * scale)))), cv2.COLOR_BGR2GRAY)
        mask = np.full(gray.shape, 255, dtype=np.uint8)
        for box in people_xyxy:
            x1, y1, x2, y2 = np.round(box * scale).astype(int)
            cv2.rectangle(mask, (max(0, x1 - 5), max(0, y1 - 5)),
                          (x2 + 5, y2 + 5), 0, -1)

        result = MotionEstimate(np.eye(3), valid=True)
        if self.previous_gray is not None:
            result = MotionEstimate(np.eye(3), valid=False, reset=True)
            if self.previous_shape == frame.shape:
                result = self._estimate(gray, mask, scale)
            if not result.valid:
                self.reference_to_current = np.eye(3, dtype=np.float64)
        self.previous_gray, self.previous_mask = gray, mask
        self.previous_shape = frame.shape
        return result

    def _estimate(self, gray, mask, scale):
        failed = MotionEstimate(np.eye(3), valid=False, reset=True)
        points = cv2.goodFeaturesToTrack(
            self.previous_gray, maxCorners=300, qualityLevel=.01,
            minDistance=8, mask=self.previous_mask)
        if points is None or len(points) < 20:
            return failed
        following, status, _ = cv2.calcOpticalFlowPyrLK(
            self.previous_gray, gray, points, None, winSize=(21, 21), maxLevel=3)
        if following is None:
            return failed
        backward, back_status, _ = cv2.calcOpticalFlowPyrLK(
            gray, self.previous_gray, following, None, winSize=(21, 21), maxLevel=3)
        if backward is None:
            return failed
        source = points.reshape(-1, 2)
        target = following.reshape(-1, 2)
        keep = (status.ravel() == 1) & (back_status.ravel() == 1)
        keep &= np.linalg.norm(source - backward.reshape(-1, 2), axis=1) < 1
        finite = np.isfinite(target).all(axis=1)
        keep &= finite
        positions = np.round(np.nan_to_num(target)).astype(int)
        inside = ((positions[:, 0] >= 0) & (positions[:, 0] < gray.shape[1])
                  & (positions[:, 1] >= 0) & (positions[:, 1] < gray.shape[0]))
        keep &= inside
        valid_indices = np.flatnonzero(keep)
        keep[valid_indices] &= mask[positions[valid_indices, 1], positions[valid_indices, 0]] != 0
        if keep.sum() < 20:
            return failed
        affine, inliers = cv2.estimateAffinePartial2D(
            source[keep], target[keep], method=cv2.RANSAC,
            ransacReprojThreshold=2, maxIters=1000, confidence=.99)
        if affine is None or inliers is None:
            return failed
        ratio = float(inliers.mean())
        zoom = float(np.linalg.norm(affine[0, :2]))
        angle = abs(float(np.degrees(np.arctan2(affine[1, 0], affine[0, 0]))))
        if (ratio < .6 or not .9 < zoom < 1.1 or angle > 5
                or abs(affine[0, 2]) > gray.shape[1] * .25
                or abs(affine[1, 2]) > gray.shape[0] * .25):
            return failed
        transform = np.eye(3)
        transform[:2] = affine
        transform[:2, 2] /= scale
        candidate = transform @ self.reference_to_current
        if not np.isfinite(candidate).all() or np.linalg.cond(candidate[:2, :2]) > 10:
            return failed
        self.reference_to_current = candidate
        return MotionEstimate(np.linalg.inv(candidate), valid=True,
                              inlier_ratio=ratio, feature_count=int(keep.sum()))


def transform_boxes(boxes: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Transform all four corners; preserve the axis-aligned enclosing rectangle."""
    if len(boxes) == 0:
        return boxes.copy()
    corners = np.stack([boxes[:, [0, 1]], boxes[:, [2, 1]],
                        boxes[:, [2, 3]], boxes[:, [0, 3]]], axis=1).astype(np.float32)
    transformed = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), matrix).reshape(-1, 4, 2)
    return np.concatenate([transformed.min(axis=1), transformed.max(axis=1)], axis=1)
