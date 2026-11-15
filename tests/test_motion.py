import unittest

import cv2
import numpy as np

from sports.common.motion import CameraMotionEstimator, transform_boxes


class CameraMotionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        gray = rng.integers(0, 256, (240, 400), dtype=np.uint8)
        self.frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        self.empty = np.empty((0, 4), dtype=np.float32)

    def test_recovers_translation_and_accumulates_in_reference_coordinates(self):
        estimator = CameraMotionEstimator()
        first = estimator.update(self.frame, self.empty)
        self.assertTrue(first.valid)
        for dx, dy in [(6, -3), (12, -6)]:
            transform = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float64)
            moved = cv2.warpAffine(self.frame, transform, (400, 240))
            estimate = estimator.update(moved, self.empty)
            self.assertTrue(estimate.valid)
            np.testing.assert_allclose(
                estimate.current_to_reference[:2], cv2.invertAffineTransform(transform), atol=.15)

    def test_recovers_small_zoom_and_rotation(self):
        estimator = CameraMotionEstimator()
        estimator.update(self.frame, self.empty)
        transform = cv2.getRotationMatrix2D((200, 120), .5, 1.02)
        moved = cv2.warpAffine(self.frame, transform, (400, 240))
        estimate = estimator.update(moved, self.empty)
        self.assertTrue(estimate.valid)
        np.testing.assert_allclose(
            estimate.current_to_reference[:2], cv2.invertAffineTransform(transform), atol=.35)

    def test_people_are_excluded_from_feature_mask(self):
        estimator = CameraMotionEstimator()
        estimator.update(self.frame, np.array([[100, 50, 200, 150]], dtype=np.float32))
        self.assertTrue(np.all(estimator.previous_mask[45:156, 95:206] == 0))
        self.assertEqual(estimator.previous_mask[10, 10], 255)

    def test_featureless_frames_and_resolution_changes_request_reset(self):
        estimator = CameraMotionEstimator()
        blank = np.zeros_like(self.frame)
        estimator.update(blank, self.empty)
        failed = estimator.update(blank, self.empty)
        self.assertFalse(failed.valid)
        self.assertTrue(failed.reset)
        changed = estimator.update(self.frame[:, :300], self.empty)
        self.assertTrue(changed.reset)

    def test_boxes_use_all_four_corners_and_empty_inputs_work(self):
        matrix = np.array([[0, -1, 100], [1, 0, 10], [0, 0, 1]], dtype=float)
        result = transform_boxes(np.array([[10, 20, 30, 40]], dtype=np.float32), matrix)
        np.testing.assert_allclose(result, [[60, 20, 80, 40]])
        self.assertEqual(transform_boxes(self.empty, matrix).shape, (0, 4))


if __name__ == '__main__':
    unittest.main()
