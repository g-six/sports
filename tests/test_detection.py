import unittest

import numpy as np
import supervision as sv

from sports.common.detection import detect_people, normalize_person_detections, tile_starts


def detections(boxes, classes, confidence):
    return sv.Detections(
        xyxy=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        class_id=np.asarray(classes, dtype=int),
        confidence=np.asarray(confidence, dtype=float))


class TiledDetectionTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((100, 160, 3), dtype=np.uint8)

    def test_grid_anchors_last_tile_without_slivers(self):
        self.assertEqual(tile_starts(1920, 960, 240), [0, 720, 960])
        self.assertEqual(tile_starts(1080, 960, 240), [0, 120])
        self.assertEqual(tile_starts(400, 960, 240), [0])
        with self.assertRaises(ValueError):
            tile_starts(100, 100, 100)

    def test_duplicate_person_suppressed_even_when_roles_disagree(self):
        baseline = detections([[80, 20, 96, 70]], [2], [.5])
        results = iter([
            baseline,
            detections([[80, 20, 96, 70]], [3], [.95]),
            detections([[20, 20, 36, 70]], [2], [.8]),
        ])
        result = detect_people(self.frame, lambda _: next(results), True, 100, 40)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.class_id[0], 3)
        np.testing.assert_array_equal(baseline.xyxy, [[80, 20, 96, 70]])

    def test_internal_edge_fragments_removed_but_real_frame_edges_kept(self):
        results = iter([
            sv.Detections.empty(),
            detections([[90, 20, 100, 70]], [2], [.9]),
            detections([[30, 20, 50, 70], [70, 20, 100, 70]], [2, 2], [.9, .8]),
        ])
        result = detect_people(self.frame, lambda _: next(results), True, 100, 40)
        self.assertEqual(len(result), 2)
        np.testing.assert_array_equal(result.xyxy, [[90, 20, 110, 70], [130, 20, 160, 70]])

    def test_small_frames_do_not_get_inferred_twice_and_balls_excluded(self):
        calls = []

        def infer(image):
            calls.append(image.shape)
            return detections([[1, 1, 10, 10], [20, 20, 40, 80]], [0, 2], [.9, .8])

        result = detect_people(self.frame, infer, True, 200, 40)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result.class_id.tolist(), [2])

    def test_empty_tiles_and_full_frame_return_empty_detections(self):
        result = detect_people(self.frame, lambda _: sv.Detections.empty(), True, 100, 40)
        self.assertEqual(len(result), 0)

    def test_general_detector_keeps_only_people_without_mutating_input(self):
        original = detections([[1, 2, 30, 90], [60, 20, 80, 80]], [0, 2], [.8, .9])
        result = normalize_person_detections(original)
        self.assertEqual(result.class_id.tolist(), [2])
        self.assertEqual(original.class_id.tolist(), [0, 2])
        np.testing.assert_array_equal(result.xyxy, [[1, 2, 30, 90]])
        self.assertEqual(result.confidence.tolist(), [.8])

    def test_general_person_class_need_not_be_zero(self):
        original = detections([[1, 2, 30, 90]], [7], [.8])
        self.assertEqual(len(normalize_person_detections(original, 7)), 1)
        self.assertEqual(len(normalize_person_detections(original)), 0)


if __name__ == '__main__':
    unittest.main()
