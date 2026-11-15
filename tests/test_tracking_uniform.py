import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import supervision as sv

from sports.common.tracking_uniform import blue_shirt_fraction, filter_blue_shirts


def people(boxes):
    return sv.Detections(xyxy=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
                         confidence=np.full(len(boxes), .95), class_id=np.full(len(boxes), 2))


class BlueTrackingTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((220, 260, 3), (40, 120, 40), dtype=np.uint8)
        self.blue = [10, 10, 90, 210]
        self.white = [140, 10, 220, 210]
        self.frame[10:210, 10:90] = (210, 80, 20)
        self.frame[10:210, 140:220] = (245, 245, 245)
        # White sleeves and a large white number still leave blue shirt fabric.
        self.frame[20:100, 10:28] = 245
        self.frame[20:100, 72:90] = 245
        self.frame[48:78, 40:62] = 245

    def test_blue_with_white_sleeves_and_digits_survives_white_opponent_does_not(self):
        detections = people([self.blue, self.white])
        result = filter_blue_shirts(self.frame, detections)
        self.assertEqual(len(result), 1)
        np.testing.assert_array_equal(result.xyxy[0], self.blue)
        self.assertEqual(blue_shirt_fraction(self.frame, self.white), 0)
        # Hue is stable under a uniform change in exposure.
        shadow = (self.frame.astype(float) * .5).astype(np.uint8)
        self.assertEqual(len(filter_blue_shirts(shadow, detections)), 1)

    def test_no_blue_evidence_is_not_associated_during_white_occlusion(self):
        tracker = sv.ByteTrack(minimum_consecutive_frames=1)
        raw = people([self.blue])
        prior = []
        for _ in range(5):
            result = tracker.update_with_detections(filter_blue_shirts(self.frame, raw))
            if len(result):
                prior.append(int(result.tracker_id[0]))
        self.assertTrue(prior)
        white = self.frame.copy()
        white[10:210, 10:90] = 245
        for _ in range(2):
            result = tracker.update_with_detections(filter_blue_shirts(white, raw))
            self.assertEqual(len(result), 0)
        result = tracker.update_with_detections(filter_blue_shirts(self.frame, raw))
        self.assertEqual(int(result.tracker_id[0]), prior[-1])

    def test_invalid_small_and_non_person_boxes_do_not_pass(self):
        self.assertEqual(blue_shirt_fraction(self.frame, np.array([-100, -100, -10, -10])), 0)
        self.assertEqual(blue_shirt_fraction(self.frame, np.array([10, 10, 15, 15])), 0)
        self.assertEqual(blue_shirt_fraction(self.frame, np.array([np.nan, 0, 50, 60])), 0)
        raw = people([self.blue])
        raw.class_id[0] = 0
        self.assertEqual(len(filter_blue_shirts(self.frame, raw)), 0)
        for threshold in (0, -1, float('nan'), 2):
            with self.assertRaises(ValueError):
                filter_blue_shirts(self.frame, raw, threshold)

    def test_player_tracking_filters_before_association(self):
        from examples.soccer import main
        raw = people([self.blue, self.white])
        received = []

        def update(detections):
            received.append(detections.xyxy.copy())
            detections.tracker_id = np.arange(len(detections)) + 1
            return detections

        with patch.object(main, 'YOLO'), patch.object(main.sv, 'ByteTrack') as tracker, \
                patch.object(main, 'read_video_info',
                             return_value=SimpleNamespace(fps=29, width=260, height=220)), \
                patch.object(main.sv, 'get_video_frames_generator', return_value=iter([self.frame])), \
                patch.object(main.sv.Detections, 'from_ultralytics', return_value=raw):
            tracker.return_value.update_with_detections.side_effect = update
            with tempfile.TemporaryDirectory() as directory:
                report = Path(directory)/'tracking.jsonl'
                list(main.run_player_tracking('test.mp4', 'cpu', str(report), tracking_team='blue'))
                row = json.loads(report.read_text())
        self.assertEqual(len(received), 1)
        np.testing.assert_array_equal(received[0], [self.blue])
        self.assertEqual(len(row['detections']), 1)
        self.assertEqual(len(row['all_person_boxes']), 2)
        np.testing.assert_array_equal(row['all_person_boxes'][1], self.white)


if __name__ == '__main__':
    unittest.main()
