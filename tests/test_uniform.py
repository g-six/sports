import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import supervision as sv

from sports.common.uniform import (
    UniformClassifier, UniformReference, shirt_descriptor,
)


def person(color):
    return np.full((100, 40, 3), color, dtype=np.uint8)


class UniformClassifierTests(unittest.TestCase):
    def setUp(self):
        self.blue = person((180, 25, 15))
        self.yellow = person((20, 220, 220))
        self.references = [
            UniformReference('blue team', 'player', np.array([shirt_descriptor(self.blue)])),
            UniformReference('yellow referee', 'referee', np.array([shirt_descriptor(self.yellow)])),
        ]
        self.classifier = UniformClassifier(self.references)

    def observe(self, crop, track=1, count=1):
        for _ in range(count):
            self.classifier.begin_frame()
            prediction = self.classifier.update(track, crop)
        return prediction

    def test_requires_several_observations_and_keeps_tracks_separate(self):
        self.assertIsNone(self.observe(self.blue, count=2).role)
        self.assertEqual(self.observe(self.blue).role, 'player')
        self.assertIsNone(self.observe(self.yellow, track=2).role)
        self.assertEqual(self.observe(self.yellow, track=2, count=2).role, 'referee')

    def test_one_bad_frame_does_not_flip_role_but_sustained_evidence_does(self):
        self.observe(self.blue, count=5)
        self.assertEqual(self.observe(self.yellow).role, 'player')
        self.assertEqual(self.observe(self.yellow, count=20).role, 'referee')

    def test_ambiguous_or_unmatched_uniform_is_uncertain(self):
        ambiguous = self.blue.copy()
        ambiguous[:, :20] = self.yellow[:, :20]
        self.assertIsNone(self.observe(ambiguous, count=8).role)
        self.assertIsNone(self.observe(person((255, 255, 255)), track=2, count=8).role)

    def test_short_empty_or_untracked_crops_are_uncertain(self):
        for crop in [np.empty((0, 0, 3), dtype=np.uint8), self.blue[:10]]:
            self.assertIsNone(self.observe(crop, count=4).role)
        self.assertIsNone(self.observe(self.blue, track=-1, count=4).role)
        self.assertNotIn(-1, self.classifier.tracks)

    def test_old_track_expires(self):
        self.observe(self.blue, count=3)
        for _ in range(151):
            self.classifier.begin_frame()
        self.assertNotIn(1, self.classifier.tracks)
        self.assertIsNone(self.observe(self.yellow).role)

    def test_descriptor_uses_shirt_and_tolerates_coloured_fabric_shading(self):
        with_different_shorts = self.blue.copy()
        with_different_shorts[60:] = (0, 255, 255)
        np.testing.assert_array_equal(
            shirt_descriptor(self.blue), shirt_descriptor(with_different_shorts))
        np.testing.assert_array_equal(
            shirt_descriptor(self.blue), shirt_descriptor((self.blue * .6).astype(np.uint8)))

    def test_config_resolves_images_relative_to_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cv2.imwrite(str(root / 'blue.png'), self.blue)
            cv2.imwrite(str(root / 'yellow.png'), self.yellow)
            config = {'uniforms': [
                {'name': 'blue', 'role': 'player', 'images': ['blue.png']},
                {'name': 'yellow', 'role': 'referee', 'images': ['yellow.png']},
            ]}
            path = root / 'config.json'
            path.write_text(json.dumps(config))
            classifier = UniformClassifier.from_config(str(path))
            self.assertEqual(len(classifier.references), 2)
            config['uniforms'][0]['images'] = []
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'No reference images'):
                UniformClassifier.from_config(str(path))

    def test_video_pipeline_corrects_roles_and_preserves_untracked_people(self):
        from examples.soccer.main import run_uniform_classification

        frame = np.full((100, 100, 3), (180, 25, 15), dtype=np.uint8)
        model = MagicMock()
        model.to.return_value = model
        model.return_value = [SimpleNamespace(names={3: 'referee'})]

        def detections(_):
            return sv.Detections(
                xyxy=np.array([[0, 0, 40, 100], [60, 0, 100, 100]], dtype=np.float32),
                confidence=np.array([.9, .26]), class_id=np.array([3, 3]))

        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / 'analysis.jsonl'
            with patch('examples.soccer.main.YOLO', return_value=model), \
                    patch('examples.soccer.main.UniformClassifier.from_config', return_value=self.classifier), \
                    patch('examples.soccer.main.read_video_info', return_value=SimpleNamespace(fps=30)), \
                    patch('examples.soccer.main.sv.get_video_frames_generator', return_value=iter([frame] * 4)), \
                    patch('examples.soccer.main.sv.Detections.from_ultralytics', side_effect=detections):
                output = list(run_uniform_classification('unused', 'cpu', 'unused', str(report)))
            rows = [json.loads(line) for line in report.read_text().splitlines()]
        self.assertEqual(len(output), 4)
        self.assertTrue(all(len(row['predictions']) == 2 for row in rows))
        tracked, untracked = rows[-1]['predictions']
        self.assertEqual(tracked['detector_role'], 'referee')
        self.assertEqual(tracked['role'], 'player')
        self.assertEqual(untracked['tracker_id'], -1)
        self.assertIsNone(untracked['role'])

    def test_general_person_pipeline_does_not_claim_detector_knows_roles(self):
        from examples.soccer.main import get_person_class_id, run_uniform_classification

        model = MagicMock()
        model.to.return_value = model
        model.names = {0: 'person', 2: 'car'}
        model.return_value = [SimpleNamespace()]
        frame = np.full((100, 100, 3), (180, 25, 15), dtype=np.uint8)

        def detections(_):
            return sv.Detections(
                xyxy=np.array([[0, 0, 40, 100], [60, 0, 100, 100]], dtype=np.float32),
                confidence=np.array([.9, .9]), class_id=np.array([0, 2]))

        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / 'analysis.jsonl'
            with patch('examples.soccer.main.YOLO', return_value=model), \
                    patch('examples.soccer.main.UniformClassifier.from_config', return_value=self.classifier), \
                    patch('examples.soccer.main.read_video_info', return_value=SimpleNamespace(fps=30)), \
                    patch('examples.soccer.main.sv.get_video_frames_generator', return_value=iter([frame] * 4)), \
                    patch('examples.soccer.main.sv.Detections.from_ultralytics', side_effect=detections):
                output = list(run_uniform_classification(
                    'unused', 'cpu', 'unused', str(report), person_model='general.pt'))
            rows = [json.loads(line) for line in report.read_text().splitlines()]
        self.assertEqual(len(output), 4)
        self.assertTrue(all(len(row['predictions']) == 1 for row in rows))
        self.assertEqual(rows[-1]['predictions'][0]['detector_role'], 'person')
        self.assertEqual(rows[-1]['predictions'][0]['role'], 'player')
        self.assertEqual(model.call_args.kwargs['classes'], [0])
        self.assertEqual(get_person_class_id(SimpleNamespace(names={7: 'person'})), 7)
        with self.assertRaisesRegex(ValueError, 'class named'):
            get_person_class_id(SimpleNamespace(names={0: 'ball'}))

    def test_motion_reset_starts_new_ids_and_clears_uniform_history(self):
        from examples.soccer.main import run_uniform_classification
        from sports.common.motion import MotionEstimate

        frame = np.full((100, 100, 3), (180, 25, 15), dtype=np.uint8)
        raw = sv.Detections(xyxy=np.array([[0, 0, 40, 100]], dtype=np.float32),
                            confidence=np.array([.9]), class_id=np.array([2]))
        motion = MagicMock()
        motion.update.side_effect = [
            MotionEstimate(np.eye(3), True), MotionEstimate(np.eye(3), False, reset=True),
            MotionEstimate(np.eye(3), True), MotionEstimate(np.eye(3), True)]
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / 'analysis.jsonl'
            with patch('examples.soccer.main.YOLO', side_effect=AssertionError('Replay must not run inference')), \
                    patch('examples.soccer.main.UniformClassifier.from_config', return_value=self.classifier), \
                    patch('examples.soccer.main.CameraMotionEstimator', return_value=motion), \
                    patch('examples.soccer.main.read_video_info', return_value=SimpleNamespace(fps=30)), \
                    patch('examples.soccer.main.sv.get_video_frames_generator', return_value=iter([frame] * 4)):
                list(run_uniform_classification(
                    'unused', 'cpu', 'unused', str(report), person_model='cached',
                    compensate_camera_motion=True, raw_frames=iter([raw] * 4)))
            rows = [json.loads(line) for line in report.read_text().splitlines()]
        self.assertNotEqual(rows[0]['predictions'][0]['tracker_id'], rows[1]['predictions'][0]['tracker_id'])
        self.assertIsNone(rows[1]['predictions'][0]['role'])
        self.assertEqual(rows[3]['predictions'][0]['role'], 'player')
        np.testing.assert_array_equal(raw.xyxy, [[0, 0, 40, 100]])


if __name__ == '__main__':
    unittest.main()
