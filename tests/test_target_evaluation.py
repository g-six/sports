import unittest

from examples.soccer.evaluate_targets import evaluate, match_targets


class TargetEvaluationTests(unittest.TestCase):
    def test_one_box_cannot_cover_two_people(self):
        targets = [{'xy': [20, 20]}, {'xy': [30, 20]}]
        boxes = [{'xyxy': [0, 0, 50, 50]}]
        self.assertEqual(len(match_targets(targets, boxes)), 1)

    def test_uncovered_targets_and_empty_inputs(self):
        self.assertEqual(match_targets([{'xy': [80, 80]}], [{'xyxy': [0, 0, 50, 50]}]), {})
        self.assertEqual(match_targets([], []), {})

    def test_missing_frame_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'missing'):
            evaluate({'frames': []}, {'frames': [
                {'timestamp_seconds': 30, 'frame_offset': 0, 'targets': []}]}, ['baseline'])


if __name__ == '__main__':
    unittest.main()
