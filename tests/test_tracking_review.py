import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import supervision as sv

from examples.soccer.review_tracking import (
    build_review, fingerprint, identity_label, render, validate_assignments,
)


class TrackingReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.video = self.root / 'tracked.mp4'
        self.analysis = self.root / 'tracks.jsonl'
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*'mp4v'), 10, (160, 100))
        self.assertTrue(writer.isOpened())
        rows = []
        for index in range(6):
            frame = np.zeros((100, 160, 3), dtype=np.uint8)
            frame[20:90, 30:70] = (255, 80, 20)
            writer.write(frame)
            rows.append({'schema': 'player-tracking-v1', 'frame_index': index,
                         'timestamp': index / 10, 'source_timestamp': 300 + index / 10,
                         'fps': 10, 'width': 160, 'height': 100,
                         'detections': [{'tracker_id': 7, 'bbox': [30, 20, 70, 90],
                                         'class_id': 2, 'confidence': 0.9}]})
        writer.release()
        self.analysis.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        self.document = {'schema': 'player-assignments-v1', 'analysis_sha256': fingerprint(self.analysis),
                         'fps': 10, 'frame_count': 6, 'assignments': [
                             {'tracker_id': 7, 'team': 'Blue', 'jersey': '19',
                              'start_frame': 0, 'end_frame': 3},
                             {'tracker_id': 7, 'team': 'Blue', 'jersey': '8',
                              'start_frame': 3, 'end_frame': 6}]}

    def test_build_and_render_real_video(self):
        page = self.root / 'review.html'
        build_review(self.video, self.analysis, page)
        content = page.read_text()
        self.assertIn('"source_time_offset": 300.0', content)
        self.assertIn('"image": "', content)
        self.assertNotIn('__TRACKING_DATA__', content)
        assignments = self.root / 'assignments.json'
        assignments.write_text(json.dumps(self.document))
        output = self.root / 'labelled.mp4'
        render(self.video, self.analysis, assignments, output)
        cap = cv2.VideoCapture(str(output))
        self.assertEqual(cap.get(cv2.CAP_PROP_FRAME_COUNT), 6)
        self.assertTrue(cap.read()[0])
        cap.release()
        with self.assertRaisesRegex(ValueError, 'already exists'):
            render(self.video, self.analysis, assignments, output)

    def test_identity_switch_boundary_and_unknown(self):
        grouped = validate_assignments(self.document, self.analysis, 10, 6)
        self.assertIn('#19', identity_label({'tracker_id': 7}, 2, grouped))
        self.assertIn('#8', identity_label({'tracker_id': 7}, 3, grouped))
        self.assertEqual(identity_label({'tracker_id': 99}, 3, grouped), 'ID 99')

    def test_rejects_overlap_and_wrong_tracking_run(self):
        self.document['assignments'][0]['end_frame'] = 4
        with self.assertRaisesRegex(ValueError, 'overlap'):
            validate_assignments(self.document, self.analysis, 10, 6)
        self.document['analysis_sha256'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'different tracking run'):
            validate_assignments(self.document, self.analysis, 10, 6)

    def test_export_records_empty_frames_and_original_timestamps(self):
        from examples.soccer import main
        frame = np.zeros((100, 160, 3), dtype=np.uint8)
        detections = sv.Detections(xyxy=np.array([[30, 20, 70, 90]], dtype=np.float32),
                                   class_id=np.array([2]), confidence=np.array([0.9]),
                                   tracker_id=np.array([7]))
        export = self.root / 'export.jsonl'
        with patch.object(main, 'YOLO'), patch.object(main.sv, 'ByteTrack') as tracker, \
                patch.object(main.sv, 'get_video_frames_generator', return_value=iter([frame, frame])), \
                patch.object(main.sv.Detections, 'from_ultralytics'):
            tracker.return_value.update_with_detections.side_effect = [detections, sv.Detections.empty()]
            frames = list(main.run_player_tracking(str(self.video), 'cpu', str(export), 300))
        rows = [json.loads(line) for line in export.read_text().splitlines()]
        self.assertEqual(len(frames), 2)
        self.assertEqual(rows[0]['detections'][0]['tracker_id'], 7)
        self.assertEqual(rows[1]['detections'], [])
        self.assertEqual(rows[1]['timestamp'], 0.1)
        self.assertEqual(rows[1]['source_timestamp'], 300.1)


if __name__ == '__main__':
    unittest.main()
