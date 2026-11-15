import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import supervision as sv

from sports.common.video import read_video_info


class VideoTimingTests(unittest.TestCase):
    def test_metadata_preserves_fractional_rate_and_releases_capture(self):
        source_fps = 30000 / 1001
        properties = {cv2.CAP_PROP_FPS: source_fps, cv2.CAP_PROP_FRAME_WIDTH: 160,
                      cv2.CAP_PROP_FRAME_HEIGHT: 100, cv2.CAP_PROP_FRAME_COUNT: 300}
        video = MagicMock()
        video.isOpened.return_value = True
        video.get.side_effect = properties.__getitem__
        with patch('sports.common.video.cv2.VideoCapture', return_value=video):
            info = read_video_info('fractional.mp4')
        self.assertEqual(info.fps, source_fps)
        self.assertEqual(info.total_frames, 300)
        video.release.assert_called_once()

    def test_invalid_or_missing_rate_is_rejected_and_capture_is_released(self):
        for fps in (0, -1, float('nan'), float('inf')):
            video = MagicMock()
            video.isOpened.return_value = True
            properties = {cv2.CAP_PROP_FPS: fps, cv2.CAP_PROP_FRAME_WIDTH: 160,
                          cv2.CAP_PROP_FRAME_HEIGHT: 100, cv2.CAP_PROP_FRAME_COUNT: 300}
            video.get.side_effect = properties.__getitem__
            with patch('sports.common.video.cv2.VideoCapture', return_value=video):
                with self.assertRaisesRegex(ValueError, 'FPS'):
                    read_video_info('invalid.mp4')
            video.release.assert_called_once()
        video = MagicMock()
        video.isOpened.return_value = False
        with patch('sports.common.video.cv2.VideoCapture', return_value=video):
            with self.assertRaisesRegex(ValueError, 'Could not open'):
                read_video_info('missing.mp4')
        video.release.assert_called_once()

    def test_tracking_output_duration_and_export_use_same_fractional_fps(self):
        from examples.soccer import main
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target, analysis = root/'source.mp4', root/'tracked.mp4', root/'tracks.jsonl'
            frame = np.zeros((100, 160, 3), dtype=np.uint8)
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'mp4v'),
                                     30000/1001, (160, 100))
            self.assertTrue(writer.isOpened())
            for _ in range(60):
                writer.write(frame)
            writer.release()
            original = read_video_info(source)
            self.assertGreater(original.fps, 29.9)
            with patch.object(main, 'YOLO'), patch.object(main.sv, 'ByteTrack') as tracker, \
                    patch.object(main.sv.Detections, 'from_ultralytics', return_value=sv.Detections.empty()):
                tracker.return_value.update_with_detections.return_value = sv.Detections.empty()
                main.main(str(source), str(target), 'cpu', main.Mode.PLAYER_TRACKING,
                          no_display=True, analysis_path=str(analysis), source_time_offset=300)
            result = read_video_info(target)
            rows = [json.loads(line) for line in analysis.read_text().splitlines()]
            self.assertEqual(result.total_frames, original.total_frames)
            self.assertAlmostEqual(result.fps, original.fps, places=4)
            self.assertAlmostEqual(result.total_frames/result.fps,
                                   original.total_frames/original.fps, places=4)
            self.assertEqual(len(rows), original.total_frames)
            self.assertTrue(all(row['fps'] == original.fps for row in rows))
            self.assertAlmostEqual(rows[-1]['timestamp'], 59/original.fps)
            self.assertAlmostEqual(rows[-1]['source_timestamp'], 300 + 59/original.fps)
            tracker.assert_called_once_with(frame_rate=original.fps, minimum_consecutive_frames=3)


if __name__ == '__main__':
    unittest.main()
