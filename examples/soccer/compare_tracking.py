"""Replay matching saved detections to isolate camera compensation from inference."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
import supervision as sv

if __package__:
    from .main import run_uniform_classification
else:
    from main import run_uniform_classification


def tracking_summary(rows):
    histories = defaultdict(list)
    untracked, observations = 0, 0
    for row in rows:
        for prediction in row['predictions']:
            observations += 1
            if prediction['tracker_id'] < 0:
                untracked += 1
            else:
                if prediction['role']:
                    histories[prediction['tracker_id']].append(prediction['uniform'])
                else:
                    histories[prediction['tracker_id']]
    return {
        'unique_track_ids': len(histories),
        'uniform_label_changes': sum(
            sum(a != b for a, b in zip(history, history[1:]))
            for history in histories.values()),
        'untracked_observations': untracked, 'person_observations': observations,
    }


def replay(source, cached_analysis, uniform_config, output_dir):
    rows = [json.loads(line) for line in Path(cached_analysis).read_text().splitlines()]
    info = sv.VideoInfo.from_video_path(source)
    if len(rows) != info.total_frames or any(row['frame'] != i for i, row in enumerate(rows)):
        raise ValueError('Saved detections must cover every source frame in order.')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    def raw_frames():
        for row in rows:
            values = row['raw_detections']
            yield sv.Detections(
                xyxy=np.array([p['xyxy'] for p in values], dtype=np.float32).reshape(-1, 4),
                confidence=np.array([p['confidence'] for p in values]),
                class_id=np.array([p['class_id'] for p in values], dtype=int))

    start = perf_counter()
    # Replay both variants with the same current references. The saved analysis
    # may have been generated before reference images or thresholds changed.
    analyses = {}
    for name, compensated in [('baseline', False), ('stabilized', True)]:
        analysis = output / (name + '.jsonl')
        frames = run_uniform_classification(
            source, 'cpu', uniform_config, str(analysis), person_model='cached-person',
            compensate_camera_motion=compensated, raw_frames=raw_frames())
        with sv.VideoSink(str(output / (name + '.mp4')), info) as sink:
            for frame in frames:
                sink.write_frame(frame)
        analyses[name] = [json.loads(line) for line in analysis.read_text().splitlines()]
    elapsed = perf_counter() - start
    stabilized = analyses['stabilized']
    summary = {
        'baseline': tracking_summary(analyses['baseline']), 'stabilized': tracking_summary(stabilized),
        'replay_seconds': elapsed,
        'motion_resets': sum(bool(row['camera_motion']['reset']) for row in stabilized),
        'frames': len(stabilized),
        'note': 'Same detections and source frames. Fewer IDs or label changes alone do not prove correct identities. This is image stabilization, not pitch calibration.',
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source_video_path', required=True)
    parser.add_argument('--cached_analysis', required=True)
    parser.add_argument('--uniform_config', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()
    replay(args.source_video_path, args.cached_analysis, args.uniform_config, args.output_dir)
