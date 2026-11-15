"""Check appearance classification of manually labelled point targets."""

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import supervision as sv

from sports.common.uniform import UniformClassifier

if __package__:
    from .evaluate_targets import match_targets
else:
    from evaluate_targets import match_targets


def evaluate(source, report, annotations, uniform_config):
    classifier = UniformClassifier.from_config(uniform_config)
    # Isolate appearance from track history; production still requires multiple
    # observations. Do not interpret this diagnostic as temporal accuracy.
    classifier.min_observations = 1
    capture = cv2.VideoCapture(source)
    results = []
    try:
        for annotated in annotations['frames']:
            row = next(r for r in report['frames']
                       if r['timestamp_seconds'] == annotated['timestamp_seconds']
                       and r['frame_offset'] == annotated['frame_offset'])
            capture.set(cv2.CAP_PROP_POS_MSEC, annotated['timestamp_seconds'] * 1000)
            for _ in range(annotated['frame_offset'] + 1):
                ok, frame = capture.read()
                if not ok:
                    raise ValueError('Could not decode an annotated source frame.')
            matches = match_targets(annotated['targets'], row['person'])
            for i, target in enumerate(annotated['targets']):
                prediction = 'not detected'
                if i in matches:
                    crop = sv.crop_image(frame, row['person'][matches[i]]['xyxy'])
                    classifier.tracks.clear()
                    classifier.begin_frame()
                    prediction = classifier.update(i, crop).name
                expected = target['uniform']
                outcome = ('correct' if prediction == expected else
                           'uncertain' if prediction == 'uncertain' else
                           'missed' if prediction == 'not detected' else 'incorrect')
                results.append({
                    'timestamp_seconds': annotated['timestamp_seconds'],
                    'target': target['id'], 'expected': expected,
                    'predicted': prediction, 'outcome': outcome,
                })
    finally:
        capture.release()
    return {'summary': dict(Counter(row['outcome'] for row in results)), 'targets': results,
            'note': 'Selected single-frame appearance checks, not temporal or general accuracy. References may include these samples; use separate held-out footage for validation.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source_video_path', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--uniform_config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = evaluate(args.source_video_path, json.loads(Path(args.report).read_text()),
                      json.loads(Path(args.annotations).read_text()), args.uniform_config)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result['summary'], indent=2))
