"""Measure coverage of manually marked visible people, without claiming mAP."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def match_targets(targets, detections):
    """One-to-one assignment: a detection can cover at most one marked person."""
    if not targets or not detections:
        return {}
    costs = np.full((len(targets), len(detections)), 1e6, dtype=float)
    for i, target in enumerate(targets):
        x, y = target['xy']
        for j, detection in enumerate(detections):
            x1, y1, x2, y2 = detection['xyxy']
            if x1 <= x <= x2 and y1 <= y <= y2:
                # Prefer a centered match when overlapping boxes cover a target.
                costs[i, j] = ((x - (x1 + x2) / 2) / max(x2 - x1, 1)) ** 2 + (
                    (y - (y1 + y2) / 2) / max(y2 - y1, 1)) ** 2
    rows, columns = linear_sum_assignment(costs)
    return {int(i): int(j) for i, j in zip(rows, columns) if costs[i, j] < 1e6}


def evaluate(report, annotations, methods):
    results = []
    for annotated in annotations['frames']:
        sample = next((row for row in report['frames']
                       if row['timestamp_seconds'] == annotated['timestamp_seconds']
                       and row['frame_offset'] == annotated['frame_offset']), None)
        if sample is None:
            raise ValueError('An annotated frame is missing from the detection report.')
        for method in methods:
            matches = match_targets(annotated['targets'], sample[method])
            results.append({
                'timestamp_seconds': annotated['timestamp_seconds'], 'method': method,
                'covered': len(matches), 'total': len(annotated['targets']),
                'targets': [{**target, 'covered': index in matches}
                            for index, target in enumerate(annotated['targets'])],
            })
    summary = {}
    for method in methods:
        rows = [row for row in results if row['method'] == method]
        summary[method] = {
            'covered': sum(row['covered'] for row in rows),
            'total': sum(row['total'] for row in rows),
        }
    return {'summary': summary, 'frames': results, 'note': (
        'Coverage of selected point targets only. This does not measure full recall, '
        'precision, box localization quality, or role-classification accuracy.')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True)
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--methods', nargs='+', default=['baseline', 'person'])
    args = parser.parse_args()
    result = evaluate(json.loads(Path(args.report).read_text()),
                      json.loads(Path(args.annotations).read_text()), args.methods)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result['summary'], indent=2))
