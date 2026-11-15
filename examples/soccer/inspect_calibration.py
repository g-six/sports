"""Produce unverified pitch-landmark overlays for manual calibration review."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from sports.configs.soccer import SoccerPitchConfiguration


def inspect(source, output_dir, timestamps):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(Path(__file__).parent / 'data/football-pitch-detection.pt'))
    config = SoccerPitchConfiguration()
    capture = cv2.VideoCapture(source)
    rows = []
    try:
        for timestamp in timestamps:
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f'Cannot read source at {timestamp} seconds')
            result = model(frame, verbose=False, device='cpu')[0]
            xy = np.empty((0, 2))
            confidence = np.empty(0)
            if result.keypoints is not None and len(result.keypoints.xy):
                xy = result.keypoints.xy[0].cpu().numpy()
                confidence = result.keypoints.conf[0].cpu().numpy()
            good = (confidence > .5) & (xy[:, 0] > 1) & (xy[:, 1] > 1)
            annotated = frame.copy()
            for index in np.flatnonzero(good):
                x, y = np.round(xy[index]).astype(int)
                cv2.circle(annotated, (x, y), 5, (0, 255, 255), -1)
                cv2.putText(annotated, f'{config.labels[index]} {confidence[index]:.2f}',
                            (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, .45,
                            (0, 255, 255), 1, cv2.LINE_AA)
            row = {'timestamp_seconds': timestamp, 'status': 'unverified',
                   'confident_landmarks': int(good.sum()),
                   'landmarks': [{'index': i, 'label': config.labels[i],
                                  'xy': point.tolist(), 'confidence': float(confidence[i])}
                                 for i, point in enumerate(xy)]}
            if good.sum() >= 4 and len(xy) == len(config.vertices):
                # Fit pitch->image so the RANSAC tolerance is in image pixels.
                matrix, inliers = cv2.findHomography(
                    np.array(config.vertices, dtype=np.float32)[good], xy[good],
                    cv2.RANSAC, 5)
                if matrix is not None and inliers is not None and inliers.sum() >= 4:
                    row['ransac_inliers'] = int(inliers.sum())
                    projected = cv2.perspectiveTransform(
                        np.array(config.vertices, dtype=np.float32).reshape(-1, 1, 2), matrix).reshape(-1, 2)
                    for a, b in config.edges:
                        pair = projected[[a - 1, b - 1]]
                        if np.isfinite(pair).all() and np.abs(pair).max() < 1e6:
                            cv2.line(annotated, tuple(pair[0].astype(int)),
                                     tuple(pair[1].astype(int)), (0, 180, 255), 2)
                    row['pitch_to_image'] = matrix.tolist()
            cv2.imwrite(str(output / f'landmarks-{timestamp:g}s.jpg'), annotated)
            rows.append(row)
            print(f"{timestamp}s: {row['confident_landmarks']} confident landmarks; "
                  f"{row.get('ransac_inliers', 0)} fit inliers; unverified", flush=True)
        (output / 'report.json').write_text(json.dumps({
            'frames': rows,
            'assumptions': 'Repository pitch template: 120m x 70m, penalty area depth 20.15m. Actual pitch dimensions and model landmark identities are not verified.',
            'note': 'Low fitting error or high confidence does not establish correct field calibration. Inspect alignment with football markings. Do not use for metric speed or distance.'}, indent=2))
    finally:
        capture.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source_video_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--timestamps', nargs='+', type=float, default=[300, 1200, 2400, 3600])
    args = parser.parse_args()
    inspect(args.source_video_path, args.output_dir, args.timestamps)
