"""Compare full-frame and tiled person detection on short samples of a video."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from sports.common.detection import detect_people, normalize_person_detections


def serialize(detections):
    return [
        {"xyxy": box.tolist(), "class_id": int(role), "confidence": float(conf)}
        for box, role, conf in zip(
            detections.xyxy, detections.class_id, detections.confidence)
    ]


def panel(frame, detections, title):
    annotated = frame.copy()
    for index, (box, confidence) in enumerate(zip(detections.xyxy, detections.confidence)):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(annotated, f"{index + 1}: {confidence:.2f}",
                    (x1, max(16, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    .45, (0, 255, 255), 1, cv2.LINE_AA)
    annotated = cv2.resize(annotated, (960, 540))
    annotated = cv2.copyMakeBorder(
        annotated, 40, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
    cv2.putText(annotated, title, (12, 27), cv2.FONT_HERSHEY_SIMPLEX,
                .65, (255, 255, 255), 1, cv2.LINE_AA)
    return annotated


def compare(source, output, timestamps, frames, device, person_model=None):
    if frames < 1 or any(timestamp < 0 for timestamp in timestamps):
        raise ValueError("Use positive frame counts and non-negative timestamps.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model_path = Path(__file__).parent / 'data/football-player-detection.pt'
    model = YOLO(str(model_path)).to(device=device)
    candidate_model = YOLO(person_model).to(device=device) if person_model else None
    person_id = None
    if candidate_model is not None:
        person_id = next((key for key, name in candidate_model.names.items() if name == 'person'), None)
        if person_id is None:
            raise ValueError("The general model must have a class named 'person'.")
    candidate_key = 'person' if person_model else 'tiled'
    candidate_title = 'general person' if person_model else 'tiled'

    def infer(image):
        result = model(image, imgsz=1280, conf=.25, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    def infer_person(image):
        result = candidate_model(image, imgsz=1280, conf=.25, classes=[person_id], verbose=False)[0]
        return normalize_person_detections(sv.Detections.from_ultralytics(result), person_id)

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise ValueError(f"Could not open {source}")
    rows, comparisons = [], []
    try:
        for timestamp in timestamps:
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            for offset in range(frames):
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f"Could not read sample at {timestamp}s + {offset} frames")
                # Warm up before timing the first sample.
                if not rows:
                    infer(frame)
                    if candidate_model is not None:
                        infer_person(frame)
                start = perf_counter()
                baseline = detect_people(frame, infer)
                baseline_seconds = perf_counter() - start
                start = perf_counter()
                # Reuse the same full-frame result; charge its measured cost to
                # both methods. Each tile still gets a fresh model inference.
                first = True

                def infer_with_cached_full(image):
                    nonlocal first
                    if first:
                        first = False
                        return baseline
                    return infer(image)

                if candidate_model is not None:
                    candidate = infer_person(frame)
                    candidate_seconds = perf_counter() - start
                else:
                    candidate = detect_people(frame, infer_with_cached_full, tiled=True)
                    candidate_seconds = perf_counter() - start + baseline_seconds
                row = {
                    'timestamp_seconds': timestamp, 'frame_offset': offset,
                    'baseline_seconds': baseline_seconds, candidate_key + '_seconds': candidate_seconds,
                    'baseline': serialize(baseline), candidate_key: serialize(candidate),
                }
                rows.append(row)
                print(f"{timestamp}s +{offset}: {len(baseline)} -> {len(candidate)} boxes; "
                      f"{baseline_seconds:.2f}s -> {candidate_seconds:.2f}s", flush=True)
                if offset == frames // 2:
                    image = np.hstack([
                        panel(frame, baseline, f"{timestamp}s: original ({len(baseline)})"),
                        panel(frame, candidate, f"{timestamp}s: {candidate_title} ({len(candidate)})"),
                    ])
                    cv2.imwrite(str(output / f'comparison-{timestamp:g}s.jpg'), image)
                    comparisons.append(image)
        report = {
            'source': str(Path(source).resolve()), 'frames': rows,
            'settings': {'imgsz': 1280, 'confidence': .25, **(
                {} if person_model else {'tile_size': 960, 'overlap': 240})},
            'candidate': candidate_title,
            'baseline_model': str(model_path.resolve()),
            'candidate_model': str(Path(person_model).resolve()) if person_model else str(model_path.resolve()),
            'note': 'Detection counts are not recall or accuracy. Inspect images for misses and duplicates.',
        }
        (output / 'report.json').write_text(json.dumps(report, indent=2))
        cv2.imwrite(str(output / 'comparison-sheet.jpg'), np.vstack(comparisons))
    finally:
        capture.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source_video_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--timestamps', nargs='+', type=float, default=[300, 1200, 2400])
    parser.add_argument('--frames', type=int, default=3)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--person_model', help='Compare a general person model instead of tiling.')
    args = parser.parse_args()
    compare(args.source_video_path, args.output_dir, args.timestamps, args.frames, args.device, args.person_model)
