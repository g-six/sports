import argparse
import json
from contextlib import nullcontext
from copy import deepcopy
from enum import Enum
from itertools import islice
from typing import Iterator, List, Optional

import os
import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.common.ball import BallTracker, BallAnnotator
from sports.common.team import TeamClassifier
from sports.common.view import ViewTransformer
from sports.common.video import read_video_info
from sports.common.uniform import UniformClassifier
from sports.common.detection import detect_people, normalize_person_detections
from sports.common.motion import CameraMotionEstimator, transform_boxes
from sports.common.tracking_uniform import filter_blue_shirts
from sports.configs.soccer import SoccerPitchConfiguration

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLAYER_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-player-detection.pt')
PITCH_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-pitch-detection.pt')
BALL_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-ball-detection.pt')

BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

STRIDE = 60
CONFIG = SoccerPitchConfiguration()

COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700']
VERTEX_LABEL_ANNOTATOR = sv.VertexLabelAnnotator(
    color=[sv.Color.from_hex(color) for color in CONFIG.colors],
    text_color=sv.Color.from_hex('#FFFFFF'),
    border_radius=5,
    text_thickness=1,
    text_scale=0.5,
    text_padding=5,
)
EDGE_ANNOTATOR = sv.EdgeAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    thickness=2,
    edges=CONFIG.edges,
)
TRIANGLE_ANNOTATOR = sv.TriangleAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    base=20,
    height=15,
)
BOX_ANNOTATOR = sv.BoxAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)
ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)
BOX_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
)
ELLIPSE_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
)


class Mode(Enum):
    """
    Enum class representing different modes of operation for Soccer AI video analysis.
    """
    PITCH_DETECTION = 'PITCH_DETECTION'
    PLAYER_DETECTION = 'PLAYER_DETECTION'
    BALL_DETECTION = 'BALL_DETECTION'
    PLAYER_TRACKING = 'PLAYER_TRACKING'
    TEAM_CLASSIFICATION = 'TEAM_CLASSIFICATION'
    UNIFORM_CLASSIFICATION = 'UNIFORM_CLASSIFICATION'
    RADAR = 'RADAR'


def get_crops(frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
    """
    Extract crops from the frame based on detected bounding boxes.

    Args:
        frame (np.ndarray): The frame from which to extract crops.
        detections (sv.Detections): Detected objects with bounding boxes.

    Returns:
        List[np.ndarray]: List of cropped images.
    """
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]


def resolve_goalkeepers_team_id(
    players: sv.Detections,
    players_team_id: np.array,
    goalkeepers: sv.Detections
) -> np.ndarray:
    """
    Resolve the team IDs for detected goalkeepers based on the proximity to team
    centroids.

    Args:
        players (sv.Detections): Detections of all players.
        players_team_id (np.array): Array containing team IDs of detected players.
        goalkeepers (sv.Detections): Detections of goalkeepers.

    Returns:
        np.ndarray: Array containing team IDs for the detected goalkeepers.

    This function calculates the centroids of the two teams based on the positions of
    the players. Then, it assigns each goalkeeper to the nearest team's centroid by
    calculating the distance between each goalkeeper and the centroids of the two teams.
    """
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    team_0_centroid = players_xy[players_team_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players_team_id == 1].mean(axis=0)
    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)
    return np.array(goalkeepers_team_id)


def render_radar(
    detections: sv.Detections,
    keypoints: sv.KeyPoints,
    color_lookup: np.ndarray
) -> np.ndarray:
    mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
    transformer = ViewTransformer(
        source=keypoints.xy[0][mask].astype(np.float32),
        target=np.array(CONFIG.vertices)[mask].astype(np.float32)
    )
    xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    transformed_xy = transformer.transform_points(points=xy)

    radar = draw_pitch(config=CONFIG)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 0],
        face_color=sv.Color.from_hex(COLORS[0]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 1],
        face_color=sv.Color.from_hex(COLORS[1]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 2],
        face_color=sv.Color.from_hex(COLORS[2]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 3],
        face_color=sv.Color.from_hex(COLORS[3]), radius=20, pitch=radar)
    return radar


def run_pitch_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run pitch detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)

        annotated_frame = frame.copy()
        annotated_frame = VERTEX_LABEL_ANNOTATOR.annotate(
            annotated_frame, keypoints, CONFIG.labels)
        yield annotated_frame


def run_player_detection(
    source_video_path: str, device: str, tiled_detection: bool = False,
    person_model: Optional[str] = None,
) -> Iterator[np.ndarray]:
    """
    Run player detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    player_detection_model = YOLO(person_model or PLAYER_DETECTION_MODEL_PATH).to(device=device)
    person_id = get_person_class_id(player_detection_model) if person_model else None
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)

    def infer(image):
        kwargs = {'classes': [person_id]} if person_model else {}
        result = player_detection_model(image, imgsz=1280, verbose=False, **kwargs)[0]
        detections = sv.Detections.from_ultralytics(result)
        return normalize_person_detections(detections, person_id) if person_model else detections

    for frame in frame_generator:
        detections = detect_people(frame, infer, tiled=True) if tiled_detection else infer(frame)

        annotated_frame = frame.copy()
        annotated_frame = BOX_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = BOX_LABEL_ANNOTATOR.annotate(annotated_frame, detections)
        yield annotated_frame


def run_ball_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run ball detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    ball_tracker = BallTracker(buffer_size=20)
    ball_annotator = BallAnnotator(radius=6, buffer_size=10)

    def callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_detection_model(image_slice, imgsz=640, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    slicer = sv.InferenceSlicer(
        callback=callback,
        overlap_filter_strategy=sv.OverlapFilter.NONE,
        slice_wh=(640, 640),
    )

    for frame in frame_generator:
        detections = slicer(frame).with_nms(threshold=0.1)
        detections = ball_tracker.update(detections)
        annotated_frame = frame.copy()
        annotated_frame = ball_annotator.annotate(annotated_frame, detections)
        yield annotated_frame


def run_player_tracking(
    source_video_path: str, device: str, analysis_path: Optional[str] = None,
    source_time_offset: float = 0,
    tracking_team: str = 'all', tracking_blue_fraction: float = .12,
) -> Iterator[np.ndarray]:
    """
    Run player tracking on a video and yield annotated frames with tracked players.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    video_info = read_video_info(source_video_path)
    tracker = sv.ByteTrack(frame_rate=video_info.fps, minimum_consecutive_frames=3)
    output = open(analysis_path, 'x') if analysis_path else nullcontext(None)
    with output as analysis:
        for frame_index, frame in enumerate(frame_generator):
            result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)
            if tracking_team == 'blue':
                detections = filter_blue_shirts(frame, detections, tracking_blue_fraction)
            detections = tracker.update_with_detections(detections)
            track_ids = detections.tracker_id if detections.tracker_id is not None else []
            if analysis is not None:
                record = {
                    'schema': 'player-tracking-v1',
                    'frame_index': frame_index,
                    'timestamp': frame_index / video_info.fps,
                    'source_timestamp': source_time_offset + frame_index / video_info.fps,
                    'fps': video_info.fps,
                    'width': video_info.width, 'height': video_info.height,
                    'tracking_team': tracking_team,
                    'tracking_blue_fraction': tracking_blue_fraction if tracking_team == 'blue' else None,
                    'detections': [
                        {'tracker_id': int(track_id), 'bbox': box.tolist(),
                         'class_id': int(class_id), 'confidence': float(confidence)}
                        for box, track_id, class_id, confidence in zip(
                            detections.xyxy, track_ids,
                            detections.class_id, detections.confidence)
                    ],
                }
                analysis.write(json.dumps(record) + '\n')

            labels = [str(tracker_id) for tracker_id in track_ids]
            annotated_frame = frame.copy()
            annotated_frame = ELLIPSE_ANNOTATOR.annotate(annotated_frame, detections)
            annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
                annotated_frame, detections, labels=labels)
            yield annotated_frame


def run_team_classification(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run team classification on a video and yield annotated frames with team colors.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=STRIDE)

    crops = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops = get_crops(frame, players)
        players_team_id = team_classifier.predict(crops)

        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(
            players, players_team_id, goalkeepers)

        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        detections = sv.Detections.merge([players, goalkeepers, referees])
        color_lookup = np.array(
                players_team_id.tolist() +
                goalkeepers_team_id.tolist() +
                [REFEREE_CLASS_ID] * len(referees)
        )
        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, detections, custom_color_lookup=color_lookup)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels, custom_color_lookup=color_lookup)
        yield annotated_frame


def get_person_class_id(model) -> int:
    """Find the person class by name instead of assuming an arbitrary model index."""
    person_id = next((key for key, name in model.names.items() if name == 'person'), None)
    if person_id is None:
        raise ValueError("The general model must have a class named 'person'.")
    return person_id


def run_uniform_classification(
    source_video_path: str, device: str, uniform_config: str,
    analysis_path: Optional[str] = None, tiled_detection: bool = False,
    person_model: Optional[str] = None,
    compensate_camera_motion: bool = False,
    raw_frames: Optional[Iterator[sv.Detections]] = None,
) -> Iterator[np.ndarray]:
    """Track people and classify uniforms using supplied full-person references."""
    classifier = UniformClassifier.from_config(uniform_config)
    model = None if raw_frames is not None else YOLO(
        person_model or PLAYER_DETECTION_MODEL_PATH).to(device=device)
    person_id = get_person_class_id(model) if person_model and model is not None else None
    video_info = read_video_info(source_video_path)
    tracker = sv.ByteTrack(frame_rate=video_info.fps, minimum_consecutive_frames=1)
    motion_estimator = CameraMotionEstimator() if compensate_camera_motion else None
    track_offset, maximum_track_id = 0, 0
    palette = sv.ColorPalette.from_hex(
        [reference.color or COLORS[i % len(COLORS)]
         for i, reference in enumerate(classifier.references)] + ['#999999'])
    boxes = sv.BoxAnnotator(color=palette, thickness=2)
    labels_annotator = sv.LabelAnnotator(
        color=palette, text_scale=.4, text_color=sv.Color.BLACK)
    report = open(analysis_path, 'w') if analysis_path else None

    def infer(image):
        kwargs = {'classes': [person_id]} if person_model else {}
        result = model(image, imgsz=1280, verbose=False, **kwargs)[0]
        detections = sv.Detections.from_ultralytics(result)
        return normalize_person_detections(detections, person_id) if person_model else detections

    try:
        frames = sv.get_video_frames_generator(source_path=source_video_path)
        for frame_index, frame in enumerate(frames):
            if raw_frames is None:
                raw = detect_people(frame, infer, tiled=tiled_detection)
            else:
                try:
                    raw = next(raw_frames)
                except StopIteration as error:
                    raise ValueError('Saved detections ended before the source video.') from error
            people = raw[np.isin(raw.class_id, [PLAYER_CLASS_ID, GOALKEEPER_CLASS_ID, REFEREE_CLASS_ID])]
            motion = motion_estimator.update(frame, people.xyxy) if motion_estimator else None
            if motion is not None and motion.reset:
                tracker.reset()
                classifier.tracks.clear()
                track_offset = maximum_track_id
            # Track in stabilized coordinates, but classify and draw the original
            # image crops. Keep unconfirmed detections visible as uncertain.
            tracking_people = deepcopy(people)
            if motion is not None:
                tracking_people.xyxy = transform_boxes(people.xyxy, motion.current_to_reference)
            tracking_people.tracker_id = np.full(len(people), -1, dtype=int)
            tracker.update_with_detections(tracking_people)
            people.tracker_id = tracking_people.tracker_id.copy()
            people.tracker_id[people.tracker_id >= 0] += track_offset
            if len(people):
                maximum_track_id = max(maximum_track_id, int(people.tracker_id.max()))
            detections = people
            classifier.begin_frame()
            predictions = []
            labels = []
            colors = []
            for detection_index, (track_id, crop) in enumerate(zip(
                    detections.tracker_id, get_crops(frame, detections))):
                prediction = classifier.update(int(track_id), crop)
                track_label = f"#{track_id}" if track_id >= 0 else "untracked"
                labels.append(f"{track_label} {prediction.name}")
                colors.append(
                    prediction.reference_index if prediction.role is not None
                    else len(classifier.references))
                predictions.append({
                    'tracker_id': int(track_id),
                    'xyxy': detections.xyxy[detection_index].tolist(),
                    'detector_role': 'person' if person_model else {
                        1: 'goalkeeper', 2: 'player', 3: 'referee'}[
                            int(detections.class_id[detection_index])],
                    'uniform': prediction.name,
                    'role': prediction.role,
                })
            if report:
                report.write(json.dumps({
                    'frame': frame_index,
                    'camera_motion': None if motion is None else {
                        'valid': motion.valid, 'reset': motion.reset,
                        'inlier_ratio': motion.inlier_ratio,
                        'feature_count': motion.feature_count,
                        'current_to_reference': motion.current_to_reference.tolist(),
                    },
                    'raw_detections': [
                        {'xyxy': box.tolist(), 'class_id': int(class_id), 'confidence': float(conf)}
                        for box, class_id, conf in zip(raw.xyxy, raw.class_id, raw.confidence)
                    ],
                    'predictions': predictions,
                }) + '\n')
            color_lookup = np.asarray(colors, dtype=int)
            annotated = boxes.annotate(frame.copy(), detections, custom_color_lookup=color_lookup)
            yield labels_annotator.annotate(
                annotated, detections, labels=labels, custom_color_lookup=color_lookup)
    finally:
        if report:
            report.close()


def run_radar(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=STRIDE)

    crops = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in frame_generator:
        result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops = get_crops(frame, players)
        players_team_id = team_classifier.predict(crops)

        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(
            players, players_team_id, goalkeepers)

        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        detections = sv.Detections.merge([players, goalkeepers, referees])
        color_lookup = np.array(
            players_team_id.tolist() +
            goalkeepers_team_id.tolist() +
            [REFEREE_CLASS_ID] * len(referees)
        )
        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, detections, custom_color_lookup=color_lookup)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels,
            custom_color_lookup=color_lookup)

        h, w, _ = frame.shape
        radar = render_radar(detections, keypoints, color_lookup)
        radar = sv.resize_image(radar, (w // 2, h // 2))
        radar_h, radar_w, _ = radar.shape
        rect = sv.Rect(
            x=w // 2 - radar_w // 2,
            y=h - radar_h,
            width=radar_w,
            height=radar_h
        )
        annotated_frame = sv.draw_image(annotated_frame, radar, opacity=0.5, rect=rect)
        yield annotated_frame


def main(
    source_video_path: str,
    target_video_path: str,
    device: str,
    mode: Mode,
    no_display: bool = False,
    max_frames: Optional[int] = None,
    uniform_config: Optional[str] = None,
    analysis_path: Optional[str] = None,
    tiled_detection: bool = False,
    person_model: Optional[str] = None,
    compensate_camera_motion: bool = False,
    source_time_offset: float = 0,
    tracking_team: str = 'all', tracking_blue_fraction: float = .12,
) -> None:
    if tracking_team not in ('all', 'blue'):
        raise ValueError('tracking_team must be all or blue.')
    if tracking_team != 'all' and mode != Mode.PLAYER_TRACKING:
        raise ValueError('--tracking_team requires PLAYER_TRACKING.')
    if not np.isfinite(tracking_blue_fraction) or not 0 < tracking_blue_fraction <= 1:
        raise ValueError('tracking_blue_fraction must be in (0, 1].')
    if not np.isfinite(source_time_offset) or source_time_offset < 0:
        raise ValueError('source_time_offset must be finite and nonnegative.')
    if mode == Mode.PLAYER_TRACKING and analysis_path:
        if os.path.exists(analysis_path):
            raise ValueError('Tracking analysis output already exists; choose a new path.')
        if os.path.abspath(analysis_path) in (
                os.path.abspath(source_video_path), os.path.abspath(target_video_path)):
            raise ValueError('Tracking analysis must use a separate path from the video files.')
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be greater than zero.")
    if tiled_detection and mode not in (Mode.PLAYER_DETECTION, Mode.UNIFORM_CLASSIFICATION):
        raise ValueError("--tiled_detection supports PLAYER_DETECTION and UNIFORM_CLASSIFICATION.")
    if person_model and mode not in (Mode.PLAYER_DETECTION, Mode.UNIFORM_CLASSIFICATION):
        raise ValueError("--person_model supports PLAYER_DETECTION and UNIFORM_CLASSIFICATION.")
    if compensate_camera_motion and mode != Mode.UNIFORM_CLASSIFICATION:
        raise ValueError('--compensate_camera_motion requires UNIFORM_CLASSIFICATION.')
    if mode == Mode.UNIFORM_CLASSIFICATION:
        if uniform_config is None:
            raise ValueError("UNIFORM_CLASSIFICATION requires --uniform_config.")
        frame_generator = run_uniform_classification(
            source_video_path, device, uniform_config, analysis_path, tiled_detection,
            person_model, compensate_camera_motion)
    elif mode == Mode.PITCH_DETECTION:
        frame_generator = run_pitch_detection(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.PLAYER_DETECTION:
        frame_generator = run_player_detection(
            source_video_path=source_video_path, device=device,
            tiled_detection=tiled_detection, person_model=person_model)
    elif mode == Mode.BALL_DETECTION:
        frame_generator = run_ball_detection(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.PLAYER_TRACKING:
        frame_generator = run_player_tracking(
            source_video_path=source_video_path, device=device,
            analysis_path=analysis_path, source_time_offset=source_time_offset,
            tracking_team=tracking_team, tracking_blue_fraction=tracking_blue_fraction)
    elif mode == Mode.TEAM_CLASSIFICATION:
        frame_generator = run_team_classification(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.RADAR:
        frame_generator = run_radar(
            source_video_path=source_video_path, device=device)
    else:
        raise NotImplementedError(f"Mode {mode} is not implemented.")

    video_info = read_video_info(source_video_path)
    if max_frames is not None:
        frame_generator = islice(frame_generator, max_frames)
    total = (
        min(video_info.total_frames, max_frames)
        if max_frames is not None else video_info.total_frames
    )
    with sv.VideoSink(target_video_path, video_info) as sink:
        for frame in tqdm(frame_generator, total=total, desc="Processing video"):
            sink.write_frame(frame)

            if not no_display:
                cv2.imshow("frame", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        if not no_display:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--source_video_path', type=str, required=True)
    parser.add_argument('--target_video_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--mode', type=Mode, default=Mode.PLAYER_DETECTION)
    parser.add_argument('--uniform_config', help='JSON file containing named uniform reference images.')
    parser.add_argument('--analysis_path', help='Per-frame JSONL output for uniform classification or player tracking.')
    parser.add_argument('--source_time_offset', type=float, default=0,
                        help='Original video start time in seconds for tracking export timestamps.')
    parser.add_argument('--tracking_team', choices=['all', 'blue'], default='all',
                        help='PLAYER_TRACKING: admit only this team colour before ID association.')
    parser.add_argument('--tracking_blue_fraction', type=float, default=.12,
                        help='Minimum blue fraction of the inner shirt region, in (0, 1].')
    parser.add_argument('--person_model', help='General person model weights; uniform mode determines roles separately.')
    parser.add_argument(
        '--compensate_camera_motion', action='store_true',
        help='Stabilize tracking with background pan/zoom estimates (experimental).')
    parser.add_argument(
        '--tiled_detection', action='store_true',
        help='Add overlapping person detection tiles (player detection or uniform mode).')
    parser.add_argument(
        '--no_display', action='store_true',
        help='Save video without opening a preview window.')
    parser.add_argument(
        '--max_frames', type=int, default=None,
        help='Limit output frames for a short smoke test.')
    args = parser.parse_args()
    main(
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        device=args.device,
        mode=args.mode,
        no_display=args.no_display,
        max_frames=args.max_frames,
        uniform_config=args.uniform_config,
        analysis_path=args.analysis_path,
        tiled_detection=args.tiled_detection,
        person_model=args.person_model,
        compensate_camera_motion=args.compensate_camera_motion,
        source_time_offset=args.source_time_offset,
        tracking_team=args.tracking_team,
        tracking_blue_fraction=args.tracking_blue_fraction,
    )
