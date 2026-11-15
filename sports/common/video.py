"""Video metadata with the source's fractional frame rate preserved."""
import math

import cv2
import supervision as sv


def read_video_info(source_path: str) -> sv.VideoInfo:
    """Avoid integer FPS truncation in older Supervision metadata readers.

    The FPS used for timestamps must also be passed to the output VideoWriter.
    This reads the nominal source rate; it does not reconstruct variable-rate
    presentation timestamps.
    """
    video = cv2.VideoCapture(str(source_path))
    try:
        if not video.isOpened():
            raise ValueError(f'Could not open video at {source_path}')
        fps = float(video.get(cv2.CAP_PROP_FPS))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0 or total_frames < 0:
            raise ValueError('Video has invalid dimensions, frame count, or FPS.')
        # VideoInfo is a dataclass; its older `fps: int` annotation does not
        # coerce this value, and VideoSink passes it directly to OpenCV.
        return sv.VideoInfo(width=width, height=height, fps=fps, total_frames=total_frames)
    finally:
        video.release()
