"""Reference-based shirt classification with evidence accumulated per track."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np


ROLE_IDS = {"player": 2, "goalkeeper": 1, "referee": 3}


def shirt_descriptor(person: np.ndarray) -> Optional[np.ndarray]:
    """Use the central upper torso of a full-person BGR crop, avoiding shorts."""
    if person is None or person.size == 0:
        return None
    height, width = person.shape[:2]
    if height < 20 or width < 8:
        return None
    shirt = person[int(height * .18):int(height * .48),
                   int(width * .25):int(width * .75)]
    hsv = cv2.cvtColor(shirt, cv2.COLOR_BGR2HSV)
    # Ignore brightness within coloured fabric so sun/shadow changes do not
    # dominate the match. Keep dark and neutral fabric in separate bins.
    dark = hsv[:, :, 2] < 45
    neutral = (hsv[:, :, 1] < 60) & ~dark
    chromatic = (~dark & ~neutral).astype(np.uint8)
    hues = cv2.calcHist([hsv], [0], chromatic, [18], [0, 180]).flatten()
    histogram = np.append(hues, [dark.sum(), neutral.sum()]).astype(np.float32)
    # Square-root histograms allow Euclidean Hellinger distance in [0, 1].
    return np.sqrt(histogram / histogram.sum())


@dataclass
class UniformReference:
    name: str
    role: str
    descriptors: np.ndarray
    color: Optional[str] = None


@dataclass
class UniformPrediction:
    name: str = "uncertain"
    role: Optional[str] = None
    reference_index: Optional[int] = None


class UniformClassifier:
    """Classify shirt appearance, independently of the detector's role label.

    Distances are appearance measurements, not calibrated probabilities. Tracks
    must have multiple observations and a sufficiently distinct reference match.
    """

    def __init__(
        self, references: List[UniformReference], min_observations: int = 3,
        smoothing: float = .8, max_distance: float = .65, min_margin: float = .08,
        max_missing_frames: int = 150,
    ):
        if len(references) < 2 or len({r.name for r in references}) != len(references):
            raise ValueError("Provide at least two references with unique names.")
        if any(r.role not in ROLE_IDS for r in references):
            raise ValueError("Uniform roles must be player, goalkeeper, or referee.")
        if min_observations < 1 or not 0 <= smoothing < 1:
            raise ValueError("Invalid observation count or smoothing factor.")
        self.references = references
        self.min_observations = min_observations
        self.smoothing = smoothing
        self.max_distance = max_distance
        self.min_margin = min_margin
        self.max_missing_frames = max_missing_frames
        self.tracks: Dict[int, dict] = {}
        self.frame_index = 0

    @classmethod
    def from_config(cls, path: str) -> "UniformClassifier":
        config_path = Path(path)
        config = json.loads(config_path.read_text())
        references = []
        for item in config["uniforms"]:
            descriptors = []
            for image in item["images"]:
                image_path = config_path.parent / image
                crop = cv2.imread(str(image_path))
                descriptor = shirt_descriptor(crop)
                if descriptor is None:
                    raise ValueError(f"Unreadable or too-small reference: {image_path}")
                descriptors.append(descriptor)
            if not descriptors:
                raise ValueError(f"No reference images for {item['name']}")
            references.append(UniformReference(
                item["name"], item["role"], np.stack(descriptors), item.get("color")))
        return cls(references)

    def begin_frame(self) -> None:
        """Expire absent tracks so memory stays bounded on full matches."""
        self.frame_index += 1
        self.tracks = {
            key: value for key, value in self.tracks.items()
            if self.frame_index - value["last_seen"] <= self.max_missing_frames
        }

    def update(self, tracker_id: int, crop: np.ndarray) -> UniformPrediction:
        descriptor = shirt_descriptor(crop)
        if descriptor is None or tracker_id < 0:
            return UniformPrediction()
        distances = np.array([
            np.min(np.linalg.norm(ref.descriptors - descriptor, axis=1)) / np.sqrt(2)
            for ref in self.references
        ])
        state = self.tracks.get(tracker_id)
        if state is None:
            state = {"distances": distances, "observations": 1}
        else:
            state["distances"] = (
                self.smoothing * state["distances"] + (1 - self.smoothing) * distances
            )
            state["observations"] += 1
        state["last_seen"] = self.frame_index
        self.tracks[tracker_id] = state
        order = np.argsort(state["distances"])
        best, second = order[:2]
        if (state["observations"] < self.min_observations
                or state["distances"][best] > self.max_distance
                or state["distances"][second] - state["distances"][best] < self.min_margin):
            return UniformPrediction()
        reference = self.references[best]
        return UniformPrediction(reference.name, reference.role, int(best))
