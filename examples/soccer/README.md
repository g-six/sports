# Soccer AI ⚽

For the current general-person detector experiments, validation results and
camera/calibration limitations, see [EXPERIMENTS.md](EXPERIMENTS.md).

## 💻 install

We don't have a Python package yet. Install from source in a
[**Python>=3.8**](https://www.python.org/) environment.

```bash
pip install git+https://github.com/roboflow/sports.git
cd examples/soccer
pip install -r requirements.txt
./setup.sh
```

### Local smoke test

For a compatible local environment, create a Python 3.11 virtual environment and
install the pinned demo dependencies from the repository root:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e . -r examples/soccer/requirements-local.txt
```

Download the model weights with `examples/soccer/setup.sh` after activating the
environment. That script also downloads sample videos. Model weights belong in
`examples/soccer/data/`.

For this recording, the opening 30 seconds show the teams huddling. Extract a
five-second gameplay clip at 05:00 with FFmpeg, then run detection from the
repository root:

```bash
mkdir -p examples/soccer/output
ffmpeg -ss 300 -i "$HOME/Downloads/coastal-fc-vs-burnaby-fc.mp4" \
  -t 5 -an -c:v libx264 -crf 18 -r 30 \
  examples/soccer/data/coastal-vs-burnaby-05m00s-05m05s.mp4
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export YOLO_CONFIG_DIR="$PWD/.cache/ultralytics"
export HF_HOME="$PWD/.cache/huggingface"
.venv/bin/python examples/soccer/main.py \
  --source_video_path examples/soccer/data/coastal-vs-burnaby-05m00s-05m05s.mp4 \
  --target_video_path examples/soccer/output/coastal-vs-burnaby-gameplay-detection.mp4 \
  --device cpu --mode PLAYER_DETECTION --no_display --max_frames 150
```

Remove `--max_frames` to process the entire video. Omit `--no_display` to open a
preview window. Team classification and radar fit their team classifier using a
separate sampling pass over the entire source video, even when output frames are
limited. Output videos contain annotated frames without source audio.

### Manually assign player identities after tracking

From the repository root, run `./track-players.sh`. It prompts for the video,
segment start/duration, device, optional frame limit and preview. Alongside the
tracking MP4 it now saves `*.tracks.jsonl` and `*.review.html`.

Open the review HTML in a browser. Each track has first, highest-confidence and
last sample crops. Enter a team/colour and jersey number for tracks you recognise;
leave unknown tracks blank. Multiple track IDs can receive the same identity.
Use **Add identity range** when a track switches players: ranges use segment frame
numbers, start included and end excluded. Delete or shorten the initial full-track
range before adding another overlapping range. Verify identities against the
video; the samples alone do not establish a player's identity for the entire track.

Click **Download assignments.json** and move the downloaded file next to the video.
The page can load that JSON again to resume editing. It does not save edits
automatically. The script prints a ready-to-run render command, for example:

```bash
.venv/bin/python examples/soccer/review_tracking.py render \
  --video /path/to/blue-19.mp4 \
  --analysis /path/to/blue-19.tracks.jsonl \
  --assignments /path/to/assignments.json \
  --output /path/to/blue-19-labelled.mp4
```

Rendering uses the saved boxes and assignments, without running the detector
again. Unknown detections keep their track IDs; assigned detections show labels
such as `Blue #19 (ID 7)`. This labels all tracked people; it does not yet extract
clips for a selected player. Audio is not included. Existing annotations remain
if you use the tracking output as the input video.

For direct CLI use, add `--analysis_path /path/to/tracks.jsonl` to `PLAYER_TRACKING`.
Optionally set `--source_time_offset 300` if frame 0 corresponds to 05:00 in the
original recording. To build the review page separately:

```bash
.venv/bin/python examples/soccer/review_tracking.py build \
  --video /path/to/tracked.mp4 \
  --analysis /path/to/tracks.jsonl \
  --output /path/to/review.html
```

Always use the exact segment analysed, or its tracking output, with frame 0
aligned. Dimensions and FPS are checked, but cannot detect a different segment
with the same format. Assignments are bound to the tracking export by its hash;
do not reuse them after rerunning tracking. Older videos without a tracking-data
export require another tracking pass. Review pages embed player crops and work
offline; keep them local when the footage is private.

### Button review, named identities and learned ranking

The named-player workflow keeps an offline workspace of sample crops and
decisions. It never automatically confirms a player. Start from an HTML review
page produced by `review_tracking.py build` or `track-players.sh`:

```bash
.venv/bin/python examples/soccer/player_review.py prepare \
  --review /path/to/match.review.html \
  --workspace examples/soccer/output/player-review --blue-only
.venv/bin/python examples/soccer/player_review.py build \
  --workspace examples/soccer/output/player-review --player Alex \
  --output examples/soccer/output/player-review/batch-1.html
```

`--blue-only` is an optional blue-shirt pixel filter; omit it for other uniforms.
This filter can miss small or obscured blue shirts. A track ID is not a jersey
number and can switch identities. Check the video before assigning a whole
track; named button decisions cover the track's full observed frame range.

Open the batch page in a browser and use **Yes**, **Not the player**, **Skip** and
**Undo**. Download `player-decisions.json` to save a durable copy. A batch can load
its downloaded file to resume. When supported inside Codex, completing a batch
attempts direct chat submission; decisions are not saved to disk until chat
confirms. Ordinary browsers use the download/copy fallback. Merge a download:

```bash
.venv/bin/python examples/soccer/player_review.py apply \
  --workspace examples/soccer/output/player-review --player Alex \
  --decisions /path/to/player-decisions.json
```

Merges are idempotent, validate the entire batch before writing, and bind decisions
to the tracking-data hash and player. A manual decision overrides an automatic
exclusion. Skipped tracks stay unresolved and are omitted from subsequent batches;
remove their entries from `skipped_matches` in `named-assignments.json` to revisit
them. Automatic exclusions live separately in `automatic_exclusions`; remove an
entry to undo it. The named file can be passed directly to
`review_tracking.py render --assignments ...` to draw player names.

Once you have at least five positive and five negative examples spread across
five 30-second windows, train an appearance ranking:

```bash
.venv/bin/python examples/soccer/learn_player_filter.py \
  --workspace examples/soccer/output/player-review --player Alex
.venv/bin/python examples/soccer/player_review.py build \
  --workspace examples/soccer/output/player-review --player Alex \
  --ranking examples/soccer/output/player-review/learned-filter-report.json \
  --output examples/soccer/output/player-review/batch-2.html
```

The model uses coarse clothing, head-region colour and texture features; these
are appearance cues, not face recognition or demographic inference. It combines
sample views per track and holds nearby tracks together during validation. The
report includes average precision, top-40 precision, and the number of held-out
confirmed tracks below a proposed exclusion threshold. Those numbers describe
the labelled sample, not guaranteed performance on unreviewed footage. Candidate
updates are compared against the previous training set on identical grouped folds;
updates are accepted only when both ranking metrics do not decrease. Otherwise
the previous training set is retained while all manual decisions remain saved.
Appearance exclusions are only proposals in this report; they are not applied by
the ranking command. More varied views and lighting are preferable to aggressive
thresholds when the report shows uncertainty.

An optional temporal check can propose tracks that appear far from an already
confirmed player at the same time:

```bash
.venv/bin/python examples/soccer/temporal_player_filter.py \
  --workspace examples/soccer/output/player-review --player Alex \
  --analysis /path/to/match.tracks.jsonl
```

Inspect `temporal-filter-report.json`. Add `--apply` to save reversible exclusions
only when there are no conflicts with confirmed tracks. The rule requires at least
five shared frames, separation exceeding two player heights in every shared
frame, no near matches, and shared evidence covering at least 60% of observations.
It respects confirmed frame ranges and avoids treating nearby duplicate detections
as different people. Camera cuts and identity switches can still invalidate its
assumptions. Keep crops, workspaces, decisions and reports local; the repository's
`examples/soccer/output/` directory is ignored by Git.

### Uniform references and stable role labels

`UNIFORM_CLASSIFICATION` uses manually selected uniform examples instead of the
detector's player/referee labels. It compares the central shirt region using hue
and dark/neutral fabric proportions, then smooths the appearance evidence per
ByteTrack ID. It needs three usable observations before assigning a label;
ambiguous matches and untracked people display as `uncertain`. Missing detections
are not recovered, and lost or switched tracker IDs can still affect labels.

Create a JSON file with at least two named uniforms. Each image should be a tight
**full-person crop**, with the head at the top and feet at the bottom; the code
extracts the shirt region. Use multiple examples for different views or lighting.
Image paths are relative to the JSON file:

```json
{
  "uniforms": [
    {"name": "blue team", "role": "player", "images": ["blue-back.jpg", "blue-front.jpg"]},
    {"name": "black team", "role": "player", "images": ["black.jpg"]},
    {"name": "yellow referee", "role": "referee", "images": ["referee.jpg"]},
    {"name": "red goalkeeper", "role": "goalkeeper", "images": ["goalkeeper.jpg"]}
  ]
}
```

With the cache environment variables from the smoke test set, run:

```bash
.venv/bin/python examples/soccer/main.py \
  --source_video_path examples/soccer/data/coastal-vs-burnaby-05m00s-05m05s.mp4 \
  --target_video_path examples/soccer/output/coastal-vs-burnaby-uniforms.mp4 \
  --device cpu --mode UNIFORM_CLASSIFICATION --no_display \
  --uniform_config examples/soccer/data/uniform-references/config.json \
  --analysis_path examples/soccer/output/uniform-analysis.jsonl
```

`--analysis_path` optionally saves raw detector boxes and the tracked uniform/role
decisions per frame for comparison. Uniform labels express the reference match;
they are not calibrated confidence scores. This mode is separate from the
existing SigLIP team classification and radar modes.

The local Coastal/Burnaby references include the blue player incorrectly called
a referee at 05:00. This clip demonstrates the correction using those references;
evaluate on separate clips before making accuracy claims. References and generated
videos stay in the ignored `data/` and `output/` directories.

This recording uses a ball-following AI camera positioned near halfway on the
sideline. Uniform matching does not assume fixed screen positions. ByteTrack
still operates in image coordinates without camera-motion compensation, so pans
and zooms can fragment tracks; the physical camera location is a calibration
prior for future pitch mapping, not a fixed image-to-pitch transform.

### Compare overlapping detection tiles

Add `--tiled_detection` to `PLAYER_DETECTION` or `UNIFORM_CLASSIFICATION` to
combine full-frame inference with overlapping 960-pixel tiles. All predictions
use an input size of 1280. Internal tile-edge fragments are rejected, and
class-agnostic NMS merges duplicate people even when their predicted roles differ.
Tiling only returns person classes; it does not replace the ball detection mode.
It is optional because it substantially increases processing time on CPU.

For a small repeatable comparison across three parts of a recording:

```bash
.venv/bin/python examples/soccer/compare_detection.py \
  --source_video_path "$HOME/Downloads/coastal-fc-vs-burnaby-fc.mp4" \
  --output_dir examples/soccer/output/tiled-comparison \
  --timestamps 300 1200 2400 --frames 3 --device cpu
```

This examines three consecutive frames at each timestamp, saves before/after
images and a JSON report with detection boxes and inference timings. Detection
counts alone do not measure accuracy: inspect the images for misses, duplicate
boxes, spectators, and other false positives. These tests do not verify pitch
calibration, which has not yet been fitted for this recording.

The initial Coastal/Burnaby run (nine frames across 05:00, 20:00, and 40:00)
produced more boxes with tiles, but visual inspection also found duplicate and
partial boxes. Mean inference time was approximately 0.79 seconds per frame for
the original detector and 9.85 seconds for tiling on CPU. These results do not
justify enabling tiling by default or claiming improved accuracy.

Run the classifier tests with:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## ⚽ datasets

Original data comes from the [DFL - Bundesliga Data Shootout](https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout) 
Kaggle competition. This data has been processed to create new datasets, which can be 
downloaded from the [Roboflow Universe](https://universe.roboflow.com/).

| use case                        | dataset                                                                                                                                                          | train model                                                                                                                                                                                            |
|:--------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| soccer player detection         | [![Download Dataset](https://app.roboflow.com/images/download-dataset-badge.svg)](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roboflow/sports/blob/main/examples/soccer/notebooks/train_player_detector.ipynb)         |
| soccer ball detection           | [![Download Dataset](https://app.roboflow.com/images/download-dataset-badge.svg)](https://universe.roboflow.com/roboflow-jvuqo/football-ball-detection-rejhg)    | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roboflow/sports/blob/main/examples/soccer/notebooks/train_ball_detector.ipynb)           |
| soccer pitch keypoint detection | [![Download Dataset](https://app.roboflow.com/images/download-dataset-badge.svg)](https://universe.roboflow.com/roboflow-jvuqo/football-field-detection-f07vi)   | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roboflow/sports/blob/main/examples/soccer/notebooks/train_pitch_keypoint_detector.ipynb) |

## 🤖 models

- [YOLOv8](https://docs.ultralytics.com/models/yolov8/) (Player Detection) - Detects 
players, goalkeepers, referees, and the ball in the video.
- [YOLOv8](https://docs.ultralytics.com/models/yolov8/) (Pitch Detection) - Identifies 
the soccer field boundaries and key points.
- [SigLIP](https://huggingface.co/docs/transformers/en/model_doc/siglip) - Extracts 
features from image crops of players.
- [UMAP](https://umap-learn.readthedocs.io/en/latest/) - Reduces the dimensionality of 
the extracted features for easier clustering.
- [KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) - 
Clusters the reduced-dimension features to classify players into two teams.

## 🛠️ modes

- `PITCH_DETECTION` - Detects the soccer field boundaries and key points in the video. 
Useful for identifying and visualizing the layout of the soccer pitch.

  ```bash
  python main.py --source_video_path data/2e57b9_0.mp4 \
  --target_video_path data/2e57b9_0-pitch-detection.mp4 \
  --device mps --mode PITCH_DETECTION
  ```

  https://github.com/user-attachments/assets/cf4df75a-89fe-4c6f-b3dc-e4d63a0ed211

- `PLAYER_DETECTION` - Detects players, goalkeepers, referees, and the ball in the 
video. Essential for identifying and tracking the presence of players and other 
entities on the field.

  ```bash
  python main.py --source_video_path data/2e57b9_0.mp4 \
  --target_video_path data/2e57b9_0-player-detection.mp4 \
  --device mps --mode PLAYER_DETECTION
  ```

  https://github.com/user-attachments/assets/c36ea2c1-b03e-4ffe-81bd-27391260b187

- `BALL_DETECTION` - Detects the ball in the video frames and tracks its position. 
Useful for following ball movements throughout the match.

  ```bash
  python main.py --source_video_path data/2e57b9_0.mp4 \
  --target_video_path data/2e57b9_0-ball-detection.mp4 \
  --device mps --mode BALL_DETECTION
  ```

  https://github.com/user-attachments/assets/2fd83678-7790-4f4d-a8c0-065ef38ca031

- `PLAYER_TRACKING` - Tracks players across video frames, maintaining consistent 
identification. Useful for following player movements and positions throughout the 
match.

  ```bash
  python main.py --source_video_path data/2e57b9_0.mp4 \
  --target_video_path data/2e57b9_0-player-tracking.mp4 \
  --device mps --mode PLAYER_TRACKING
  ```
  
  https://github.com/user-attachments/assets/69be83ac-52ff-4879-b93d-33f016feb839

- `TEAM_CLASSIFICATION` - Classifies detected players into their respective teams based 
on their visual features. Helps differentiate between players of different teams for 
analysis and visualization.

  ```bash
  python main.py --source_video_path data/2e57b9_0.mp4 \
  --target_video_path data/2e57b9_0-team-classification.mp4 \
  --device mps --mode TEAM_CLASSIFICATION
  ```

  https://github.com/user-attachments/assets/239c2960-5032-415c-b330-3ddd094d32c7

- `RADAR` - Combines pitch detection, player detection, tracking, and team 
classification to generate a radar-like visualization of player positions on the 
soccer field. Provides a comprehensive overview of player movements and team formations 
on the field.

  ```bash
  python main.py --source_video_path data/2e57b9_0.mp4 \
  --target_video_path data/2e57b9_0-radar.mp4 \
  --device mps --mode RADAR
  ```

  https://github.com/user-attachments/assets/263b4cd0-2185-4ed3-9be2-cf4d8f5bfa67

## 🗺️ roadmap

- [ ] Add smoothing to eliminate flickering in RADAR mode.
- [ ] Add a notebook demonstrating how to save data and perform offline data analysis.

## © license

This demo integrates two main components, each with its own licensing:

- ultralytics: The object detection model used in this demo, YOLOv8, is distributed 
under the [AGPL-3.0 license](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).
- sports: The analytics code that powers the sports analysis in this demo is based on 
the [Supervision](https://github.com/roboflow/supervision) library, which is licensed 
under the [MIT license](https://github.com/roboflow/supervision/blob/develop/LICENSE.md). 
This makes the sports part of the code fully open source and freely usable in your 
projects.
