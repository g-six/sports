#!/usr/bin/env bash
# Interactively extract a video segment and run PLAYER_TRACKING.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

fail() { printf 'Error: %s\n' "$*" >&2; exit 1; }

expand_path() {
    case "$1" in
        '~') printf '%s' "$HOME" ;;
        '~/'*) printf '%s/%s' "$HOME" "${1:2}" ;;
        /*) printf '%s' "$1" ;;
        *) printf '%s/%s' "$PWD" "$1" ;;
    esac
}

command -v ffmpeg >/dev/null 2>&1 || fail 'Install FFmpeg first (brew install ffmpeg).'
PYTHON="$ROOT_DIR/.venv/bin/python"
[[ -x "$PYTHON" ]] || fail "Python environment missing: $PYTHON. Follow examples/soccer/README.md to set it up."
[[ -f "$ROOT_DIR/examples/soccer/data/football-player-detection.pt" ]] ||
    fail 'Model weights missing. Run bash examples/soccer/setup.sh first.'

printf 'Player tracking — enter paths without surrounding quotes.\n'
read -r -p "Source video [$HOME/Downloads/cfc-vs-lusa.mp4]: " SOURCE
SOURCE="$(expand_path "${SOURCE:-$HOME/Downloads/cfc-vs-lusa.mp4}")"
[[ -f "$SOURCE" ]] || fail "Source video does not exist: $SOURCE"

DEFAULT_TARGET="${SOURCE%.*}-player-tracking.mp4"
read -r -p "Output video [$DEFAULT_TARGET]: " TARGET
TARGET="$(expand_path "${TARGET:-$DEFAULT_TARGET}")"
[[ "$TARGET" == *.mp4 ]] || fail 'Output filename must end in .mp4.'
[[ ! -e "$TARGET" ]] || fail "Output already exists; choose a new filename: $TARGET"
[[ -d "$(dirname -- "$TARGET")" ]] || fail 'Output directory does not exist.'
ANALYSIS="${TARGET%.mp4}.tracks.jsonl"
REVIEW="${TARGET%.mp4}.review.html"
[[ ! -e "$ANALYSIS" && ! -e "$REVIEW" ]] || fail 'Tracking data or review page already exists; choose a new output filename.'

read -r -p 'Start time, HH:MM:SS or seconds [00:00:00]: ' START
START="${START:-00:00:00}"
[[ "$START" =~ ^([0-9]+([.][0-9]+)?|[0-9]+:[0-5][0-9]:[0-5][0-9]([.][0-9]+)?)$ ]] ||
    fail 'Start time must be HH:MM:SS (for example 00:05:00) or seconds (300).'

read -r -p 'Duration in seconds [Enter = through end of video]: ' DURATION
if [[ -n "$DURATION" ]]; then
    [[ "$DURATION" =~ ^[0-9]+([.][0-9]+)?$ && "$DURATION" =~ [1-9] ]] ||
        fail 'Duration must be a positive number of seconds.'
fi

read -r -p 'Device: mps (Apple GPU), cpu, or cuda (NVIDIA) [mps]: ' DEVICE
DEVICE="${DEVICE:-mps}"
case "$DEVICE" in mps|cpu|cuda) ;; *) fail 'Choose mps, cpu, or cuda.' ;; esac

read -r -p 'Track team: blue or all [blue]: ' TRACKING_TEAM
TRACKING_TEAM="${TRACKING_TEAM:-blue}"
case "$TRACKING_TEAM" in blue|all) ;; *) fail 'Choose blue or all.' ;; esac

read -r -p 'Maximum frames [Enter = all frames in segment]: ' MAX_FRAMES
if [[ -n "$MAX_FRAMES" ]]; then
    [[ "$MAX_FRAMES" =~ ^[0-9]+$ && "$MAX_FRAMES" =~ [1-9] ]] ||
        fail 'Maximum frames must be a positive integer.'
fi

read -r -p 'Show preview window? [y/N]: ' PREVIEW
case "$PREVIEW" in ''|n|N|no|NO|y|Y|yes|YES) ;; *) fail 'Enter y or n.' ;; esac
START_SECONDS="$("$PYTHON" -c 'import sys; parts=sys.argv[1].split(":"); print(sum(float(v)*60**i for i,v in enumerate(reversed(parts))))' "$START")"

TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/player-tracking.XXXXXX")"
trap 'rm -rf -- "$TEMP_DIR"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
CLIP="$TEMP_DIR/segment.mp4"

FFMPEG_ARGS=(-hide_banner -nostdin -n -ss "$START" -i "$SOURCE")
if [[ -n "$DURATION" ]]; then FFMPEG_ARGS+=(-t "$DURATION"); fi
FFMPEG_ARGS+=(-map 0:v:0 -an -c:v libx264 -crf 18 "$CLIP")
printf '\nExtracting segment starting at %s…\n' "$START"
ffmpeg "${FFMPEG_ARGS[@]}"

TRACKING_ARGS=(
    "$ROOT_DIR/examples/soccer/main.py"
    --source_video_path "$CLIP"
    --target_video_path "$TARGET"
    --device "$DEVICE"
    --mode PLAYER_TRACKING
    --tracking_team "$TRACKING_TEAM"
    --analysis_path "$ANALYSIS"
    --source_time_offset "$START_SECONDS"
)
if [[ -n "$MAX_FRAMES" ]]; then TRACKING_ARGS+=(--max_frames "$MAX_FRAMES"); fi
case "$PREVIEW" in y|Y|yes|YES) ;; *) TRACKING_ARGS+=(--no_display);; esac

cd "$ROOT_DIR"
printf '\nTracking team: %s…\n' "$TRACKING_TEAM"
"$PYTHON" "${TRACKING_ARGS[@]}"
printf '\nSaved tracking video: %s\n' "$TARGET"
"$PYTHON" "$ROOT_DIR/examples/soccer/review_tracking.py" build \
    --video "$TARGET" --analysis "$ANALYSIS" --output "$REVIEW"
printf '\nOpen this review page in your browser: %s\n' "$REVIEW"
printf 'Assign team/jersey labels and download assignments.json.\n'
printf '\nRender those assignments with:\n'
printf '%q ' "$PYTHON" "$ROOT_DIR/examples/soccer/review_tracking.py" render \
    --video "$TARGET" --analysis "$ANALYSIS" \
    --assignments "$(dirname -- "$TARGET")/assignments.json" \
    --output "${TARGET%.mp4}-labelled.mp4"
printf '\n'
