#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PD="$(cd "${HERE}/../.." && pwd)"

IN_DIR="${PD}/videos"
VIDEO_NAMES_FILE="${PD}/public_test_video_names.txt"
ARCHIVE_DIR="${HERE}/archive"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --in-dir|--in_dir)
      IN_DIR="${2%/}"; shift 2 ;;
    --video-names-file|--video_names_file)
      VIDEO_NAMES_FILE="$2"; shift 2 ;;
    *)
      echo "Unknown arg: $1" >&2
      echo "Usage: $0 [--in-dir <dir>] [--video-names-file <file>]" >&2
      exit 2 ;;
  esac
done

echo "Starting HNeRV-LRConv compression script..."
python "${HERE}/compress.py" \
  --in-dir "$IN_DIR" \
  --archive-dir "$ARCHIVE_DIR" \
  --video-names-file "$VIDEO_NAMES_FILE" \
  --epochs 250 \
  --lr 0.005 \
  --batch-size 4 \
  --embed-dim 32 \
  --fc-dim 128 \
  --bottleneck-ratio 0.50 \
  --ft-epochs 50 \
  --seed 1234

echo "HNeRV-LRConv compression finished."
