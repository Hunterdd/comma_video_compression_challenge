#!/usr/bin/env bash
# Must produce a raw video file at <output_dir>/<base_name>.raw.
# A .raw file is a flat binary dump of uint8 RGB frames, shape (N, H, W, 3),
# where H=874 W=1164, no header.
#
# Called by evaluate.sh as:
#   bash submissions/my_submission/inflate.sh <archive_dir> <output_dir> <file_list>
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
SUB_NAME="$(basename "$HERE")"

DATA_DIR="$1"
OUTPUT_DIR="$2"
FILE_LIST="$3"

mkdir -p "$OUTPUT_DIR"

# Use venv Python if present, otherwise fall back to system python
if [ -f "$ROOT/.venv/bin/python" ]; then
    PYTHON="$ROOT/.venv/bin/python"
elif [ -f "$ROOT/.venv/bin/python3" ]; then
    PYTHON="$ROOT/.venv/bin/python3"
else
    PYTHON="python"
fi

while IFS= read -r line; do
  [ -z "$line" ] && continue
  BASE="${line%.*}"
  SRC="${DATA_DIR}/${BASE}.bin"
  DST="${OUTPUT_DIR}/${BASE}.raw"

  [ ! -f "$SRC" ] && echo "ERROR: ${SRC} not found" >&2 && exit 1

  printf "Inflating %s ... " "$line"
  cd "$ROOT"
  "$PYTHON" -m "submissions.${SUB_NAME}.inflate" "$SRC" "$DST"
done < "$FILE_LIST"
