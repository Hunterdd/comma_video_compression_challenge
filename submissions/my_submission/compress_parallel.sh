#!/usr/bin/env bash
# compress_parallel.sh supporting running either model 0 or 1 in the background with logging.

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 [0|1]" >&2
  echo "  0: Run original HNeRVDecoder model" >&2
  echo "  1: Run HNeRVDecoder_grouped model (via hook.py)" >&2
  exit 1
fi

MODEL_ARG="$1"

if [ "$MODEL_ARG" != "0" ] && [ "$MODEL_ARG" != "1" ]; then
  echo "ERROR: Invalid argument. Must be 0 or 1." >&2
  exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

# Define log and archive names
LOG_FILE="$HERE/log_model_${MODEL_ARG}.txt"
ARCHIVE_ZIP="$HERE/archive_model_${MODEL_ARG}.zip"

echo "Starting model ${MODEL_ARG} training in background..."
echo "Log file: ${LOG_FILE}"

# Run the training and subsequent zipping in a background subshell
(
  cd "$ROOT"
  
  # Ensure the submissions/my_submission/src directory is in PYTHONPATH
  export PYTHONPATH="$HERE/src"

  if [ "$MODEL_ARG" = "0" ]; then
    echo "=== Running Original Model ==="
    python -m train
  else
    echo "=== Running Grouped Model (hooked) ==="
    python -c "import hook; import train; train.main()"
  fi

  # After training finishes successfully, extract final 0.bin location and zip it
  echo "Training completed. Preparing archive..."
  
  # Extract the path from the "Final archive: " print statement, removing carriage returns
  ARCHIVE_BIN=$(grep -a "Final archive:" "$LOG_FILE" | head -n1 | sed 's/Final archive: //' | tr -d '\r')
  
  if [ -n "$ARCHIVE_BIN" ] && [ -f "$ARCHIVE_BIN" ]; then
    echo "Found archive at: $ARCHIVE_BIN"
    cd "$(dirname "$ARCHIVE_BIN")"
    rm -f "$ARCHIVE_ZIP"
    zip -j "$ARCHIVE_ZIP" "0.bin"
    echo "Wrote $ARCHIVE_ZIP"
  else
    echo "ERROR: Could not locate final archive 0.bin path in log." >&2
    exit 1
  fi
) 2>&1 | tee "$LOG_FILE" | while read -r line; do echo "[Model ${MODEL_ARG}] $line"; done &

BG_PID=$!
echo "Process started with PID ${BG_PID}"
