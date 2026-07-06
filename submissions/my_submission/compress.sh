#!/usr/bin/env bash
# Build archive.zip for challenge submission.
#
# 1. Reads best_archive.bin from the 3 trained models in full_run/
# 2. Packs them into a single 0.bin (multi-archive format, see inflate.py)
# 3. Zips 0.bin as archive.zip — ready for evaluate.sh
#
# Usage (from challenge root or submission dir):
#   bash submissions/my_submission/compress.sh           # uses final stage s8_muon
#   bash submissions/my_submission/compress.sh s7_sigma  # use a different stage
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
STAGE="${1:-s8_muon}"

# Use venv Python if present
if [ -f "$ROOT/.venv/bin/python" ]; then
    PYTHON="$ROOT/.venv/bin/python"
elif [ -f "$ROOT/.venv/bin/python3" ]; then
    PYTHON="$ROOT/.venv/bin/python3"
else
    PYTHON="python"
fi

echo "============================================================"
echo "Compressing stage=${STAGE} → archive.zip"
echo "============================================================"

# Pack archives into 0.bin (written to submission dir)
cd "$HERE"
"$PYTHON" compress.py "$STAGE"

# Zip 0.bin → archive.zip
rm -f "$HERE/archive.zip"
zip -j "$HERE/archive.zip" "$HERE/0.bin"
echo "Done: archive.zip  ($(wc -c < "$HERE/archive.zip" | tr -d ' ') bytes)"
echo ""
echo "To evaluate locally:"
echo "  bash ${ROOT}/evaluate.sh --submission-dir ${HERE} --device cuda"
