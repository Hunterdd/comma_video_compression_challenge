#!/usr/bin/env bash
# Inflate and score our 3-model ensemble at a specific training stage.
#
# Usage (from challenge root):
#   bash submissions/my_submission/inflate.sh          # scores stage 8 (final)
#   bash submissions/my_submission/inflate.sh 1        # scores after stage 1
#   bash submissions/my_submission/inflate.sh 5        # scores after stage 5
#
# Stages:
#   1=CE  2=Softplus  3=Smooth  4=Smooth+QAT
#   5=L7+C1a(λ=0.01)  6=L7+C1a(λ=0.02)  7=L7+C1a(σ=0.1)  8=Muon

set -e

STAGE=${1:-8}

echo "============================================================"
echo "Scoring ensemble at stage ${STAGE}"
echo "============================================================"

# Run from challenge root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHALLENGE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${CHALLENGE_ROOT}"

# Use venv Python if present, otherwise fall back to system Python
if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
elif [ -f ".venv/bin/python3" ]; then
    PYTHON=".venv/bin/python3"
else
    PYTHON="python3"
fi

echo "Python : ${PYTHON}"
echo "Root   : ${CHALLENGE_ROOT}"
echo ""

"${PYTHON}" submissions/my_submission/eval_stage.py --stage "${STAGE}"
