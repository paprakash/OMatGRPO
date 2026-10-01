#!/bin/bash
# Run LeMat-GenBench on a folder of CIFs with the setup of the paper.
#
#   LEMAT_GENBENCH_ROOT=/path/to/lemat-genbench scripts/eval/run_lemat_genbench.sh <cif_dir> <name>
#
# - LEMAT_GENBENCH_ROOT: a checkout of https://github.com/LeMaterial/lemat-genbench at the pinned commit
#   below, with its environment installed (LGB_PYTHON: its python, default `python`).
# - Preset comprehensive_multi_mlip_hull: stability from an ensemble of three potentials (ORB, MACE,
#   UMA). UMA is gated on Hugging Face: without a token, LeMat-GenBench silently drops the UMA leg and
#   reports a two-potential ensemble, so this script refuses to start without one. Check the result
#   with lgb_report.py, which verifies that all three legs scored the same number of structures.
# - Results land in $LEMAT_GENBENCH_ROOT/results_final/<name>_comprehensive_multi_mlip_hull_<time>.json.
# - The run is CPU-only and takes several hours for 2,500 structures.
set -eo pipefail

PINNED_COMMIT=58e6eae3e4a6c87c22171cf069123ecc4e2fa7e6
PRESET=comprehensive_multi_mlip_hull

CIF_DIR="${1:?usage: run_lemat_genbench.sh <cif_dir> <name>}"
NAME="${2:?usage: run_lemat_genbench.sh <cif_dir> <name>}"
: "${LEMAT_GENBENCH_ROOT:?set LEMAT_GENBENCH_ROOT to the lemat-genbench checkout}"
LGB_PYTHON="${LGB_PYTHON:-python}"
CIF_DIR="$(cd "$CIF_DIR" && pwd)"

commit="$(git -C "$LEMAT_GENBENCH_ROOT" rev-parse HEAD)"
echo "lemat-genbench: $LEMAT_GENBENCH_ROOT @ $commit (pinned: $PINNED_COMMIT)"
echo "preset: $PRESET"
if [ "$commit" != "$PINNED_COMMIT" ]; then
  echo "ERROR: the checkout is not at the pinned commit; results would not be comparable to the paper." >&2
  echo "       git -C \"$LEMAT_GENBENCH_ROOT\" checkout $PINNED_COMMIT" >&2
  exit 1
fi

if [ -z "${HF_TOKEN:-}" ] && [ -f "${HF_HOME:-$HOME/.cache/huggingface}/token" ]; then
  HF_TOKEN="$(cat "${HF_HOME:-$HOME/.cache/huggingface}/token")"
  export HF_TOKEN
fi
if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: no Hugging Face token (HF_TOKEN or \`huggingface-cli login\`). The UMA leg of the ensemble" >&2
  echo "       needs the gated facebook/UMA model; without it the results are not the paper's setup." >&2
  exit 1
fi

export PYTHONUTF8=1 PYTHONIOENCODING=utf-8      # smact needs UTF-8
n=$(find "$CIF_DIR" -name '*.cif' | wc -l)
echo "cifs: $CIF_DIR ($n files)  name: $NAME  start: $(date -Is)"
cd "$LEMAT_GENBENCH_ROOT"
"$LGB_PYTHON" -u scripts/run_benchmarks.py --cifs "$CIF_DIR" --config "$PRESET" --name "$NAME"
echo "end: $(date -Is)"
