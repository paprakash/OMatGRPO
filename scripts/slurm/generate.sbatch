#!/bin/bash
# Generate, relax and export 2,500 structures from a checkpoint (paper protocol), on one GPU.
#   sbatch scripts/slurm/generate.sbatch <weights.safetensors> <out_dir>
# Writes <out_dir>/{structures.csv, gen.extxyz, relaxed.extxyz, meta.json} and <out_dir>/export/{cifs, structures_summary.csv}.
#SBATCH --job-name=omatgrpo-generate
#SBATCH --account=<your-account>
#SBATCH --partition=<gpu-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
set -eo pipefail
CKPT="${1:?usage: sbatch generate.sbatch <checkpoint> <out_dir>}"; OUT="${2:?usage}"
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate omatgrpo || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u scripts/eval/generate.py --checkpoint "$CKPT" --out_dir "$OUT" --n 2500 --chunk 100 --seed 42
python -u scripts/eval/export_cifs.py --run_dir "$OUT" --dest "$OUT/export"
