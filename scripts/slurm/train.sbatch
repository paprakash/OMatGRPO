#!/bin/bash
# Train one recipe on one GPU.  sbatch scripts/slurm/train.sbatch configs/recipes/arityguard_creatrelax.yaml
# Edit the account, partition and environment lines for your cluster. One run of 750 rollouts took 7 h 41 min
# to 8 h 35 min on one NVIDIA B200 (B=4, K=16) for the paper's creativity runs. Those jobs requested 120-178 GB
# of host memory; SLURM recorded a maximum resident set size of about 3.9 GB.
#SBATCH --job-name=omatgrpo-train
#SBATCH --account=<your-account>
#SBATCH --partition=<gpu-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=178G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
set -eo pipefail
RECIPE="${1:?usage: sbatch train.sbatch <recipe.yaml> [extra flags]}"; shift
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate omatgrpo || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u -m omg.grpo.train --recipe "$RECIPE" "$@"
