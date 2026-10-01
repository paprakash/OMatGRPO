#!/bin/bash
# LeMat-GenBench on an exported CIF folder (CPU only; several hours for 2,500 structures).
#   LEMAT_GENBENCH_ROOT=... LGB_PYTHON=... sbatch scripts/slurm/lemat_genbench.sbatch <cif_dir> <name>
#SBATCH --job-name=omatgrpo-lgb
#SBATCH --account=<your-account>
#SBATCH --partition=<cpu-partition>
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%j.out
set -eo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
bash scripts/eval/run_lemat_genbench.sh "${1:?usage: <cif_dir> <name>}" "${2:?usage: <cif_dir> <name>}"
