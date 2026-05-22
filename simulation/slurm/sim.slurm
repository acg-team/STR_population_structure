#!/bin/bash
#SBATCH --job-name=sim
#SBATCH --array=1-5
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --partition=earth-3
#SBATCH --constraint=rhel8
#SBATCH --output=slurm_logs/sim_rep%a_%j.out
#SBATCH --error=slurm_logs/sim_rep%a_%j.err

set -euo pipefail

# Go to project folder
cd /cfs/earth/scratch/xiaf/popstr/sim

# Load your environment
# module load miniconda
source /net/home/xiaf/miniconda3/etc/profile.d/conda.sh
conda activate sim

mkdir -p slurm_logs

REP=${SLURM_ARRAY_TASK_ID}

echo "Running replicate ${REP}"
echo "Started at: $(date)"
echo "Host: $(hostname)"

python run_pipeline.py \
  --grid sim_rep.csv \
  --replicate "${REP}" \
  --n-runs 5 \
  --max-k 8

echo "Finished replicate ${REP}"
echo "Finished at: $(date)"
