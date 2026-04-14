#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --partition=long-cpu
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=15:00:00
#SBATCH --output=/home/mila/a/abdallah/scratch/jobs_log/mlp_infer_job_%j.out
#SBATCH --error=/home/mila/a/abdallah/scratch/jobs_log/mlp_infer_job_%j.err

# Exit on error
set -e

# Check if crawl list file is provided
if [ -z "$1" ]; then
      month=dec
else
      month=$1
fi

if [ -z "$2" ]; then
      dataset=dqr
else
      dataset=$2
fi

# Echo time and hostname into log
echo "Date:     $(date)"
echo "Hostname: $(hostname)"
echo "month: $(month)"
echo "dataset: $(dataset)"

# Execute Python script
# Use `uv run --offline` on clusters without internet access on compute nodes.
uv run python  mlp_inference.py --month=$month --dataset=$dataset --start_idx=11 --end_idx=15