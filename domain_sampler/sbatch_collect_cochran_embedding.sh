#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --partition=long-cpu
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/mila/a/abdallah/scratch/jobs_log/dec2024/domainrel_cochran_emb_%j.out
#SBATCH --error=/home/mila/a/abdallah/scratch/jobs_log/dec2024/domainrel_cochran_emb_%j.err

# Exit on error
set -e
if [ -z "$1" ]; then
      start_idx=1
else
      start_idx=$1
fi

if [ -z "$2" ]; then
      end_idx=10
else
      end_idx=$2
fi
if [ -z "$3" ]; then
      task=TopicModeling
else
      task=$3
fi

echo sbatch python sample_month_content.py 
python sample_month_content.py 

