#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --partition=long-cpu
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=/home/mila/a/abdallah/scratch/jobs_log/topic_modeling/topic_modeling_job_%j.out
#SBATCH --error=/home/mila/a/abdallah/scratch/jobs_log/topic_modeling/topic_modeling_job_%j.err

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

echo sbatch python TopicModeling.py --task=$task --start_idx=$start_idx --end_idx=$end_idx
python TopicModeling.py --task=$task --start_idx=$start_idx --end_idx=$end_idx

