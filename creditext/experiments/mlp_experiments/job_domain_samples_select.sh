#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --partition=long-cpu
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/home/mila/a/abdallah/scratch/jobs_log/sampleDomainDocs/sampleDomainDocs_job_%j.out
#SBATCH --error=/home/mila/a/abdallah/scratch/jobs_log/sampleDomainDocs/sampleDomainDocs_job_%j.err

# Exit on error
set -e


if [ -z "$1" ]; then
      CRAWL=ccmain202508
else
      CRAWL=$1
fi
python sampleDomainsContent.py