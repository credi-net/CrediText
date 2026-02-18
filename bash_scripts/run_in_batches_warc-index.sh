#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --partition=long-cpu
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=/home/mila/a/abdallah/scratch/jobs_log/cc-index-build/cc-index-build_job_%j.out
#SBATCH --error=/home/mila/a/abdallah/scratch/jobs_log/cc-index-build/cc-index-build_job_%j.err

# Exit on error
set -e


if [ -z "$1" ]; then
      CRAWL=ccmain202508
else
      CRAWL=$1
fi
CRAWL=${CRAWL,,}
if [ -z "$2" ]; then
      Month=Feb2025
else
      Month=$2
fi

if [ -z "$3" ]; then
      sidx=0
else
      sidx=$3
fi

if [ -z "$4" ]; then
      eidx=10
else
      eidx=$4
fi

if [ -z "$5" ]; then
      batch_size=10
else
      batch_size=$5
fi
# Echo time and hostname into log
echo "Date:     $(date)"
echo "Hostname: $(hostname)"
echo "CRAWL:     $CRAWL"
echo "Month:     $Month"
echo "sidx:     $sidx"
echo "eidx:     $eidx"
echo "batch_size:     $batch_size"
export JAVA_HOME=~/jdk-17.0.12/
export PATH=$PATH:$JAVA_HOME/bin

# Execute Python script
# Use `uv run --offline` on clusters without internet access on compute nodes.

for ((i=$sidx; i<$eidx; i+=$batch_size)); do
    echo "#########################################################################################"
    echo "/end-to-end.sh  CC-Crawls/$Month.txt $i $((i+batch_size-1)) [cc-index-table] ../data/${Month}/${Month}_domains.csv"
    ./end-to-end.sh  CC-Crawls/$Month.txt $i $((i+batch_size-1)) [cc-index-table] ../data/${Month}/${Month}_domains.csv
    # rm -r ~/scratch/cc-index/table/cc-main/warc/crawl=CC-MAIN-2025-08/subset=warc
done