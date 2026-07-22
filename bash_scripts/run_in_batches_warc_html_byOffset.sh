#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --partition=long-cpu
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/home/mila/a/abdallah/scratch/jobs_log/cc-html-ext/domin_rel_cc-warc_byoffset_job_%j.out
#SBATCH --error=/home/mila/a/abdallah/scratch/jobs_log/cc-html-ext/domin_rel_cc-warc_byoffset_job_%j.err

# Exit on error
set -e


if [ -z "$1" ]; then
      CRAWL=ccmain2022451
else
      CRAWL=$1
fi
CRAWL=${CRAWL,,}
if [ -z "$2" ]; then
      Month=Dec2024
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

set -e
batch_size=50
for ((i=$sidx; i<$eidx; i+=$batch_size)); do
    echo "#########################################################################################"
    ./end-to-end.sh CC-Crawls/Dec2024.txt $i $((i+$batch_size-1)) [warc] ../bash_scripts/spark-warehouse/cc_full_index_domain_rel_cc_index_table_ccmain202451_sampled_eng_400.parquet domain_rel_warc_byoffset spark-warehouse/cc_full_index_domain_rel_cc_index_table_ccmain202451_sampled_eng_400.txt byoffset
    # rm -r  ~/scratch/crawl-data/CC-MAIN-2025-08/segments
    #  rm -r ../data/crawl-data/CC-MAIN-2024-42/segments
    # rm -r ../data/crawl-data/CC-MAIN-2024-46/segments
    # rm -r ../data/crawl-data/CC-MAIN-2024-51/segments
    # rm -r /shared_mnt/tmp/*
done

