# CRAWL=ccmain202508
CRAWL=None
# Month=Feb2025
# Month=Jan2025
# Month=Mar2025
# Month=Dec2024
# Month=Oct2024
Month=Nov2024
batch_size=5
echo sbatch ./run_in_batches_warc-index.sh $CRAWL $Month 270 300 $batch_size
sbatch ./run_in_batches_warc-index.sh $CRAWL $Month 270 300 $batch_size

# sbatch ./run_in_batches_warc-index.sh None Dec2024 0 300 5
