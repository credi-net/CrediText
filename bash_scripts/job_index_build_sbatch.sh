# CRAWL=ccmain202508
# CRAWL=ccmain202508
# CRAWL=ccmain202508
# CRAWL=ccmain202451

CRAWL=None
# Month=Feb2025
# Month=Jan2025
# Month=Mar2025
Month=Dec2024
# Month=Nov2024
# Month=Oct2024
# batch_size=10
batch_size=30
# echo sbatch ./run_in_batches_warc-index.sh $CRAWL $Month 270 300 $batch_size
# sbatch ./run_in_batches_warc-index.sh $CRAWL $Month 270 300 $batch_size
# sbatch ./run_in_batches_warc-index_domain_rel.sh $CRAWL $Month 0 300 $batch_size
sbatch ./run_in_batches_warc-index_dqr.sh $CRAWL $Month 0 300 $batch_size


# sbatch ./run_in_batches_warc-index.sh None Dec2024 0 300 5
