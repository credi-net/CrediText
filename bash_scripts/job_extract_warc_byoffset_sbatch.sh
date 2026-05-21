# CRAWL=ccmain202508
# CRAWL=ccmain202508
# CRAWL=ccmain202508
CRAWL=ccmain202451

# CRAWL=None
# Month=Feb2025
# Month=Jan2025
# Month=Mar2025
Month=Dec2024
# Month=Nov2024
# Month=Oct2024
# batch_size=10
batch_size=50
# echo sbatch ./run_in_batches_warc-index.sh $CRAWL $Month 270 300 $batch_size
# sbatch ./run_in_batches_warc-index.sh $CRAWL $Month 270 300 $batch_size
# sbatch ./run_in_batches_warc-index_domain_rel.sh $CRAWL $Month 0 300 $batch_size

batch_pairs=(
9500 10000
12050 16000
16000 20000
22050 26000
26000 30000
32000 35950
35950 40000
43500 46700
46700 50000
53600 56750
56750 60000
63650 66800
66800 70000
73650 76800
76800 80000
83950 86950
86950 90000
)

# Loop through 2 entries at a time
for ((i=0; i<${#batch_pairs[@]}; i+=2)); do
    start=$((batch_pairs[i]))
    end=$((batch_pairs[i+1] + 1))
    # echo "$start -> $end"
    sbatch run_in_batches_warc_html_byOffset.sh $CRAWL $Month $start $end $batch_size 
done

# sbatch ./run_in_batches_warc-index_dqr.sh $CRAWL $Month 0 300 $batch_size
# sbatch ./run_in_batches_warc-index.sh None Dec2024 0 300 5
