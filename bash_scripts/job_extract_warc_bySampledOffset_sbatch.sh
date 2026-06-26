# CRAWL=ccmain202508
# CRAWL=ccmain202508
# CRAWL=ccmain202508
# CRAWL=ccmain202451
CRAWL=ccmain202446

# CRAWL=None
# Month=Feb2025
# Month=Jan2025
# Month=Mar2025
# Month=Dec2024
Month=Nov2024
# Month=Oct2024
# batch_size=10
batch_size=10000

####### sequantial ###########
for ((i=0; i<90000; i+=$batch_size)); do
    start=$(($i+0))
    end=$(($i+$batch_size))

######### by predefined pairs ############
# batch_pairs=(
# 9400 10000
# )

# for ((i=0; i<${#batch_pairs[@]}; i+=2)); do
#     start=$((batch_pairs[i]))
#     end=$((batch_pairs[i+1]))
    
    echo "sbatch run_in_batches_warc_html_bySampledOffset.sh $CRAWL $Month $start $end 50 "
    sbatch run_in_batches_warc_html_bySampledOffset.sh $CRAWL $Month $start $end 50
done


