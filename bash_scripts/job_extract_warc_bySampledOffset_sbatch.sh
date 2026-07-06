# CRAWL=ccmain202508
# CRAWL=ccmain202508
# CRAWL=ccmain202508
CRAWL=ccmain202451
# CRAWL=ccmain202446

# CRAWL=None
# Month=Feb2025
# Month=Jan2025
# Month=Mar2025
Month=Dec2024
# Month=Nov2024
# Month=Oct2024


# ####### sequantial ###########
# batch_size=10
# batch_size=10000
# for ((i=0; i<90000; i+=$batch_size)); do
#     start=$(($i+0))
#     end=$(($i+$batch_size))

######### by predefined pairs ############
# batch_pairs=(
# 48000 49950
# 66150 70000
# 48000 50000
# 37950 40000
# 5450  6000
# 7350  8000
# 86450 86500
# )

batch_pairs=(
13950	14000
33950	34000
39100	40000
49350	50000
55900	56000
68800	70000
76400	76500
)

for ((i=0; i<${#batch_pairs[@]}; i+=2)); do
    start=$((batch_pairs[i]))
    end=$((batch_pairs[i+1]))
    
    echo "sbatch run_in_batches_warc_html_bySampledOffset.sh $CRAWL $Month $start $end 50 "
    sbatch run_in_batches_warc_html_bySampledOffset.sh $CRAWL $Month $start $end 50
done


