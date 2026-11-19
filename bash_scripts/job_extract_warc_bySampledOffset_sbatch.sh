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


# ####### sequantial ###########
batch_size=10000
WarcPerProcess=50
batch_pairs=()
batch_start=0
batch_end=90000
# for ((j=$batch_start; j<$batch_end; j+=batch_size)); do
#   batch_pairs+=("($j, $((j+batch_size)))")
# done
######### by predefined pairs ############
batch_pairs=(
"(2250 , 5000)"
"(5000 , 10000)"
"(10000 , 15000)"
"(15000 , 20000)"
"(21850 , 25000)"
"(25000 , 30000)"
"(31900 , 35000)"
"(35000 , 40000)"
"(41250 , 45000)"
"(45000 , 50000)"
"(51850 , 55000)"
"(55000 , 60000)"
"(62200 , 65000)"
"(65000 , 70000)"
"(72200 , 75000)"
"(75000 , 80000)"
"(80300 , 85000)"
"(85000 , 90000)"
)

# batch_pairs=(
# 13950	14000
# 33950	34000
# 39100	40000
# 49350	50000
# 55900	56000
# 68800	70000
# 76400	76500
# )
echo "${batch_pairs[@]}"

for pair in "${batch_pairs[@]}"; do
    pair="${pair//[()]/}"      # strip the parentheses
    start="${pair%%,*}"        # everything before the comma
    end="${pair##*, }"         # everything after ", "
    # echo "start=$start end=$end -> processing batch [$start, $end)"  
    echo "sbatch run_in_batches_warc_html_bySampledOffset.sh $CRAWL $Month $start $end $WarcPerProcess "
    sbatch run_in_batches_warc_html_bySampledOffset.sh $CRAWL $Month $start $end $WarcPerProcess
    # bash run_in_batches_warc_html_bySampledOffset.sh $CRAWL $Month $start $end $WarcPerProcess
done


