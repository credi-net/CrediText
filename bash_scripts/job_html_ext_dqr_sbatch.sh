# CRAWL=ccmain202508
# Month=Feb2025

# CRAWL=ccmain202505
# Month=Jan2025

# CRAWL=ccmain202513
# Month=Mar2025

# CRAWL=ccmain202451
# Month=Dec2024

# CRAWL=ccmain202446
# Month=Nov2024

CRAWL=ccmain202442
Month=Oct2024


batch_size=5000
for ((i=50; i<2600; i+=$batch_size)); do
    if ((i >= 20000)); then
        batch_size=10000
    fi
    end=$(($i+$batch_size))
    echo sbatch ./job_html_ext_dqr.sh $CRAWL $Month $i $end 50
    sbatch ./job_html_ext_dqr.sh $CRAWL $Month $i $end 50
done