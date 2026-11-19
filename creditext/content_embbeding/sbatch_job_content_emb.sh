# CRAWL=ccmain202508
# Month=Feb2025



batch_size=1
numbers_list="21 22 24 25 27 28 30 31 34 35 37 38 40 41 43 44 46 47 49 50 52 53 55"
# for ((i=15; i<=19; i+=$batch_size)); do
for i in $numbers_list; do
    echo sbatch ./job_content_emb_jan2025.sh $i $i 0 256 5000 google/embeddinggemma-300m
    sbatch ./job_content_emb_jan2025.sh $i $i 0 256 5000 google/embeddinggemma-300m
done