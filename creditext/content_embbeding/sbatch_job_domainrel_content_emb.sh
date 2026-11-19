# CRAWL=ccmain202508
# Month=Feb2025



# batch_size=1
numbers_list="0 4 8 12"
# for ((i=15; i<=19; i+=$batch_size)); do
for i in $numbers_list; do
    echo sbatch ./job_content_emb_domainrel_dec2024.sh google/embeddinggemma-300m $i $((i+3))
    sbatch ./job_content_emb_domainrel_dec2024.sh $i $((i+3)) 
done