# CRAWL=ccmain202508
# Month=Feb2025
emb_model=paraphrase-multilingual-MiniLM-L12-v2


batch_pairs=(
0 1
1 2
2 3
3 4
4 5
5 6
6 7
7 8
8 9
9 10
)
for ((i=0; i<${#batch_pairs[@]}; i+=2)); do
    start=$((batch_pairs[i]))
    end=$((batch_pairs[i+1]-1))
    echo sbatch ./job_content_emb_SampledOffset_dec2024.sh $start $end $emb_model
    sbatch ./job_content_emb_SampledOffset_dec2024.sh $start $end $emb_model
done