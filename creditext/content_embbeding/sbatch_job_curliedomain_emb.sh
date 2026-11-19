# CRAWL=ccmain202508
# Month=Feb2025

############## MiniLLM###########
# emb_model=paraphrase-multilingual-MiniLM-L12-v2
# batch_pairs=(
# 7600	7700
# 38200	38300
# 66400	66500
# 68750	70000
# )
######Gemma #############
emb_model=google/embeddinggemma-300m
# emb_model=google/embeddinggemma-2
# emb_model=Alibaba-NLP/gte-large-en-v1.5
batch_pairs=(
0	100
1	100
2	100
3	100
4	100
5	100
6   100
)

for ((i=0; i<${#batch_pairs[@]}; i+=2)); do
    start=$((batch_pairs[i]))
    end=$((batch_pairs[i+1]-1))
    echo sbatch ./job_content_emb_curliedomains.sh  $start $end $emb_model
    sbatch ./job_content_emb_curliedomains.sh $start $end $emb_model
    # bash ./job_content_emb_curliedomains.sh $start $end $emb_model
done