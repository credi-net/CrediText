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
# emb_model=google/embeddinggemma-300m
# emb_model=google/embeddinggemma-2
emb_model=Alibaba-NLP/gte-large-en-v1.5
batch_pairs=(
0   20
20  40
40  60
60  80
80  100 
100 120
120 140
)

for ((i=0; i<${#batch_pairs[@]}; i+=2)); do
    start=$((batch_pairs[i]))
    end=$((batch_pairs[i+1]-1))
    echo sbatch ./job_content_emb_credigrain_L3S3.sh  0 10 $emb_model $start $end
    sbatch ./job_content_emb_credigrain_L3S3.sh 0 10 $emb_model $start $end
    # bash ./job_content_emb_credigrain_L3S3.sh $start $end $emb_model
done