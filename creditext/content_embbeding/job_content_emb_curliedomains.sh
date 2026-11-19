#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --mem-per-gpu=64G
#SBATCH --cpus-per-task=16
# #SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/home/mila/a/abdallah/scratch/jobs_log/dec2024/curliedomian_emb_l3s3_job_%j.out
#SBATCH --error=/home/mila/a/abdallah/scratch/jobs_log/dec2024/curliedomian_emb_l3s3_job_%j.err

# Exit on error
set -e

# Echo time and hostname into log
echo "Date:     $(date)"
echo "Hostname: $(hostname)"

# Execute Python script
# Use `uv run --offline` on clusters without internet access on compute nodes.

# podman run -v /home/mila/a/abdallah/scratch/LLM_Models:/models --rm --device nvidia.com/gpu=all  -p 8080:8080 ghcr.io/ggml-org/llama.cpp:server-cuda -hf nomic-ai/nomic-embed-text-v1.5-GGUF:Q4_K_S --embedding -ngl 99  -c 16384 -n 16384 --port 8080 --host 0.0.0.0  --log-disable &
# sleep 30s

# podman run -v /home/mila/a/abdallah/scratch/LLM_Models:/models --rm --device nvidia.com/gpu=all  -p 8080:8080 ghcr.io/ggml-org/llama.cpp:server-cuda -hf unsloth/embeddinggemma-300m-GGUF:BF16  --embedding -ngl 99 -np 32 -c 65536 -b 16384 -ub 16384  --port 8080 --host 0.0.0.0 --log-disable &
# sleep 30s
python  generate_content_emb_from_parquet_file.py  --files_start_idx=$1 --files_end_idx=$2 --emb_model=$3

# Qwen/Qwen3-Embedding-8B
# --log-disable
# emb_model_name="$(S5##*/)"
# echo "$emb_model_name"
# sentence-transformers/all-MiniLM-L6-v2
# nomic-ai/nomic-embed-text-v1
# podman run -v /home/mila/a/abdallah/scratch/LLM_Models:/models --rm --device nvidia.com/gpu=all  -p 8080:8080 ghcr.io/ggml-org/llama.cpp:server-cuda -hf nomic-ai/nomic-embed-text-v1-GGUF  --embedding -ngl 99  -c 16384 -n 16384 --port 8080 --host 0.0.0.0  --log-disable 
# -m /models/Qwen3-Embedding-8B-Q5_K_M.gguf


# podman run -v /home/mila/a/abdallah/scratch/LLM_Models:/models --rm --device nvidia.com/gpu=all  -p 6060:6060 ghcr.io/ggml-org/llama.cpp:server-cuda -hf unsloth/Qwen3.5-9B-GGUF:UD-Q4_K_XL    -ngl 99  --ctx-size 16384     --temp 1.0     --top-p 0.95     --top-k 20     --min-p 0.00     --alias "unsloth/Qwen3.5-9B-GGUF"     --port 6060     --chat-template-kwargs '{"enable_thinking":true}' --port 8080 --host 0.0.0.0  
