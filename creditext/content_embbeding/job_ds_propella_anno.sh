#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=32G
#SBATCH --time=30:00:00
#SBATCH --output=/home/mila/a/abdallah/scratch/jobs_log/propella/propella_anno_job_%j.out
#SBATCH --error=/home/mila/a/abdallah/scratch/jobs_log/propella/propella_anno_job_%j.err

# Exit on error
set -e
if [ -z "$1" ]; then
      api_port=6060
else
      api_port=$1
fi
# Echo time and hostname into log
echo "Date:     $(date)"
echo "Hostname: $(hostname)"

# Execute Python script
# Use `uv run --offline` on clusters without internet access on compute nodes.
podman run --gpus all  -v /home/mila/a/abdallah/scratch/LLM_Models:/models  --rm --device nvidia.com/gpu=all -p $api_port:$api_port    -e HF_TOKEN=your_huggingface_token   lmsysorg/sglang:latest python -m sglang.launch_server  --model-path ellamind/propella-1-0.6b     --host 0.0.0.0     --port $api_port --context-length 65536   --max-running-requests 256  --chunked-prefill-size 8192  --enable-mixed-chunk  --num-continuous-decode-steps 8     --grammar-backend llguidance     --mem-fraction-static 0.7 --log-level error --disable-piecewise-cuda-graph  &
sleep 600s
uv run python  propella_doc_annotation.py --port=$api_port --batchsize=100

# podman run --gpus all  -v /home/mila/a/abdallah/scratch/LLM_Models:/models  --rm --device nvidia.com/gpu=all -p 6060:6060    lmsysorg/sglang:latest python  -m sglang.launch_server --model-path Qwen/Qwen3.5-9B --host 0.0.0.0 --port 6060 --tp-size 8 --mem-fraction-static 0.8 --context-length 262144 --reasoning-parser qwen3 --log-level error

