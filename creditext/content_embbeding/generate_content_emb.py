import pandas as pd
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from huggingface_hub import list_repo_files
from sentence_transformers import SentenceTransformer
import pyarrow.parquet as pq
import pickle
import argparse
import torch
from vllm import LLM, SamplingParams
from openai import OpenAI
import os
def get_ds_files(repo_id,pattern="*.parquet"):
    files = list_repo_files(
        repo_id=repo_id,
        repo_type="dataset")
    files_lst=[]
    for f in files:
        if "wetcontent" in f and f.endswith(pattern[1:]):
            files_lst.append(f)
    return files_lst


def print_gpu_memory_info(device_id=0):
    if torch.cuda.is_available():
        # Set the device
        torch.cuda.set_device(device_id)        
        # Get memory stats in bytes
        total_memory = torch.cuda.get_device_properties(device_id).total_memory
        reserved_memory = torch.cuda.memory_reserved(device_id)
        allocated_memory = torch.cuda.memory_allocated(device_id)        
        # Calculate free memory within the PyTorch reserved pool
        free_in_reserved = reserved_memory - allocated_memory        
        # Convert bytes to GiB for readability
        total_memory_gib = total_memory / (1024**3)
        reserved_memory_gib = reserved_memory / (1024**3)
        allocated_memory_gib = allocated_memory / (1024**3)
        free_in_reserved_gib = free_in_reserved / (1024**3)
        print(f"--- GPU Memory Info (Device {device_id}) ---")
        print(f"Total Memory: {total_memory_gib:.2f} GiB")
        print(f"Allocated Memory (PyTorch): {allocated_memory_gib:.2f} GiB")
        print(f"Reserved Memory (PyTorch cache): {reserved_memory_gib:.2f} GiB")
        print(f"Free in Reserved Pool: {free_in_reserved_gib:.2f} GiB")
    else:
        print("CUDA not available.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Content Embedding")
    parser.add_argument("--hf_repo_id", type=str, default="Hussein-Abdallah/CrediBench-WebContent-Jan2025", help="dqr dataset path")
    parser.add_argument("--emb_model", type=str, default="google/embeddinggemma-300m", help="LLM model Name",
                        choices=["Qwen/Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-8B", "Qwen/Qwen3-Embedding-4B",
                                 "google/embeddinggemma-300m","nomic-ai/nomic-embed-text-v1"])
    parser.add_argument("--local_dir", type=str, default="/home/mila/a/abdallah/hsh_projects/content_emb/content_emb", help="document key column")
    parser.add_argument("--hf_files_start_idx", type=int, default=0, help="start file idx")
    parser.add_argument("--hf_files_end_idx", type=int, default=100, help="end file idx")
    parser.add_argument("--parquet_batch_size", type=int, default=100000, help="parquet file read rows batch_size")
    parser.add_argument("--parquet_start_batch_idx", type=int, default=0, help="start batch idx")
    parser.add_argument("--parquet_end_batch_idx", type=int, default=1000, help="end batch idx")
    parser.add_argument("--emb_batch_size", type=int, default=5000, help="GPU emb batch size")
    parser.add_argument("--emb_dim", type=int, default=256, help="resize embedding dim")
    parser.add_argument("--inferEngine", type=str, default="Transformers", choices=["Transformers","LLamaCPP","Ollama" "VLLM"], help="resize embedding dim")
    args = parser.parse_args()
    print(f"args={args}")
    assert torch.cuda.is_available() and torch.cuda.device_count() > 0
    device = torch.device("cuda", 0)
    print(f"device={torch.cuda.get_device_name(0)}")
    print_gpu_memory_info(0)
    ds_name=args.hf_repo_id.split("/")[-1]
    repo_parquet_files_lst = get_ds_files(args.hf_repo_id)
    if args.inferEngine=="Transformers":
        model = SentenceTransformer(args.emb_model)
    elif args.inferEngine=="VLLM":
        max_model_len=8096
        model = LLM(model=args.emb_model,dtype="bfloat16",gpu_memory_utilization=0.8,quantization="fp8") # vllm model max_model_len=max_model_len
    elif args.inferEngine=="LLamaCPP":
        client = OpenAI(
                base_url="http://0.0.0.0:8080/v1",
                api_key="llama-cpp"  
            )
    print("########## GPU Memory After Loading  Model ###############")
    print_gpu_memory_info(0)
    for f_idx, f in tqdm(enumerate(repo_parquet_files_lst[args.hf_files_start_idx:args.hf_files_end_idx+1])):
        hf_hub_download(repo_id=args.hf_repo_id, filename=f, repo_type="dataset", local_dir=f"{args.local_dir}/{ds_name}/")
        # print(f"{local_dir}/{ds_name}/{f}")
        parquet_file = pq.ParquetFile(f"{args.local_dir}/{ds_name}/{f}")
        print(f"file={f}")
        print(f"batches_count={1+int(parquet_file.metadata.num_rows / args.parquet_batch_size)}")
        out_folder_path= f'{args.local_dir}/{args.hf_repo_id.split("/")[-1]}_{args.emb_model.split("/")[-1]}'
        os.makedirs(out_folder_path, exist_ok=True)
        for parquet_batch_idx, batch in tqdm(enumerate(parquet_file.iter_batches(batch_size=args.parquet_batch_size)),
                             total=1 + int(parquet_file.metadata.num_rows / args.parquet_batch_size)):
            if parquet_batch_idx>=args.parquet_start_batch_idx and parquet_batch_idx<=args.parquet_end_batch_idx:
                df_chunk = batch.to_pandas()
                urls_lst = df_chunk["WARC_Target_URI"].tolist()
                domains_lst = df_chunk["Domain_Name"].tolist()
                domains_txt = df_chunk["wet_record_txt"].astype(str).tolist()
                embds = {}
                for emb_batch_idx in tqdm(range(0, len(domains_lst), args.emb_batch_size)):
                    # print("########## GPU Memory before encode ###############")
                    # print_gpu_memory_info(0)
                    ############## transformers ###############
                    if args.inferEngine=="Transformers":
                        batch_emb = model.encode(domains_txt[emb_batch_idx:emb_batch_idx + args.emb_batch_size])
                    ############ VLLM ##########################
                    elif args.inferEngine=="VLLM":
                        max_input_len=max_model_len
                        batch_emb = model.embed([elem[0:max_input_len] for elem in domains_txt[emb_batch_idx:emb_batch_idx + args.emb_batch_size]],use_tqdm=False)
                        batch_emb=[elem.outputs.embedding for elem in batch_emb]
                    ################ LLama-CPP ##################
                    elif args.inferEngine=="LLamaCPP":
                        response = client.embeddings.create(
                        model="local-embedding-model",
                        input=[doc[0:int(2048)] for doc in domains_txt[emb_batch_idx:emb_batch_idx + args.emb_batch_size]])
                        batch_emb=[emb.embedding for emb in response.data]  
                    for emb_idx in range(len(batch_emb)):
                        # print("domain_idx=",emb_batch_idx+emb_idx)
                        if domains_lst[emb_batch_idx + emb_idx] not in embds:
                            embds[domains_lst[emb_batch_idx + emb_idx]] = []
                        embds[domains_lst[emb_batch_idx + emb_idx]].append(
                            [urls_lst[emb_batch_idx + emb_idx], batch_emb[emb_idx][0:args.emb_dim]])
                emb_file_name = f'{out_folder_path}/{f.split(".")[0]}_{args.emb_model.split("/")[-1]}_{parquet_batch_idx}.pkl'
                print(f"emb_file_name={emb_file_name}")
                with open(emb_file_name, 'wb') as emb_file:
                    pickle.dump(embds, emb_file)
            else:
                print(f"the parquet_batch_idx={parquet_batch_idx} is not included for processing.")
