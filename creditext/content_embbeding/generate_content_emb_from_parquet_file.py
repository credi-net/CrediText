from isort import file
import pandas as pd
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from huggingface_hub import list_repo_files
from sentence_transformers import SentenceTransformer
import pyarrow.parquet as pq
import pyarrow as pa
import pickle
import argparse
import os
import glob
from pathlib import Path
import torch
# from vllm import LLM, SamplingParams
from openai import OpenAI
import os
def get_ds_files(dataset_type,dataset,regex="*.parquet"):
    def list_all_files(root_path: str, regex: str = "*.pkl", recursive: bool = False):
        if recursive:
            return [str(p) for p in Path(root_path).rglob(regex)]
        else: 
            return glob.glob(f'{root_path}/{regex}', recursive=False)
    
    if dataset_type=="local_file" and dataset.endswith(".parquet"):
        files=[dataset]
    elif dataset_type=="local_directory":
        files=list_all_files(dataset, regex=regex, recursive=True)
    elif dataset_type=="hf_dataset":
        files = list_repo_files(repo_id=dataset,repo_type="dataset")
    elif dataset_type=="hf_file" and dataset.endswith(".parquet"):
        files = [dataset]
    files_lst=[]
    for f in files:
        if ("warc" in f or "wetcontent" in f) and f.endswith(regex.split(".")[-1]):
            files_lst.append(f)
    return files_lst
    
def embed_long(client, text,tokenizer, max_tok=2000):
    ids = tokenizer.encode(text)
    chunks = [ids[i:i+max_tok] for i in range(0, len(ids), max_tok)]
    vecs = [client.embeddings.create(model="...", input=tokenizer.decode(c)).data[0].embedding
            for c in chunks]
    # mean-pool across chunks (normalize after if you use cosine sim)
    return [sum(col)/len(col) for col in zip(*vecs)]

def embed_short(client, batch_text, tokenizer,max_tok=2000):
    batch_ids = [tokenizer.encode(text)[0:max_tok] for text in batch_text]
    batch_text=None
    response = client.embeddings.create(model="...", input=[tokenizer.decode(c) for c in batch_ids])
    batch_emb=[emb.embedding for emb in response.data]  
    return batch_emb
    
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

def write_domain_emb_parquet(rows: dict, directory_path: str, file_name: str):
    '''rows format: {'domain':domains_lst, 'embeddings':[ [{"page":url,"emb":[float]}] ] }'''
    schema = pa.schema([
        ("domain", pa.string()),
        ("embeddings", pa.list_( pa.struct([
                ("page", pa.string()),
                ("emb", pa.list_(pa.float32()))
            ]) ))])    
    table = pa.Table.from_pydict(rows, schema=schema)
    table = table.sort_by("domain")
    pq.write_table(table, f"{directory_path}/{file_name}",row_group_size=100,use_dictionary=["domain"])

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Content Embedding")    
    # parser.add_argument("--dataset", type=str, default="warc_domain_rel_warc_byoffset_ccmain202451_0_90000.parquet", help="dataset path")

    # parser.add_argument("--dataset", type=str, default="/home/mila/a/abdallah/scratch/hsh_projects/CrediText/bash_scripts/spark-warehouse/warc_warc_curlie_domain_ccmain202451", help="dataset path")        
    # parser.add_argument("--out_dir", type=str, default="/home/mila/a/abdallah/scratch/hsh_projects/CrediText/bash_scripts/spark-warehouse/", help="embeddings out path")
    # parser.add_argument("--dataset_type", type=str, default="local_directory", choices=["hf_dataset","hf_file","local_file","local_directory"], help="the content dataset_type")

    parser.add_argument("--dataset", type=str, default="/home/mila/a/abdallah/scratch/hsh_projects/CrediText/bash_scripts/spark-warehouse/warc_credigrain_l3s3_ccmain202451.parquet", help="dataset path")        
    parser.add_argument("--out_dir", type=str, default="/home/mila/a/abdallah/scratch/hsh_projects/CrediText/bash_scripts/spark-warehouse/", help="embeddings out path")
    parser.add_argument("--dataset_type", type=str, default="local_file", choices=["hf_dataset","hf_file","local_file","local_directory"], help="the content dataset_type")

    # parser.add_argument("--dataset", type=str, default="Hussein-Abdallah/CrediBench-WebContent-Oct2024_CochranSampled", help="dataset path")        
    # parser.add_argument("--dataset", type=str, default="Hussein-Abdallah/CrediBench-WebContent-Dec2024_CochranSampled", help="dataset path")        
    # parser.add_argument("--dataset_type", type=str, default="hf_dataset", choices=["hf_dataset","hf_file","local_file","local_directory"], help="the content dataset_type")
    # parser.add_argument("--out_dir", type=str, default="/home/mila/a/abdallah/scratch/hsh_projects/CrediText/bash_scripts/spark-warehouse/", help="embeddings out path")

    parser.add_argument("--emb_model", type=str, default="google/embeddinggemma-2", help="LLM model Name",
                        choices=["Alibaba-NLP/gte-large-en-v1.5","Qwen/Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-8B", "Qwen/Qwen3-Embedding-4B",
                                 "google/embeddinggemma-300m","google/embeddinggemma-2","nomic-ai/nomic-embed-text-v1","paraphrase-multilingual-MiniLM-L12-v2"])        
    parser.add_argument("--files_start_idx", type=int, default=0, help="start file idx")
    parser.add_argument("--files_end_idx", type=int, default=100, help="end file idx")
    parser.add_argument("--parquet_batch_size", type=int, default=2**14, help="parquet file read rows batch_size")
    parser.add_argument("--parquet_start_batch_idx", type=int, default=0, help="start batch idx")
    parser.add_argument("--parquet_end_batch_idx", type=int, default=1000, help="end batch idx")
    parser.add_argument("--emb_batch_size", type=int, default=2**13, help="GPU emb batch size")
    parser.add_argument("--emb_dim", type=int, default=10**4, help="resize embedding dim")
    parser.add_argument("--inferEngine", type=str, default="Transformers", choices=["Transformers","LLamaCPP","Ollama" "VLLM"], help="resize embedding dim")
    args = parser.parse_args()
    tokenizer=None
    print(f"args={args}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print_gpu_memory_info(0)
    if "/" in args.dataset:
        ds_name=args.dataset.split("/")[-1]
    else:
        ds_name=None
    repo_parquet_files_lst = get_ds_files(args.dataset_type,args.dataset,regex="warc_warc*[0-9][0-9].parquet")
    ############# sort parquet files based on the warc offset in the file name #############
    files_to_process=[]
    if "local" in args.dataset_type:        
        for elem in repo_parquet_files_lst:
            try:
                key=int(f"{elem.split('/')[-2].split('_')[-2]}")
            except (ValueError, IndexError):
                key=-1 ### local file without start & end index
                continue
            if key>=args.files_start_idx and key<=args.files_end_idx:
                files_to_process.append(elem)
        if key==-1 and args.files_start_idx<len(repo_parquet_files_lst): 
            files_to_process.append(repo_parquet_files_lst[args.files_start_idx])
        else:
            print(f"the parquet files are not in the expected format") 
            exit(0)
            
        print(f"Sample of repo_parquet_files after sorting={files_to_process[0:5]}")
    else:
        for idx,elem in enumerate(repo_parquet_files_lst):
            file_idx=elem.split(".")[0].split("_")[-2]
            file_idx=int(file_idx) if file_idx.isdigit() else -1
            if file_idx>=args.files_start_idx and file_idx<=args.files_end_idx:
                files_to_process.append(elem)
        print(f"Sample of repo_parquet_files after sorting={files_to_process[0:5]}")
    #########################
    if args.inferEngine=="Transformers":
        model = SentenceTransformer(args.emb_model,trust_remote_code=True) 
        model = model.to(device) 
        model.eval()        
    elif args.inferEngine=="VLLM":
        max_model_len=8096
        model = LLM(model=args.emb_model,dtype="bfloat16",gpu_memory_utilization=0.8,quantization="fp8") # vllm model max_model_len=max_model_len
    elif args.inferEngine=="LLamaCPP":
        from transformers import AutoTokenizer
        tokenizer= AutoTokenizer.from_pretrained(args.emb_model)
        client = OpenAI(
                base_url="http://0.0.0.0:8080/v1",
                api_key="llama-cpp"  
            )
    print("########## GPU Memory After Loading  Model ###############")
    print_gpu_memory_info(0)
    for f_idx, f in tqdm(enumerate(files_to_process)):
        # hf_hub_download(repo_id=args.dataset, filename=f, repo_type="dataset", local_dir=f"{args.out_dir}/{ds_name}/")
        # print(f"{out_dir}/{ds_name}/{f}")
        if "local" in args.dataset_type:
            emb_file_name = f'{f.split(".")[0]}_{args.emb_model.split("/")[-1]}_{args.parquet_start_batch_idx}_{args.parquet_end_batch_idx}.pkl'            
            if os.path.isfile(emb_file_name):
                print(f"Embedding File Exists: ({emb_file_name})")
                continue
            parquet_file=pq.ParquetFile(f)       
        elif ds_name:
            input_file_path=f"{args.out_dir}/{ds_name}/{f}"
            os.makedirs(f"{args.out_dir}/{ds_name}", exist_ok=True)
            if args.dataset_type=="hf_dataset" and not os.path.exists(input_file_path):                
                hf_hub_download(repo_id=args.dataset, filename=f, repo_type="dataset", local_dir=f"{args.out_dir}/{ds_name}/")
            parquet_file = pq.ParquetFile(f"{args.out_dir}/{ds_name}/{f}")
        else:
            parquet_file = pq.ParquetFile(f"{args.out_dir}/{f}")
        print(f"Processing File={f}")
        print(f"batches_count={1+int(parquet_file.metadata.num_rows / args.parquet_batch_size)}")
        
        if args.dataset_type=="local_file":
            out_folder_path= args.out_dir
        elif ds_name:
            out_folder_path= f'{args.out_dir}/{ds_name}'
        else:
            out_folder_path= f'{args.out_dir}/{args.dataset.split(".")[0]}_{args.emb_model.split("/")[-1]}'
        os.makedirs(out_folder_path, exist_ok=True)

        embds = {}
        for parquet_batch_idx, batch in tqdm(enumerate(parquet_file.iter_batches(batch_size=args.parquet_batch_size)),
                             total=1 + int(parquet_file.metadata.num_rows / args.parquet_batch_size)):
            if parquet_batch_idx>=args.parquet_start_batch_idx and parquet_batch_idx<=args.parquet_end_batch_idx:
                df_chunk = batch.to_pandas()
                urls_lst = df_chunk["WARC_Target_URI"].tolist()
                domains_lst = df_chunk["Domain_Name"].tolist()
                if "warc_record_text_resili" in df_chunk.columns:
                    domains_txt = df_chunk["warc_record_text_resili"].astype(str).tolist()                                
                else:
                    domains_txt = df_chunk["wet_record_txt"].astype(str).tolist()                
                for emb_batch_idx in tqdm(range(0, len(domains_lst), args.emb_batch_size)):
                    # print("########## GPU Memory before encode ###############")
                    # print_gpu_memory_info(0)
                    ############## transformers ###############
                    if args.inferEngine=="Transformers":
                        with torch.no_grad():
                            batch_emb = model.encode(domains_txt[emb_batch_idx:emb_batch_idx + args.emb_batch_size],device=device,show_progress_bar=True)
                    ############ VLLM ##########################
                    elif args.inferEngine=="VLLM":
                        max_input_len=max_model_len
                        batch_emb = model.embed([elem[0:max_input_len] for elem in domains_txt[emb_batch_idx:emb_batch_idx + args.emb_batch_size]],use_tqdm=False)
                        batch_emb=[elem.outputs.embedding for elem in batch_emb]
                    ################ LLama-CPP ##################
                    elif args.inferEngine=="LLamaCPP":   
                        ########## long Embedding with chunking and mean pooling ##########
                        # batch_emb=[embed_long(client, doc,tokenizer) for doc in domains_txt[emb_batch_idx:emb_batch_idx + args.emb_batch_size]]
                        ########## short embedding without chunking ##########
                        batch_emb=embed_short(client, batch_text=domains_txt[emb_batch_idx:emb_batch_idx + args.emb_batch_size], tokenizer=tokenizer, max_tok=8000)
                        
                        # response = client.embeddings.create(
                        # model="local-embedding-model",
                        # input=[doc[0:int(1800)] for doc in domains_txt[emb_batch_idx:emb_batch_idx + args.emb_batch_size]])
                        # batch_emb=[emb.embedding for emb in response.data]  
                    for emb_idx in range(len(batch_emb)):
                        # print("domain_idx=",emb_batch_idx+emb_idx)
                        if domains_lst[emb_batch_idx + emb_idx] not in embds:
                            embds[domains_lst[emb_batch_idx + emb_idx]] = []
                        embds[domains_lst[emb_batch_idx + emb_idx]].append([urls_lst[emb_batch_idx + emb_idx], batch_emb[emb_idx][0:args.emb_dim]])
            else:
                print(f"the parquet_batch_idx={parquet_batch_idx} is not included for processing.")

        
        if "local" in args.dataset_type:
            emb_file_name = f'{f.split(".")[0]}_{args.emb_model.split("/")[-1]}_{args.parquet_start_batch_idx}_{args.parquet_end_batch_idx}.pkl'
        else:
            emb_file_name = f'{out_folder_path}/{f.split("/")[-1].split(".")[0]}_{args.emb_model.split("/")[-1]}_{parquet_batch_idx}.pkl'
            
        print(f"emb_file_name={emb_file_name}")
        with open(emb_file_name, 'wb') as emb_file:
            pickle.dump(embds, emb_file)
        ############# to Parquet ################
        embds={k:[{"page":elem[0], "emb":elem[1]} for elem in v] for k,v in embds.items()}
        content_emb_dict={'domain':list(embds.keys()), 'embeddings':list(embds.values())}  
        write_domain_emb_parquet(content_emb_dict,out_folder_path,emb_file_name.split("/")[-1].replace(".pkl",".parquet"))
        ################### delete parquet file after processing if it is from hf_dataset #####################
        if args.dataset_type=="hf_dataset" and os.path.exists(input_file_path): 
            os.remove(input_file_path)
        print(f"Finished Processing File={f}")
        
