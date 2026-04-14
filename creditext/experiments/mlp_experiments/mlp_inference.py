import argparse
import logging
from typing import Dict, cast
import pickle
import numpy as np
from creditext.utils.args import parse_args
from creditext.utils.logger import setup_logging
from creditext.utils.path import get_root_dir
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from statistics import mean
import pyarrow.parquet as pq
import pyarrow.compute as pc
import multiprocessing
from multiprocessing import Pool, cpu_count
import threading
import logging
from creditext.utils.logger import setup_logging
from utils import search_parquet_duckdb , load_emb_index,list_all_files
import resource
import sys
# shard_domains_dict={}
# def process_chunk(chunk):
#     global shard_domains_dict
#     for k, v in chunk:
#         shard_domains_dict[v].append(k)
text_emb_shards_dict={}
domain_index_text_dict, domain_index_set={},()
domain_index_gnn_dict, index_domain_gnn_dict={},{}
def load_emb_dict(path="../../../data/Dec2024/gnn_random_v0", emb_pickle="domain_shard_index.pkl"):
    shared_key=emb_pickle.split('.')[0]
    if shared_key not in text_emb_shards_dict:        
        with open(f'{path}/{emb_pickle}', 'rb') as f:
            embd_dict = pickle.load(f)
        if isinstance(embd_dict[list(embd_dict.keys())[0]], list):
            embd_dict={ k:v[0][1] for k,v in embd_dict.items()}
        text_emb_shards_dict[shared_key]=embd_dict
    return text_emb_shards_dict[shared_key]

def process_inference_batch(idx,mlp_reg,batch_domains,gnn_emb_path,text_emb_path,gnn_emb_idx):
    try:
        # batch_domains = gnn_v[i:i + batch_size]
        logging.info(f"started: gnn_emb_idx={gnn_emb_idx}\t batch_idx={idx}")    
        global index_domain_text_dict,domain_index_set
        ############ collect gnn domains from gnn_emb_idx file ########
        gnn_emb_dict=search_parquet_duckdb(f_path=f"{gnn_emb_path}/shard_{gnn_emb_idx}.parquet",col="domain",q_domains=batch_domains,max_memory="8GB",threads=8,batch_size=1e4,schema={'key':'domain','val':'emb'})
        # gnn_emb_dict=search_parquet_duckdb(gnn_emb_path, parquet_name=f"shard_{gnn_emb_idx}.parquet", column_name="domain",search_values=batch_domains)
        ############ search for text emb across text emb shards and append with GNN emb ############
        text_emb_index_batch_dict={}
        text_emb_index_batch_dict[-1]=[] ## domains without text content emb
        for k in domain_index_set: ## for each shard in text emb shards
            text_emb_index_batch_dict[k]=[] 

        for k in gnn_emb_dict: ## for each domain in the batch gnn emb dict
            if k not in domain_index_text_dict:                    
                    domain_index_text_dict[k]=-1
            text_emb_index_batch_dict[domain_index_text_dict[k]].append(k) ## build -1 key domains list
        
        batch_scores_dict={}
        batch_text_emb_dict={}
        for text_emb_shard,domains in tqdm(text_emb_index_batch_dict.items()):
            iter_text_emb_dict={}
            if text_emb_shard==-1:
                iter_text_emb_dict={domain:[0]*256 for domain in domains}
            elif len(domains)>0:
                iter_text_emb_dict=search_parquet_duckdb(f_path=f"{text_emb_path}/{text_emb_shard}.parquet",col="domain",q_domains=domains,max_memory="8GB",threads=8,batch_size=1e4,schema={'key':'domain','val':'embeddings'},keep_emb_frist_elem="emb")
                # text_emb_dict=search_parquet_duckdb(text_emb_path, parquet_name=f"{text_emb_shard}.parquet", column_name="domain",search_values=domains)            
            else:
                continue     
            batch_text_emb_dict.update(iter_text_emb_dict)
            for domain in domains: ## Text + GNN emb
                iter_text_emb_dict[domain].extend(gnn_emb_dict[domain]) ## concat text and gnn emb
                
        final_emb_dict=batch_text_emb_dict        
        logging.info(f"Start MLP predict at gnn_emb_idx={gnn_emb_idx}\t batch_idx={idx}")
        if len(final_emb_dict)>0:
            pred_scores=mlp_reg.predict(list(final_emb_dict.values()))
            socres_dict=dict(zip(list(final_emb_dict.keys()), list(pred_scores.tolist())))
            batch_scores_dict.update(socres_dict)
                # logging.info(f"finished: gnn_emb_idx={gnn_emb_idx}\t batch_idx={idx}\ttext_emb_shard{text_emb_shard}")    
        logging.info(f"finished: gnn_emb_idx={gnn_emb_idx}\t batch_idx={idx}")
        return (batch_scores_dict,idx)
    except Exception as e:
        logging.error(f"Error in batch idx={idx} of gnn_emb_idx={gnn_emb_idx} with Exception: {e}")
        return ({},idx)
def result_callback(result):
    logging.info(f"\nCallback result received for batch idx={result[1]}\n")

def do_inference(args):
    args.text_emb_path=args.text_emb_path+f"{args.month}2024_wetcontent_embeddinggemma-300m/Credibench_{args.month}2024_wetcontent_embeddinggemma-300m/"
    args.gnn_emb_path=args.gnn_emb_path+f"{args.month[0].upper()}{args.month[1:]}2024/gnn_random_Feb2026/"
    #########################
    run_file_name=f"mlp_infer_{args.dataset}_{args.month}_{args.exec_mode}_embeddinggemma-300m"
    setup_logging(f"{args.model_path}/{run_file_name}.log")          
    logging.info(f"args={args}")  
    # set_ulimit()
    ############## Load Model ###############  
    if args.dataset=="dqr":     
        # modelname = f"dqr_{args.month}_pc1_pytorch_text_embeddinggemma-300m_GAT-text-avg_run0_agg_credibench_MLP_Model.pkl"
        modelname = f"dqr_{args.month}_pc1_pytorch_text_embeddinggemma-300m_GAT-text_20260403T001550_run0_credibench_MLP_Model.pkl"
        # modelname = f"dqr_{args.month}_pc1_sklearn_text_embeddinggemma-300m_GAT-text_20260404T210121_run0_credibench_MLP_Model.pkl"
    else:
        # modelname = f"weaksupervision_{args.month}_weak_label_pytorch_text_embeddinggemma-300m_GAT-text_run0_credibench_MLP_Model.pkl"
        modelname = f"DomainRel_{args.month}_weak_label_pytorch_text_embeddinggemma-300m_GAT-text_20260403T005912_run0_credibench_MLP_Model.pkl"
    mlp_reg=None
    with open(f'{args.model_path}/{modelname.split(args.month)[0]}{args.month}/{modelname}', 'rb') as f:
        mlp_reg = pickle.load(f)    
    # X_test_feat = [[0]*256*2]
    # pred = mlp_reg.predict(X_test_feat)
    ############## Load emb index ###############
    global domain_index_text_dict, domain_index_set   
    global domain_index_gnn_dict, index_domain_gnn_dict    
    domain_index_text_dict, domain_index_set    = load_emb_index(args.text_emb_path, f"{args.month}2024_wetcontent_domains_index.pkl",False)
    domain_index_set=set(domain_index_text_dict.values())
    domain_index_gnn_dict, index_domain_gnn_dict = load_emb_index(args.gnn_emb_path, "domain_shard_index.pkl",True)
    batch_size=int(1e5)
    logging.info(f'batch_size={batch_size}')
    logging.info(f'cpu_count={cpu_count()}')
    for gnn_k,gnn_v in tqdm(index_domain_gnn_dict.items()):
        if int(gnn_k) < args.start_idx or int(gnn_k) > args.end_idx:
            logging.info(f"Skipping shard:{gnn_k}")
            continue            
        ############### serial Execution ###############
        logging.info(f"shared {gnn_k} len of domains={len(gnn_v)}")
        if args.exec_mode=="serial":
            results=[]
            for i in tqdm(range(0, len(gnn_v), batch_size)):
                results.append(process_inference_batch(i,mlp_reg,gnn_v[i:i + batch_size],args.gnn_emb_path,args.text_emb_path,gnn_k))
        ############### parallel Execution ###############            
        elif args.exec_mode == "multiprocess":
            data=[(i,mlp_reg,gnn_v[i:i + batch_size],args.gnn_emb_path,args.text_emb_path,gnn_k) for i in range(0, len(gnn_v), batch_size)]       
            logging.info(f"Total number of batches={len(data)}")
            ########## Multiprocessing Pool ###############        
            with multiprocessing.Pool(processes=4) as pool:
                results = pool.starmap(process_inference_batch, data)
            # logging.info(results)
            pool.close()
        ########### Multithreading ###############
        elif args.exec_mode == "multithread":
            data=[(i,mlp_reg,gnn_v[i:i + batch_size],args.gnn_emb_path,args.text_emb_path,gnn_k) for i in range(0, len(gnn_v), batch_size)]       
            threads = []
            for d in data:
                threads.append(threading.Thread(target=process_inference_batch, args=d))
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        ########### Save Results###############
        batch_infer_dict={}
        for res in results:           
            batch_infer_dict.update(res[0])
        results=None
        
        col_name = "pc1_score" if args.dataset=="dqr" else "binary_score"
        pd.DataFrame(list(zip(batch_infer_dict.keys(), batch_infer_dict.values())),columns=["domain", col_name])\
        .to_parquet(f"{args.model_path}/mlpInfer_{modelname.split('.')[0]}_{gnn_k}.parquet",
                    engine='pyarrow',row_group_size=50_000,use_dictionary=True,compression="snappy")
def merge_pyarrow(files,outputpath):
    schema = pq.ParquetFile(files[0]).schema_arrow
    with pq.ParquetWriter(outputpath,schema=schema) as writer:
        for file in tqdm(files):
            writer.write_table(pq.read_table(file, schema=schema))
def merge_files(args):
    file_path_list=list_all_files(args.model_path, rgex= f"mlpInfer_{args.dataset}_*.parquet", recursive=False)
    merge_pyarrow(file_path_list,f"{args.model_path}/mlpInfer_{args.dataset}.parquet")
def main() -> None:
    root = str(get_root_dir())
    parser = argparse.ArgumentParser(description="MLP Inference")
    parser.add_argument("--text_emb_path", type=str, default=str("/home/mila/a/abdallah/scratch/hsh_projects/content_emb/content_emb/") ,help="text emb files path")
    parser.add_argument("--gnn_emb_path", type=str, default=str(root + "/data/") ,help="gnn emb files path")
    parser.add_argument("--model_path", type=str, default=str("/home/mila/a/abdallah/scratch/hsh_projects/CrediText/plots"),help="model path")
    parser.add_argument("--month", type=str, default="dec", choices=["oct", "nov", "dec"],help="CrediBench month snapshot")
    parser.add_argument("--exec_mode", type=str, default="multiprocess", choices=["serial", "multithread", "multiprocess"],help="execution mode")
    parser.add_argument("--dataset", type=str, default="dqr", choices=["dqr", "domainRel"],help="execution mode")
    parser.add_argument("--start_idx", type=int, default=0,help="GNN min shard idx")
    parser.add_argument("--end_idx", type=int, default=100,help="GNN max shard idx")
    args = parser.parse_args()
    do_inference(args)
    # merge_files(args)

         
def set_ulimit():
    # Define new soft and hard limits (e.g., 2048 for both)
    # Use resource.RLIM_INFINITY for an unlimited resource
    # Note: You may need root privileges to raise the hard limit
    new_soft_limit = 2048
    new_hard_limit = 4096 # Cannot exceed the system's current hard limit without root
    try:
        # Get the current limits (optional, for checking)
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        logging.info(f"Current open file limits (soft, hard): ({soft_limit}, {hard_limit})")

        # Set new limits
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft_limit, new_hard_limit))
        logging.info(f"New open file limits set to: ({new_soft_limit}, {new_hard_limit})")

        # Verify the new limits
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        logging.info(f"Verified limits (soft, hard): ({soft_limit}, {hard_limit})")

    except ValueError as e:
        logging.info(f"Error setting resource limit: {e}")
        logging.info("Check if the limits are valid (soft <= hard, and hard <= system hard limit).")
    except Exception as e:
        logging.info(f"An unexpected error occurred: {e}")
if __name__ == '__main__':   
    main()