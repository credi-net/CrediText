import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import pyarrow.parquet as pq
import pickle
import argparse
import torch
import os
import pickle
import gc
import pyarrow as pa
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm   
from sampler import DomainSampler
from  creditext.experiments.mlp_experiments.utils import search_parquet_duckdb, list_all_files,write_domain_emb_parquet

month_to_cc_dict={"dec":"ccmain202451","nov":"ccmain202446","oct":"ccmain202442"}

def collect_embeddings(args):
    index_file_path=f"{args.index_path}/cc_full_index_{args.dataset}_cc_index_table_{month_to_cc_dict[args.month]}_0_299.parquet"
    index_df=search_parquet_duckdb(f_path=index_file_path,q_domains=None,filter_by_col=None,projection_cols=None,max_memory="8GB",
                                    threads =8,batch_size=int(1e4),schema=None)
    domain_urls_df=index_df[['url_host_name','url']]
    domain_urls_vc=domain_urls_df['url_host_name'].value_counts()
    domain_urls_sampled_dict={}
    for k,v in domain_urls_vc.items():
        domain_urls_sampled_dict[k]=DomainSampler.get_min_sample_size(v,confidence= 0.95, margin_error = 0.05)

    ########## Collect Embeddings for the sampled domains #############
    emb_path=f"{args.index_path}/warc_warc_bysampledoffset_{month_to_cc_dict[args.month]}/ln_gemma_emb/"
    emb_files_lst=list_all_files(root_path=emb_path, regex="*.parquet", recursive=False)
    dataset_emb_dict={}
    for i, f in enumerate(tqdm(emb_files_lst, desc="Processing embedding files")):
        print(f"Processing idx {i} file: {f}")
        emb_df=search_parquet_duckdb(f_path=f,q_domains=list(domain_urls_sampled_dict.keys()),filter_by_col="domain",projection_cols=None,max_memory="8GB",
                                        threads =8,batch_size=int(1e4),schema=None)
        # emb_df=emb_df[emb_df['domain'].isin(domain_urls_sampled_dict.keys())]        
        for row in emb_df.itertuples(index=False):
            domain = row.domain
            embeddings = row.embeddings
            if domain not in dataset_emb_dict:
                dataset_emb_dict[domain] = []

            for doc_emb in embeddings:
                dataset_emb_dict[domain].append(doc_emb)
                domain_urls_sampled_dict[domain] -= 1
        del emb_df
        gc.collect()
        domain_urls_sampled_dict={k: v for k, v in domain_urls_sampled_dict.items() if v > 0}
        print(f"Remaining domains to sample max-min: {max(domain_urls_sampled_dict.values())}, {min(domain_urls_sampled_dict.values())}")
        # break
        if i%50==0:
            with open(f"{args.data_path}/{args.dataset}_{args.month}2024_gemma300m_cochan_sampled.pkl", "wb") as file:
                pickle.dump(dataset_emb_dict, file)

    with open(f"{args.data_path}/{args.dataset}_{args.month}2024_gemma300m_cochran_sampled_all.pkl", "wb") as file:
                    pickle.dump(dataset_emb_dict, file)
    # domain_rel_p1_df=search_parquet_duckdb(f_path=f"{args.index_path}/warc_warc_bysampledoffset_ccmain202451/domainRel_ccmain202451_sampled_gemma300_emb_p1.parquet",q_domains=None,filter_by_col=None,projection_cols=None,max_memory="8GB",
            #                                        threads =8,batch_size=int(1e4),schema=None)
            # domain_rel_p2_df=search_parquet_duckdb(f_path=f"{args.index_path}/warc_warc_bysampledoffset_ccmain202451/domainRel_ccmain202451_sampled_gemma300_emb_p2.parquet",q_domains=None,filter_by_col=None,projection_cols=None,max_memory="8GB",
            #                                            threads =8,batch_size=int(1e4),schema=None)  
            # content_df=search_parquet_duckdb(f_path="/home/mila/a/abdallah/scratch/hsh_projects/CrediText/data/weaksupervision/weak_sampled_content_emb_dec2024_embeddinggemma-300m_768.parquet",q_domains=None,filter_by_col=None,projection_cols=None,max_memory="8GB",
            #                                        threads =8,batch_size=int(1e4),schema=None)
     

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='TE3L', choices=['Qwen3-Embedding-8B-Q5_K_M','TE3L'], help='Name of the pre-trained model to use for embeddings')
    parser.add_argument('--month', type=str, default='dec', choices=['oct', 'nov', 'dec'], help='Month for which to sample content')
    parser.add_argument('--dataset', type=str, default='dqr', choices=['fullMonth', 'domain_rel', 'dqr'], help='Month subset')
    parser.add_argument('--data_path', type=str, default='/home/mila/a/abdallah/scratch/hsh_projects/CrediText/data',  help='data path')
    parser.add_argument('--index_path', type=str, default='/home/mila/a/abdallah/scratch/hsh_projects/CrediText/bash_scripts/spark-warehouse',  help='data path')
    parser.add_argument('--task', type=str, default='doc_cluster_weighting', choices=['collect_embeddings', 'sample_content', 'doc_cluster_weighting'], help='Task to perform')
    args = parser.parse_args()
    print("args=", args)

    if args.task=="collect_embeddings":
        collect_embeddings(args)
    elif args.task=="doc_cluster_weighting":
        with open(f"{args.data_path}/{args.dataset}/cochran/{args.dataset}_dec2024_gemma300m_cochran_sampled_all.pkl", "rb") as file:
                    domainrel_emb_dict = pickle.load(file)
        parquet_rows={'domain':domainrel_emb_dict.keys(),'embeddings':domainrel_emb_dict.values()}
        write_domain_emb_parquet(parquet_rows,f"{args.data_path}/{args.dataset}/cochran/",f"{args.dataset}_dec2024_gemma300m_cochan_sampled.parquet")
        # del parquet_rows
        ds=DomainSampler()
        dataset_cluster_props_lst=[]
        for k,v in tqdm(domainrel_emb_dict.items()):
            emb_lst=[elem['emb'] for elem in v]
            urls_lst=[elem['page'] for elem in v]
            topics_df=ds.topic_analysis_cosin_sim(emb_lst,topk=3)
            for i,row in enumerate(topics_df.itertuples(index=False), start=0):
                page_attributes_lst=[k,urls_lst[i]]
                for topic_idx in range(0,3):                    
                    page_attributes_lst.append(getattr(row, f'topic{topic_idx}_id'))
                    # v[i][f'topic{topic_idx}_name']=getattr(row, f'topic{topic_idx}_name')
                    page_attributes_lst.append(getattr(row, f'topic{topic_idx}_prop'))
                dataset_cluster_props_lst.append(page_attributes_lst)
        dataset_cluster_props_df=pd.DataFrame(dataset_cluster_props_lst,columns=['domain','url','topic0_id','topic0_prop','topic1_id','topic1_prop','topic2_id','topic2_prop'])
        # domainrel_cluster_props_df.set_index('domain', inplace=True)
        dataset_cluster_props_df.to_parquet(f"{args.data_path}/{args.dataset}/cochran/{args.dataset}_dec2024_gemma300m_cochan_sampled_cluster_props.parquet",engine="pyarrow",row_group_size=1000,use_dictionary=["domain","topic0_id"])
        # print(f"Domain: {k}, Number of embeddings: {len(v)}")
        


        


        
    
    

        



        
        