from matplotlib.pyplot import table
import pandas as pd
import glob
import os
from tqdm import tqdm
import urllib
import gc
import pyarrow.parquet as pq
import argparse
import duckdb
import pickle
import pyarrow as pa
from creditext.experiments.mlp_experiments.utils import search_parquet_duckdb, list_all_files,write_domain_emb_parquet,query_parquet_duckdb,write_domain_topics_parquet 
from creditext.experiments.mlp_experiments.dataset_loader import DQR, DomainRel
from domain_sampler import DomainSampler


base_path=None
def list_parquet_files(path="warc_index_table_ccmain202508",start=0,end=300,batch_size=10,matching_paths=[],match_regex="*.snappy.parquet",nested_folders=True):    
    content_batches_df_lst=[]
    if len(matching_paths)==0 and nested_folders:
        matching_paths=[]
        for i in range(start,end,batch_size):
            matching_paths.append(f"{base_path}/{path}_{i}_{i+batch_size-1}")
    elif not nested_folders:
        matching_paths.append(f"{base_path}/{path}")
    file_path_list=[]
    for p in tqdm(matching_paths):
        files = glob.glob(p+f"/{match_regex}")
        for f in files:
            file_path_list.append(f)
    return file_path_list
def merge_pyarrow(files,outputpath):
    schema = pq.ParquetFile(files[0]).schema_arrow
    with pq.ParquetWriter(outputpath,schema=schema) as writer:
        for file in tqdm(files):
            tbl=pq.read_table(file, schema=schema)
            writer.write_table(tbl)
def pkl_to_parquet(file_path_dict):
    '''rows format: {domain:[{"page":page,"emb":[float]}]}'''
    domains_index_dict={}
    for k, v in tqdm(file_path_dict.items()):
        k_dict = {}
        for f in v:
            f_dict = pickle.load(open(f, "rb"))
            print(f"processing file {f.split('/')[-2:]} with {len(f_dict)} domains")
            for domain, url_emb_lst in f_dict.items():
                if domain not in k_dict:
                    k_dict[domain] = [ {'page':elem[0],'emb':elem[1]} for elem in url_emb_lst]
                else:
                    k_dict[domain].extend([{'page':elem[0],'emb':elem[1]} for elem in url_emb_lst])
            del f_dict
            gc.collect()
        out_file_index=f.split('/')[-2].split("_")[-2]
        if len(k_dict)>0:
            content_emb_dict={'domain':k_dict.keys(), 'embeddings':k_dict.values()}  
            for domain in content_emb_dict:
                if domain in domains_index_dict:
                    domains_index_dict[domain].appned(out_file_index)
                else:
                    domains_index_dict[domain]=[out_file_index]
            del k_dict
            gc.collect()
            write_domain_emb_parquet(content_emb_dict,f"{"/".join(f.split('/')[:-1])}", f"{k}_gemma300m_emb.parquet")
            del content_emb_dict
            gc.collect()
            ######### delete pkl files to save storage #######
            for f in v:
                os.remove(f)
    pd.DataFrame(zip(list(domains_index_dict.keys()),list(domains_index_dict.values)),columns=["domain","shards"])\
        .to_parquet(f"{'/'.join(f.split('/')[:-2])}_shards_index.parquet")

def topic_modeling(emb_files_lst:list=[],bert_model_path:str="safe_bertopic", start_idx:int=0,end_idx:int=-1,topk=3):
    end_idx=len(emb_files_lst) if end_idx==-1 else end_idx
    topicModler=DomainSampler(bert_model_path)
    for emb_f in tqdm(emb_files_lst[start_idx:end_idx]):
        print(f"file={emb_f}")
        SQL_Query=f"SELECT * FROM read_parquet('{emb_f}')"
        res_df=query_parquet_duckdb(SQL_Query=SQL_Query)
        file_topics_df_lst=[]
        bs=10**4
        for i in tqdm(range(0,len(res_df),bs)):
            batch_df=res_df[i:i+bs]
            domains_lst=batch_df["domain"].tolist()
            url_emb_lst=batch_df["embeddings"].tolist() 
            url_domain_lst =[[element['page'],domains_lst[idx]] for idx,domain_emb in enumerate(url_emb_lst) for element in domain_emb]
            urls_lst =[elem[0] for elem in url_domain_lst]
            domains_lst =[elem[1] for elem in url_domain_lst]
            emb_lst =[element['emb'] for domain_emb in url_emb_lst for element in domain_emb]
            # topicModler.topic_analysis(urls_lst,urls_lst,emb_lst,deep_analysis=False,is_html=False)
            doc_topics_df=topicModler.topic_analysis_cosin_sim(emb_lst,topk)
            doc_topics_df["url"]=urls_lst
            doc_topics_df["domain"]=domains_lst
            file_topics_df_lst.append(doc_topics_df)
        file_topics_df=pd.concat(file_topics_df_lst)
        del file_topics_df_lst
        gc.collect()  
        cols_to_keep = [col for col in file_topics_df.columns if col != 'domain']
        grouped_topics_df=file_topics_df.groupby('domain')[cols_to_keep].apply(lambda x: x.to_dict('records')).to_dict()
        grouped_topics_dict={'domain':list(grouped_topics_df.keys()), 'topics':list(grouped_topics_df.values())}              
        write_domain_topics_parquet(grouped_topics_dict,f"{"/".join(emb_f.split('/')[:-1])}", emb_f.split("/")[-1].replace("emb",f"topics_top{topk}"))
        del grouped_topics_dict
        del grouped_topics_df
        gc.collect()        
        




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ccmain content merge")
    parser.add_argument("--task", type=str, default="PklToParquet", choices=["TopicModeling","PklToParquet"], help="task type")
    parser.add_argument("--ccmain", type=str, default="ccmain202451", help="ccmain")
    parser.add_argument("--base_path", type=str, default="/home/mila/a/abdallah/scratch/hsh_projects/CrediText/bash_scripts/spark-warehouse", help="base path")    
    parser.add_argument("--start_idx", type=int, default=0, help="frist chunck idx")    
    parser.add_argument("--end_idx", type=int, default=90000, help="end chunck idx")     
    args = parser.parse_args()
    print(f"args={args}")
 
    ############### Merge DQR HTML Parquet Files #############
    if args.task=="PklToParquet":
        files_path=f"warc_warc_bysampledoffset_{args.ccmain}"
        base_path=f"{args.base_path}/{files_path}"    
        file_path_list=list_parquet_files(files_path,args.start_idx,args.end_idx,50,[],match_regex="*paraphrase-multilingual-MiniLM-L12-v2.pkl")
        file_path_dict={}
        for f in file_path_list:
            k=f.split("/")[-2]
            if k not in file_path_dict:
                file_path_dict[k]=[]
            file_path_dict[k].append(f)
        pkl_to_parquet(file_path_dict)
    elif args.task=="TopicModeling":
        files_path=f"warc_warc_bysampledoffset_{args.ccmain}"
        base_path=f"{args.base_path}/{files_path}"   
        bert_model_path=f"/home/mila/a/abdallah/scratch/hsh_projects/CrediText/domain_sampler/safe_bertopic" 
        emb_files_lst=list_parquet_files(files_path,args.start_idx,args.end_idx,50,[],match_regex=f"*_gemma300m_emb*.parquet")
        topic_modeling(emb_files_lst,bert_model_path,0,len(emb_files_lst))
       
   
    