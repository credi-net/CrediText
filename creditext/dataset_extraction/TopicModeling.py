from importlib.resources import path

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
import numpy as np
import pyarrow as pa
from creditext.experiments.mlp_experiments.utils import search_parquet_duckdb, list_all_files,write_domain_emb_parquet,query_parquet_duckdb,write_domain_topics_parquet 
from creditext.experiments.mlp_experiments.dataset_loader import DQR, DomainRel
from domain_sampler import DomainSampler


def list_parquet_files(base_path="",path="warc_index_table_ccmain202508",start=0,end=300,batch_size=10,matching_paths=[],match_regex="*.snappy.parquet",nested_folders=True):    
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
def build_domains_index_parquet(emb_files_lst:list=[],start_idx:int=0,end_idx:int=-1):
    end_idx=len(emb_files_lst) if end_idx==-1 else end_idx
    domains_index_dict={}
    for emb_f in tqdm(emb_files_lst[start_idx:end_idx]):
        # print(f"file={emb_f}")
        SQL_Query=f"SELECT domain FROM read_parquet('{emb_f}')"
        res_df=query_parquet_duckdb(SQL_Query=SQL_Query)
        domains_lst=res_df["domain"].tolist()        
        out_file_index=emb_f.split('/')[-2].split("_")[-2]
        for domain in domains_lst:
            if domain in domains_index_dict:
                domains_index_dict[domain].append(out_file_index)
            else:
                domains_index_dict[domain]=[out_file_index]
    index_df=pd.DataFrame(zip(list(domains_index_dict.keys()),list(domains_index_dict.values())),columns=["domain","shards"])
    index_df.to_parquet(f"{'/'.join(emb_f.split('/')[:-2])}_shards_index_{start_idx}_{end_idx}.parquet")
    return index_df

def pkl_to_parquet(file_path_dict,embedding_model_name="gemma300",start_idx=0,end_idx=10):
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
        # out_file_index=f.split('/')[-1].split("_")[-4]
        if len(k_dict)>0:
            domains_lst=list(k_dict.keys())
            content_emb_dict={'domain':domains_lst, 'embeddings':list(k_dict.values())}  
            for domain in domains_lst:
                if domain in domains_index_dict:
                    domains_index_dict[domain].append(out_file_index)
                else:
                    domains_index_dict[domain]=[out_file_index]
            del k_dict
            del domains_lst
            gc.collect()
            write_domain_emb_parquet(content_emb_dict,f"{"/".join(f.split('/')[:-1])}", f"{k.split(".")[0]}_{embedding_model_name}_emb.parquet")
            del content_emb_dict
            gc.collect()
            ######### delete pkl files to save storage #######
            for f in v:
                os.remove(f)
    pd.DataFrame(zip(list(domains_index_dict.keys()),list(domains_index_dict.values())),columns=["domain","shards"])\
        .to_parquet(f"{'/'.join(f.split('/')[:-2])}_shards_index_{start_idx}_{end_idx}.parquet")

def topic_modeling(emb_files_lst:list=[],bert_model_path:str="safe_bertopic", start_idx:int=0,end_idx:int=-1,topk=3,query_batch_size=int(10**5)):
    end_idx=len(emb_files_lst) if end_idx==-1 else end_idx
    topicModler=DomainSampler(bert_model_path)
    for emb_f in tqdm(emb_files_lst[start_idx:end_idx]):
        print(f"file={emb_f}")
        Count_Query=f"SELECT count(*) as count FROM read_parquet('{emb_f}')"
        count_df=query_parquet_duckdb(SQL_Query=Count_Query)
        rows_count=int(count_df["count"][0:1].values[0])
        print(f"rows_count={rows_count}")
        file_topics_df_lst=[]        
        for off in tqdm(range(0,rows_count,query_batch_size)):
            SQL_Query=f"SELECT * FROM read_parquet('{emb_f}') offset {off} limit {query_batch_size}"
            res_df=query_parquet_duckdb(SQL_Query=SQL_Query)            
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
                file_topics_df_lst.append(doc_topics_df.copy())
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
def get_domainRel_text_embeddings(emb_model="gemma300",ccmain="ccmain202451",base_path="/home/mila/a/abdallah/scratch/hsh_projects/CrediText"):
    # domains_lst=DomainRel.get_domains_lst()
    train_df,val_df,test_df=DomainRel.load_splits(f"{base_path}/data/weaksupervision")
    domainRel_domains_set=set(train_df["domain"].tolist()).union(set(val_df["domain"].tolist())).union(set(test_df["domain"].tolist()))
    con = duckdb.connect()
    con.register("domain_list_tbl", pd.DataFrame({"domain": list(domainRel_domains_set)}))
    files_path=f"warc_warc_bysampledoffset_{ccmain}"
    base_path=f"{base_path}/bash_scripts/spark-warehouse/{files_path}"   
    emb_files_lst=list_parquet_files(base_path,files_path,0,90000,50,[],match_regex=f"*_{emb_model}*_emb*.parquet",nested_folders=True)                
    domain_rel_emb_dict={}
    for idx,emb_f in tqdm(enumerate(emb_files_lst)):
        # if idx<=1000:
        #     continue
        # print(f"file={emb_f}")
        res_df=con.sql(f"""
        SELECT b.*
        FROM read_parquet('{emb_f}') b
        SEMI JOIN domain_list_tbl l ON b.domain = l.domain
        """).df()
        iter=zip(res_df['domain'].to_numpy(), res_df['embeddings'].to_numpy())
        del res_df
        for domain, emb in iter:
            if domain in domain_rel_emb_dict:
                domain_rel_emb_dict[domain].extend(list(emb))
            else:
                domain_rel_emb_dict[domain]=list(emb)
        del iter
        gc.collect() 
        if idx %100==0:
            domainRel_sampled_emb_dict={'domain':list(domain_rel_emb_dict.keys()), 'embeddings':list(domain_rel_emb_dict.values())}              
            write_domain_emb_parquet(domainRel_sampled_emb_dict,f"{base_path}", f"domainRel_{ccmain}_sampled_{emb_model}_emb_{idx}.parquet")
            del domainRel_sampled_emb_dict
            gc.collect() 
    domainRel_sampled_emb_dict={'domain':list(domain_rel_emb_dict.keys()), 'embeddings':list(domain_rel_emb_dict.values())}              
    write_domain_emb_parquet(domainRel_sampled_emb_dict,f"{base_path}", f"domainRel_{ccmain}_sampled_{emb_model}_emb.parquet")
    del domainRel_sampled_emb_dict
    gc.collect() 
    return domain_rel_emb_dict


def merge_2parquet_emb_files(file1,file2):    
    df1 = pd.read_parquet(file1, engine="pyarrow")
    domain_rel_emb_dict=dict(zip(df1["domain"], df1["embeddings"]))
    del df1
    gc.collect()
    df2 = pd.read_parquet(file2, engine="pyarrow")   
    iter=zip(df2['domain'].to_numpy(), df2['embeddings'].to_numpy())
    del df2
    gc.collect()
    for domain, emb in iter:
        if domain in domain_rel_emb_dict:
            domain_rel_emb_dict[domain]=np.append(domain_rel_emb_dict[domain], emb)
        else:
            domain_rel_emb_dict[domain]=emb
    del iter
    gc.collect() 
    domainRel_sampled_emb_dict={'domain':list(domain_rel_emb_dict.keys()), 'embeddings':list(domain_rel_emb_dict.values())}
    del domain_rel_emb_dict
    gc.collect()
    merged_file_path=file1.split("/")[-1].replace(".parquet", "_merged.parquet")
    base_path="/".join(file1.split("/")[:-1])
    write_domain_emb_parquet(domainRel_sampled_emb_dict,base_path, merged_file_path)
    return domain_rel_emb_dict
def compare_2parquet_emb_files(file1,file2):
    ds1_dict=search_parquet_duckdb(file1, filter_by_col="domain",q_domains=None,max_memory="8GB",schema={'key':'domain','val':'embeddings'})
    ds2_dict=search_parquet_duckdb(file2, filter_by_col="domain",q_domains=None,max_memory="8GB",schema={'key':'domain','val':'embeddings'})
    return len(ds1_dict),len(ds2_dict),len(set(ds1_dict.keys()).intersection(set(ds2_dict.keys())))
def query_topics_per_doc(topics_parquet_file_path):
    SQL_count_query=f""" select count(*) as count from read_parquet('{topics_parquet_file_path}') """
    res_df=query_parquet_duckdb(SQL_Query=SQL_count_query,max_memory="16GB")
    rows_count=int(res_df["count"][0:1].values[0])
    bs=int(10**6)
    results=[]
    for offset in tqdm(range(0,rows_count,bs)):     
        SQL_Query=f""" select topic, count(*) as count from (
                SELECT domain,X['unnest'].topic0_id as topic FROM (
                select * from read_parquet('{topics_parquet_file_path}') offset {offset} limit {bs}) ,
                UNNEST(topics) AS X) 
                group by topic ORDER BY count(*) DESC """
        res_df=query_parquet_duckdb(SQL_Query=SQL_Query,max_memory="16GB")        
        # res_df.to_csv(topics_parquet_file_path.replace(".parquet",f"_counts_{offset}.csv"),index=None)
        results.append(res_df)
    final_result=pd.concat(results,ignore_index=True)
    final_result = final_result.groupby('topic')['count'].sum().reset_index()
    final_result=final_result.sort_values(by=["count"])
    final_result.to_csv(topics_parquet_file_path.replace(".parquet",f"_counts.csv"),index=None)
    return final_result 

def query_topics_list(topics_parquet_file_path):
    SQL_count_query=f""" select count(*) as count from read_parquet('{topics_parquet_file_path}') """
    res_df=query_parquet_duckdb(SQL_Query=SQL_count_query,max_memory="16GB")
    rows_count=int(res_df["count"][0:1].values[0])
    bs=int(10**6)
    results=[]
    for offset in tqdm(range(0,rows_count,bs)):     
        SQL_Query=f"""
                SELECT distinct X['unnest'].topic0_name as topic FROM (
                select * from read_parquet('{topics_parquet_file_path}') offset {offset} limit {bs}) ,
                UNNEST(topics) AS X """
        res_df=query_parquet_duckdb(SQL_Query=SQL_Query,max_memory="16GB")        
        # res_df.to_csv(topics_parquet_file_path.replace(".parquet",f"_counts_{offset}.csv"),index=None)
        results.append(res_df)
    final_result=pd.concat(results,ignore_index=True)
    final_result=final_result.drop_duplicates()
    final_result.to_csv(topics_parquet_file_path.replace(".parquet",f"_names.csv"),index=None)
    return final_result 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ccmain content merge")
    parser.add_argument("--task", type=str, default="compare2datasets", choices=["TopicModeling","PklToParquet","buildIndex","queryTopics","domainRel","merge2_emb_parquet","compare2datasets"], help="task type")
    parser.add_argument("--emb_model", type=str, default="gemma300", choices=["gemma300","paraphrase-multilingual-MiniLM-L12-v2"], help="task type")
    parser.add_argument("--ccmain", type=str, default="ccmain202451", help="ccmain")
    parser.add_argument("--base_path", type=str, default="/home/mila/a/abdallah/scratch/hsh_projects/CrediText/bash_scripts/spark-warehouse", help="base path")    
    parser.add_argument("--start_idx", type=int, default=0, help="frist chunck idx")    
    parser.add_argument("--end_idx", type=int, default=90000, help="end chunck idx")     
    args = parser.parse_args()
    print(f"args={args}")     
    
    if args.task=="compare2datasets":
        file1=f"/home/mila/a/abdallah/scratch/hsh_projects/CrediText/data/weaksupervision/weak_sampled_content_emb_dec2024_embeddinggemma-300m_768.parquet"
        file2=f"/home/mila/a/abdallah/scratch/hsh_projects/CrediText/data/weaksupervision/weak_content_emb_dec2024_embeddinggemma-300m_768.parquet"
        compare_2parquet_emb_files(file1,file2)
    elif args.task=="domainRel":
        get_domainRel_text_embeddings(emb_model=args.emb_model,ccmain=args.ccmain)
    elif args.task=="merge2_emb_parquet":
        file1=f"{args.base_path}/warc_warc_bysampledoffset_ccmain202451/domainRel_ccmain202451_sampled_gemma300_emb_1000_p1.parquet"
        file2=f"{args.base_path}/warc_warc_bysampledoffset_ccmain202451/domainRel_ccmain202451_sampled_gemma300_emb_p2.parquet"
        merge_2parquet_emb_files(file1,file2)
    elif args.task=="PklToParquet":
        file_path_dict={}        
        # files_path=f"CrediBench-WebContent-Dec2024"
        # base_path=f"{args.base_path}" 
        # file_path_list=list_parquet_files(base_path,files_path,args.start_idx,args.end_idx,50,[],match_regex=f"*{args.emb_model}*.pkl",nested_folders=False)
        # for f in file_path_list:
        #     k=f.split("/")[-1] 
        #     file_path_dict[k]=[f]
         
        files_path=f"warc_warc_bysampledoffset_{args.ccmain}"
        base_path=f"{args.base_path}/{files_path}" 
        file_path_list=list_parquet_files(base_path,files_path,args.start_idx,args.end_idx,50,[],match_regex=f"*{args.emb_model}.pkl")
        for f in file_path_list:
            k=f.split("/")[-2]
            if k not in file_path_dict:
                file_path_dict[k]=[]
            file_path_dict[k].append(f)
        pkl_to_parquet(file_path_dict,args.emb_model,args.start_idx,args.end_idx)
    elif args.task=="buildIndex":
        files_path=f"warc_warc_bysampledoffset_{args.ccmain}"
        base_path=f"{args.base_path}/{files_path}"   
        emb_files_lst=list_parquet_files(base_path,files_path,args.start_idx,args.end_idx,50,[],match_regex=f"*{args.emb_model}_emb.parquet")
        build_domains_index_parquet(emb_files_lst,0,len(emb_files_lst))
    elif args.task=="TopicModeling":        
        bert_model_path=f"/home/mila/a/abdallah/scratch/hsh_projects/CrediText/domain_sampler/safe_bertopic" 
        
        # files_path=f"warc_warc_bysampledoffset_{args.ccmain}"
        # base_path=f"{args.base_path}/{files_path}"   
        # emb_files_lst=list_parquet_files(base_path,files_path,args.start_idx,args.end_idx,50,[],match_regex=f"*_{args.emb_model}_emb*.parquet",nested_folders=False)                
        
        files_path=f"CrediBench-WebContent-Dec2024"
        base_path=f"{args.base_path}"   
        emb_files_lst=list_parquet_files(base_path,files_path,args.start_idx,args.end_idx,50,[],match_regex=f"*_{args.emb_model}_emb*.parquet",nested_folders=False)
        
        topic_modeling(emb_files_lst,bert_model_path,args.start_idx,args.end_idx+1)
    elif args.task=="queryTopics":
        # topics_file_path=f"{args.base_path}/warc_bysampledoffset_topics_ccmain202451.parquet"
        topics_file_path=f"{args.base_path}/CrediBench-WebContent-Dec2024/Dec2024_max6_topics_top3.parquet"
        query_topics_per_doc(topics_file_path)
        query_topics_list(topics_file_path)
       
   
    