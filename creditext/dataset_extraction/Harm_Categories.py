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
from sentence_transformers import SentenceTransformer
import torch
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



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ccmain content merge")
    parser.add_argument("--task", type=str, default="WebCategory", choices=["EmbCategory","WebCategory"], help="task type")
    parser.add_argument("--emb_model", type=str, default="google/embeddinggemma-300m", choices=["google/embeddinggemma-300m","paraphrase-multilingual-MiniLM-L12-v2"], help="task type")
    parser.add_argument("--ccmain", type=str, default="ccmain202451", help="ccmain")
    parser.add_argument("--base_path", type=str, default="/home/mila/a/abdallah/scratch/hsh_projects/CrediText/", help="base path")    
    parser.add_argument("--start_idx", type=int, default=0, help="frist chunck idx")    
    parser.add_argument("--end_idx", type=int, default=90000, help="end chunck idx")     
    args = parser.parse_args()
    print(f"args={args}") 
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
        print(f"device={torch.cuda.get_device_name(0)}")
    if args.task=="WebCategory":
        web_category_df=pd.read_csv(f"{args.base_path}/creditext/dataset_extraction/24_web_categories.csv")
        cat_lst=web_category_df["category"].tolist()
        cat_desc_lst=web_category_df["new description + keywords"].tolist()
        model = SentenceTransformer(args.emb_model)
        category_emb = model.encode(cat_desc_lst, batch_size=32, show_progress_bar=True)
        web_category_emb_df=pd.DataFrame({"category":cat_lst,"category_emb":list(category_emb)})
        web_category_emb_df.to_csv(f"{args.base_path}/creditext/dataset_extraction/24_web_category_{args.emb_model.split('/')[-1]}_emb.csv",index=False)

        bert_model_path=f"/home/mila/a/abdallah/scratch/hsh_projects/CrediText/domain_sampler/new_safe_bertopic" 
        embed_KeyBERT=False
        topicModler=DomainSampler(bert_model_path,embed_KeyBERT=embed_KeyBERT)
        category_topics_df=topicModler.category_topic_cosin_sim(category_emb,cat_lst,3)
        category_topics_df.to_parquet(f"{args.base_path}/creditext/dataset_extraction/24_web_category_{args.emb_model.split('/')[-1]}_topics{"_KeyBERT" if embed_KeyBERT else ''}.parquet",index=False)
        category_topics_df.to_csv(f"{args.base_path}/creditext/dataset_extraction/24_web_category_{args.emb_model.split('/')[-1]}_topics{"_KeyBERT" if embed_KeyBERT else ''}.csv",index=False)

    elif args.task=="EmbCategory":
        harm_taxonmy_df=pd.read_csv(f"{args.base_path}/creditext/dataset_extraction/harm_taxonomy.csv")
        harm_taxonmy_df=harm_taxonmy_df[["path","topical_description","harm_description"]]
        path_lst,topical_lst,harm_lst=[],[],[]
        for row in harm_taxonmy_df.itertuples():
            path_lst.append(row[1])
            topical_lst.append(row[2])
            harm_lst.append(row[3])

        model = SentenceTransformer(args.emb_model)
        topical_emb = model.encode(topical_lst, batch_size=32, show_progress_bar=True)
        harm_emb = model.encode(harm_lst, batch_size=32, show_progress_bar=True)
        taxonmy_topics_df=pd.DataFrame({"path":path_lst,"topical_emb":list(topical_emb),"harm_emb":list(harm_emb)})
        
        bert_model_path=f"/home/mila/a/abdallah/scratch/hsh_projects/CrediText/domain_sampler/safe_bertopic" 
        topicModler=DomainSampler(bert_model_path)
        topical_category_topics_df=topicModler.category_topic_cosin_sim(topical_emb,path_lst,3)
        topical_category_topics_df.to_parquet(f"{args.base_path}/creditext/dataset_extraction/harm_taxonomy_topical_{args.emb_model.split('/')[-1]}.parquet",index=False)
        topical_category_topics_df.to_csv(f"{args.base_path}/creditext/dataset_extraction/harm_taxonomy_topical_{args.emb_model.split('/')[-1]}_topics.csv",index=False)
        harm_category_topics_df=topicModler.category_topic_cosin_sim(harm_emb,path_lst,3)
        harm_category_topics_df.to_parquet(f"{args.base_path}/creditext/dataset_extraction/harm_taxonomy_harm_{args.emb_model.split('/')[-1]}.parquet",index=False)
        harm_category_topics_df.to_csv(f"{args.base_path}/creditext/dataset_extraction/harm_taxonomy_harm_{args.emb_model.split('/')[-1]}_topics.csv",index=False)

        # for col in category_topics_df.columns:
        #     taxonmy_topics_df["topical_"+col]=category_topics_df[col]        
        # for col in harm_topics_df.columns:
        #     taxonmy_topics_df["harm_"+col]=harm_topics_df[col]
        # taxonmy_topics_df.to_parquet(f"{args.base_path}/creditext/dataset_extraction/harm_taxonomy_{args.emb_model}_emb.parquet",index=False)
        # taxonmy_topics_df[['path', 'topical_topic0_id','topical_topic0_name', 'topical_topic0_prop', 'harm_topic0_id','harm_topic0_name', 'harm_topic0_prop']].to_csv(f"{args.base_path}/creditext/dataset_extraction/harm_taxonomy_{args.emb_model}_topics.csv",index=False)
    
    