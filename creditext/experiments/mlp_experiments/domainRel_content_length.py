import os
import pickle
from typing import Counter
from tqdm import tqdm
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from utils import search_parquet_duckdb,list_all_files
base_path="/home/mila/a/abdallah/scratch/hsh_projects"
def get_scraped_content_length_dict():
    scraped_content_df=pd.read_csv(f"{base_path}/CrediText/data/scrapedContent/weaklabels/WeakLabelsScrapedContent.csv")
    scraped_content_df["text"]=scraped_content_df["text"].astype(str)
    scraped_content_df["Content_Length"]=scraped_content_df.apply(lambda row:len(row["url"]) if row["text"]=="nan" else len(row["text"]), axis=1)
    scraped_pages_lengths_dict=dict(zip(scraped_content_df["url"].tolist(), scraped_content_df["Content_Length"].tolist()))
    pickle.dump(scraped_pages_lengths_dict, open(f'{base_path}/CrediText/creditext/experiments/mlp_experiments/domainrel_scraped_homepage_length_dict.pkl', 'wb'))
    return scraped_pages_lengths_dict
    
def get_monthly_content_length_dict(month:str = "Dec2024"):
    scraped_pages_df=pd.read_csv(f"{base_path}/CrediText/creditext/experiments/mlp_experiments/scraped_{month}_pages_urls.csv")
    scraped_pages_set=set(scraped_pages_df["page_url"].tolist()) 
    month_content_path=f"{base_path}/content_emb/content_emb/CrediBench-WebContent-{month}"
    all_content_files=list_all_files(month_content_path, rgex= "*.parquet", recursive=False)
    month_pages_lengths_dict={}
    for f in tqdm(all_content_files):
        f_lengths_dict=search_parquet_duckdb(f_path=f,col="WARC_Target_URI",q_domains=list(scraped_pages_set),max_memory="8GB",threads=8,batch_size=int(1e4),schema={'key':'WARC_Target_URI','val':'Content_Length'},keep_emb_frist_elem=None)
        month_pages_lengths_dict.update(f_lengths_dict)
        scraped_pages_set-=set(f_lengths_dict.keys())
    pickle.dump(month_pages_lengths_dict, open(f'{base_path}/CrediText/creditext/experiments/mlp_experiments/domainrel_{month}_pages_length_dict.pkl', 'wb'))
    return month_pages_lengths_dict
if __name__ == "__main__":
    # get_monthly_content_length_dict("Dec2024")
    # get_scraped_content_length_dict()
    scraped_pages_lengths_dict=pickle.load(open(f'{base_path}/CrediText/data/weaksupervision/domainrel_scraped_homepage_length_dict.pkl', 'rb'))
    month_pages_lengths_dict=pickle.load(open(f'{base_path}/CrediText/data/weaksupervision/domainrel_dec2024_pages_length_dict.pkl', 'rb'))
    all_pages_lengths_dict={**scraped_pages_lengths_dict, **month_pages_lengths_dict}
    doc_lengths=list(all_pages_lengths_dict.values())
    doc_lengths=sorted(doc_lengths)
    frequancy_dict=dict(Counter(doc_lengths))
    pinned_frequancy_dict={}
    pin_size=100
    for i in range (0, int(1e6), pin_size):
        pinned_frequancy_dict[i]=sum([frequancy_dict[k] if k in frequancy_dict else 0 for k in range(i,i+pin_size+1)])
    
    x = list(pinned_frequancy_dict.keys())
    y = list(pinned_frequancy_dict.values())
    plt.figure(figsize=(8,5))
    # plt.bar(x, y)
    plt.plot(x, y, linewidth=1)
    plt.yscale('log') 
    plt.xlabel("Content Length")
    plt.ylabel("Frequency")
    plt.title("frequancy of content lengths for Dec2024 domain's contents")
    plt.savefig(f"domainrel_content_length_frequancy.pdf", format="pdf", bbox_inches='tight')
    plt.show()
    