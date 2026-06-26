from calendar import c
import multiprocessing
from duckdb import df
import pandas as pd
from regex import P
from shap import sample
from tqdm import tqdm
from creditext.experiments.mlp_experiments.utils  import query_parquet_duckdb, list_all_files
import random
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
logging.basicConfig(level=logging.INFO)
from collections import Counter
import numpy as np
global_path="/home/mila/a/abdallah/scratch/hsh_projects/CrediText/bash_scripts/spark-warehouse"
def get_doc_domain_counts_vc(index_path="warc_index_table_content_table_ccmain202446",month="202446"):
    logging.info(f"Processing index {index_path}...")
    month_dir=f"{global_path}/{index_path}"
    parquet_files_lst=list_all_files(month_dir, "*.parquet", recursive=True)
    counts_df_lst=[] 
    for f_path in tqdm(parquet_files_lst):    
        domain_doc_count_df=query_parquet_duckdb(SQL_Query=f"SELECT url_host_name,count(url_host_name) as doc_count FROM read_parquet('{f_path}') group by url_host_name order by count(url_host_name) desc")
        counts_df_lst.append(domain_doc_count_df)
    all_counts_df=pd.concat(counts_df_lst)
    doc_domain_count_df=all_counts_df.groupby("url_host_name").sum().reset_index().sort_values("doc_count", ascending=False)
    doc_domain_count_vc=doc_domain_count_df["doc_count"].value_counts().sort_index()
    doc_domain_count_df=pd.DataFrame({"doc_count":list(doc_domain_count_vc.index), "domains_count":list(doc_domain_count_vc.values)})
    doc_domain_count_df.to_parquet(f"{global_path}/month_doc_domain_frequency_{month}.parquet", compression='gzip')
    return doc_domain_count_df["doc_count"].value_counts().sort_index()

def get_doc_length_counts_vc(index_path="warc_index_table_content_table_ccmain202446",month="202446"):
    logging.info(f"Processing index {index_path}...")
    month_dir=f"{global_path}/{index_path}"
    parquet_files_lst=list_all_files(month_dir, "*.parquet", recursive=True)
    counts_df_lst=[] 
    doc_length_dict={}
    for f_path in tqdm(parquet_files_lst):    
        doc_length_df=query_parquet_duckdb(SQL_Query=f"SELECT warc_record_length FROM read_parquet('{f_path}')")
        for idx,row in doc_length_df.itertuples():
            doc_length_dict[row] = doc_length_dict.get(row, 0) + 1
    month_doc_length_df=pd.DataFrame(list(doc_length_dict.items()), columns=['doc_length', 'count'])
    month_doc_length_df["doc_length_bin"] = month_doc_length_df["doc_length"].apply(lambda x: int(x/1000))
    length_dict = month_doc_length_df.groupby('doc_length_bin')['count'].sum().to_dict()
    doc_len_df=pd.DataFrame(list(length_dict.items()), columns=['length', 'count'])
    doc_len_df.to_parquet(f"{global_path}/month_doc_length_frequency_{month}.parquet", compression='gzip')
    return pd.Series(length_dict).sort_index(ascending=True)
def get_languages_doc_counts_vc(index_path="warc_index_table_content_table_ccmain202446",month="202446"):
    logging.info(f"Processing index_path {index_path}...")
    month_dir=f"{global_path}/{index_path}"
    parquet_files_lst=list_all_files(month_dir, "*.parquet", recursive=True)
    counts_df_lst=[] 
    languages_dict={}
    for f_path in tqdm(parquet_files_lst):    
        languages_df=query_parquet_duckdb(SQL_Query=f"SELECT content_languages FROM read_parquet('{f_path}')")
        for idx,row in languages_df.itertuples():
            for leng in row.split(","):
                languages_dict[leng] = languages_dict.get(leng, 0) + 1
    month_lang_df=pd.DataFrame(list(languages_dict.items()), columns=['language', 'count'])
    month_lang_df.to_parquet(f"{global_path}/month_languages_frequency_{month}.parquet", compression='gzip')
    return pd.Series(languages_dict).sort_values(ascending=False)
def get_topic_doc_counts_vc(index_path="warc_index_table_content_table_ccmain202446",month="202446"):
    logging.info(f"Processing index_path {index_path}...")
    month_dir=f"{global_path}/{index_path}"
    parquet_files_lst=list_all_files(month_dir, "*gemma300m_topics_top3.parquet", recursive=True)
    counts_df_lst=[] 
    docs_per_topic_counts_dict=None
    domain_prt_topic_counts_dict={}
    for f_path in tqdm(parquet_files_lst):    
        topics_df=query_parquet_duckdb(SQL_Query=f"SELECT * FROM read_parquet('{f_path}')")
        batch_topics_count_dict=dict(Counter([topic['topic0_name'] for doc in topics_df["topics"].tolist() for topic in doc ]))
        if not docs_per_topic_counts_dict:
            docs_per_topic_counts_dict=batch_topics_count_dict
        else:
            for k in batch_topics_count_dict.keys():
                if k in docs_per_topic_counts_dict:
                    docs_per_topic_counts_dict[k]+=batch_topics_count_dict[k]
                else:
                    docs_per_topic_counts_dict[k]=batch_topics_count_dict[k]

    # month_lang_df.to_parquet(f"{global_path}/month_languages_frequency_{month}.parquet", compression='gzip')
    # return pd.Series(languages_dict).sort_values(ascending=False)
    return pd.Series(docs_per_topic_counts_dict).sort_values(ascending=False)

def plot_doc_domain_frequency(vc,month="dec2024", output="doc_domain_frequency.pdf", y_log_scale=True):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 5))
    plt.bar(vc.index, vc.values, width=0.8,alpha=0.7)
    plt.xticks(range(0, max(vc.index), 10))
    if y_log_scale:
        plt.yscale("log")   
    plt.xlabel("Docs Count")
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Domains Count")
    plt.title("Document-Domain Frequency Plot")
    plt.tight_layout()
    plt.savefig(f"{month}_{'log_' if y_log_scale else ''}{output}", format="pdf", bbox_inches="tight")
    plt.show()
    plt.close()
def plot_doc_domain_frequency_multi(vc_dict={},month="dec2024", output="doc_domain_frequency_multi.pdf", y_log_scale=True,chart_type="bar"):
    import matplotlib.pyplot as plt    
    plt.figure(figsize=(15, 5) if y_log_scale else (8, 3))
    # vc_dict=dict(reversed(list(vc_dict.items()))) # reverse becouse Max6 has more docs per bin than others
    max_doc_count=10000 if y_log_scale else 100
    width=0.8 if  y_log_scale else 2
    doc_step=int(max_doc_count/10)
    for key,vc in vc_dict.items():
        if key=="All":
            vc=vc[vc.index<=max_doc_count]
        plt.bar(vc.index, vc.values, width=width, label=key,alpha=0.7)
    plt.xticks(range(0, max_doc_count, doc_step),labels=[f"{x}" for x in range(0, max_doc_count, doc_step)])
    if y_log_scale:
        plt.yscale("log")   
    plt.legend(vc_dict.keys())
    plt.xlabel("Docs Count")
    plt.xticks(rotation=45, ha='right')
    plt.ylabel(f"Domain's Count{' (log scale)' if y_log_scale else ''}")
    plt.title("Document-Domain Frequency Plot")
    plt.tight_layout()
    plt.savefig(f"{month}_{chart_type}_{'log_' if y_log_scale else ''}{output}", format="pdf", bbox_inches="tight")
    plt.show()
    plt.close()

def plot_doc_length_frequency(vc,month="dec2024", output="doc_length_frequency.pdf", y_log_scale=True):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 5))
    plt.bar(vc.index, vc.values, width=0.8,alpha=0.7)
    plt.xticks(range(0, 1000, 100))
    if y_log_scale:
        plt.yscale("log")   
    plt.xlabel("HTML Doc Length (in K Chars)")
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Docs Count")
    plt.title("Document Length Frequency Plot")
    plt.tight_layout()
    plt.savefig(f"{month}_{'log_' if y_log_scale else ''}_{output}", format="pdf", bbox_inches="tight")
    plt.show()
    plt.close()
def plot_doc_length_frequency_multi(vc_dict={},month="dec2024", output="doc_length_frequency_multi.pdf", y_log_scale=True,chart_type="bar"):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 5))
    for key,vc in vc_dict.items():
        plt.bar(vc.index, vc.values, width=0.8, label=key,alpha=0.7)
    plt.xticks(range(0, 1000, 100),labels=[f"{x}K" for x in range(0, 1000, 100)])
    if y_log_scale:
        plt.yscale("log")   
    plt.legend(vc_dict.keys())
    plt.xlabel("HTML Doc Length")
    plt.xticks(rotation=45, ha='right')
    plt.ylabel(f"Docs Count{' (log scale)' if y_log_scale else ''}")
    plt.title("Document Length Frequency Plot")
    plt.tight_layout()
    plt.savefig(f"{month}_{chart_type}_{'log_' if y_log_scale else ''}{output}", format="pdf", bbox_inches="tight")
    plt.show()
    plt.close()
    
def plot_doc_language_frequency(vc,month="dec2024", output="doc_language_frequency.pdf", y_log_scale=True):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 5))
    plt.bar(vc.index[0:50], vc.values[0:50], width=0.8,alpha=0.7)
    # plt.xticks(range(0, 351, 50))
    if y_log_scale:
        plt.yscale("log")   
    plt.xlabel("Language")
    plt.xticks(rotation=45, ha='right')
    plt.ylabel(f"Docs Count{' (log scale)' if y_log_scale else ''}")
    plt.title("Document Languages Frequency Plot")
    plt.tight_layout()
    plt.savefig(f"{month}_{'log_' if y_log_scale else ''}{output}", format="pdf", bbox_inches="tight")
    plt.show()
    plt.close()
def plot_doc_language_frequency_multi(vc_dict={},month="dec2024", output="doc_language_frequency_multi.pdf", y_log_scale=True,chart_type="bar"):
    import matplotlib.pyplot as plt
    language_lst=[]
    for key,vc in vc_dict.items():
        language_lst.extend(vc.index[0:50])
    language_set=set(language_lst)    
    for lang in language_set:
        for key,vc in vc_dict.items():        
            if lang not in vc.index:            
                vc[lang] = 0

    plt.figure(figsize=(15, 5))
    sorted_lang_keys=vc_dict["All"][vc_dict["All"].index.isin(language_set)].sort_values(ascending=False).index
    if chart_type=="stacked_bar":
        series1_lst=[]    
        for key,vc in vc_dict.items():
            vc_filtered=vc[vc.index.isin(language_set)].reindex(sorted_lang_keys)
            series1_lst.insert(0,vc_filtered.values)    
        x = sorted_lang_keys
        y = np.vstack(series1_lst)
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.stackplot(x, y,alpha=0.6)
    elif chart_type=="bar":
        for key,vc in vc_dict.items():
            vc_filtered=vc[vc.index.isin(language_set)].reindex(sorted_lang_keys)
            plt.bar(vc_filtered.index, vc_filtered.values, width=0.8, label=key,alpha=0.7)

    # plt.xticks(range(0, 351, 50))
    if y_log_scale:
        plt.yscale("log") 
    plt.legend(vc_dict.keys())  
    plt.xlabel("Language")
    plt.xticks(rotation=45, ha='right')
    plt.ylabel(f"Docs Count{' (log scale)' if y_log_scale else ''}")
    plt.title("Document Languages Frequency Plot")
    plt.tight_layout()
    plt.savefig(f"{month}_{chart_type}_{'log_' if y_log_scale else ''}{output}", format="pdf", bbox_inches="tight")
    plt.show()
    plt.close()

if __name__ == "__main__":
    plot_multi_dist = False
    y_log_scale = False
    # for month in ['202451','202446','202442']:
    for month in ['202451']:
        if not plot_multi_dist:
            # for version in tqdm(['max6']):
            for version in['topics']:
                month_suffix = f"{month}{'_'+version if version else ''}"
                if version in['sampled']:
                    index_path=f"intermediate_sampled_offsets_{month}"                
                elif version in['topics']:
                    index_path=f"warc_warc_bysampledoffset_ccmain{month}"
                else:
                    index_path=f"warc_index_table_content_table_ccmain{month_suffix}"                    
                
                
                tpoic_doc_counts_vc=get_topic_doc_counts_vc(index_path,month=month_suffix)

                doc_domain_counts_vc=get_doc_domain_counts_vc(index_path,month=month_suffix)
                plot_doc_domain_frequency(doc_domain_counts_vc, month=month_suffix, output=f"doc_domain_frequency.pdf")
                languages_doc_counts_vc=get_languages_doc_counts_vc(index_path,month=month_suffix)
                plot_doc_language_frequency(languages_doc_counts_vc, month=month_suffix, output=f"doc_language_frequency.pdf")
                doc_length_counts_vc=get_doc_length_counts_vc(index_path,month=month_suffix)
                plot_doc_length_frequency(doc_length_counts_vc, month=month_suffix, output=f"doc_length_frequency.pdf")
        else:
            vc_dict={}  
            for plot_type in ['topics','languages','doc_domain','doc_length']:  
                print(f"Processing plot type: {plot_type}...")                            
                vc_dict={}
                for version in[None,'sampled','max6']:
                    file_path=f"{global_path}/month_{plot_type}_frequency_{month}{'_'+version if version else ''}.parquet"
                    vc_df=query_parquet_duckdb(SQL_Query=f"SELECT * FROM read_parquet('{file_path}')")
                    if plot_type=="languages":
                        vc_dict[version if version else 'All']=pd.Series(vc_df['count'].values, index=vc_df['language'].values)                        
                    elif plot_type=="doc_length":
                        vc_dict[version if version else 'All']=pd.Series(vc_df['count'].values, index=vc_df['length'].values).sort_index(ascending=True)
                    elif plot_type=="doc_domain":
                        vc_dict[version if version else 'All']=pd.Series(vc_df['domains_count'].values, index=vc_df['doc_count'].values)
                    elif plot_type=="topics":
                        vc_dict[version if version else 'All']=pd.Series(vc_df['domains_count'].values, index=vc_df['doc_count'].values)
                        
                
                if plot_type=="languages":                        
                    plot_doc_language_frequency_multi(vc_dict, month=month, output=f"doc_language_frequency_multi.pdf",y_log_scale=y_log_scale,chart_type="bar")
                elif plot_type=="doc_length":                        
                    plot_doc_length_frequency_multi(vc_dict, month=month, output=f"doc_length_frequency_multi.pdf",y_log_scale=y_log_scale,chart_type="bar")
                elif plot_type=="doc_domain":                        
                    plot_doc_domain_frequency_multi(vc_dict, month=month, output=f"doc_domain_frequency_multi.pdf",y_log_scale=y_log_scale,chart_type="bar")
                
                        