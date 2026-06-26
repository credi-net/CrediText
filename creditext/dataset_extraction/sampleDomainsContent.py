from calendar import c
import multiprocessing
from duckdb import df
import pandas as pd
from regex import P
from shap import sample
from tqdm import tqdm
from creditext.experiments.mlp_experiments.utils import query_parquet_duckdb, list_all_files
from domain_sampler.sampler import DomainSampler
import random
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
logging.basicConfig(level=logging.INFO)
import gc

global_path="/home/mila/a/abdallah/scratch/hsh_projects/CrediText/bash_scripts/spark-warehouse"
def generate_domain_doc_count():
    for month in ['202451','202446','202442']:
        logging.info(f"Processing month {month}...")
        month_dir=f"{global_path}/warc_index_table_content_table_ccmain{month}"
        parquet_files_lst=list_all_files(month_dir, "*.parquet", recursive=True)
        # f_path=f"{global_path}/cc_full_index_ccmain{month}_0_299.parquet"   
        counts_df_lst=[] 
        for f_path in tqdm(parquet_files_lst):    
            domain_doc_count_df=query_parquet_duckdb(SQL_Query=f"SELECT url_host_name,count(url_host_name) as doc_count FROM read_parquet('{f_path}') group by url_host_name order by count(url_host_name) desc")
            counts_df_lst.append(domain_doc_count_df)
        all_counts_df=pd.concat(counts_df_lst)
        domain_doc_count_df=all_counts_df.groupby("url_host_name").sum().reset_index().sort_values("doc_count", ascending=False)
        domain_doc_count_df.to_parquet(f"{global_path}/domain_docs_count_{month}.parquet", compression='gzip')
        logging.info(domain_doc_count_df)
def plot_sample_frequency(vc,output="dec2024_sampled_docs_frequance.pdf", y_log_scale=True):
    import matplotlib.pyplot as plt
    plt.bar(vc.index, vc.values, width=0.8)
    plt.xticks(range(0, 351, 50))
    if y_log_scale:
        plt.yscale("log")   
    plt.xlabel("# Docs")
    plt.ylabel("Domains Count")
    plt.title("Sampled Docs per domain Frequency Plot")
    plt.tight_layout()
    plt.savefig(output, format="pdf", bbox_inches="tight")
    plt.show()
def generate_domain_doc_samples_count():    
    # for month in ['202451','202446','202442']:
    for month in ['202446','202442']:
        logging.info(f"Processing month {month}...")  
        f_path=f"{global_path}/domain_docs_count_{month}.parquet"
        domain_doc_count_df=query_parquet_duckdb(SQL_Query=f"SELECT url_host_name,doc_count FROM read_parquet('{f_path}')")
        domains_sample_size_lst=[]
        for idx,row in tqdm(domain_doc_count_df.iterrows(), total=domain_doc_count_df.shape[0]):
            domain_name=row['url_host_name']
            doc_count=row['doc_count']
            sample_size=DomainSampler.get_min_sample_size(pop_size=doc_count, confidence=0.95, margin_error=0.05)
            domains_sample_size_lst.append([domain_name,doc_count, sample_size])
        domains_sample_size_df=pd.DataFrame(domains_sample_size_lst, columns=['url_host_name','doc_count','sample_size'])
        vc = domains_sample_size_df["sample_size"].value_counts().sort_index()
        plot_sample_frequency(vc, output=f"{global_path}/sampled_docs_frequency_{month}.pdf")
        domains_sample_size_df.to_parquet(f"{global_path}/domain_docs_sample_size_{month}.parquet", compression='gzip')
        logging.info(domains_sample_size_df)

def generate_sampled_domains_doc_warc_offset():    
    # for month in ['202451','202446','202442']:
    domain_sample_count_dict={} ## dict to keep track of remaining sample size for each domain
    month_offset_lst=[]
    def process_row(row):
        if row.url_host_name in domain_sample_count_dict and domain_sample_count_dict[row.url_host_name]>0:
                month_offset_lst.append([row.url_host_name, row.warc_filename, row.warc_record_offset, row.content_languages, row.warc_record_length])
                domain_sample_count_dict[row.url_host_name]-=1
    # for month in ['202446','202442']:
    for month in ['202451']:
        logging.info(f"Processing month {month}...")
        f_path=f"{global_path}/domain_docs_sample_size_{month}.parquet"
        domain_sample_count_df=query_parquet_duckdb(SQL_Query=f"SELECT url_host_name,sample_size FROM read_parquet('{f_path}')")
        domain_sample_count_dict=dict(zip(domain_sample_count_df['url_host_name'], domain_sample_count_df['sample_size']))
        del domain_sample_count_df
        gc.collect()
        month_dir=f"{global_path}/warc_index_table_content_table_ccmain{month}"
        parquet_files_lst=list_all_files(month_dir, "*.parquet", recursive=True)
        random.shuffle(parquet_files_lst) ## shuffle for randomness 
        # parquet_files_lst=parquet_files_lst[0:1] 
        # f_path=f"{global_path}/cc_full_index_ccmain{month}_0_299.parquet" 
        for idx,f_path in tqdm(enumerate(parquet_files_lst), total=len(parquet_files_lst)):
            logging.info(f"Processing file {idx+1}/{len(parquet_files_lst)}: {f_path}")
            if len(domain_sample_count_dict) == 0:
                break
            domain_doc_count_df=query_parquet_duckdb(SQL_Query=f"SELECT url_host_name,warc_filename, warc_record_offset,content_languages,warc_record_length FROM read_parquet('{f_path}')")
            for row in domain_doc_count_df.itertuples(): # Serial Exec
                process_row(row)
            # with ThreadPoolExecutor(max_workers=8) as executor:
            #     executor.map(process_row, domain_doc_count_df.itertuples())
            # with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor: # parllel Exec
            #     executor.map(process_row, domain_doc_count_df.itertuples())                
            del domain_doc_count_df
            gc.collect()
            domain_sample_count_dict={k:v for k,v in domain_sample_count_dict.items() if v>0} ## remove domains that already have their sample size fulfilled
            logging.info(f"Remaining domains to sample: {len(domain_sample_count_dict)}")
            if (idx+1) % 10 == 0: ## save intermediate results every 20 files
                 intermediate_offset_df=pd.DataFrame(month_offset_lst , columns=['url_host_name', 'warc_filename', 'warc_record_offset', 'content_languages', 'warc_record_length'])
                 month_offset_lst=[] ## reset the list to save memory
                 gc.collect()
                 intermediate_offset_df=intermediate_offset_df.sort_values(by=['warc_filename'])
                 intermediate_offset_df.to_parquet(f"{global_path}/intermediate_sampled_offsets_{month}_{idx+1}.parquet", compression='gzip')
                 del intermediate_offset_df
                 gc.collect()
                 logging.info(f"Saved intermediate results after processing {idx+1} files.")
                 
                 pd.DataFrame(list(domain_sample_count_dict.items()), columns=['url_host_name', 'sample_size']).to_parquet(f"{global_path}/remaining_sample_size_{month}.parquet", compression='gzip')

        month_offset_df=pd.DataFrame(month_offset_lst , columns=['url_host_name', 'warc_filename', 'warc_record_offset', 'content_languages', 'warc_record_length'])
        month_offset_df=month_offset_df.sort_values(by=['warc_filename'])
        month_offset_df.to_parquet(f"{global_path}/intermediate_sampled_offsets_{month}_{idx+1}.parquet", compression='gzip')
        logging.info(month_offset_df)

def explore_sampled_offsets(month='202451'):
    f_path=f"{global_path}/month_sampled_offsets_{month}.parquet"
    month_offset_df=query_parquet_duckdb(SQL_Query=f"SELECT url_host_name,warc_filename, warc_record_offset FROM read_parquet('{f_path}')")
    logging.info(month_offset_df)

if __name__ == "__main__":
    # generate_domain_doc_count()
    # generate_domain_doc_samples_count()
    generate_sampled_domains_doc_warc_offset()
    # explore_sampled_offsets()