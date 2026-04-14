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
from creditext.experiments.mlp_experiments.utils import search_parquet_duckdb
from creditext.experiments.mlp_experiments.dataset_loader import DQR, DomainRel

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

def build_warc_min_index_per_domain(ccmain,full_index_file_name):
    ############### PyArrow #############
    # table = pq.read_table(f"{base_path}/{full_index_file_name}", columns=["url_host_name","warc_filename"])
    # grouped_df = table.group_by('url_host_name').aggregate([('warc_filename', 'min')])
    # grouped_df=grouped_df.sort_values(by=["url_host_name"])
    # grouped_df["Domain_Name"]=grouped_df.index
    # grouped_df=grouped_df.reset_index(drop=True)
    # grouped_df=grouped_df.drop_duplicates()
    ################## pandas ##########
    # cc_full_index_df=pd.read_parquet(f"{base_path}/{full_index_file_name}",engine='pyarrow')
    # grouped_df =cc_full_index_df[["url_host_name","warc_filename"]].groupby("url_host_name")["warc_filename"].min()
    # grouped_df=grouped_df.to_frame()
    # grouped_df=grouped_df.sort_values(by=["url_host_name"])
    # grouped_df["Domain_Name"]=grouped_df.index
    # grouped_df=grouped_df.reset_index(drop=True)
    # grouped_df=grouped_df.drop_duplicates()
    ################## DuckDB #########
    grouped_df = duckdb.query(f"""SELECT distinct url_host_name as Domain_Name, MIN(warc_filename) as warc_filename
        FROM '{base_path}/{full_index_file_name}'
        GROUP BY url_host_name
        order by url_host_name """).df()
    grouped_df.to_parquet(f"{base_path}/{ccmain}_warc_min_index.parquet", engine='pyarrow', compression='snappy',index=False)    
    grouped_df.to_csv(f"{base_path}/{ccmain}_warc_min_index.csv", header=True,index=None)    
def build_labeled_dataset_warc_index(ccmain="ccmain202508"):
    dqr_domain_lst=DQR.get_domains_lst()
    dqr_warc_index_df=search_parquet_duckdb(f"{base_path}/{ccmain}_warc_min_index.parquet", col="Domain_Name",q_domains=dqr_domain_lst,max_memory="8GB",schema=None)
    dqr_warc_index_df.to_parquet(f"{base_path}/dqr_{ccmain}_warc_min_index.parquet", engine='pyarrow', compression='snappy',index=False)  
    dqr_warc_index_df.to_csv(f"{base_path}/dqr_{ccmain}_warc_min_index.csv", index=None)

    drel_domain_lst=DomainRel.get_domains_lst()
    drel_df_lst=[]
    bs=int(1e5)
    # for i in tqdm(range(0,len(drel_domain_lst),bs)):
    #     drel_warc_index_df=search_parquet_duckdb(f"{base_path}/{ccmain}_warc_min_index.parquet", col="Domain_Name",q_domains=drel_domain_lst[i:i+bs],max_memory="8GB",schema=None)
    #     drel_df_lst.append(drel_warc_index_df)
    # drel_warc_index_df=pd.concat(drel_df_lst, ignore_index=True)
    # drel_warc_index_df.to_parquet(f"{base_path}/drel_{ccmain}_warc_min_index_{i}_{i+100000}.parquet", engine='pyarrow', compression='snappy',index=False)  
    # drel_warc_index_df.to_csv(f"{base_path}/drel_{ccmain}_warc_min_index_{i}_{i+100000}.csv", index=None)

def write_index_file_order(ccmain,labeled_ds=None):
    domains_df=pd.read_parquet(f"{base_path}/{labeled_ds+"_" if labeled_ds else ""}{ccmain}_warc_min_index.parquet")
    pd.DataFrame(set(domains_df["Domain_Name"]),columns=["domain"]).to_csv(f"{base_path}/{labeled_ds+"_" if labeled_ds else ""}{ccmain}_domains.csv",index=None)
    counts_df=domains_df["warc_filename"].value_counts().to_frame()
    counts_df["warc_filename"]=counts_df.index
    counts_df=counts_df.reset_index(drop=True)
    counts_df[["warc_filename"]].to_csv(f"{base_path}/{labeled_ds+"_" if labeled_ds else ""}{ccmain}_warc_FilesOrder.txt",header=None,index=None)    
    counts_df["warc_filename"]=counts_df["warc_filename"].apply(lambda x: str(x).replace("/warc/","/wet/").replace("warc.gz","warc.wet.gz"))
    counts_df[["warc_filename"]].to_csv(f"{base_path}/{labeled_ds+"_" if labeled_ds else ""}{ccmain}_wet_FilesOrder.txt",header=None,index=None)    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ccmain content merge")
    parser.add_argument("--file_type", type=str, choices=['html','wet','wat','index','infer'], default="infer", help="type of file to merge")
    parser.add_argument("--ccmain", type=str, default="ccmain202451", help="ccmain")
    # parser.add_argument("--base_path", type=str, default="/home/mila/a/abdallah/scratch/hsh_projects/CrediText/bash_scripts/spark-warehouse", help="base path")
    parser.add_argument("--base_path", type=str, default="/home/mila/a/abdallah/scratch/hsh_projects/CrediText/plots", help="base path")
    parser.add_argument("--org_batch_size", type=int, default=5, help="orginal parquet files batch_size ")    
    parser.add_argument("--group_size", type=int, default=300, help="parquet files batch_size per merge, e.g. merge every 300 files into one out file")    
    parser.add_argument("--group_size_double_factor", type=int, default=10000, help="double the batch size every 10k files into one out file")    
    parser.add_argument("--start_idx", type=int, default=0, help="frist parquet file idx")    
    # parser.add_argument("--end_idx", type=int, default=300, help="end parquet file idx")    
    parser.add_argument("--end_idx", type=int, default=14, help="end parquet file idx")    
    args = parser.parse_args()
    print(f"args={args}")
    ############### Merge DQR HTML Parquet Files #############
    if args.file_type=="html":
        dqr_path="warc_dqr_html_ext_table_ccmain202442"
        base_path=f"/home/mila/a/abdallah/scratch/hsh_projects/CrediText/bash_scripts/spark-warehouse/{dqr_path}"    
        file_path_list=list_parquet_files(f"{dqr_path}",0,3000,50,[])
        merge_pyarrow(file_path_list,f"{base_path}/{dqr_path}.parquet")
    
    base_path=args.base_path
    ccmain=args.ccmain
    group_size=args.group_size
    buckets_lst=[]
    start_idx=args.start_idx
    ############# build buckets groups to merge #############
    while start_idx<=args.end_idx-1:
        buckets_lst.append((start_idx, start_idx+group_size-1))
        start_idx+=group_size
        if start_idx>0 and start_idx%args.group_size_double_factor==0:
            group_size*=4
    ##########################################
    print(buckets_lst)
    if args.file_type=="index":
        mid_path=f"warc_index_table_content_table_{args.ccmain}"
        out_file_name=f"cc_full_index_{args.ccmain}"
    elif args.file_type=="wet":
        mid_path=f"wet_content_table_{args.ccmain}"
        out_file_name=f"wet_content_table_{args.ccmain}"
    elif args.file_type=="infer":
        mid_path=None
        # out_file_name=f"mlpInfer_dqr_dec_pc1_pytorch_text_embeddinggemma-300m_GAT-text-avg_run0_agg_credibench_MLP_Model"
        # out_file_name=f"mlpInfer_weaksupervision_dec_weak_label_pytorch_text_embeddinggemma-300m_text__run0_agg_credibench_MLP_Model"
        out_file_name=f"mlpInfer_dqr_dec_pc1_sklearn_text_embeddinggemma-300m_GAT-text_20260404T210121"
    for (start_idx,end_idx) in tqdm(buckets_lst):
        print(f"##########({start_idx},{end_idx})##########")
        if args.file_type=="index":
            # file_path_list=merge_parquet_files(f"{mid_path}/{mid_path}",start_idx,end_idx,args.org_batch_size,[])
            # full_index_file_name=f"{out_file_name}_{start_idx}_{end_idx}.parquet"
            # merge_pyarrow(file_path_list,f"{base_path}/{full_index_file_name}")
            # build_warc_min_index_per_domain(ccmain,full_index_file_name)
            # build_labeled_dataset_warc_index(ccmain=ccmain)
            # write_index_file_order(ccmain=ccmain)
            write_index_file_order(ccmain=ccmain,labeled_ds="dqr")
        elif args.file_type=="infer":
            file_path_list=list_parquet_files(f"{out_file_name}/",start_idx,end_idx,args.org_batch_size,[],match_regex=f"{out_file_name.split('_')[0]}*.parquet",nested_folders=False)
            merge_pyarrow(file_path_list,f"{base_path}/{out_file_name}/{out_file_name}.parquet")
        else:
            file_path_list=list_parquet_files(f"{mid_path}/{mid_path}",start_idx,end_idx,args.org_batch_size,[])
            merge_pyarrow(file_path_list,f"{base_path}/{mid_path}/{out_file_name}_{start_idx}_{end_idx}.parquet")
        # cc_index_df.to_parquet()
        # cc_index_df=None
        # collected = gc.collect()
        # print(f"Garbage collector collected {collected} objects.")

