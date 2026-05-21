import os
import sys
import pandas as pd
sys.path.insert(1, "/".join(os.getcwd().split("/")[0:-2]))
from creditext.experiments.mlp_experiments.utils import search_parquet_duckdb, reverse_domain_name
if __name__ == '__main__':
    domains_per_month_dict={}
    base_path="/home/mila/a/abdallah/scratch/hsh_projects/CrediText"
    dqr_domains_df=pd.read_csv(f"{base_path}/data/dqr/domain_pc1.csv")
    dqr_domains_set=set(dqr_domains_df["domain"])
    domain_rel_domains_df=pd.read_csv(f"{base_path}/data/weaksupervision/domain_rel_domains.csv")
    domain_rel_domains_set=set(domain_rel_domains_df["domain"])
    key_column="url_host_name"
    for month in ["42","46","51"]:
        content_path=f"{base_path}/bash_scripts/spark-warehouse/cc_full_index_ccmain2024{month}_0_299.parquet"
        dqr_res_df=search_parquet_duckdb(f_path=content_path,filter_by_col =key_column,q_domains=dqr_domains_set,batch_size=int(1e4),schema=None)
        dqr_res_df=dqr_res_df.sort_values(by=[key_column])
        dqr_res_df.to_csv(f"{base_path}/data/dqr/dqr_cc_full_index_ccmain2024{month}.csv",index=None)
        domain_rel_res_df=search_parquet_duckdb(f_path=content_path,filter_by_col ="url_host_name",q_domains=domain_rel_domains_set,batch_size=int(1e4),schema=None)                
        domain_rel_res_df=domain_rel_res_df.sort_values(by=[key_column])
        domain_rel_res_df.to_csv(f"{base_path}/data/weaksupervision/domainRel_cc_full_index_ccmain2024{month}.csv",index=None)
    

    
    

        
