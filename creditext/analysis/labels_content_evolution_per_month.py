import os
import sys
import pandas as pd
sys.path.insert(1, "/".join(os.getcwd().split("/")[0:-2]))
from creditext.experiments.mlp_experiments.utils import search_parquet_duckdb, reverse_domain_name
if __name__ == '__main__':
    domains_per_month_dict={}
    base_path="/home/mila/a/abdallah/scratch/hsh_projects/CrediText/data"
    for month in ["Oct","Nov","Dec"]:
        content_path=f"{base_path}/dqr/dqr_content_{month}2024.parquet"
        res_df=search_parquet_duckdb(f_path=content_path,projection_cols=['domain'],filter_by_col ="domain",q_domains=None,batch_size=int(1e4),schema=None)
        domains_per_month_dict[month]=set(res_df["domain"].tolist())
    for split in ["test","train","val"]:
        content_path=f"{base_path}/dqr/splits/{split}_regression_domains.parquet"
        res_df=search_parquet_duckdb(f_path=content_path,projection_cols=['domain'],filter_by_col ="domain",q_domains=None,batch_size=int(1e4),schema=None)
        res_df["domain"]=res_df["domain"].apply(lambda x: reverse_domain_name(x))
        domains_per_month_dict[split]=res_df

    for split in ["test","train","val"]:
        print(f"General DQR {split} count={len(domains_per_month_dict[split])}")

    for month in ["Oct","Nov","Dec"]:
        print(f" ########## DQR {month} count={len(domains_per_month_dict[month])} ############")
        for split in ["test","train","val"]:  
            res_df=domains_per_month_dict[split]      
            lables_set=set(res_df["domain"].tolist())
            print(f"DQR {month}:{split} count={len(domains_per_month_dict[month].intersection(lables_set))}")
    
    print("######################### Domain Rel ####################")
    domains_per_month_dict={}   
    for month in ["Oct","Nov","Dec"]:
        content_path=f"{base_path}/weaksupervision/weaksupervision_content_{month}2024.parquet"
        res_df=search_parquet_duckdb(f_path=content_path,projection_cols=['domain'],filter_by_col ="domain",q_domains=None,batch_size=int(1e4),schema=None)
        domains_per_month_dict[month]=set(res_df["domain"].tolist())
    for split in ["test","train","val"]:
        content_path=f"{base_path}/weaksupervision/all_splits/balanced/{split}_domains.parquet"
        res_df=search_parquet_duckdb(f_path=content_path,projection_cols=['domain','label'],filter_by_col ="domain",q_domains=None,batch_size=int(1e4),schema=None)
        res_df["domain"]=res_df["domain"].apply(lambda x: reverse_domain_name(x))
        domains_per_month_dict[split]=res_df
    
    domain_rel_set=set()
    for split in ["test","train","val"]:
        print(f"General domain Rel {split} count={len(domains_per_month_dict[split])}")
        res_df=domains_per_month_dict[split]
        domain_rel_set.update(set(res_df["domain"].tolist()))      
        cred_set=set(res_df[res_df["label"]==1]["domain"].tolist())
        non_cred_set=set(res_df[res_df["label"]==0]["domain"].tolist())
        print(f"Cred count={len(cred_set)}")
        print(f"Non-Cred count={len(non_cred_set)}")

    for month in ["Oct","Nov","Dec"]:
        print(f" ########## domain Rel {month} count={len(domains_per_month_dict[month])} ############")
        for split in ["test","train","val"]:  
            res_df=domains_per_month_dict[split]      
            cred_set=set(res_df[res_df["label"]==1]["domain"].tolist())
            non_cred_set=set(res_df[res_df["label"]==0]["domain"].tolist())
            print(f"domain Rel {month}:{split} Cred count={len(domains_per_month_dict[month].intersection(cred_set))}")
            print(f"domain Rel {month}:{split} Non-Cred count={len(domains_per_month_dict[month].intersection(non_cred_set))}")

    pd.DataFrame(list(domain_rel_set),columns=["domain"]).to_csv(f"{base_path}/weaksupervision/domain_rel_domains.csv",index=None)
    

    
    

        
