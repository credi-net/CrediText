import os
import sys
import pandas as pd
sys.path.insert(1, "/".join(os.getcwd().split("/")[0:-2]))
from creditext.experiments.mlp_experiments.utils import search_parquet_duckdb, reverse_domain_name
if __name__ == '__main__':
    base_path="/home/mila/a/abdallah/scratch/hsh_projects/CrediText/bash_scripts/spark-warehouse"
    for cc in ["202451","202446","202442"]:
        # parquet_file_path=f"{base_path}/cc_full_index_domain_rel_cc_index_table_ccmain{cc}_0_299.parquet"
        # parquet_file_path=f"{base_path}/cc_full_dqr_cc_index_table_ccmain{cc}_0_299.parquet"
        parquet_file_path=f"{base_path}/warc_domain_rel_warc_byoffset_ccmain202451/warc_domain_rel_warc_byoffset_ccmain202451_0_90000.parquet"
        res_df=search_parquet_duckdb(f_path=parquet_file_path,q_domains=None,batch_size=int(1e4),schema=None)
        print(res_df)