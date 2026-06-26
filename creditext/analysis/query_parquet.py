import os
import sys
import pandas as pd
sys.path.insert(1, "/".join(os.getcwd().split("/")[0:-2]))
from creditext.experiments.mlp_experiments.utils import search_parquet_duckdb, reverse_domain_name, query_parquet_duckdb
if __name__ == '__main__':
    base_path="/home/mila/a/abdallah/scratch/hsh_projects/CrediText/bash_scripts/spark-warehouse"
    for cc in ["202451","202446","202442"]:
        # parquet_file_path=f"{base_path}/cc_full_index_domain_rel_cc_index_table_ccmain{cc}_0_299.parquet"
        # parquet_file_path=f"{base_path}/cc_full_dqr_cc_index_table_ccmain{cc}_0_299.parquet"
        # parquet_file_path=f"{base_path}/warc_domain_rel_warc_byoffset_ccmain202451/warc_domain_rel_warc_byoffset_ccmain202451_0_90000.parquet"
        # res_df=search_parquet_duckdb(f_path=parquet_file_path,q_domains=None,batch_size=int(1e4),schema=None)

        # parquet_file_path=f"{base_path}/intermediate_sampled_offsets_202451/sampled_offsets_index_ccmain202451.parquet"
        # parquet_file_path=f"{base_path}/cc_full_index_ccmain202451_0_299_max6.parquet"
        # SQL_Query=f"SELECT url_host_name, count(url) FROM read_parquet('{parquet_file_path}') group by url_host_name order by count(url) desc"
        # SQL_Query=f"SELECT * FROM read_parquet('{parquet_file_path}') limit 10"

        parquet_file_path=f"{base_path}/warc_warc_bysampledoffset_ccmain202451/warc_warc_bysampledoffset_ccmain202451_0_49/warc_warc_bysampledoffset_ccmain202451_0_49_gemma300m_emb.parquet"
        SQL_Query=f"SELECT * FROM read_parquet('{parquet_file_path}') limit 10"
        res_df=query_parquet_duckdb(SQL_Query=SQL_Query)
        
        # print(res_df)

        # ############### read courrpted parquet file and restore correctly written groups #############
        # from pyarrow.parquet import ParquetFile
        # # Attempt extraction while ignoring incomplete trailing checks
        # f = ParquetFile(parquet_file_path)
        # for i in range(f.num_row_groups):
        #     try:
        #         df = f.read_row_group(i).to_pandas()
        #     except Exception:
        #         print(f"Row group {i} is corrupted.")
        