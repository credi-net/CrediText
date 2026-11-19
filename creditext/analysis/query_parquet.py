import os
import sys
import duckdb
from tokenize import group
import pandas as pd
import matplotlib.pyplot as plt
# from domain_sampler import DomainSampler
from urllib.parse import urlparse
sys.path.insert(1, "/".join(os.getcwd().split("/")[0:-2]))
from creditext.experiments.mlp_experiments.utils import search_parquet_duckdb, reverse_domain_name, query_parquet_duckdb
if __name__ == '__main__':
    base_path="/home/mila/a/abdallah/scratch/hsh_projects/CrediText/bash_scripts/spark-warehouse"
    for cc in ["202446","202442","202451"]:
        # parquet_file_path=f"{base_path}/cc_full_index_domain_rel_cc_index_table_ccmain{cc}_0_299.parquet"
        # parquet_file_path=f"{base_path}/cc_full_dqr_cc_index_table_ccmain{cc}_0_299.parquet"
        # parquet_file_path=f"{base_path}/warc_domain_rel_warc_byoffset_ccmain202451/warc_domain_rel_warc_byoffset_ccmain202451_0_90000.parquet"
        # res_df=search_parquet_duckdb(f_path=parquet_file_path,q_domains=None,batch_size=int(1e4),schema=None)

        # parquet_file_path=f"{base_path}/intermediate_sampled_offsets_202451/sampled_offsets_index_ccmain202451.parquet"
        # parquet_file_path=f"{base_path}/cc_full_index_ccmain202451_0_299_max6.parquet"
        # SQL_Query=f"SELECT url_host_name, count(url) FROM read_parquet('{parquet_file_path}') group by url_host_name order by count(url) desc"
        # SQL_Query=f"SELECT * FROM read_parquet('{parquet_file_path}') limit 10"

        # parquet_file_path=f"{base_path}/warc_warc_bysampledoffset_ccmain202451/warc_warc_bysampledoffset_ccmain202451_0_49/warc_warc_bysampledoffset_ccmain202451_0_49_gemma300m_emb.parquet"
        # SQL_Query=f"SELECT * FROM read_parquet('{parquet_file_path}') limit 10"
        # res_df=query_parquet_duckdb(SQL_Query=SQL_Query)

        # parquet_file_path=f"{base_path}/CrediBench-WebContent-Dec2024/dec2024_wetcontent_0_199_paraphrase-multilingual-MiniLM-L12-v2_topics_top3.parquet"
        # SQL_Query=f"SELECT count(*) FROM read_parquet('{parquet_file_path}')"
        # res_df=query_parquet_duckdb(SQL_Query=SQL_Query)

        # parquet_file_path=f"{base_path}/warc_warc_bysampledoffset_ccmain202451_domain_buckets_map.parquet"
        # SQL_Query=f"SELECT * FROM read_parquet('{parquet_file_path}') limit 10"
        # res_df=query_parquet_duckdb(SQL_Query=SQL_Query)

        # parquet_file_path=f"/home/mila/a/abdallah/scratch/hsh_projects/CrediText/data/domainrel_dec2024_gemma300m_cochan_sampled.parquet"
        # SQL_Query=f"SELECT count(*) FROM read_parquet('{parquet_file_path}') limit 10"
        # res_df=query_parquet_duckdb(SQL_Query=SQL_Query)
        # print(res_df)


        # ####################### cochran samples variations ###########################\
        # {
        #     use_dqr=True
        #     dqr_df=pd.read_csv("/home/mila/a/abdallah/scratch/hsh_projects/CrediText/data/dqr/domain_pc1.csv")
        #     dqr_domains_lst=list(dqr_df["domain"].unique())
        #     con = duckdb.connect()
        #     con.execute(f"SET memory_limit='8GB'")
        #     con.execute(f"SET threads=8")
        #     # con.register("dqr", pd.DataFrame({"domain": list(dqr_domains_lst)}))
        #     if use_dqr:
        #         parquet_file_path="/home/mila/a/abdallah/scratch/hsh_projects/CrediText/data/dqr/cochran/dqr_dec2024_gemma300m_cochan_sampled_cluster_props.parquet"
        #     else:
        #         parquet_file_path="/home/mila/a/abdallah/scratch/hsh_projects/CrediText/data/weaksupervision/cochran/domainrel_dec2024_embeddinggemma-300m_cochran_sampled_all_cluster_props.parquet"
        #     # SQL_Query=f"SELECT * FROM read_parquet('{parquet_file_path}') "

        #     k=6
        #     order_by_col_dict={ "top":"topic0_prop DESC",
        #                     "lower":"topic0_prop ASC",
        #                     "random":"random()",
        #                     "free_random":"random()",
        #                     "free_top":"topic0_prop DESC",
        #                     "free_lower":"topic0_prop ASC"}
        #     for order_key,v in order_by_col_dict.items():
        #         print(f"Processing order_key: {order_key} with order_by_col: {v}")
        #         SQL_Query=f"""WITH ranked AS (
        #                 SELECT
        #                     domain,url,topic0_id,topic0_prop,
        #                     ROW_NUMBER() OVER (
        #                     PARTITION BY domain {"" if "free_" in order_key else ", topic0_id"}
        #                     ORDER BY {v} 
        #                     ) AS rn
        #                 FROM read_parquet('{parquet_file_path}')) 
        #                 SELECT ranked.domain, url, topic0_id, topic0_prop
        #                 FROM ranked 
        #                 WHERE rn <= {k} 
        #                 ORDER BY ranked.domain, topic0_id, topic0_prop DESC"""
        #         # res_df=query_parquet_duckdb(SQL_Query=SQL_Query)
        #         res_df=con.execute(SQL_Query).df()
        #         # res_df["domain"] = res_df["url"].apply(lambda url: urlparse(url).netloc)
        #         if use_dqr:
        #             res_df.to_parquet(f"/home/mila/a/abdallah/scratch/hsh_projects/CrediText/data/dqr/cochran/dqr_dec2024_embeddinggemma-300m_cochran_sampled_{order_key}{k}_cluster_props.parquet",engine="pyarrow",row_group_size=1000,use_dictionary=["domain","topic0_id"])
        #         else:
        #             res_df.to_parquet(f"/home/mila/a/abdallah/scratch/hsh_projects/CrediText/data/weaksupervision/cochran/domainrel_dec2024_embeddinggemma-300m_cochran_sampled_{order_key}{k}_cluster_props.parquet",engine="pyarrow",row_group_size=1000,use_dictionary=["domain","topic0_id"])
        #         print(res_df)
        # }       
        ####################### cochran docs counts ###########################
        # parquet_file_path="/home/mila/a/abdallah/scratch/hsh_projects/CrediText/data/weaksupervision/cochran/domainrel_dec2024_embeddinggemma-300m_cochran_sampled_all_cluster_props.parquet"                    
        # SQL_Query=f""" SELECT domain,count(url)                    
        #         FROM read_parquet('{parquet_file_path}')
        #         group by domain
        #         ORDER BY count(url) DESC"""
        # res_df=query_parquet_duckdb(SQL_Query=SQL_Query)
        # vc=res_df[res_df.columns[1]].value_counts().sort_index(ascending=False)
        # # 2. Set up the plot
        # fig, ax = plt.subplots(figsize=(10, 4))
        # ax.plot(vc.index, vc.values, marker='.', linestyle='-')

        # # 3. Configure the x-axis to run from 390 to 1, decreasing by 50
        # plt.yscale('log')
        # x_ticks = list(range(390, 0, -50))
        # ax.set_xticks(x_ticks)
        # ax.set_xlim(390, 1)  # Setting the max first and min second reverses the axis direction
        # plt.xticks(fontsize=16)
        # plt.yticks(fontsize=16)
        # # 4. Add labels and grid
        # ax.set_xlabel("Docs Count",fontsize=16)
        # ax.set_ylabel("Domains Count (Log)",fontsize=16)
        # ax.grid(True, linestyle='--', alpha=0.9)

        # # 5. Save the output as a PDF
        # plt.tight_layout()
        # plt.savefig("frequency_plot.pdf", format="pdf")
        # plt.show()

        # # res_df["domain"] = res_df["url"].apply(lambda url: urlparse(url).netloc)
        # res_df.to_parquet(f"/home/mila/a/abdallah/scratch/hsh_projects/CrediText/data/domainrel_dec2024_gemma300m_cochran_sampled_{order_key}{k}_cluster_props.parquet",engine="pyarrow",row_group_size=1000,use_dictionary=["domain","topic0_id"])
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
        
        # #################### list longest 3 Shortest 3 #################        
        parquet_file_path=f"/home/mila/a/abdallah/scratch/hsh_projects/CrediText/bash_scripts/spark-warehouse/cc_full_index_credigrain_cc_index_table_ccmain{cc}_0_299.parquet"
        SQL_Query=f"SELECT * FROM read_parquet('{parquet_file_path}')"
        ############ 3L3S ###########
        k = 3
        L3s3_query=f"""
        WITH ranked AS (
            SELECT
                url_host_name,
                url,
                warc_record_length,
                warc_filename,
                warc_record_offset,
                ROW_NUMBER() OVER (PARTITION BY url_host_name
                                ORDER BY warc_record_length DESC, url) AS rn_long,
                ROW_NUMBER() OVER (PARTITION BY url_host_name
                                ORDER BY warc_record_length ASC,  url) AS rn_short
            FROM read_parquet('{parquet_file_path}')
            WHERE warc_record_length IS NOT NULL
        )
        SELECT 
            url_host_name,
            rn_long,
            rn_short,
            url,
            warc_record_length,
            warc_filename,
            warc_record_offset
        FROM ranked
        WHERE rn_long <= {k} OR rn_short <= {k}
        ORDER BY url_host_name,rn_long"""        
        res_df=query_parquet_duckdb(SQL_Query=L3s3_query)
        res_df=res_df.drop_duplicates()
        res_df.to_parquet(f"{base_path}/cc_full_index_credigrain_cc_index_table_{cc}_L3S3.parquet",engine="pyarrow",row_group_size=1000,use_dictionary=["Domain_Name"])
        print(res_df)

        ################ Query CrediGrain L3S3 content from parquet file #############
        # parquet_file_path=f"/home/mila/a/abdallah/scratch/hsh_projects/CrediText/bash_scripts/spark-warehouse/warc_warc_credigrain_l3s3_ccmain202451_0_9999.parquet"
        # SQL_Query=f"SELECT * FROM read_parquet('{parquet_file_path}') limit 10"
        # res_df=query_parquet_duckdb(SQL_Query=SQL_Query)
        # print(res_df)