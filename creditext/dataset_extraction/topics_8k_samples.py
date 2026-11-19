import argparse
import os
import sys
import pandas as pd
import duckdb
from domain_sampler import DomainSampler
sys.path.insert(1, "/".join(os.getcwd().split("/")[0:-2]))
from creditext.experiments.mlp_experiments.utils import search_parquet_duckdb, reverse_domain_name, query_parquet_duckdb
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ccmain content merge")
    parser.add_argument("--task", type=str, default="CosineSimilarity", choices=["deepAnalysis","CosineSimilarity","preComputed"], help="task type")
    parser.add_argument("--emb_model", type=str, default="gemma300m", choices=["gemma300m","paraphrase-multilingual-MiniLM-L12-v2"], help="task type")
    parser.add_argument("--ccmain", type=str, default="ccmain202451", help="ccmain")
    parser.add_argument("--base_path", type=str, default="/home/mila/a/abdallah/scratch/hsh_projects/CrediText/bash_scripts/spark-warehouse", help="base path")    
    parser.add_argument("--start_idx", type=int, default=0, help="frist chunck idx")    
    parser.add_argument("--end_idx", type=int, default=100, help="end chunck idx")     
    args = parser.parse_args()
    print(f"args={args}")     
    base_path=args.base_path    
    bert_model_path=f"/home/mila/a/abdallah/scratch/hsh_projects/CrediText/domain_sampler/new_safe_bertopic"         

    parquet_file_path=f"{base_path}/warc_warc_bysampledoffset_ccmain202451_0_49.parquet"
    SQL_Query=f"""select * from read_parquet('{parquet_file_path}') limit 10000"""
    res_df=query_parquet_duckdb(SQL_Query=SQL_Query)
    urls_lst=res_df["WARC_Target_URI"].tolist()
    articles=res_df["warc_record_text_resili"].tolist()
    
    if args.task=="deepAnalysis":
        ############### Calaculate Topics Deep Analysis ##############
        topicModler=DomainSampler(bert_model_path)
        anlsysis_res_lst=topicModler.topic_analysis(urls=urls_lst, articles=articles, embedings=None, deep_analysis=True, is_html=False)
        analysis_df, result_df=anlsysis_res_lst[0], anlsysis_res_lst[1]
        ########################
        res_df=res_df.groupby('Domain_Name').head(2)
        res_df[["WARC_Target_URI"]].to_csv("dec2024_doc-url_samples_1k.csv",index=None)
        res_df[["WARC_Target_URI","warc_record_text_resili"]].to_csv("dec2024_doc-content_samples_1k.csv",index=None)
        print(res_df)
    elif args.task=="preComputed":
        ############ Select PreComputed Topics ##################
        parquet_file_path=f"{base_path}/warc_warc_bysampledoffset_ccmain202451/warc_warc_bysampledoffset_ccmain202451_0_49/warc_warc_bysampledoffset_ccmain202451_0_49_{args.emb_model}_topics_top3.parquet"
        # parquet_file_path=f"{base_path}/warc_bysampledoffset_topics_ccmain202451.parquet"
        # SQL_Query=f"SELECT * FROM read_parquet('{parquet_file_path}') limit 1000"
        parquet_file_path=f"{base_path}/warc_warc_bysampledoffset_ccmain202451_0_49.parquet"
        SQL_Query=f"""select * from read_parquet('{parquet_file_path}') limit 10000"""
        res_df=query_parquet_duckdb(SQL_Query=SQL_Query)

        urls_lst=res_df["WARC_Target_URI"].tolist()
        SQL_Query=f"""SELECT domain,X['unnest'].topic0_id,X['unnest'].topic0_prop,X['unnest'].topic1_id,X['unnest'].topic1_prop,X['unnest'].topic2_id,X['unnest'].topic2_prop ,X['unnest'].url as topic FROM (
                        select * from read_parquet('{parquet_file_path}') limit 100000) ,
                        UNNEST(topics) AS X
                        where list_contains($1, X['unnest'].url) """
        res_df=pd.DataFrame(duckdb.execute(SQL_Query,[urls_lst]).fetchall()) 
        res_df.columns=['domain', 'topic0_id', 'topic0_prop', 'topic1_id', 'topic1_prop', 'topic2_id', 'topic2_prop', 'url']
        # res_df=query_parquet_duckdb(SQL_Query=SQL_Query)
        # res_df.to_csv("dec2024_MiniLLM_doc_topic_samples_8k.csv",index=None)
        res_df.to_csv(f"dec2024_{args.emb_model}_doc_topic_samples_8k.csv",index=None)
    elif args.task=="CosineSimilarity":
        ############### Calaculate Topics Cosine Similarity ##############
        parquet_file_path=f"{base_path}/warc_warc_bysampledoffset_ccmain202451/warc_warc_bysampledoffset_ccmain202451_0_49/warc_warc_bysampledoffset_ccmain202451_0_49_{args.emb_model}_emb.parquet"
        # SQL_Query=f"SELECT * FROM read_parquet('{parquet_file_path}') limit 1000"
        SQL_Query=f"""SELECT domain, e.page, e.emb
                    FROM (
                        SELECT domain,
                            list_filter(embeddings, x -> list_contains($1, x.page)) AS matched --filter to pages in input list
                        FROM read_parquet('{parquet_file_path}')
                    ) sub
                    CROSS JOIN UNNEST(matched) AS t(e) -- unnest struct list, alias element as e for each dict
                    WHERE len(matched) > 0 """
        res_df=pd.DataFrame(duckdb.execute(SQL_Query,[urls_lst]).fetchall()) 
        res_df.columns=['domain', 'url', 'emb']
        bert_model_path=f"/home/mila/a/abdallah/scratch/hsh_projects/CrediText/domain_sampler/new_safe_bertopic"         
        topicModler=DomainSampler(bert_model_path)
        urls_lst=res_df["url"].tolist()
        emb_lst=res_df["emb"].tolist()
        topics_df=topicModler.topic_analysis_cosin_sim(emb_lst=emb_lst)
        topics_df['url']=urls_lst
        topics_df.to_csv(f"dec2024_topics_CosineSimilarity_{args.emb_model}_samples_1k.csv",index=None)
    
    print(res_df)
        