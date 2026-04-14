from re import sub

from creditext.experiments.mlp_experiments.utils import mean, search_parquet_duckdb,normalize_embeddings,train_valid_test_split,resize_emb
from duckdb import table
import pandas as pd
import numpy as np
import pickle
import logging

def load_splits(path,split_mode="balanced", test_mode="credible-non"):
    if test_mode=="credible-non":
        ######### test and validate on all domains (balanced set) but train with either all balanced or a sub-category only domains as label 0 (igonre other subcategories) ############
        test_domains_df=search_parquet_duckdb(f'{path}/all_splits/balanced/test_domains.parquet', col=None,q_domains=None,max_memory="8GB",schema=None)        
        test_domains_df['domain']=test_domains_df['domain'].apply(lambda x: '.'.join(str(x).split('.')[::-1]))        

        valid_domains_df=search_parquet_duckdb(f'{path}/all_splits/balanced/val_domains.parquet', col=None,q_domains=None,max_memory="8GB",schema=None)        
        valid_domains_df['domain']=valid_domains_df['domain'].apply(lambda x: '.'.join(str(x).split('.')[::-1]))

        train_domains_df=search_parquet_duckdb(f'{path}/all_splits/{split_mode}/train_domains.parquet', col=None,q_domains=None,max_memory="8GB",schema=None)
        train_domains_df['domain']=train_domains_df['domain'].apply(lambda x: '.'.join(str(x).split('.')[::-1]))

    elif test_mode=="sub-category":
        ######### subcategory classifier: consider subcategory domains as label 1 and others categories as label 0 ############
        test_domains_df=search_parquet_duckdb(f'{path}/all_splits/{split_mode}/labeled_test_domains.parquet', col=None,q_domains=None,max_memory="8GB",schema=None)        
        test_domains_df['domain']=test_domains_df['domain'].apply(lambda x: '.'.join(str(x).split('.')[::-1]))        

        valid_domains_df=search_parquet_duckdb(f'{path}/all_splits/{split_mode}/labeled_val_domains.parquet', col=None,q_domains=None,max_memory="8GB",schema=None)        
        valid_domains_df['domain']=valid_domains_df['domain'].apply(lambda x: '.'.join(str(x).split('.')[::-1]))

        train_domains_df=search_parquet_duckdb(f'{path}/all_splits/{split_mode}/labeled_train_domains.parquet', col=None,q_domains=None,max_memory="8GB",schema=None)
        train_domains_df['domain']=train_domains_df['domain'].apply(lambda x: '.'.join(str(x).split('.')[::-1]))

    return train_domains_df,valid_domains_df,test_domains_df
    
if __name__=="__main__":
    path="/home/mila/a/abdallah/scratch/hsh_projects/CrediText/data/weaksupervision"
    domain_rel_annotations_df=pd.read_csv(f'{path}/labels_annot.csv')
    domain_rel_annotations_dict={}
    category_to_sub_category_dict= {
    'wikipedia':'general',
    'phish-and-legit':'phishing',
    'url-phish':'phishing',
    'phish-dataset':'phishing',
    'legit-phish':'phishing',
    'misinfo-domains':'misinfo',        
    'nelez':'misinfo',
    'urlhaus': 'malware'}

    for col in list(domain_rel_annotations_df.columns[1:]):
        domains_set=set(domain_rel_annotations_df[~domain_rel_annotations_df[col].isna()]["domain"])
        # domain_rel_annotations_dict.update(dict(zip(domains_set, [[category_to_sub_category_dict[col],col]] * len(domains_set))))
        domain_rel_annotations_dict.update(dict(zip(domains_set, [category_to_sub_category_dict[col]]* len(domains_set))))    
    # pickle.dump(domain_rel_annotations_dict, open(f'{path}/domain_rel_annotations_dict.pkl', 'wb'))

    train_domains_df,valid_domains_df,test_domains_df=load_splits(path)
    train_domains_df["category"]=train_domains_df["domain"].map(domain_rel_annotations_dict)
    valid_domains_df["category"]=valid_domains_df["domain"].map(domain_rel_annotations_dict)
    test_domains_df["category"]=test_domains_df["domain"].map(domain_rel_annotations_dict)
    print("train domains distribution:")
    print(train_domains_df["category"].value_counts())
    print("##########")
    print(train_domains_df["label"].value_counts())
    print("##########")
    print(train_domains_df.groupby(['category','label']).count())
    print("#########################################")
    print("valid domains distribution:")
    print(valid_domains_df["category"].value_counts())
    print("##########")
    print(valid_domains_df["label"].value_counts())
    print("##########")      
    print(valid_domains_df.groupby(['category','label']).count())
    print("#########################################")
    print("test domains distribution:")
    print(test_domains_df["category"].value_counts())
    print("##########")
    print(test_domains_df["label"].value_counts())
    print("##########")
    print(test_domains_df.groupby(['category','label']).count())
