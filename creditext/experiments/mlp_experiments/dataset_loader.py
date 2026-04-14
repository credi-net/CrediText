from re import S
from turtle import st

from creditext.experiments.mlp_experiments.utils import fuse_1d_emb, mean, search_parquet_duckdb,normalize_embeddings,train_valid_test_split,resize_and_fuse_emb
import pandas as pd
import numpy as np
import pickle
import logging
agg_months_dict={"oct":["oct"],
                "nov":["oct","nov"],
                "dec":["oct","nov","dec"]}

class DomainRel(object):
    @staticmethod
    def load_agg_Nmonth_emb_dict(embed_type: str, path:str ="../../../data", model_name :str ="embeddinggemma-300m",
                                month_lst: list[str]=["dec", "nov", "oct"], agg:str ="avg",gnn_encoder:str="text",normalize: bool=False,emb_dim: int=256,original_emb_dim: int=256):
        """load and aggregate N-month embedding dictionaries for both text and GNN embeddings
            Args:
                embed_type: The type of the embedding i.e text,GN_GAT, others
                path: The embedding pickle file or parquet file path
                model_name: the LLM embeding model name
                month_lst: list of months to aggregate
                agg: the ggregation function i.e. avg,cat,min,max
                gnn_encoder: the GNN embedding encoder i.e RNI or text
                normalize: boolean to normalize the embeddings
                emb_dim: the embedding diminsion to trim at
                original_emb_dim: the original full length embedding size

            Returns:
                The aggerated N-Month embeddings
            """
        months_emb_lst = []
        
        for month in month_lst:
            if embed_type == "GNN_GAT":
                if gnn_encoder=="RNI":
                    with open(f'{path}/gnn_embedding/{gnn_encoder}/{month}_binary_labelled_set_domain_rni_embeddings_updated_balanced.pkl', 'rb') as f:
                        embd_dict = pickle.load(f)
                elif gnn_encoder=="text":
                    file_path=f'{path}/gnn_embedding/{gnn_encoder}/{month}_binary_labelled_set_domain_from_text_embeddings_updated_balanced.parquet'
                    logging.info(f"GNN emb file path={file_path}")
                    embd_dict=search_parquet_duckdb(file_path, col=None,q_domains=None,max_memory="8GB",schema={'key':'domain','val':'embeddings'})
            elif embed_type == "text":
                embd_dict=DomainRel.load_emb_dict_from_parquet(embed_type, path, model_name, month,normalize=False,emb_dim=emb_dim, original_emb_dim=original_emb_dim)            
            if normalize:
                embd_dict=normalize_embeddings(embd_dict)
            months_emb_lst.append(embd_dict)
        
        common_domains_set=set(months_emb_lst[0].keys())    
        for lst in months_emb_lst[1:]:
            common_domains_set = common_domains_set.intersection(lst.keys())
        
        diff_domains_set=set(months_emb_lst[-1].keys())-common_domains_set
        for key in diff_domains_set:
            if agg == "cat":
                months_emb_lst[-1][key].extend(months_emb_lst[-1][key]*len(months_emb_lst))

        for key in common_domains_set:
            for i in range(0, len(months_emb_lst)-1):
                months_emb_lst[-1][key]=fuse_1d_emb(months_emb_lst[-1][key], months_emb_lst[i][key], fusion_mode=agg)
        return months_emb_lst[-1]

    @staticmethod
    def load_agg_Nmonth_weaksupervision_emb_dict(embed_type: str, path="../../../data", model_name: str="embeddinggemma-300m",
                                                month_lst: list[str]=["dec", "nov", "oct"], target: str="pc1", agg: str="avg"):
        months_emb_PhishTank_lst = []
        months_emb_URLhaus_lst = []
        months_emb_legit_lst = []
        for month in month_lst:
            with open(f'{path}/PhishTank_{target}_rni_{month}_2024_embeddings.pkl', 'rb') as f:
                months_emb_PhishTank_lst.append(pickle.load(f))
            with open(f'{path}/URLHaus_{target}_rni_{month}_2024_embeddings.pkl', 'rb') as f:
                months_emb_URLhaus_lst.append(pickle.load(f))
            with open(f'{path}/IP2Location_{target}_rni_{month}_2024_embeddings.pkl', 'rb') as f:
                months_emb_legit_lst.append(pickle.load(f))

        for ds_months in [months_emb_PhishTank_lst, months_emb_URLhaus_lst, months_emb_legit_lst]:
            for key in ds_months[0].keys():
                for i in range(1, len(ds_months)):  # loop on dataset months
                    if key in ds_months[i]:
                        if agg == "concat":
                            ds_months[0][key].extend(ds_months[i][key])
                            # logging.info(len(ds_months[0][key]))
                        elif agg == "min":
                            ds_months[0][key] = [min(a, b) for a, b in zip(ds_months[0][key], ds_months[i][key])]
                        elif agg == "max":
                            ds_months[0][key] = [max(a, b) for a, b in zip(ds_months[0][key], ds_months[i][key])]
                        elif agg == "avg":
                            ds_months[0][key] = [(a + b) / 2 for a, b in zip(ds_months[0][key], ds_months[i][key])]
                            # logging.info(len(ds_months[0][key]))
        return months_emb_PhishTank_lst[0], months_emb_URLhaus_lst[0], months_emb_legit_lst[0]

    @staticmethod
    def load_emb_dict(embed_type: str, path:str="../../../data",pickle_name:str=None, model_name: str="embeddinggemma-300m", month: str="dec", target: str="pc1", emb_dim: int=8192,normalize: bool=False,gnn_encoder: str="RNI"):
        if pickle_name:
            with open(f'{path}/{pickle_name}', 'rb') as f:
                embd_dict = pickle.load(f)
        elif embed_type == "text":
            if model_name == "embeddinggemma-300m":
                with open(f'{path}/dqr_{month}_text_embeddinggemma-300m_{emb_dim}.pkl', 'rb') as f:
                    embd_dict = pickle.load(f)
            elif model_name == "embeddingQwen3-0.6B":
                with open(f'{path}/dqr_{month}_text_embeddingQwen3-0.6B_1024.pkl', 'rb') as f:
                    embd_dict = pickle.load(f)
            elif model_name == "embeddingQwen3-8B":
                with open(f'{path}/dqr_{month}_text_embeddingQwen3-8B_4096.pkl', 'rb') as f:
                    embd_dict = pickle.load(f)
            elif model_name == "embeddingTE3L":
                with open(f'{path}/dqr_{month}_text_embeddingTE3L_3072.pkl', 'rb') as f:
                    embd_dict = pickle.load(f)
            elif model_name == "RoBERTa":
                file_path=f"{path}/weak_content_emb_{month}2024_RoBERTa_768.parquet"
                embd_dict=search_parquet_duckdb(file_path, col=None,q_domains=None,max_memory="8GB",schema={'key':'domain','val':'embeddings'})                
        elif embed_type == "domainName":
            with open(f'{path}/dqr_domainName_embeddingQwen3-0.6B_1024.pkl', 'rb') as f:
                embd_dict = pickle.load(f)
        elif embed_type == "GNN_GAT":
            if gnn_encoder=="RNI":
                # with open(f'{path}/gnn_embedding/{gnn_encoder}/{month}_binary_dqr_domain_rni_embeddings.pkl', 'rb') as f:
                #     embd_dict = pickle.load(f)               

                # file_path=f"{path}/gnn_embedding/{gnn_encoder}_23032026/{month}_domainRel_gat-text_emb.parquet"
                file_path=f"{path}/gnn_embedding/{gnn_encoder}_31032026/{month}_domainRel_gat-RNI_emb.parquet"
                logging.info(f"GNN emb file path={file_path}")
                embd_dict=search_parquet_duckdb(file_path, col=None,q_domains=None,max_memory="8GB",schema={'key':'domain','val':'emb'})
            elif gnn_encoder=="text":
                # file_path=f'{path}/gnn_embedding/{gnn_encoder}_15032026/{month}_binary_labelled_set_domain_from_text_embeddings.parquet'
                file_path=f"{path}/gnn_embedding/{gnn_encoder}/Feb2026/{month}_binary_labelled_set_domain_from_text_embeddings_updated_balanced.parquet"
                logging.info(f"GNN emb file path={file_path}")
                embd_dict=search_parquet_duckdb(file_path, col=None,q_domains=None,max_memory="8GB",schema={'key':'domain','val':'embeddings'})
        elif embed_type == "FQDN":
            # fqdn_file_name="weaklabels_fqdn_features.pkl"
            fqdn_file_name="weaklabels_dec_domains_fqdn_features.pkl"
            with open(f'{path}/{fqdn_file_name}', 'rb') as f:
                embd_dict = pickle.load(f)

        if isinstance(embd_dict[list(embd_dict.keys())[0]][0], dict): #list of dicts per domain pages (parquet format)
            embd_dict={ k:v[0]['emb'][0:emb_dim] for k,v in embd_dict.items()}
        elif isinstance(embd_dict[list(embd_dict.keys())[0]][0], list) and isinstance(embd_dict[list(embd_dict.keys())[0]][0][0], str) : #list of lists per domain pages (parquet format)
            embd_dict={ k:v[0][1][0:emb_dim] for k,v in embd_dict.items()}
        if normalize:
            embd_dict=normalize_embeddings(embd_dict)
        return embd_dict
    @staticmethod
    def load_emb_dict_from_parquet(embed_type: str, path:str="../../../data", model_name:str="embeddinggemma-300m", month:str="dec", target:str="pc1", emb_dim:int=8192,normalize:bool=False,original_emb_dim:int=1024,keep_content:str="all",keep_count:int=3):
        embd_dict=None
        if embed_type == "text":
                embd_dict=search_parquet_duckdb(f'{path}/weak_content_emb_{month}2024_{model_name}_{original_emb_dim}.parquet', col="domain",q_domains=None,max_memory="8GB",schema={'key':'domain','val':'embeddings'})
        elif embed_type == "GNN_GAT":
            with open(f'{path}/{month}_{target}_dqr_domain_rni_embeddings.pkl', 'rb') as f:
                embd_dict = pickle.load(f)

        if embed_type!="text" or keep_content=="all":
            if isinstance(embd_dict[list(embd_dict.keys())[0]][0], dict): #list of dicts per domain pages (parquet format)                
                if embed_type=="text":
                    domains_bylength_embd_dict={}
                    for d in embd_dict.keys():
                        d_pages_emb_dict={elem['page']:elem['emb'] for elem in embd_dict[d]}
                        selected_pages_lst=list(d_pages_emb_dict.keys())
                        domains_bylength_embd_dict[d]=d_pages_emb_dict[selected_pages_lst[0]][0:emb_dim]
                        for i in range(1,len(selected_pages_lst)):
                            domains_bylength_embd_dict[d]=fuse_1d_emb(domains_bylength_embd_dict[d],d_pages_emb_dict[selected_pages_lst[i]][0:emb_dim],fusion_mode="avg")
                    embd_dict=domains_bylength_embd_dict
                else:
                    embd_dict={ k:v[0]['emb'][0:emb_dim] for k,v in embd_dict.items()}
                    
            elif isinstance(embd_dict[list(embd_dict.keys())[0]][0], list) and isinstance(embd_dict[list(embd_dict.keys())[0]][0][0], str) : #list of lists per domain pages (parquet format)
                embd_dict={ k:v[0][1][0:emb_dim] for k,v in embd_dict.items()}
        else:
            logging.info(f"keep_content={keep_content}\tkeep_count={keep_count}")
            month_lengths_dict=pickle.load(open(f'{path}/domainrel_dec2024_pages_length_dict.pkl', 'rb'))
            domains_bylength_embd_dict={}
            if isinstance(embd_dict[list(embd_dict.keys())[0]][0], dict):
                for d in embd_dict.keys():
                    d_pages_emb_dict={elem['page']:elem['emb'] for elem in embd_dict[d]}
                    d_pages_length_dict={page:month_lengths_dict[page] for page in d_pages_emb_dict}
                    if keep_content=="longest":
                        sorted_dict = dict(sorted(d_pages_length_dict.items(), key=lambda x: x[1],reverse=True)) ## sort desc
                    elif keep_content=="shortest":
                        sorted_dict = dict(sorted(d_pages_length_dict.items(), key=lambda x: x[1])) ## sort asc
                    selected_pages_lst=list(sorted_dict.keys())[0:keep_count]
                    domains_bylength_embd_dict[d]=d_pages_emb_dict[selected_pages_lst[0]][0:emb_dim]
                    for i in range(1,len(selected_pages_lst)):
                        domains_bylength_embd_dict[d]=fuse_1d_emb(domains_bylength_embd_dict[d],d_pages_emb_dict[selected_pages_lst[i]][0:emb_dim],fusion_mode="avg")
            embd_dict=domains_bylength_embd_dict      
        if normalize:
            embd_dict=normalize_embeddings(embd_dict)
        return embd_dict
    @staticmethod
    def load_splits(path:str,split_mode:str="balanced", test_mode:str="credible-non"):
        domain_rel_annotations_dict=pickle.load(open(f'{path}/domain_rel_annotations_dict.pkl', 'rb'))
        domain_rel_annotations_dict={k:v[0] for k,v in domain_rel_annotations_dict.items()}
        category_set=set([k for k,v in domain_rel_annotations_dict.items() if v==split_mode])
        splits_lst=[]
        for split in["train","val","test"]:
            split_df=search_parquet_duckdb(f'{path}/all_splits/balanced/{split}_domains.parquet', col=None,q_domains=None,max_memory="8GB",schema=None)        
            split_df['domain']=split_df['domain'].apply(lambda x: '.'.join(str(x).split('.')[::-1])) 
            splits_lst.append(split_df)

        if test_mode=="credible-non":
            ######### test and validate on all domains (balanced set) but train with either all balanced or a sub-category only domains as label 0 (igonre other subcategories) ############                                  
            if split_mode!="balanced":                
                splits_lst[0]=splits_lst[0][splits_lst[0]["domain"].isin(category_set)]            

        elif test_mode=="sub-category":
            ######### subcategory classifier: consider subcategory domains with label 0 as label 1 and all others categories as label 0 ############
            true_labels_set=set()
            for idx in range(0,len(splits_lst)):
                cat_df=splits_lst[idx][splits_lst[idx]["domain"].isin(category_set)]
                true_labels_set.update(cat_df[cat_df["label"]==0]["domain"].tolist())
            for idx in range(0,len(splits_lst)):         
                splits_lst[idx]["label"]=splits_lst[idx]["domain"].apply(lambda x:1 if x in true_labels_set else 0)

        return splits_lst[0],splits_lst[1],splits_lst[2]
    
    @staticmethod
    def load_run_embeddings(args: dict):
        global agg_months_dict
        full_emb_dict={}
        ############## Load text embeddings and labels ###############
        if args.embed_type=="FQDN":
            month_emb_dict =DomainRel.load_emb_dict(args.embed_type, args.domainRel_text_emb_path, pickle_name=None,emb_dim=args.emb_dim,normalize=True)
            full_emb_dict=month_emb_dict
        else:
            if args.emb_model=="embeddingTE3L":
                args.emb_model="embeddinggemma-300m"
            # args.emb_model="Qwen3-Embedding-0.6B"
            full_emb_dict =DomainRel.load_emb_dict(args.embed_type, args.domainRel_text_emb_path, pickle_name=f"weak_content_emb_{args.emb_model}_{args.original_emb_dim}.pkl",emb_dim=args.emb_dim,normalize=False)

        if args.agg_text_emb:
            month_emb_dict = DomainRel.load_agg_Nmonth_emb_dict("text",args.domainRel_gnn_emb_path,model_name=args.emb_model, agg=args.agg_function,gnn_encoder=args.gnn_encoder,month_lst=agg_months_dict[args.month],emb_dim=args.emb_dim, original_emb_dim=args.original_emb_dim)
        elif args.embed_type not in ["FQDN"]:
            month_emb_dict = DomainRel.load_emb_dict_from_parquet(args.embed_type, args.domainRel_text_emb_path, args.emb_model, args.month,normalize=False,emb_dim=args.emb_dim, original_emb_dim=args.original_emb_dim,keep_content=args.keep_content,keep_count=args.keep_content_count)

        weaklabeles_df = pd.read_csv(f"{args.domainRel_path}/weaklabels.csv")
        weaklabeles_df = weaklabeles_df[weaklabeles_df["domain"].isin(full_emb_dict)]
        weaklabeles_df = weaklabeles_df.reset_index(drop=True)
        text_emb_dict={}
        text_emb_dict.update(month_emb_dict)
        # text_emb_dict.update({k:v for k,v in full_emb_dict.items() if k not in month_emb_dict})
        text_emb_dict.update({k:v for k,v in full_emb_dict.items() })
        acc_lst,f1_lst=[],[]
        ############### filter by the GNN graph node splits ###################
        gnn_emb_dict = None
        gnn_emb_dict = DomainRel.load_emb_dict("GNN_GAT", args.domainRel_gnn_emb_path,month=args.month,gnn_encoder=args.gnn_encoder) 
        postfix_len_avg=np.mean([len(elem.split(".")[-1]) for elem in list(gnn_emb_dict.keys())[0:10]])
        logging.info(f"GNN domains postfix_len_avg={postfix_len_avg}")
        if postfix_len_avg>3: ## domain names are reversed in the GNN embedding dict i.e. com.domain instead of domain.com, so we reverse them back to match the weaklabels domains format
            gnn_emb_dict={".".join(k.split(".")[::-1]):v for k,v in gnn_emb_dict.items()}    
        ######### handel reversed domains ################
        missing_domains_set=set(gnn_emb_dict.keys())-set(weaklabeles_df[weaklabeles_df["domain"].isin(gnn_emb_dict.keys())]["domain"])
        for k in missing_domains_set:
            gnn_emb_dict[".".join(k.split(".")[::-1])]=gnn_emb_dict[k]

        missing_domains_set=set(gnn_emb_dict.keys())-set(weaklabeles_df[weaklabeles_df["domain"].isin(gnn_emb_dict.keys())]["domain"])    
        weaklabeles_df=weaklabeles_df[weaklabeles_df["domain"].isin(gnn_emb_dict.keys())]    
        logging.info(f"len missing domains set={len(missing_domains_set)}")
        # ############### Load splits ###################
        if args.filter_by_GNN_nodes:   
            train_domains_df,valid_domains_df,test_domains_df=DomainRel.load_splits(args.domainRel_path, split_mode=args.split_mode, test_mode=args.test_mode)
            ############ assign splts`labels ################
            lables_dict={}
            for split in [train_domains_df,valid_domains_df,test_domains_df]:
                lables_dict.update(dict(zip(split["domain"],split["label"])))
            weaklabeles_df["weak_label"]=weaklabeles_df["domain"].apply(lambda x:-1 if x not in lables_dict else lables_dict[x]).astype(int)

            test_domains_set=set(test_domains_df['domain']) 
            valid_domains_set=set(valid_domains_df['domain']) 
            train_domains_set=set(train_domains_df['domain']) 
            # filter_by_domains_set=test_domains_set.union(valid_domains_set).union(train_domains_set)
            logging.info(f"test set labels count ={weaklabeles_df[weaklabeles_df["domain"].isin(test_domains_df['domain'])]["weak_label"].value_counts()}")
            logging.info(f"valid set labels count ={weaklabeles_df[weaklabeles_df["domain"].isin(valid_domains_df['domain'])]["weak_label"].value_counts()}")
            logging.info(f"train set labels count ={weaklabeles_df[weaklabeles_df["domain"].isin(train_domains_df['domain'])]["weak_label"].value_counts()}")
        ################### GNN Embedding #############    
        features_emb_dict = None   
        if args.use_gnn_emb:
            if args.agg_month_emb:
                gnn_emb_dict = DomainRel.load_agg_Nmonth_emb_dict("GNN_GAT",args.domainRel_gnn_emb_path, agg=args.agg_function,gnn_encoder=args.gnn_encoder,month_lst=agg_months_dict[args.month])
                gnn_emb_dict={".".join(k.split(".")[::-1]):v for k,v in gnn_emb_dict.items()}
            elif not gnn_emb_dict:
                gnn_emb_dict = DomainRel.load_emb_dict("GNN_GAT", args.domainRel_gnn_emb_path,month=args.month,gnn_encoder=args.gnn_encoder)
                gnn_emb_dict={".".join(k.split(".")[::-1]):v for k,v in gnn_emb_dict.items()}
            weaklabeles_df = weaklabeles_df[weaklabeles_df["domain"].isin(gnn_emb_dict.keys())]
            weaklabeles_df = weaklabeles_df.reset_index(drop=True)
        else:
            gnn_emb_dict=None   
        
        if args.use_FQDN:                                         
            features_emb_dict =DomainRel.load_emb_dict("FQDN", args.domainRel_text_emb_path, pickle_name=None,model_name=None,emb_dim=args.emb_dim,normalize=True)
        ############### Split #####################
        if args.filter_by_GNN_nodes: 
            X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(args.domainRel_target, weaklabeles_df,key='domain',test_valid_size=args.test_valid_size,regressor=False,train_lst=train_domains_set,valid_lst=valid_domains_set,test_lst=test_domains_set)
        else:
            X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(args.domainRel_target, weaklabeles_df,key='domain',test_valid_size=args.test_valid_size,regressor=False)

        logging.info(f"len(X_train)={len(X_train)}\tlen(X_valid)={len(X_valid)}\tlen(X_test)={len(X_test)}\t")
        X_train_feat, X_valid_feat, X_test_feat = resize_and_fuse_emb(text_emb_dict, args.domainRel_target, X_train, X_valid,X_test, gnn_emb=gnn_emb_dict,topic_emb=features_emb_dict,trim_to=args.emb_dim,fusion_mode=args.fusion_mode)
        logging.info(f"X_train_feat.shape={len(X_train_feat[0]) if type(X_train_feat[0]) == list else X_train_feat[0].shape}")
        return X_train, y_train, X_valid, y_valid, X_test, y_test,X_train_feat, X_valid_feat, X_test_feat
    def get_domains_lst(path:str="~/scratch/hsh_projects/CrediText/data/weaksupervision/weaklabels.csv"):
        labels_df = pd.read_csv(path)
        return labels_df["domain"].tolist()
class DQR (object):
    @staticmethod
    def load_emb_dict(embed_type: str, path:str="../../../data",pickle_name:str=None, model_name:str="embeddinggemma-300m", month:str="dec", target:str="pc1", emb_dim:int=256,normalize:bool=False,gnn_encoder:str="RNI"):
        if pickle_name:
            with open(f'{path}/{pickle_name}', 'rb') as f:
                embd_dict = pickle.load(f)
        elif embed_type == "text":
            if model_name == "embeddinggemma-300m":
                with open(f'{path}/dqr_{month}_text_embeddinggemma-300m_{emb_dim}.pkl', 'rb') as f:
                    embd_dict = pickle.load(f)
            elif model_name == "embeddingQwen3-0.6B":
                with open(f'{path}/dqr_{month}_text_embeddingQwen3-0.6B_1024.pkl', 'rb') as f:
                    embd_dict = pickle.load(f)
            elif model_name == "embeddingQwen3-8B":
                with open(f'{path}/dqr_{month}_text_embeddingQwen3-8B_4096.pkl', 'rb') as f:
                    embd_dict = pickle.load(f)
            elif model_name == "embeddingTE3L":
                with open(f'{path}/dqr_{month}_text_embeddingTE3L_3072.pkl', 'rb') as f:
                    embd_dict = pickle.load(f)
            elif model_name == "IPTC_Topic_emb":
                with open(f'IPTCTopicModeling/dqr_dec_IPTC_predFinalLayer_emb_dict.pkl','rb') as f:
                    embd_dict = pickle.load(f)
        elif embed_type == "domainName":
            with open(f'{path}/dqr_domainName_embeddingQwen3-0.6B_1024.pkl', 'rb') as f:
                embd_dict = pickle.load(f)
        elif embed_type == "GNN_GAT":
            if gnn_encoder=="RNI":
                # file_path=f'{path}/gnn_embedding/RNI/{target}/{month}_{target}_dqr_domain_rni_embeddings.pkl'  # Jan 2026 version with RNI emb
                # with open(file_path, 'rb') as f: 
                #     embd_dict = pickle.load(f)

                # file_path=f'{path}/gnn_embedding/{gnn_encoder}_23032026/{target}/{month}_dqr_gat-text_emb.parquet'  # 23 March 2026 version with gat RNI
                file_path=f'{path}/gnn_embedding/{gnn_encoder}_31032026/{target}/{month}_dqr_gat-RNI_emb.parquet'
                embd_dict=search_parquet_duckdb(file_path, col=None,q_domains=None,max_memory="8GB",schema={'key':'domain','val':'emb'})

            elif gnn_encoder=="text":
                # file_path=f'{path}/gnn_embedding/{gnn_encoder}_15032026/{target}/{month}_dqr_gat-text_emb.parquet' # Jan 2026 version with gat text emb
                # embd_dict=search_parquet_duckdb(file_path, col=None,q_domains=None,max_memory="8GB",schema={'key':'domain','val':'emb'})              

                file_path=f'{path}/gnn_embedding/{gnn_encoder}_Feb2026/{target}/{month}_dqr_domain_gat_from_text_embeddings_updated.parquet' # Feb 2026 version with gat text emb  
                embd_dict=search_parquet_duckdb(file_path, col=None,q_domains=None,max_memory="8GB",schema={'key':'domain','val':'embeddings'})                    

            logging.info(f"GNN emb file path={file_path}")

        elif embed_type == "IPTC_Topic":
            with open(f'IPTCTopicModeling/dqr_IPTC-news-topic_scores.pkl', 'rb') as f:
                embd_dict = pickle.load(f)
        elif embed_type == "IPTC_Topic_freq":
            with open(f'IPTCTopicModeling/dqr_topics_frequency_norm_dict.pkl','rb') as f:
                embd_dict = pickle.load(f)
        elif embed_type == "IPTC_Topic_emb":
            with open(f'IPTCTopicModeling/dqr_dec_IPTC_predFinalLayer_emb_dict.pkl','rb') as f:
                embd_dict = pickle.load(f)
            # logging.info(list(embd_dict.keys())[0],embd_dict[list(embd_dict.keys())[0]])
        elif embed_type == "3Feat":
            with open(f'/shared_mnt/github_repos/CrediGraph/data/dqr/dqr_3Feat_dict.pkl', 'rb') as f:
                embd_dict = pickle.load(f)
        elif embed_type == "3Feat2":
            with open(f'/shared_mnt/github_repos/CrediGraph/data/dqr/dqr_3Feat_dict2.pkl', 'rb') as f:
                embd_dict = pickle.load(f)

        elif embed_type == "TFIDF":
            # with open(f'{path}/dqr_TFIDF_emb.pkl', 'rb') as f:
            #     embd_dict=pickle.load(f)
            # with open(f'{path}/dqr_TFIDF_emb_8465.pkl', 'rb') as f:
            #     embd_dict=pickle.load(f)
            # with open(f'{path}/dqr_TFIDF_emb_19437.pkl', 'rb') as f:
            #     embd_dict=pickle.load(f)
            if month == "dec":
                with open(f'{path}/dqr_dec_TFIDF_weaksupervision_emb_222755.pkl', 'rb') as f:
                    embd_dict = pickle.load(f)
            elif month == "nov":
                with open(f'{path}/dqr_nov_TFIDF_weaksupervision_emb_258729.pkl', 'rb') as f:
                    embd_dict = pickle.load(f)
            elif month == "oct":
                with open(f'{path}/dqr_oct_TFIDF_weaksupervision_emb_19085.pkl', 'rb') as f:
                    embd_dict = pickle.load(f)
        elif embed_type == "PASTEL":
            with open(f'{path}/dqr_pastel_dict.pkl', 'rb') as f:
                embd_dict = pickle.load(f)
        elif embed_type == "PASTEL_hasContent":
            with open(f'{path}/dqr_hasContent_pastel_dict.pkl', 'rb') as f:
                embd_dict = pickle.load(f)
        elif embed_type == "propella_annotations":
            # emb_file_name="dqr_propella_annotations_features.pkl"
            # emb_file_name="dqr_propella_annotations_html_emb_e5-small-v2.pkl"
            # emb_file_name="dqr_propella_annotations_html_emb_F2LLM-0.6B.pkl"
            emb_file_name="dqr_propella_annotations_html_features.pkl"
            with open(f'{path}/{emb_file_name}', 'rb') as f:
                embd_dict = pickle.load(f)
        elif embed_type == "FQDN":
            with open(f'{path}/dqr_fqdn_features.pkl', 'rb') as f:
                embd_dict = pickle.load(f)
        if isinstance(embd_dict[list(embd_dict.keys())[0]][0], dict): #list of dicts per domain pages (parquet format)
            embd_dict={ k:v[0]['emb'] for k,v in embd_dict.items()}
        elif isinstance(embd_dict[list(embd_dict.keys())[0]][0], list) and isinstance(embd_dict[list(embd_dict.keys())[0]][0][0], str) : #list of lists per domain pages (parquet format)
            embd_dict={ k:v[0][1] for k,v in embd_dict.items()}
        if normalize:
            embd_dict=normalize_embeddings(embd_dict)
        return embd_dict

    @staticmethod
    def load_weaksupervision_emb_dict(embed_type: str, path: str="../../../data", model_name: str="embeddinggemma-300m", month: str="dec",
                                    target:str="pc1", gnn_emb:str=None, agg:str=None):
        embd_dict_phishtank, embd_dict_URLhaus, embd_dict_PhishDataset_legit = {}, {}, {}
        if embed_type == "text":
            if model_name == "embeddinggemma-300m":
                with open(f'{path}/dqr_{month}_text_embeddinggemma-300m_768.pkl', 'rb') as f:
                    embd_dict = pickle.load(f)
            elif model_name == "embeddingQwen3-0.6B":
                with open(f'{path}/dqr_{month}_text_embeddingQwen3-0.6B_1024.pkl', 'rb') as f:
                    embd_dict = pickle.load(f)
            elif model_name == "embeddingQwen3-8B":
                with open(f'{path}/cc_dec_2024_phishtank_Qwen3-Embedding-8B_4096.pkl', 'rb') as f:
                    embd_dict_phishtank = pickle.load(f)
                with open(f'{path}/cc_dec_2024_URLhaus_Qwen3-Embedding-8B_4096.pkl', 'rb') as f:
                    embd_dict_URLhaus = pickle.load(f)
                with open(f'{path}/cc_dec_2024_PhishDataset_legit_Qwen3-Embedding-8B_4096.pkl', 'rb') as f:
                    embd_dict_PhishDataset_legit = pickle.load(f)
            elif model_name == "embeddingTE3L":
                with open(f'{path}/phishtank_{month}_TE3L_weaksupervision_emb_3072.pkl', 'rb') as f:
                    embd_dict_phishtank = pickle.load(f)
                with open(f'{path}/URLhaus_{month}_TE3L_weaksupervision_emb_3072.pkl', 'rb') as f:
                    embd_dict_URLhaus = pickle.load(f)
                with open(f'{path}/PhishDataset_legit_{month}_TE3L_weaksupervision_emb_3072.pkl', 'rb') as f:
                    embd_dict_PhishDataset_legit = pickle.load(f)

            if gnn_emb == True:
                logging.info("len of embd_dict_phishtank before appending GNN=",
                    len(embd_dict_phishtank[list(embd_dict_phishtank.keys())[0]]))
                if agg is None:
                    with open(f'{path}/PhishTank_{target}_rni_embeddings.pkl', 'rb') as f:
                        gnn_embd_dict_phishtank = pickle.load(f)
                    with open(f'{path}/URLHaus_{target}_rni_embeddings.pkl', 'rb') as f:
                        gnn_embd_dict_URLhaus = pickle.load(f)
                    with open(f'{path}/IP2Location_{target}_rni_embeddings.pkl', 'rb') as f:
                        gnn_embd_dict_PhishDataset_legit = pickle.load(f)
                else:
                    gnn_embd_dict_phishtank, gnn_embd_dict_URLhaus, gnn_embd_dict_PhishDataset_legit = load_agg_Nmonth_weaksupervision_emb_dict(
                        embed_type, path, model_name, month_lst=["dec", "nov", "oct"], target=target, agg=agg)

                ############# Append Emb #############
                for k in embd_dict_phishtank:
                    embd_dict_phishtank[k] = gnn_embd_dict_phishtank[k] + embd_dict_phishtank[k]
                for k in embd_dict_URLhaus:
                    embd_dict_URLhaus[k] = gnn_embd_dict_URLhaus[k] + embd_dict_URLhaus[k]
                for k in embd_dict_PhishDataset_legit:
                    embd_dict_PhishDataset_legit[k] = gnn_embd_dict_PhishDataset_legit[k] + embd_dict_PhishDataset_legit[k]
                logging.info("len of embd_dict_phishtank after appending GNN=",
                    len(embd_dict_phishtank[list(embd_dict_phishtank.keys())[0]]))


        elif embed_type == "domainName":
            with open(f'{path}/dqr_domainName_embeddingQwen3-0.6B_1024.pkl', 'rb') as f:
                embd_dict = pickle.load(f)
        elif embed_type == "GNN_GAT":
            with open(f'{path}/11Kdataset_GAT_targets_connected_edges_GNN_textE_300E_pc1_emb.pkl', 'rb') as f:
                embd_dict = pickle.load(f)
        elif embed_type == "TFIDF":
            # with open(f'{path}/dqr_TFIDF_emb.pkl', 'rb') as f:
            #     embd_dict=pickle.load(f)
            # with open(f'{path}/dqr_TFIDF_emb_8465.pkl', 'rb') as f:
            #     embd_dict=pickle.load(f)
            # with open(f'{path}/dqr_dec_TFIDF_emb_19437.pkl', 'rb') as f:
            #     embd_dict=pickle.load(f)
            # with open(f'{path}/dqr_TFIDF_emb_19437.pkl', 'rb') as f:
            #     embd_dict=pickle.load(f)
            emb_size = "222755" if month == "dec" else "258729" if month == "nov" else "19085"
            with open(f'{path}/phishtank_{month}_TFIDF_weaksupervision_emb_{emb_size}.pkl', 'rb') as f:
                embd_dict_phishtank = pickle.load(f)
            with open(f'{path}/URLhaus_{month}_TFIDF_weaksupervision_emb_{emb_size}.pkl', 'rb') as f:
                embd_dict_URLhaus = pickle.load(f)
            with open(f'{path}/phishDataset_legit_{month}_TFIDF_weaksupervision_emb_{emb_size}.pkl', 'rb') as f:
                embd_dict_PhishDataset_legit = pickle.load(f)

        return embd_dict_phishtank, embd_dict_URLhaus, embd_dict_PhishDataset_legit

    @staticmethod
    def load_agg_Nmonth_emb_dict(embed_type: str, path: str="../../../data", model_name: str="embeddinggemma-300m",
                            month_lst: list[str]=["oct", "nov", "dec"], target: str="pc1", agg: str="avg",emb_dim: int=256,normalize: bool=False,gnn_encoder: str="text",original_emb_dim: int=256):
        """load and aggregate N-month embedding dictionaries for both text and GNN embeddings
        Args:
            embed_type: The type of the embedding i.e text,GN_GAT, others
            path: The embedding pickle file or parquet file path
            model_name: the LLM embeding model name
            target: the regression target i.e PC!, MBFC or others
            month_lst: list of months to aggregate
            agg: the ggregation function i.e. avg,cat,min,max
            gnn_encoder: the GNN embedding encoder i.e RNI or text
            normalize: boolean to normalize the embeddings
            emb_dim: the embedding diminsion to trim at
            original_emb_dim: the original full length embedding size

        Returns:
            The aggerated N-Month embeddings
        """
        months_emb_lst = []
        for month in month_lst:
            if embed_type == "text":
                    if model_name == "embeddinggemma-300m":
                        with open(f'{path}/dqr_{month}_text_embeddinggemma-300m_{emb_dim}.pkl', 'rb') as f:
                            embd_dict = pickle.load(f)
                    elif model_name == "embeddingQwen3-0.6B":
                        with open(f'{path}/dqr_{month}_text_embeddingQwen3-0.6B_1024.pkl', 'rb') as f:
                            embd_dict = pickle.load(f)
                    elif model_name == "embeddingQwen3-8B":
                        with open(f'{path}/dqr_{month}_text_embeddingQwen3-8B_4096.pkl', 'rb') as f:
                            embd_dict = pickle.load(f)
                    elif model_name == "embeddingTE3L":
                        with open(f'{path}/dqr_{month}_text_embeddingTE3L_3072.pkl', 'rb') as f:
                            embd_dict = pickle.load(f)            
            elif embed_type == "GNN_GAT":          
                if gnn_encoder=="RNI":
                    with open(f'{path}/gnn_embedding/RNI/{target}/{month}_{target}_dqr_domain_rni_embeddings.pkl', 'rb') as f:
                        embd_dict = pickle.load(f)
                elif gnn_encoder=="text":
                    file_path=f'{path}/gnn_embedding/text/{target}/{month}_dqr_domain_gat_from_text_embeddings_updated.parquet'
                    embd_dict=search_parquet_duckdb(file_path, col=None,q_domains=None,max_memory="8GB",schema={'key':'domain','val':'embeddings'})

            if isinstance(embd_dict[list(embd_dict.keys())[0]][0], dict): #list of dicts per domain pages (parquet format)
                embd_dict={ k:v[0]['emb'] for k,v in embd_dict.items()}
            elif isinstance(embd_dict[list(embd_dict.keys())[0]][0], list) and isinstance(embd_dict[list(embd_dict.keys())[0]][0][0], str) : #list of lists per domain pages (parquet format)
                embd_dict={ k:v[0][1] for k,v in embd_dict.items()}
            if normalize:
                embd_dict=normalize_embeddings(embd_dict)
            months_emb_lst.append(embd_dict)

        common_domains_set=set(months_emb_lst[0].keys())    
        for lst in months_emb_lst[1:]:
            common_domains_set = common_domains_set.intersection(lst.keys())
        
        diff_domains_set=set(months_emb_lst[-1].keys())-common_domains_set
        for key in diff_domains_set:
            if agg == "cat":
                months_emb_lst[-1][key].extend(months_emb_lst[-1][key]*len(months_emb_lst))

        for key in common_domains_set:
            for i in range(0, len(months_emb_lst)-1):
                if agg == "cat":
                    months_emb_lst[-1][key].extend(months_emb_lst[i][key])
                elif agg == "min":
                    months_emb_lst[-1][key] = [min(a, b) for a, b in zip(months_emb_lst[-1][key], months_emb_lst[i][key])]
                elif agg == "max":
                    months_emb_lst[-1][key] = [max(a, b) for a, b in zip(months_emb_lst[-1][key], months_emb_lst[i][key])]
                elif agg == "avg":
                    months_emb_lst[-1][key] = [(a + b) / 2 for a, b in zip(months_emb_lst[-1][key], months_emb_lst[i][key])]

        # concat_dict = {k: v for k, v in months_emb_lst[-1].items() if k in common_domains_set}
        # logging.info(f"concat Nmonth emb size={len(concat_dict[list(concat_dict.keys())[0]])}")
        # logging.info(f"len of keys={len(concat_dict.keys())}")
        return months_emb_lst[-1]

    @staticmethod
    def load_agg_Nmonth_weaksupervision_emb_dict(embed_type: str, path: str="../../../data", model_name: str="embeddinggemma-300m",
                                                month_lst:list[str]=["dec", "nov", "oct"], target:str="pc1", agg:str="avg"):
        months_emb_PhishTank_lst = []
        months_emb_URLhaus_lst = []
        months_emb_legit_lst = []
        for month in month_lst:
            with open(f'{path}/PhishTank_{target}_rni_{month}_2024_embeddings.pkl', 'rb') as f:
                months_emb_PhishTank_lst.append(pickle.load(f))
            with open(f'{path}/URLHaus_{target}_rni_{month}_2024_embeddings.pkl', 'rb') as f:
                months_emb_URLhaus_lst.append(pickle.load(f))
            with open(f'{path}/IP2Location_{target}_rni_{month}_2024_embeddings.pkl', 'rb') as f:
                months_emb_legit_lst.append(pickle.load(f))

        for ds_months in [months_emb_PhishTank_lst, months_emb_URLhaus_lst, months_emb_legit_lst]:
            for key in ds_months[0].keys():
                for i in range(1, len(ds_months)):  # loop on dataset months
                    if key in ds_months[i]:
                        if agg == "concat":
                            ds_months[0][key].extend(ds_months[i][key])
                            # logging.info(len(ds_months[0][key]))
                        elif agg == "min":
                            ds_months[0][key] = [min(a, b) for a, b in zip(ds_months[0][key], ds_months[i][key])]
                        elif agg == "max":
                            ds_months[0][key] = [max(a, b) for a, b in zip(ds_months[0][key], ds_months[i][key])]
                        elif agg == "avg":
                            ds_months[0][key] = [(a + b) / 2 for a, b in zip(ds_months[0][key], ds_months[i][key])]
                            # logging.info(len(ds_months[0][key]))
        return months_emb_PhishTank_lst[0], months_emb_URLhaus_lst[0], months_emb_legit_lst[0]

    @staticmethod
    def load_run_embeddings(args:dict):
        global agg_months_dict
        ############## Load training data and split ###############
        if args.agg_text_emb:
            text_emb_dict = DQR.load_agg_Nmonth_emb_dict("text", path= args.dqr_text_emb_path,model_name= args.emb_model,month_lst=agg_months_dict[args.month], agg="avg",normalize=False)
        else:
            text_emb_dict = DQR.load_emb_dict(args.embed_type,path=args.dqr_text_emb_path, model_name=args.emb_model, month=args.month,normalize=True,emb_dim=args.emb_dim)

        labeled_11k_df = pd.read_csv(f"{args.dqr_path}/domain_ratings.csv")
        labeled_11k_df[f"{args.dqr_target}_norm"]=labeled_11k_df[args.dqr_target].apply(lambda x: round(float(x)*10))
        ######################## Filter by GNN montly nodes ####################
        targets_nodes_df = pd.read_csv(f"{args.dqr_path}/targets_nodes_df.csv")
        targets_nodes_df["domain_rev"] = targets_nodes_df["domain"].apply(lambda x: '.'.join(str(x).split('.')[::-1]))
        labeled_11k_df = labeled_11k_df[labeled_11k_df["domain"].isin(text_emb_dict)]
        ############### filter_by_PASTEL_domains ###################
        if 'filter_by_PASTEL_domains' in args and args.filter_by_PASTEL_domains:
            pastel_emb_dict = {}
            with open(f'{args.dqr_path}/dqr_hasContent_pastel_dict.pkl', 'rb') as f:
                pastel_emb_dict = pickle.load(f)
            labeled_11k_df = labeled_11k_df[labeled_11k_df["domain"].isin(pastel_emb_dict.keys())]   
        
        # ############### filter by the 8K GNN Nodes ###################
        # if args.filter_by_GNN_nodes:   
        #     labeled_11k_df = labeled_11k_df[labeled_11k_df["domain"].isin(targets_nodes_df["domain_rev"])]
        ############### filter by the train/val/test domains ###################
        test_domains_df=search_parquet_duckdb(f'{args.dqr_path}/splits/test_regression_domains.parquet', col=None,q_domains=None,max_memory="8GB",schema=None)
        test_domains_df['domain']=test_domains_df['domain'].apply(lambda x: '.'.join(str(x).split('.')[::-1]))
        filtered_test_df=labeled_11k_df[labeled_11k_df["domain"].isin(test_domains_df['domain'])]
        logging.info(f"test set lables count ={len(filtered_test_df),filtered_test_df[f"{args.dqr_target}_norm"].value_counts()}")

        valid_domains_df=search_parquet_duckdb(f'{args.dqr_path}/splits/val_regression_domains.parquet', col=None,q_domains=None,max_memory="8GB",schema=None)
        valid_domains_df['domain']=valid_domains_df['domain'].apply(lambda x: '.'.join(str(x).split('.')[::-1]))
        filtered_val_df=labeled_11k_df[labeled_11k_df["domain"].isin(valid_domains_df['domain'])]
        logging.info(f"val set lables count ={len(filtered_val_df),filtered_val_df[f"{args.dqr_target}_norm"].value_counts()}")

        train_domains_df=search_parquet_duckdb(f'{args.dqr_path}/splits/train_regression_domains.parquet', col=None,q_domains=None,max_memory="8GB",schema=None)
        train_domains_df['domain']=train_domains_df['domain'].apply(lambda x: '.'.join(str(x).split('.')[::-1]))
        filtered_train_df=labeled_11k_df[labeled_11k_df["domain"].isin(train_domains_df['domain'])]
        logging.info(f"train set lables count ={len(filtered_train_df),filtered_train_df[f"{args.dqr_target}_norm"].value_counts()}")

        test_domains_set=set(test_domains_df['domain']) 
        valid_domains_set=set(valid_domains_df['domain']) 
        train_domains_set=set(train_domains_df['domain']) 
        filter_by_domains_set=test_domains_set.union(valid_domains_set).union(train_domains_set)
        labeled_11k_df = labeled_11k_df[labeled_11k_df["domain"].isin(filter_by_domains_set)]
        non_exist_domains_df = labeled_11k_df[~labeled_11k_df["domain"].isin(filter_by_domains_set)]
        logging.info(f"non_exist_domains_df={non_exist_domains_df}")
        labeled_11k_df = labeled_11k_df.reset_index(drop=True)
        text_emb_dict={k:v for k,v in text_emb_dict.items() if k in filter_by_domains_set}  

        
        features_emb_dict = None 
        gnn_emb_dict=None
        if args.use_gnn_emb:
            if args.agg_month_emb:
                agg_months_dict={"oct":["oct"],
                            "nov":["oct","nov"],
                            "dec":["oct","nov","dec"]}
                gnn_emb_dict = DQR.load_agg_Nmonth_emb_dict("GNN_GAT", args.dqr_gnn_emb_path, agg=args.agg_function,gnn_encoder=args.gnn_encoder,month_lst=agg_months_dict[args.month])
            else:
                gnn_emb_dict = DQR.load_emb_dict("GNN_GAT", args.dqr_gnn_emb_path,month=args.month,gnn_encoder=args.gnn_encoder,normalize=False)
            labeled_11k_df = labeled_11k_df[labeled_11k_df["domain"].isin(gnn_emb_dict.keys())]
        if 'use_topic_emb' in args and args.use_topic_emb:
            # features_emb_dict = load_emb_dict("IPTC_Topic", args.emb_path)
            # features_emb_dict = load_emb_dict("IPTC_Topic_freq", args.emb_path)
            # features_emb_dict = load_emb_dict("IPTC_Topic_emb", args.emb_path)
            # features_emb_dict = load_emb_dict("3Feat", args.emb_path)
            # features_emb_dict = load_emb_dict("3Feat2", args.emb_path)
            features_emb_dict = DQR.load_emb_dict("PASTEL_hasContent", args.dqr_text_emb_path)
        if 'use_FQDN' in args and args.use_FQDN:
            features_emb_dict = DQR.load_emb_dict("FQDN", args.dqr_text_emb_path,normalize=False)
            # features_emb_dict = DQR.load_emb_dict(embed_type="FQDN",args.dqr_text_emb_path, pickle_name=None,model_name=None,emb_dim=args.emb_dim,normalize=True)
        
        ############ Use 3M fixed Split #################
        X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(args.dqr_target, labeled_11k_df,key='domain',test_valid_size=args.test_valid_size,regressor=False,train_lst=train_domains_set,valid_lst=valid_domains_set,test_lst=test_domains_set)
        ############ Use Startified random split per month #################
        # X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(args.dqr_target, labeled_11k_df,key='domain',test_valid_size=args.test_valid_size)
        ############ resize Emb #################
        X_train_feat, X_valid_feat, X_test_feat = resize_and_fuse_emb(text_emb_dict, args.dqr_target, X_train, X_valid,X_test, gnn_emb=gnn_emb_dict,topic_emb=features_emb_dict,trim_to=args.emb_dim)
        logging.info(f"X_train_feat.shape={len(X_train_feat[0]) if type(X_train_feat[0]) == list else X_train_feat[0].shape}")
        return X_train, y_train, X_valid, y_valid, X_test, y_test,X_train_feat, X_valid_feat, X_test_feat
    
    @staticmethod
    def get_domains_lst(path:str="~/scratch/hsh_projects/CrediText/data/dqr/domain_ratings.csv"):        
        labels_df = pd.read_csv(path)
        return labels_df["domain"].tolist()
        
