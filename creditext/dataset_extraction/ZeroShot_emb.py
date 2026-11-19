import argparse
import gc
from math import e
import os
import sys
import pandas as pd
import duckdb
from domain_sampler import DomainSampler
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
sys.path.insert(1, "/".join(os.getcwd().split("/")[0:-2]))
import os
import torch
from tqdm import tqdm

from creditext.experiments.mlp_experiments.utils import search_parquet_duckdb, reverse_domain_name, query_parquet_duckdb
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ccmain content merge")
    parser.add_argument("--task", type=str, default="CalcAccuracy", choices=["deepAnalysis","CosineSimilarity","preComputed","CalcAccuracy"], help="task type")
    parser.add_argument("--emb_model", type=str, default="Alibaba-NLP/gte-large-en-v1.5", choices=["BAAI/bge-base-en-v1.5","BAAI/bge-m3","Alibaba-NLP/gte-multilingual-base","microsoft/harrier-oss-v1-270m","Alibaba-NLP/gte-large-en-v1.5","google/embeddinggemma-300m","paraphrase-multilingual-MiniLM-L12-v2","DatologyAI/luxical-one"], help="task type")
    parser.add_argument("--taxonomy", type=str, default="webOrganize_topic24", choices=["webOrganize_format24","webOrganize_topic24"], help="task type")
    parser.add_argument("--ccmain", type=str, default="ccmain202451", help="ccmain")
    parser.add_argument("--base_path", type=str, default="/home/mila/a/abdallah/scratch/hsh_projects/CrediText/creditext/dataset_extraction", help="base path")    
    parser.add_argument("--start_idx", type=int, default=0, help="frist chunck idx")    
    parser.add_argument("--end_idx", type=int, default=100, help="end chunck idx")     
    parser.add_argument("--dataset", type=str, default="WebOrganizer-TopicAnnotations-Llama-3.1-405B-FP8", choices=["WebOrganizer-FormatAnnotations-Llama-3.1-405B-FP8","WebOrganizer-TopicAnnotations-Llama-3.1-405B-FP8","dec2024_doc-content_samples_1k"], help="dataset name")     
    args = parser.parse_args()
    print(f"args={args}")     
    base_path=args.base_path    
    bert_model_path=f"/home/mila/a/abdallah/scratch/hsh_projects/CrediText/domain_sampler/new_safe_bertopic"             
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.task=="CalcAccuracy":
        lables_df=pd.read_csv(f"{base_path}/{args.dataset}_true_lables.csv")
        if args.taxonomy=="webOrganize_topic24":
            test_paper_lables_map_dict={'Adult Content':'Adult','Art & Design':'Art & Design','Crime & Law':'Crime & Law','Education & Jobs':'Education & Jobs','Electronics & Hardare':'Hardware','Entertainment':'Entertainment','Fashion & Beauty':'Fashion & Beauty','Finance & Business':'Finance & Business','Food & Dining':'Food & Dining','Games':'Games','Health':'Health','History & Geography':'History','Home & Hobbies':'Home & Hobbies','Industrial':'Industrial','Literature':'Literature','Politics':'Politics','Religion':'Religion','Science Math & Technology':'Science & Technology','Social Life':'Social Life','Software':'Software','Software Development':'Software Development','Sports & Fitness':'Sports & Fitness','Transportation':'Transportation','Travel & Tourism':'Travel'}
        elif args.taxonomy=="webOrganize_format24":
            test_paper_lables_map_dict={"Organizational About Page":"About (Org.)","Personal About Page":"About (Personal)","Academic Writing":"Academic Writing","Transcript / Interview":"Audio Transcript","Discussion Forum / Comment Section":"Comment Section","Content Listing":"Content Listing","Creative Writing":"Creative Writing","Customer Support Page":"Customer Support","Technical Writing":"Documentation","FAQs":"FAQ","Knowledge Article":"Knowledge Article","Legal Notices":"Legal Notices","Listicle":"Listicle","Organizational Announcement":"News (Org.)","News Article":"News Article","Nonfiction Writing":"Nonfiction Writing","Personal Blog":"Personal Blog","Product Page":"Product Page","Q&A Forum":"Q&A Forum","Spam / Ads":"Spam / Ads","Structured Data":"Structured Data","Incomplete Content":"Truncated","Tutorial / How-To Guide":"Tutorial","User Reviews":"User Review"}

        lables_df["trueLabelMapped"]=lables_df["true_label"].map(test_paper_lables_map_dict)
        for model_name in tqdm(["BAAI/bge-base-en-v1.5","BAAI/bge-m3","Alibaba-NLP/gte-multilingual-base","microsoft/harrier-oss-v1-270m","Alibaba-NLP/gte-large-en-v1.5","google/embeddinggemma-300m","paraphrase-multilingual-MiniLM-L12-v2"]):
              short_model_name=model_name.split('/')[-1]
              model_result_df=pd.read_csv(f"{base_path}/{args.dataset}_{short_model_name}_predictions.csv")
              lables_df[f"{short_model_name}_label"]=model_result_df["predicted_label"]
        if args.taxonomy=="webOrganize_format24":
            web_organize_df=pd.read_csv(f"{base_path}/{args.dataset}_FormatClassifier_predictions.csv")
            lables_df[f"FormatClassifier_label"]=web_organize_df["24_format_name"]
        elif args.taxonomy=="webOrganize_topic24":
            web_organize_df=pd.read_csv(f"{base_path}/{args.dataset}_TopicClassifier_predictions.csv")
            lables_df[f"TopicClassifier_label"]=web_organize_df["24_cat_name"]
        cols_lst=list(lables_df.columns)
        cols_lst=[elem for elem in cols_lst if "_label" in elem]
        for col in cols_lst:
            acc=lables_df.apply(lambda x: 1 if x["trueLabelMapped"]==x[col] else 0, axis=1).sum()/lables_df.shape[0]
            print(f"col={col}, acc={acc}")
        gte_df=lables_df[['url','true_label','trueLabelMapped','gte-large-en-v1.5_label']]
        gte_df['is_true']=gte_df.apply(lambda x: 1 if x["trueLabelMapped"]==x['gte-large-en-v1.5_label'] else 0, axis=1)
        incorrect_gte_df=gte_df[gte_df['is_true']==0]
              
    else:
        # for model_name in [args.emb_model]:
        # "DatologyAI/luxical-one" 
        # for model_name in tqdm(["BAAI/bge-m3","microsoft/harrier-oss-v1-270m","google/embeddinggemma-300m","Alibaba-NLP/gte-large-en-v1.5","Alibaba-NLP/gte-multilingual-base","paraphrase-multilingual-MiniLM-L12-v2"]):
        for model_name in tqdm(["BAAI/bge-base-en-v1.5"]):
            print(f"model_name={model_name}")
            model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
            model.eval()
            label_texts=[]
            if args.taxonomy=="webOrganize_topic24":
                catg24_df = pd.read_csv(f"{base_path}/24_web_categories.csv")
                label_names = catg24_df["category"].tolist()
                for row in catg24_df.itertuples():
                    label_texts.append(f"Topic Category: {row[1]}\n Topic Description:{row[3]}")
            elif args.taxonomy=="webOrganize_format24":
                catg24_df = pd.read_csv(f"{base_path}/24_web_formats.csv")
                label_names = catg24_df["Format"].tolist()
                for row in catg24_df.itertuples():
                        label_texts.append(f"Document Format: {row[1]}\n Format Description:{row[3]} \n Format Examples:{row[4]}")

            max_doc_size=10**4
            with torch.no_grad():
                # 2. Embed labels once
                label_embeddings = model.encode(label_texts, normalize_embeddings=True)
                documents=[]
                if args.dataset=="WebOrganizer-TopicAnnotations-Llama-3.1-405B-FP8":
                    parquet_file_path=f"{base_path}/{args.dataset}/test-00000-of-00001.parquet"
                    SQL_Query=f"SELECT * FROM read_parquet('{parquet_file_path}') limit {max_doc_size}"
                    content_df=query_parquet_duckdb(SQL_Query=SQL_Query)
                    for row in content_df.itertuples():
                        documents.append(f"Document URL: {row[2]}\n Document Content:{row[1]}")
                    urls = content_df["url"].tolist()
                    true_lables = content_df["top_choice"].apply(lambda x:x.split("\n")[0]).tolist()
                    true_labels_df=pd.DataFrame({"url": urls, "true_label": true_lables})
                    true_labels_df.to_csv(f"{base_path}/{args.dataset}_true_lables.csv", index=False)
                elif args.dataset=="WebOrganizer-FormatAnnotations-Llama-3.1-405B-FP8":
                    parquet_file_path=f"{base_path}/{args.dataset}/test-00000-of-00001.parquet"
                    SQL_Query=f"SELECT * FROM read_parquet('{parquet_file_path}') limit {max_doc_size}"
                    content_df=query_parquet_duckdb(SQL_Query=SQL_Query)
                    for row in content_df.itertuples():
                        documents.append(f"Document URL: {row[2]}\n Document Content:{row[1]}")
                    urls = content_df["url"].tolist()
                    true_lables = content_df["top_choice"].apply(lambda x:x.split("\n")[0]).tolist()
                    true_labels_df=pd.DataFrame({"url": urls, "true_label": true_lables})
                    true_labels_df.to_csv(f"{base_path}/{args.dataset}_true_lables.csv", index=False)            
                elif args.dataset=="dec2024_doc-content_samples_1k":
                    content_df = pd.read_csv(f"{base_path}/dec2024_doc-content_samples_1k.csv")            
                    content_df=content_df[0:max_doc_size]         
                    for row in content_df.itertuples():
                        documents.append(f"Document URL: {row[1]}\n Document Content:{row[2]}")
                    urls = content_df["WARC_Target_URI"].tolist()

                doc_embeddings = model.encode(
                    documents, normalize_embeddings=True, batch_size=1, show_progress_bar=True
                )

            # 4. Cosine similarity (dot product, since embeddings are normalized) -> argmax
            sims = doc_embeddings @ label_embeddings.T   # shape: [n_docs, n_labels]
            predicted_idx = sims.argmax(axis=1)
            predicted_labels = [label_names[i] for i in predicted_idx]
            confidences = sims.max(axis=1)
            res_df=pd.DataFrame({"url": urls, "emb": list(doc_embeddings)})    
            res_df.to_csv(f"{base_path}/{args.dataset}_{model_name.split('/')[-1]}_embeddings.csv", index=False)
            res_df=pd.DataFrame({"url": urls, "predicted_label": predicted_labels, "confidence": confidences})    
            res_df.to_csv(f"{base_path}/{args.dataset}_{model_name.split('/')[-1]}_predictions.csv", index=False)

            ########## clean model #######
            del model
            del doc_embeddings
            gc.collect()
            torch.cuda.empty_cache()

            