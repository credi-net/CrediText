import argparse
import pickle
import ast
from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import threading

from propella import (
    create_messages,
    AnnotationResponse,
    get_annotation_response_schema,
    ann_property_dict
)

base_data_path="/home/mila/a/abdallah/scratch/hsh_projects/CrediText/data"
# dataset_content_path = f"{base_data_path}/scrapedContent/dqr_active_domains_scraped_homepage_html/dqr_active_domains_scraped_homepage_resiliparse.csv"
# text_col="html"
# dataset_content_path = f"{base_data_path}/scrapedContent/weaklabels/weaklabel_domains_with_content.csv"
dataset_content_path = f"{base_data_path}/scrapedContent/weaklabels/weaklabel_domains_without_content.csv"
text_col="text"
ds_name=dataset_content_path.split('/')[-1].split('.')[0]
dataset_annotations_out_path = f"{base_data_path}/weaksupervision"
dataset_propella_annotations_dict = {}
def get_doc_annotation(client,url,content):
    # print(f"Annotating {url}...")
    # document=f"Web Domain URL: {url}\nHome Page Content: {content}"
    global dataset_propella_annotations_dict
    response = client.chat.completions.create(
        model="ellamind/propella-1-0.6b",
        messages=create_messages(content[0:4000]),
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "AnnotationResponse",
                "schema": get_annotation_response_schema(flatten=True, compact_whitespace=True),
                "strict": True,
            }
        },
    )
    response_content = response.choices[0].message.content
    # result = AnnotationResponse.model_validate_json(response_content)
    dataset_propella_annotations_dict[url]=ast.literal_eval(response_content)
     
def generate_dataset_annotation(dataset_content_path,dataset_annotations_out_path,content_col="resiliparse_text", batchsize=10,port=7070,NoContent=False):    
    global dataset_propella_annotations_dict
    print(f"Loading dataset from {dataset_content_path}...")
    dataset_df = pd.read_csv(dataset_content_path)
    client = OpenAI(base_url=f"http://0.0.0.0:{port}/v1", api_key="llama-cpp")   
    for i in tqdm(range(0,len(dataset_df),batchsize)):
        # if i<10000:
        #     continue
        print(f"Processing batch {i} to {i+batchsize}...")
        batch_df = dataset_df.iloc[i:i+batchsize]
        thread_lst=[]
        if NoContent:
            for row in batch_df.itertuples():            
                url = row.domain
                content= f"Web Domain URL: {url}\nHome Page Content: Domain is not active ."
                thread_lst.append(threading.Thread(target=get_doc_annotation, args=(client,url,content)))            
        else:
            for row in batch_df.itertuples():            
                url = row.url
                content= str(row.__getattribute__(content_col))
                thread_lst.append(threading.Thread(target=get_doc_annotation, args=(client,url,content)))            
        for thread in thread_lst:
            thread.start()
        for thread in thread_lst:
            thread.join()
        # print(result.model_dump_json(indent=4))
        if ((i+batchsize)%1000)==0 or (i+batchsize)>=len(dataset_df):
            with open(f"{dataset_annotations_out_path}/{ds_name}_propella_annotations_{content_col}_{i}.pkl", "wb") as f:
                pickle.dump(dataset_propella_annotations_dict, f)
            dataset_propella_annotations_dict={}

def load_annotations(dataset_annotations_out_path,content_col):
    with open(f"{dataset_annotations_out_path}/{ds_name}_propella_annotations_{content_col}.pkl", "rb") as f:
        dataset_propella_annotations_dict = pickle.load(f)
    return dataset_propella_annotations_dict
def encode_annotation(annotation_dict):
    features_dict={}
    for k,v in annotation_dict.items():
        doc_ann_features=[]
        for ann_k,ann_v in v.items():
            if ann_k not in ann_property_dict:
                continue
            property=ann_property_dict[ann_k]
            if type(ann_v) is str:
                doc_ann_features.append(list(property).index(property(ann_v)))
            else:
                num = 0
                if property==ann_property_dict["country_relevance"]:
                    num = len(ann_v)
                else:
                    for elem in ann_v:
                        if elem not in property:
                            continue
                        num |= (1 << list(property).index(property(elem)))
                doc_ann_features.append(num)
        features_dict[k]=doc_ann_features
    return features_dict
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLP Inference")
    parser.add_argument("--port", type=int, default=6060,help="API port")
    parser.add_argument("--batchsize", type=int, default=100,help="Batch size for annotation")
    args = parser.parse_args()
    generate_dataset_annotation(dataset_content_path,dataset_annotations_out_path,text_col, batchsize=args.batchsize,port=args.port,NoContent=True)
    dqr_propella_annotations_dict = load_annotations(dataset_annotations_out_path,text_col)
    feat_dict=encode_annotation(dqr_propella_annotations_dict)    
    with open(f"{dataset_annotations_out_path}/dqr_propella_annotations_features.pkl", "wb") as f:
        with open(f"{dataset_annotations_out_path}/{ds_name}_propella_annotations_{text_col}_features.pkl", "wb") as f:
            pickle.dump(feat_dict, f)
#    idx=list(ContentIntegrity).index(ContentIntegrity.COMPLETE)
#33500