from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import normalize 
from sympy import fu
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean
from sklearn.model_selection import KFold
from typing import Dict, cast
import pickle
import pyarrow.parquet as pq
import duckdb
import pyarrow as pa
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import os
import glob
import re
import logging
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import shap
from typing import Any
import torch
from itertools import zip_longest
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score as Recall,roc_auc_score as AUROC,average_precision_score as AUPRC
from pathlib import Path

def list_all_files(root_path: str, regex: str = "*.pkl", recursive: bool = False):
    if recursive:
        return [str(p) for p in Path(root_path).rglob(regex)]
    else: 
        return glob.glob(f'{root_path}/{regex}', recursive=recursive)

def normalize_embeddings(emb_dict: Dict[Any, list], norm_type: str = "min-max"):
    X=list(emb_dict.values())
    if norm_type == "l2":
        norm_arr=normalize(X, norm=norm_type)
    elif norm_type == "min-max":
        scaler = MinMaxScaler()
        norm_arr = scaler.fit_transform(X)
    elif norm_type == "standard":
        scaler = StandardScaler()
        norm_arr = scaler.fit_transform(X)

    return {k: norm_arr[idx] for idx,k in enumerate(emb_dict.keys())}


def kfold_split(X: pd.DataFrame, y: pd.Series, n_splits: int = 5):
    kf = KFold(n_splits = n_splits, shuffle = True, random_state = 42)
    splits = []
    for train_index, test_index in kf.split(X):
        X_tr_va, X_test = X.iloc[train_index], X.iloc[test_index]
        y_tr_va, y_test = y[train_index], y[test_index]
        X_train, X_val, y_train, y_val = train_test_split(X_tr_va, y_tr_va, test_size=0.25)
        splits.append((X_train, y_train,X_val, y_val, X_test, y_test))
    return splits
def train_valid_test_split(target: str, labeled_df: pd.DataFrame, key: str = 'domain', test_valid_size: float = 0.4, regressor: bool = True, train_lst: list = None, valid_lst: list = None, test_lst: list = None):
    X = labeled_df[[key, target]]
    if train_lst is not None and valid_lst is not None and test_lst is not None:
        X_train = X[X[key].isin(train_lst)]
        X_valid = X[X[key].isin(valid_lst)]
        X_test = X[X[key].isin(test_lst)]
        return X_train,X_train[target].tolist(), X_valid,X_valid[target].tolist(), X_test,X_test[target].tolist()
    else:
        if regressor:
            if target == "mbfc_bias":
                quantiles = labeled_df[target].quantile([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            else:
                quantiles = labeled_df[target].quantile([0.2, 0.4, 0.6, 0.8, 1.0])
            bins = [labeled_df[target].min()] + quantiles.tolist()
            labeled_df[target + '_cat'] = pd.cut(labeled_df[target], bins=bins, labels=quantiles, include_lowest=True)
            y = labeled_df[target + '_cat']
        else:
            # bins = np.linspace(0, 0.9, 10)
            # y = np.digitize(labeled_11k_df[target].tolist(), bins) 
            y=labeled_df[target].tolist()
            
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_valid_size, stratify=y, random_state=42
        )
        X_valid, X_test, y_valid, y_test = train_test_split(
            X_test, y_test, test_size=0.5, stratify=y_test, random_state=42)
        if regressor:
            return X_train, labeled_df.iloc[X_train.index][target].tolist(), X_valid, labeled_df.iloc[X_valid.index][
                target].tolist(), X_test, labeled_df.iloc[X_test.index][target].tolist()
        else:
            return X_train, np.array(y_train), X_valid,np.array(y_valid) , X_test,np.array(y_test)

def resize_and_fuse_emb(text_emb: Dict[Any, list], target: str, X_train: pd.DataFrame, X_valid: pd.DataFrame, X_test: pd.DataFrame, gnn_emb: Dict[Any, list] = None, topic_emb: Dict[Any, list] = None, trim_to: int = 1024, fusion_mode:str="cat"):
        train_lst,val_lst,test_lst=[],[],[]
        X_train_feat = [text_emb[d][0:trim_to] for d in X_train["domain"].tolist()]
        X_valid_feat = [text_emb[d][0:trim_to] for d in X_valid["domain"].tolist()]
        X_test_feat = [text_emb[d][0:trim_to] for d in X_test["domain"].tolist()]
        train_lst.append(X_train_feat)
        val_lst.append(X_valid_feat)
        test_lst.append(X_test_feat)
        logging.info(f"text emb-size={len(X_test_feat[0])}")
        if gnn_emb is not None:
            X_train_feat_gnn = [gnn_emb[d][0:256] for d in X_train["domain"].tolist()]
            X_valid_feat_gnn = [gnn_emb[d][0:256] for d in X_valid["domain"].tolist()]
            X_test_feat_gnn = [gnn_emb[d][0:256] for d in X_test["domain"].tolist()]
            train_lst.append(X_train_feat_gnn)
            val_lst.append(X_valid_feat_gnn)
            test_lst.append(X_test_feat_gnn)
            logging.info(f"gnn emb-size={len(X_train_feat_gnn[0])}")
        if topic_emb is not None:
            X_train_feat_topic = [topic_emb[d] if d in topic_emb else [0] * len(list(topic_emb.values())[0]) for d in X_train["domain"].tolist()]          
            X_valid_feat_topic = [topic_emb[d] if d in topic_emb else [0] * len(list(topic_emb.values())[0]) for d in X_valid["domain"].tolist()]    
            X_test_feat_topic = [topic_emb[d] if d in topic_emb else [0] * len(list(topic_emb.values())[0]) for d in X_test["domain"].tolist()]   
            train_lst.append(X_train_feat_topic)
            val_lst.append(X_valid_feat_topic)
            test_lst.append(X_test_feat_topic)
            logging.info(f"topic emb-size={len(X_train_feat_topic[0])}")        
        for i in range(1,len(train_lst)):
                train_lst[0] =fuse_2d_emb(train_lst[0], train_lst[i], fusion_mode=fusion_mode)
                val_lst[0] = fuse_2d_emb(val_lst[0], val_lst[i], fusion_mode=fusion_mode)
                test_lst[0] = fuse_2d_emb(test_lst[0], test_lst[i], fusion_mode=fusion_mode)  
        logging.info(f"final fused emb-size={len(train_lst[0][0])}")
        return train_lst[0], val_lst[0], test_lst[0]

def fuse_2d_emb(emb_lst1:list[list],emb_lst2:list[list], fusion_mode:str="cat"):    
    fused_emb = [fuse_1d_emb(row1, row2, fusion_mode=fusion_mode) for row1, row2 in zip_longest(emb_lst1, emb_lst2, fillvalue=0)]
    return fused_emb
def fuse_1d_emb(row1:list,row2:list, fusion_mode:str="cat"):
    if fusion_mode == "cat":
        fused_emb = [*row1, *row2]
    elif fusion_mode == "sum":
        fused_emb = [a + b for a, b in zip_longest(row1, row2, fillvalue=0)]
    elif fusion_mode == "min":
        fused_emb = [min(a, b) for a, b in zip_longest(row1, row2, fillvalue=0)]
    elif fusion_mode == "max":
        fused_emb = [max(a, b) for a, b in zip_longest(row1, row2, fillvalue=0)]
    elif fusion_mode == "avg":
        fused_emb = [(a + b) / 2 for a, b in zip_longest(row1, row2, fillvalue=0)]
    elif fusion_mode == "mul":
        fused_emb = [a * b for a, b in zip_longest(row1, row2, fillvalue=1)]
    return fused_emb

def plot_loss(train_loss:list[float], valid_loss:list[float], test_loss:list[float], mean_loss:list[float], out_file_path:str ="loss_plot.pdf",ylabel:str="MSE"):
    plt.figure(figsize=(5, 4))
    plt.rc('font', size=16)
    plt.plot(range(len(train_loss)), train_loss, label="train loss")
    plt.plot(range(len(train_loss)), valid_loss, label="validation loss")
    plt.plot(range(len(train_loss)), test_loss, label="test loss")
    if mean_loss is not None and len(mean_loss)>0:
        plt.plot(range(len(train_loss)), mean_loss, label="mean loss")
    plt.xticks(range(0, len(train_loss) + 1, 1 if len(train_loss) <= 10 else len(train_loss) // 10))
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend(fontsize=14)
    plt.savefig(out_file_path,bbox_inches='tight', pad_inches=0.1)
    plt.show()


def plot_histogram(true: list[float], pred: list[float], out_file_path: str = "_testset_true_vs_pred_frequency.pdf", target: str = "pc1"):
    plt.figure(figsize=(5, 4))
    plt.hist(pred, bins=50, range=(0, 1), edgecolor='black', color='lightblue', label="Pred")
    plt.hist(true, bins=50, range=(0, 1), edgecolor='black', color='orange', alpha=0.6, label="True")
    y_max = max(np.histogram(true, bins=50, range=(0, 1))[0].max(),
                np.histogram(pred, bins=50, range=(0, 1))[0].max())
    plt.rc('font', size=15)
    plt.xticks(np.arange(0, 1.1, 0.2), rotation=0, ha='right')
    plt.yticks(np.arange(0, y_max + 50, 100), rotation=0, ha='right')
    plt.xlabel(target.upper().replace("_", "-"))
    plt.ylabel('frequency')
    plt.legend()
    plt.savefig(out_file_path,bbox_inches='tight', pad_inches=0.1)
    plt.show()

def plot_classesCount(cm: np.ndarray, out_file_path:str ="_testset_true_vs_pred_frequency.pdf",target:str ="pc1"):
    plt.figure(figsize=(5, 4))
    per_class_acc = np.diag(cm) / cm.sum(axis=1)
    per_class_acc = np.nan_to_num(per_class_acc)
    classes = np.arange(len(per_class_acc))
    plt.bar(classes, per_class_acc)
    plt.xlabel("Class Label")
    plt.ylabel("Accuracy")
    plt.title("Accuracy per Label")
    plt.ylim(0, 1)
    plt.xticks(classes, classes)
    for i, v in enumerate(per_class_acc):
        plt.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_file_path,bbox_inches='tight', pad_inches=0.1)
    plt.show()
def plot_confusion_matrix(cm: np.ndarray, out_file_path: str = "_testset_confusion_matrix.pdf"):
    fig_size=max(3, len(cm))
    plt.figure(figsize=(fig_size, fig_size))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[f'T:{idx}' for idx in range(len(cm))],
                yticklabels=[f'P:{idx}' for idx in range(len(cm))])
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(out_file_path,bbox_inches='tight', pad_inches=0.1)
    plt.show()



def plot_regression_scatter(true: list[float], pred: list[float], out_file_path: str = "_testset_true_vs_pred_scatter.pdf", target: str = "pc1"):
    plt.figure(figsize=(5, 4))
    plt.rc('font', size=16)
    plt.scatter(true, pred, alpha=0.7)
    plt.plot([0, 1], [0, 1], color='red', linestyle='-', label='regression line')
    plt.xticks(np.arange(0, 1.1, 0.2), rotation=0, ha='right')
    plt.yticks(np.arange(0, 1.1, 0.2), rotation=0, ha='right')
    plt.xlabel(target.upper().replace("_", "-"))
    plt.ylabel(f"Predicted {target.upper().replace('_', '-')}")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_file_path,bbox_inches='tight', pad_inches=0.1)
    plt.show()


def eval(pred: list[float], true: list[float]):
    max(abs(x - y) for x, y in zip(true, pred))
    res_df = pd.DataFrame(zip(true, pred), columns=['true', 'pred'])
    res_df["diff"] = res_df.apply(lambda row: abs(row['true'] - row['pred']), axis=1)
    max_idx = res_df.idxmax()["diff"]
    min_idx = res_df.idxmin()["diff"]
    max_diff_row = res_df.iloc[max_idx]
    min_diff_row = res_df.iloc[min_idx]
    max_diff_dict = max_diff_row.to_dict()
    max_diff_dict["test_idx"] = max_idx
    min_diff_dict = min_diff_row.to_dict()
    min_diff_dict["test_idx"] = min_idx

    mse = mean_squared_error(true, pred)
    # logging.info(f"mse={mse}")
    r2 = r2_score(true, pred)
    # logging.info(f"r2={r2}")
    mae = mean_absolute_error(true, pred)
    true_mean = mean(true)
    mean_mae = mean_absolute_error(true, [true_mean for elem in true])
    # logging.info(f"MAE={mae}")
    return mse, mae, r2, mean_mae, min_diff_dict, max_diff_dict


def load_emb_index(path:str="../../../data/Dec2024/gnn_random_v0", index_pickle:str="domain_shard_index.pkl",invert:str =True ):
    with open(f'{path}/{index_pickle}', 'rb') as f:
            index_dict = pickle.load(f)
    # global shard_domains_dict
    shard_domains_dict=None
    if invert:
        shard_domains_dict={}
        val_set=set(index_dict.values())
        for v in val_set:
            shard_domains_dict[v]=[]

        for k,v in index_dict.items():
            shard_domains_dict[v].append(k)
    return index_dict,shard_domains_dict



def search_parquet_emb(path:str="../../../data/Dec2024/gnn_random_v0", parquet_name:str="shard_0.parquet", column_name:str="",search_values:list=[],schema:dict={'key':'domain','val':'emb'}):
    column_name = 'domain'
    if not search_values or len(search_values)==0:
        table = pq.read_table(f'{path}/{parquet_name}', filters=None)
    else:
        filters = [(column_name, 'in', search_values)]
        table = pq.read_table(f'{path}/{parquet_name}', filters=filters)

    emb_dict={}
    for batch in table.to_batches():
        for i in range(len(batch)):
            emb_dict[str(batch[schema['key']][i])]=batch[schema['val']][i].as_py()
    return emb_dict
def search_parquet_content(path:str="../../../data/Dec2024/gnn_random_v0", parquet_name:str="shard_0.parquet", column_name:str="",search_values:list=[]):
    column_name = 'domain'
    filters = [(column_name, 'in', search_values)]
    table = pq.read_table(f'{path}/{parquet_name}', filters=filters)
    emb_dict={}
    for batch in table.to_batches():
        for i in range(len(batch)):
            emb_dict[str(batch['domain'][i])]=batch['embeddings'][i][0]['emb'].as_py() ## first doc emb
    return emb_dict

def search_parquet_duckdb(f_path:str,q_domains:list,filter_by_col:str=None,projection_cols:list[str]=None,max_memory:str="4GB",threads:int =8,batch_size:int =int(1e4),schema:dict={'key':'domain','val':'emb'},keep_emb_frist_elem:int =None):
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{max_memory}'")
    con.execute(f"SET threads={threads}")
    if  not q_domains or len(q_domains)==0:
        query=f"SELECT {'*' if projection_cols is None else ",".join(projection_cols)} FROM read_parquet('{f_path}')"
        result = con.execute(query)
    else:
        query=f"SELECT {'*' if projection_cols is None else ",".join(projection_cols)}  FROM read_parquet('{f_path}') WHERE {filter_by_col} IN ?"
        result = con.execute(query, [list(q_domains)])    

    cols_map={name[0]:idx for idx,name in enumerate(result.description)}
    if schema is None:
        res=[]
        while True:
            rows = result.fetchmany(int(batch_size))
            if not rows:
                break
            for row in rows:
                res.append([elem for elem in row])
        res=pd.DataFrame(res,columns=cols_map.keys())
    else:
        res={}
        while True:
            rows = result.fetchmany(int(batch_size))
            if not rows:
                break
            for row in rows:
                if keep_emb_frist_elem is not None:
                    res[row[cols_map[schema['key']]]]=row[cols_map[schema['val']]][0][keep_emb_frist_elem] ## first doc emb
                else:
                    res[row[cols_map[schema['key']]]]=row[cols_map[schema['val']]]
    return res
def query_parquet_duckdb(SQL_Query:str,max_memory:str="4GB",threads:int =8,batch_size:int =int(1e4)):
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{max_memory}'")
    con.execute(f"SET threads={threads}")
    try:
        result = con.execute(SQL_Query)
    except Exception as e:
        logging.error(f"Error occurred while executing SQL query:{e}\nSQL_Query={SQL_Query}")
        raise

    cols_map={name[0]:idx for idx,name in enumerate(result.description)}
    res=[]
    while True:
        rows = result.fetchmany(int(batch_size))
        if not rows:
            break
        for row in rows:
            res.append([elem for elem in row])
    res=pd.DataFrame(res,columns=cols_map.keys())
    return res
def write_domain_emb_parquet(rows: dict, directory_path: str, file_name: str):
    '''rows format: {'domain':domains_lst, 'embeddings':[ [{"page":url,"emb":[float]}] ] }'''
    schema = pa.schema([
        ("domain", pa.string()),
        ("embeddings", pa.list_( pa.struct([
                ("page", pa.string()),
                ("emb", pa.list_(pa.float32()))
            ]) ))])    
    table = pa.Table.from_pydict(rows, schema=schema)
    table = table.sort_by("domain")
    pq.write_table(table, f"{directory_path}/{file_name}",row_group_size=100,use_dictionary=["domain"])

def write_domain_topics_parquet(rows: dict, directory_path: str, file_name: str):
    '''rows format: {'domain':domains_lst, 'topics':[ [{"topic0_id":2,"topic0_name":"",topic0_prop:0.5, ..}'''
    schema = pa.schema([
        ("domain", pa.string()),
        ("topics", pa.list_( pa.struct([
                ("topic0_id", pa.int32()),
                ("topic0_name", pa.string()),
                ("topic0_prop", pa.float32()),
                ("topic1_id", pa.int32()),
                ("topic1_name", pa.string()),
                ("topic1_prop", pa.float32()),
                ("topic2_id", pa.int32()),
                ("topic2_name", pa.string()),
                ("topic2_prop", pa.float32()),
                ("url", pa.string())                
            ]) ))])  
    table = pa.Table.from_pydict(rows, schema=schema)
    table = table.sort_by("domain")
    pq.write_table(table, f"{directory_path}/{file_name}",row_group_size=100,use_dictionary=["domain"])

def save_shaply_plots(model:object, X_train: list[list[Any]], X_test: list[list[Any]], out_file_path:str ="shaply",model_type:str="regressor"):
    # Calculate SHAP values using the KernelExplainer (logistic regression is linear)
    background_data = shap.utils.sample(np.array(X_train), 200)
    # background_data = np.array(X_train)
    if model_type=="classifier":
        explainer = shap.Explainer(model.predict, background_data)
    elif model_type=="regressor":
        explainer = shap.KernelExplainer(model.predict,background_data)

    test_data=shap.utils.sample(np.array(X_test),100)
    shap_values = explainer.shap_values(test_data,silent=True)
    # Step 6: Visualize SHAP results
    # labels=["Has_WWW","DNS_A_Record","DNS_AAAA_Record","DNS_MX_Record","DNS_TXT_Record","DNS_CNAME_Record","DNS_CNAME_Resolution","Certificate_Valid","Status_Code_OK","Final_Protocol_HTTPS","HTTP_to_HTTPS_Redirect","High_Redirects","HSTS_Present","Has_Suspicious_Keywords","SSL_Verification_Failed","Is_Risky_TLD","Domain_Length","Num_Hyphens","Num_Digits","Contains_IP_Address","URL_Shortener","Subdomain_Count","Title_Length","Body_Length","A","AAAA","MX","TXT","CNAME"]
    labels=["Has_WWW","DNS_A_Record","DNS_AAAA_Record","DNS_MX_Record","DNS_TXT_Record","DNS_CNAME_Record","DNS_CNAME_Resolution","Certificate_Valid","Status_Code_OK","Final_Protocol_HTTPS","HTTP_to_HTTPS_Redirect","High_Redirects","HSTS_Present","Has_Suspicious_Keywords","SSL_Verification_Failed","Is_Risky_TLD","Domain_Length","Num_Hyphens","Num_Digits","Contains_IP_Address","URL_Shortener","Subdomain_Count","Title_Length","Body_Length","WHOIS_Creation_Date","WHOIS_Expiration_Date","WHOIS_Updated_Date","WHOIS_Age","WHOIS_Info_Available","A","AAAA","MX","TXT","CNAME"]
    shap.summary_plot(shap_values, test_data,show=False,feature_names=labels,max_display=len(labels))  # For the positive class
    plt.savefig(f"{out_file_path}_shap_summary_plot.pdf", format="pdf", bbox_inches='tight')
    plt.close()

    # shap.dependence_plot('feature_0', shap_values[1], test_data,show=False)
    # plt.savefig(f"{out_file_path}_shap_dependence_plot.pdf", format="pdf", bbox_inches='tight')
    # plt.close()

def eval_binary_classification(pred: np.ndarray[np.int_], true: np.ndarray[np.int_])-> dict:
    
    accuracy = accuracy_score(true, pred)   
    logging.info(f"Accuracy: {accuracy:.4f}")

    f1 = f1_score(true,pred, average='macro')    
    logging.info(f"F1 score (macro): {f1:.4f}")

    recall = Recall(true, pred)    
    logging.info(f"Recall: {recall:.4f}")

    auc_roc = AUROC(true, pred)    
    logging.info(f"AUROC: {auc_roc:.4f}")

    auc_pr = AUPRC(true, pred)   
    logging.info(f"AUPRC: {auc_pr:.4f}")

    cm = confusion_matrix(true, pred)
    logging.info(f"cm: {cm}")  

    return {"accuracy": accuracy, "f1_score": f1, "recall": recall, "auc_roc": auc_roc, "auc_pr": auc_pr, "cm": cm} 


def compute_ece(smx: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error (ECE).

    Measures how well predicted probabilities match actual correctness.
    Bins samples by confidence (max softmax prob), computes
    |accuracy - confidence| per bin, weighted by bin size.

    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|

    Lower is better. 0 = perfectly calibrated.
    """
    confidences = smx.max(axis=1)
    predictions = smx.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (
            confidences <= bin_boundaries[i + 1]
        )
        if mask.sum() == 0:
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / len(labels)) * abs(bin_acc - bin_conf)
    return ece


def compute_nll(smx: np.ndarray, labels: np.ndarray) -> float:
    """Negative Log-Likelihood (NLL).

    NLL = -mean(log(p(y_true)))

    Measures quality of predicted probability for the true class.
    Lower is better. Equivalent to cross-entropy on test set.
    """
    probs_clipped = np.clip(smx, 1e-7, 1.0)
    return float(-np.log(probs_clipped[np.arange(len(labels)), labels]).mean())
def reverse_domain_name(domain_name: str):
    return  '.'.join(str(domain_name).split('.')[::-1])