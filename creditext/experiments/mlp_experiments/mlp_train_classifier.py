import argparse
from datetime import datetime
import pickle
import numpy as np
from creditext.utils.path import get_root_dir
import torch
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier as  Sklearn_MLPClassifier
from mlp_modules import MLPRegressor, train_classifier_unbalanced,train_classifier_unbalanced_halo
from mlp_modules import MLP3LayersPredictor
from mlp_modules import MultiTaskMLP,train_scikitlearn_classifier,LabelPredictor,train_classifier_unbalanced
from sklearn.preprocessing import normalize 
from utils import train_valid_test_split,resize_and_fuse_emb,\
                  plot_loss,plot_regression_scatter,eval,\
                  plot_classesCount,plot_confusion_matrix,\
                  save_shaply_plots,eval_binary_classification,\
                  compute_ece,compute_nll


from dataset_loader import DomainRel
import logging
import copy
from creditext.utils.logger import setup_logging

def write_testset_emb(run_file_name,X_test,X_test_feat):
    test_set_emb_dict=dict(zip(X_test["domain"].tolist(),X_test_feat))
    test_emb_file_name=f"{run_file_name}_test_set_emb_dict.pkl"
    with open(test_emb_file_name, 'wb') as file:
        pickle.dump(test_set_emb_dict, file)
def mlp_classifier(args) -> None:
    now = datetime.now()
    iso_compact = now.strftime("%Y%m%dT%H%M%S")
    run_file_name=f"DomainRel_{args.month}_{args.domainRel_target}_{args.library}_{args.embed_type}_{args.emb_model}_{f'GAT-{args.gnn_encoder}' if args.use_gnn_emb else ''}{'-'+args.agg_function if args.agg_month_emb else ''}{'_topic-emb' if args.use_topic_emb else ''}_{args.split_mode}_{args.test_mode}_{args.fusion_mode}_{args.keep_content}-{args.keep_content_count}_{iso_compact}"
    setup_logging(f"{args.logs_out_path}/{run_file_name}.log")          
    logging.info(f"args={args}")  
    run_file_name=f"{args.plots_out_path}/{run_file_name}"
    ############################
    X_train, y_train, X_valid, y_valid, X_test, y_test,X_train_feat, X_valid_feat, X_test_feat=DomainRel.load_run_embeddings(args)
    ############## Save Test set Embeddings dict pickle #############
    write_testset_emb(run_file_name,X_test,X_test_feat)
    ################# Train #####################
    results_dict={"accuracy": [], "f1_score": [], "recall": [], "auc_roc": [], "auc_pr": [],'cm':[], "ece": [], "nll": []}
    for i in range(args.runs):        
        logging.info(f"#################### Run {i} ##################")    
        run_file_name+=f"_run{str(i)}"   
        ###################### PYTorch Regressor  ######################    
        logging.info(f" hidden_dim_multiplier={args.hidden_dim_multipler}")
        if args.library=="pytorch":
            mlp_clf = LabelPredictor(len(X_train_feat[0]),hidden_dim_multiplier=args.hidden_dim_multipler, out_dim=2)
            logging.info(f"MLP Classifier Architecture: {mlp_clf}")
            if args.loss_fun=="nl_loss":
                mlp_clf, train_loss, valid_loss, test_loss, mean_loss = train_classifier_unbalanced(mlp_clf, X_train_feat, y_train,
                                                                                X_valid_feat, y_valid, X_test_feat,
                                                                                y_test, epochs=args.epochs)
            elif args.loss_fun=="halo":
                mlp_clf, train_loss, valid_loss, test_loss,mean_loss, (halo_model,gamma,abstain_bias)= train_classifier_unbalanced_halo(mlp_clf, X_train_feat, y_train,
                                                                                    X_valid_feat, y_valid, X_test_feat,
                                                                                    y_test, epochs=args.epochs)
        ######################## Scikit-Learn ###################
        elif args.library=="sklearn":
            mlp_clf = Sklearn_MLPClassifier(hidden_layer_sizes=(128, 32),
                                activation='relu', solver='adam',max_iter=args.max_iter, random_state=42,
                                verbose=False, learning_rate_init=args.lr,warm_start=True)
            mlp_clf.out_activation_ = 'sigmoid'
            mlp_clf, train_loss, valid_loss, test_loss, mean_loss = train_scikitlearn_classifier(mlp_clf, X_train_feat, y_train,
                                                                            X_valid_feat, y_valid, X_test_feat,
                                                                            y_test, epochs=args.epochs)
                                                                              
       
        ###################### Save Model ####################
        with open(f"{run_file_name}_credibench_MLP_Model.pkl", 'wb') as file:
            pickle.dump(mlp_clf, file)        
        ############# plot Shaply ################
        if args.embed_type=="FQDN":
            save_shaply_plots(mlp_clf, X_train_feat, X_test_feat, out_file_path=run_file_name, model_type="classifier")
        ################## Pred ###############        
        true = np.array(y_test)
        X_test_feat=torch.tensor(X_test_feat)
        if args.loss_fun=="nl_loss":
            mlp_clf.eval()
            pred =np.array(mlp_clf.predict(X_test_feat))
            logits =mlp_clf(X_test_feat).detach()
            probs=np.array(logits.softmax(dim=1))
        elif args.loss_fun=="halo":
            halo_model.eval()
            pos, centroids = halo_model(X_test_feat.float())
            pos = pos.to(torch.float32)
            centroids = centroids.to(torch.float32)
            x_sq = pos.pow(2).mean(dim=-1, keepdim=True)
            y_sq = centroids.pow(2).mean(dim=-1, keepdim=True)
            # Native dot product, then scaled by D
            dot_product = (pos @ centroids.T) / pos.size(-1)
            r_sq = x_sq + y_sq.T - 2.0 * dot_product
            r_sq = torch.clamp(r_sq, min=0.0)
            logits_k = -(r_sq * gamma)
            logit_abstain = -(x_sq * gamma) + abstain_bias
            logits_k_plus_1 = torch.cat([logits_k, logit_abstain], dim=-1)
            pred = 1 - torch.nn.functional.softmax(logits_k_plus_1, dim=-1)[:, -1]
            pred=pred.detach()
        ################## Eval ###############        
        clf_res_dict =eval_binary_classification(true, pred)
        for k in clf_res_dict.keys():
            results_dict[k].append(clf_res_dict[k])
        ################## Compute Calibration Metrics ###############
        results_dict["ece"].append(compute_ece(probs,true))
        logging.info(f"ECE={results_dict['ece'][-1]}")
        results_dict["nll"].append(compute_nll(probs,true))
        logging.info(f"NLL={results_dict['nll'][-1]}")
        ############# Plot loss, confusion matrix, class frequancy, and true vs pred scatter ###############
        plot_loss(train_loss, valid_loss, test_loss, mean_loss, run_file_name+"_loss.pdf",ylabel="CrossEntropy Loss")
        plot_classesCount(results_dict['cm'][-1],run_file_name+"_class_frequancy.pdf")
        plot_regression_scatter(true, pred, run_file_name+"_testset_true_vs_pred_scatter.pdf")
        plot_confusion_matrix(results_dict['cm'][-1], run_file_name+"_testset_confusion_matrix.pdf")
        ###############################################################
        if args.month and args.month not in [args.transferability_test_month]:
            trans_args=copy.deepcopy(args)
            trans_args.month=args.transferability_test_month
            _, _, _, _, trans_X_test, trans_y_test,_, _, trans_X_test_feat=DomainRel.load_run_embeddings(trans_args)
            trans_true = np.array(trans_y_test)
            trans_X_test_feat=torch.tensor(trans_X_test_feat)
            if args.loss_fun=="nl_loss":
                mlp_clf.eval()
                trans_pred =np.array(mlp_clf.predict(trans_X_test_feat))
                trans_logits =mlp_clf(trans_X_test_feat).detach()
                trans_probs=np.array(trans_logits.softmax(dim=1))
                ################## Eval ###############        
                trans_clf_res_dict =eval_binary_classification(trans_true, trans_pred)
                for k in clf_res_dict.keys():
                    logging.info(f"transferability from ({args.month}) to ({args.transferability_test_month})-> {k}= {trans_clf_res_dict[k]}")
                ################## Compute Calibration Metrics ###############
                results_dict["ece"].append(compute_ece(trans_probs,trans_true))
                logging.info(f"transferability from ({args.month}) to ({args.transferability_test_month})-> ECE={results_dict['ece'][-1]}")
                results_dict["nll"].append(compute_nll(trans_probs,trans_true))
                logging.info(f"transferability from ({args.month}) to ({args.transferability_test_month})-> NLL={results_dict['nll'][-1]}")



    for k in results_dict.keys():
        logging.info(f"{k.upper()} MAEN={np.mean(results_dict[k])} STD={np.std(results_dict[k])}")
    results_val_lst=[v for k,v in results_dict.items()]
    results_val_lst.append([args]*(len(results_dict['accuracy'])))
    results_label_lst=[k for k,v in results_dict.items()]
    results_label_lst.append('args')
    results_df = pd.DataFrame(list(zip(*results_val_lst)),columns=results_label_lst)
    results_df.to_csv(f"{run_file_name}_dqr_results.csv",index=None)

if __name__ == '__main__':
    root = str(get_root_dir())
    parser = argparse.ArgumentParser(description="MLP Experiments")
    parser.add_argument("--domainRel_target", type=str, default="weak_label", choices=["weak_label"], help="the credability target")
    parser.add_argument("--domainRel_text_emb_path", type=str, default=str(root + "/data/weaksupervision") ,help="emb files path")
    parser.add_argument("--domainRel_gnn_emb_path", type=str, default=str(root + "/data/weaksupervision") ,help="emb files path")
    parser.add_argument("--domainRel_path", type=str, default=str(root + "/data/weaksupervision"),help="dqr dataset path")
    parser.add_argument("--embed_type", type=str, default="text",
                        choices=["text", "domainName", "GNN_GAT", "TFIDF", "PASTEL","FQDN"], help="domains embedding technique")
    parser.add_argument("--emb_model", type=str, default="embeddinggemma-300m",
                        choices=["Qwen3-Embedding-8B", "Qwen3-Embedding-0.6B", "embeddinggemma-300m", "TE3L","Qwen3-Embedding-8B-Q5_K_M",
                                 "IPTC_Topic_emb","RoBERTa"],help="LLM embedding model")
    parser.add_argument("--batch_size", type=int, default=5000,help="training batch size")
    parser.add_argument("--test_valid_size", type=float, default=0.4,help="ratio of test and vaild sets")
    parser.add_argument("--emb_dim", type=int, default=256,help="embedding size")
    parser.add_argument("--hidden_dim_multipler", type=float, default=0.5,help="hidden_dimision_size= input_dimision_size*hidden_dim_multipler")
    parser.add_argument("--original_emb_dim", type=int, default=256,help="The original embedding model dim size")
    parser.add_argument("--max_iter", type=int, default=200,help="MLP regressor max iteration count")
    parser.add_argument("--lr", type=float, default=1e-1,help="learning rate")
    # parser.add_argument("--lr", type=float, default=5e-1,help="learning rate")
    parser.add_argument("--epochs", type=int,default=500, help="# training epochs") 
    parser.add_argument("--plots_out_path", type=str, default=str(root + "/plots"),help="plots and results store path")
    parser.add_argument("--logs_out_path", type=str, default=str(root + "/logs"),help="logging path")
    parser.add_argument("--runs", type=int, default=1,help="# training runs")
    parser.add_argument("--use_gnn_emb", action='store_true',help="append GNN embedding")
    parser.add_argument("--gnn_encoder",   default="text", choices=["RNI","text"],help="append GNN node embedding intialization")
    parser.add_argument("--agg_month_emb",  action='store_true', help="aggregate montly GNN embeddings")
    parser.add_argument("--agg_text_emb", action='store_true', help="aggregate montly text embeddings")
    parser.add_argument("--agg_function", type=str, default="cat",choices=["avg","cat", "sum", "min", "max"], help="aggregate function")
    parser.add_argument("--use_topic_emb", action='store_true',help="use topic modeling features")
    parser.add_argument("--filter_by_GNN_nodes", action='store_false',help="filter by domains has GNN embeddings")
    parser.add_argument("--num_classes", type=int, default=2,help="# classifcation classes for Multihead model")
    parser.add_argument("--use_FQDN", action='store_true',help="use fqdn_features")
    parser.add_argument("--generate_weaksupervision_scores", action='store_true', help="generate weak supervision datasets scores")
    parser.add_argument("--month", type=str, default="nov", choices=["oct", "nov", "dec"],help="CrediBench month snapshot")
    parser.add_argument("--library", type=str, default="pytorch", choices=["pytorch", "sklearn"],help="ML library to use")
    parser.add_argument("--split_mode", type=str, default="balanced", choices=["balanced", "phishing","malware","misinfo","general"],help="split mode for the dataset")
    parser.add_argument("--test_mode", type=str, default="credible-non", choices=["credible-non", "sub-category"],help="split mode for the dataset")
    parser.add_argument("--fusion_mode", type=str, default="cat", choices=["avg","cat", "sum", "min", "max","mul","gated"],help="embedding fusion method")
    parser.add_argument("--keep_content", type=str, default="all", choices=["all","longest", "shortest"],help="which content to keep for fusion")
    parser.add_argument("--keep_content_count", type=int, default=4, help="number of pages content to keep for fusion")
    parser.add_argument("--loss_fun", type=str, default="nl_loss", choices=["nl_loss", "halo"],help="the loss function to use for training the MLP regressor")
    parser.add_argument("--transferability_test_month", type=str, default="oct", choices=["oct", "nov","dec"],help="transferabilty of the model to a new month test set")
    args = parser.parse_args()
    mlp_classifier(args)
    