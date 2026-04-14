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
                  save_shaply_plots,expected_calibration_error
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score as Recall,roc_auc_score as AUROC,average_precision_score as AUPRC
from dataset_loader import DomainRel
import logging
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
    results = []
   
    for i in range(args.runs):        
        logging.info(f"#################### Run {i} ##################")    
        run_file_name+=f"_run{str(i)}"   
        ###################### PYTorch Regressor  ######################
        # dim_multiplier=2
        dim_multiplier=0.5
        logging.info(f" hidden_dim_multiplier={dim_multiplier}")
        if args.library=="pytorch":
            mlp_clf = LabelPredictor(len(X_train_feat[0]),hidden_dim_multiplier=dim_multiplier, out_dim=2)
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
        ################## Eval and Plot ###############
        acc_lst,f1_lst,Recall_lst,AUROC_lst,AUPRC_lst=[],[],[],[],[]
        true = y_test
        if args.loss_fun=="nl_loss":
            pred = mlp_clf.predict(X_test_feat)
        elif args.loss_fun=="halo":
            pos, centroids = halo_model(torch.tensor(X_test_feat).float())
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

        # ece = expected_calibration_error(torch.tensor(pred), torch.tensor(true))
        # logging.info(f"Expected Calibration Error: {ece:.4f}")
        pred=pred.round()
        accuracy = accuracy_score(true, pred)
        acc_lst.append(accuracy)
        logging.info(f"Accuracy: {accuracy:.4f}")

        f1 = f1_score(true,pred, average='macro')
        f1_lst.append(f1)
        logging.info(f"F1 score (macro): {f1:.4f}")

        recall = Recall(true, pred)
        Recall_lst.append(recall)
        logging.info(f"Recall: {recall:.4f}")

        auc_roc = AUROC(true, pred)
        AUROC_lst.append(auc_roc)
        logging.info(f"AUROC: {auc_roc:.4f}")

        auc_pr = AUPRC(true, pred)
        AUPRC_lst.append(auc_pr)
        logging.info(f"AUPRC: {auc_pr:.4f}")

        cm = confusion_matrix(true, pred)
        logging.info(f"cm: {cm}")        

        
        plot_loss(train_loss, valid_loss, test_loss, mean_loss, run_file_name+"_loss.pdf",ylabel="CrossEntropy Loss")
        plot_classesCount(cm,run_file_name+"_class_frequancy.pdf")
        plot_regression_scatter(true, pred, run_file_name+"_testset_true_vs_pred_scatter.pdf")
        plot_confusion_matrix(cm, run_file_name+"_testset_confusion_matrix.pdf")
        ###############################################################

    logging.info(f"ACC MAEN={np.mean(acc_lst)}")
    logging.info(f"ACC STD={np.std(acc_lst)}")
    results_df = pd.DataFrame(list(zip(acc_lst,f1_lst,[args,args,args])), columns=['ACC', 'F1', 'args'])
    results_df.to_csv(f"{run_file_name}{'_agg' if args.agg_month_emb else ''}_dqr_results.csv",index=None)

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
    parser.add_argument("--original_emb_dim", type=int, default=768,help="The original embedding model dim size")
    parser.add_argument("--max_iter", type=int, default=200,help="MLP regressor max iteration count")
    parser.add_argument("--lr", type=float, default=1e-1,help="learning rate")
    # parser.add_argument("--lr", type=float, default=5e-1,help="learning rate")
    parser.add_argument("--epochs", type=int,default=500, help="# training epochs") 
    parser.add_argument("--plots_out_path", type=str, default=str(root + "/plots"),help="plots and results store path")
    parser.add_argument("--logs_out_path", type=str, default=str(root + "/logs"),help="logging path")
    parser.add_argument("--runs", type=int, default=1,help="# training runs")
    parser.add_argument("--use_gnn_emb", action='store_false',help="append GNN embedding")
    parser.add_argument("--gnn_encoder",   default="RNI", choices=["RNI","text"],help="append GNN node embedding intialization")
    parser.add_argument("--agg_month_emb",  action='store_true', help="aggregate montly GNN embeddings")
    parser.add_argument("--agg_text_emb", action='store_true', help="aggregate montly text embeddings")
    parser.add_argument("--agg_function", type=str, default="cat",choices=["avg","cat", "sum", "min", "max"], help="aggregate function")
    parser.add_argument("--use_topic_emb", action='store_true',help="use topic modeling features")
    parser.add_argument("--filter_by_GNN_nodes", action='store_false',help="filter by domains has GNN embeddings")
    parser.add_argument("--num_classes", type=int, default=2,help="# classifcation classes for Multihead model")
    parser.add_argument("--use_FQDN", action='store_true',help="use fqdn_features")
    parser.add_argument("--generate_weaksupervision_scores", action='store_true', help="generate weak supervision datasets scores")
    parser.add_argument("--month", type=str, default="dec", choices=["oct", "nov", "dec"],help="CrediBench month snapshot")
    parser.add_argument("--library", type=str, default="pytorch", choices=["pytorch", "sklearn"],help="ML library to use")
    parser.add_argument("--split_mode", type=str, default="balanced", choices=["balanced", "phishing","malware","misinfo","general"],help="split mode for the dataset")
    parser.add_argument("--test_mode", type=str, default="credible-non", choices=["credible-non", "sub-category"],help="split mode for the dataset")
    parser.add_argument("--fusion_mode", type=str, default="cat", choices=["avg","cat", "sum", "min", "max","mul","gated"],help="embedding fusion method")
    parser.add_argument("--keep_content", type=str, default="all", choices=["all","longest", "shortest"],help="which content to keep for fusion")
    parser.add_argument("--keep_content_count", type=int, default=2, help="number of pages content to keep for fusion")
    parser.add_argument("--loss_fun", type=str, default="nl_loss", choices=["nl_loss", "halo"],help="the loss function to use for training the MLP regressor")
    args = parser.parse_args()
    mlp_classifier(args)
    