import argparse
from datetime import datetime
import pickle
import numpy as np
import os
import random
import time
from creditext.utils.path import get_root_dir
import torch
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier as  Sklearn_MLPClassifier
from creditext.experiments.mlp_experiments.mlp_modules import MLPRegressor, train_classifier_unbalanced,train_classifier_unbalanced_halo
from creditext.experiments.mlp_experiments.mlp_modules import MLP3LayersPredictor
from creditext.experiments.mlp_experiments.mlp_modules import MultiTaskMLP,train_scikitlearn_classifier,LabelPredictor,train_classifier_unbalanced
from creditext.experiments.mlp_experiments.mlp_modules import CrediGrainMultiTaskMLP, train_credigrain_multitask
from sklearn.preprocessing import normalize 
from creditext.experiments.mlp_experiments.utils import train_valid_test_split,resize_and_fuse_emb,\
                  plot_loss,plot_regression_scatter,eval,\
                  plot_classesCount,plot_confusion_matrix,\
                  save_shaply_plots,expected_calibration_error,absolute_error_summary
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score as Recall,roc_auc_score as AUROC,average_precision_score as AUPRC, precision_recall_curve, precision_recall_fscore_support
from creditext.experiments.mlp_experiments.dataset_loader import DomainRel, CrediGrain
import logging
from creditext.utils.logger import setup_logging


def configure_determinism(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_reliability_component_weights(bin_weight: float, cts_weight: float) -> tuple[float, float]:
    if bin_weight < 0.0 or cts_weight < 0.0:
        raise ValueError("CrediGrain reliability component weights must be non-negative.")

    total = bin_weight + cts_weight
    if total <= 0.0:
        raise ValueError("CrediGrain reliability component weights cannot both be zero.")

    return bin_weight / total, cts_weight / total


def validate_task_profile_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.task_profile != "credigrain":
        return

    if args.split_mode != CrediGrain.SUPPORTED_SPLIT_MODE:
        parser.error(
            f"task_profile=credigrain only supports --split_mode={CrediGrain.SUPPORTED_SPLIT_MODE} "
            f"(got {args.split_mode})"
        )
    if args.test_mode != CrediGrain.SUPPORTED_TEST_MODE:
        parser.error(
            f"task_profile=credigrain only supports --test_mode={CrediGrain.SUPPORTED_TEST_MODE} "
            f"(got {args.test_mode})"
        )
    if args.class_support_cutoff < 0:
        parser.error("--class_support_cutoff must be non-negative")

def write_testset_emb(run_file_name,X_test,X_test_feat):
    test_set_emb_dict=dict(zip(X_test["domain"].tolist(),X_test_feat))
    test_emb_file_name=f"{run_file_name}_test_set_emb_dict.pkl"
    with open(test_emb_file_name, 'wb') as file:
        pickle.dump(test_set_emb_dict, file)


def get_credibility_values(y_dict: dict, pred_dict: dict) -> tuple[np.ndarray, np.ndarray]:
    class_names=y_dict.get("label_maps", {}).get("epistemic_reliability.bin", [])
    for positive_label in ["reliable", "credible"]:
        if positive_label in class_names:
            positive_idx=class_names.index(positive_label)
            return y_dict["epistemic_reliability.bin"][:, positive_idx], pred_dict["epistemic_reliability.bin"][:, positive_idx]
    for negative_label in ["unreliable", "non_credible"]:
        if negative_label in class_names:
            negative_idx=class_names.index(negative_label)
            return 1.0-y_dict["epistemic_reliability.bin"][:, negative_idx], 1.0-pred_dict["epistemic_reliability.bin"][:, negative_idx]
    raise ValueError(f"CrediGrain reliability labels must include a reliable/unreliable class, got {class_names}.")


def get_stream_mask(y_dict: dict, stream_name: str) -> np.ndarray:
    mask_name=f"{stream_name}_mask"
    if mask_name not in y_dict:
        raise KeyError(f"Missing availability mask for CrediGrain stream {stream_name}: {mask_name}")
    return np.asarray(y_dict[mask_name])>0


def get_stream_row_mask(y_dict: dict, stream_name: str) -> np.ndarray:
    stream_mask=get_stream_mask(y_dict, stream_name)
    if stream_mask.ndim==1:
        return stream_mask
    return np.any(stream_mask, axis=1)


def get_classification_streams(y_dict: dict) -> list[str]:
    return [
        stream_name
        for stream_name in (
            "functional_category",
            "cybersecurity",
            "epistemic_reliability",
            "epistemic_reliability.bin",
        )
        if stream_name in y_dict and f"{stream_name}_mask" in y_dict
    ]


def tune_credigrain_thresholds(y_valid: dict, pred_valid: dict) -> dict[str, np.ndarray]:
    tuned_thresholds={}
    for stream_name in get_classification_streams(y_valid):
        stream_mask=get_stream_mask(y_valid, stream_name)
        y_true_mat=np.asarray(y_valid[stream_name])
        y_score_mat=np.asarray(pred_valid[stream_name])
        stream_thresholds=np.full(y_true_mat.shape[1], 0.5, dtype=np.float64)

        for class_idx in range(y_true_mat.shape[1]):
            class_mask=stream_mask[:,class_idx]
            y_true_col=y_true_mat[class_mask,class_idx]
            y_score_col=y_score_mat[class_mask,class_idx]
            if len(np.unique(y_true_col)) < 2:
                continue

            precision,recall,candidate_thresholds=precision_recall_curve(y_true_col, y_score_col)
            candidate_f1=np.divide(
                2.0*precision[:-1]*recall[:-1],
                precision[:-1]+recall[:-1],
                out=np.zeros_like(candidate_thresholds, dtype=np.float64),
                where=(precision[:-1]+recall[:-1])>0,
            )
            best_f1=np.max(candidate_f1)
            best_indices=np.flatnonzero(np.isclose(candidate_f1, best_f1))
            stream_thresholds[class_idx]=candidate_thresholds[best_indices[-1]]

        tuned_thresholds[stream_name]=stream_thresholds
    return tuned_thresholds


def _threshold_predictions(predictions: np.ndarray, stream_name: str, thresholds: dict[str, np.ndarray] | None) -> np.ndarray:
    threshold=0.5 if thresholds is None else np.asarray(thresholds[stream_name])
    return (predictions>=threshold).astype(int)


def compute_credigrain_metrics(y_test: dict, pred_dict: dict, rel_bin_weight: float, rel_cts_weight: float, thresholds: dict[str, np.ndarray] | None=None) -> dict:
    metrics={}
    for stream_name in get_classification_streams(y_test):
        stream_mask=get_stream_mask(y_test, stream_name)
        y_true_mat=y_test[stream_name]
        y_pred_mat=_threshold_predictions(pred_dict[stream_name], stream_name, thresholds)
        class_f1=[]
        class_support=[]
        micro_true=[]
        micro_pred=[]
        for class_idx in range(y_true_mat.shape[1]):
            class_mask=stream_mask[:,class_idx]
            if not np.any(class_mask):
                continue
            y_true_col=y_true_mat[class_mask,class_idx]
            y_pred_col=y_pred_mat[class_mask,class_idx]
            micro_true.append(y_true_col)
            micro_pred.append(y_pred_col)
            if np.any(y_true_col):
                class_f1.append(f1_score(y_true_col, y_pred_col, zero_division=0))
                class_support.append(np.sum(y_true_col))
        if class_f1:
            metrics[f"{stream_name}_f1_macro"]=float(np.mean(class_f1))
            metrics[f"{stream_name}_f1_micro"]=f1_score(
                np.concatenate(micro_true),
                np.concatenate(micro_pred),
                zero_division=0,
            )
            metrics[f"{stream_name}_f1_weighted"]=float(np.average(class_f1, weights=class_support))
        else:
            metrics[f"{stream_name}_f1_macro"]=float("nan")
            metrics[f"{stream_name}_f1_micro"]=float("nan")
            metrics[f"{stream_name}_f1_weighted"]=float("nan")

    cts_mask=get_stream_mask(y_test, "epistemic_reliability.cts")
    if np.any(cts_mask):
        cts_mae,cts_min_ae,cts_max_ae=absolute_error_summary(
            y_test["epistemic_reliability.cts"][cts_mask],
            pred_dict["epistemic_reliability.cts"][cts_mask],
        )
        metrics["epistemic_reliability_cts_mae"]=cts_mae
        metrics["epistemic_reliability_cts_min_ae"]=cts_min_ae
        metrics["epistemic_reliability_cts_max_ae"]=cts_max_ae
    else:
        metrics["epistemic_reliability_cts_mae"]=float("nan")
        metrics["epistemic_reliability_cts_min_ae"]=float("nan")
        metrics["epistemic_reliability_cts_max_ae"]=float("nan")

    rel_bin_true,rel_bin_pred=get_credibility_values(y_test, pred_dict)
    score_mask=get_stream_row_mask(y_test, "epistemic_reliability.bin") & cts_mask
    if np.any(score_mask):
        reliability_score_true=rel_bin_weight * rel_bin_true + rel_cts_weight * y_test["epistemic_reliability.cts"]
        reliability_score_pred=rel_bin_weight * rel_bin_pred + rel_cts_weight * pred_dict["epistemic_reliability.cts"]
        score_mae,score_min_ae,score_max_ae=absolute_error_summary(
            reliability_score_true[score_mask],
            reliability_score_pred[score_mask],
        )
        metrics["reliability_score_mae"]=score_mae
        metrics["reliability_score_min_ae"]=score_min_ae
        metrics["reliability_score_max_ae"]=score_max_ae
    else:
        metrics["reliability_score_mae"]=float("nan")
        metrics["reliability_score_min_ae"]=float("nan")
        metrics["reliability_score_max_ae"]=float("nan")
    return metrics


def build_credigrain_summary_metrics(y_test: dict, metrics: dict) -> pd.DataFrame:
    classification_streams=get_classification_streams(y_test)
    continuous_stream="epistemic_reliability.cts"
    summary={}

    for stream_name in classification_streams:
        summary[stream_name]={
            "f1_macro":metrics[f"{stream_name}_f1_macro"],
            "f1_micro":metrics[f"{stream_name}_f1_micro"],
            "f1_weighted":metrics[f"{stream_name}_f1_weighted"],
            "mae":float("nan"),
            "max_ae":float("nan"),
            "min_ae":float("nan"),
            "n":int(np.sum(get_stream_row_mask(y_test, stream_name))),
            "n_classes":int(y_test[stream_name].shape[1]),
        }

    continuous_mask=get_stream_mask(y_test, continuous_stream)
    summary[continuous_stream]={
        "f1_macro":float("nan"),
        "f1_micro":float("nan"),
        "f1_weighted":float("nan"),
        "mae":metrics["epistemic_reliability_cts_mae"],
        "max_ae":metrics["epistemic_reliability_cts_max_ae"],
        "min_ae":metrics["epistemic_reliability_cts_min_ae"],
        "n":int(np.sum(continuous_mask)),
        "n_classes":1,
    }
    return pd.DataFrame(summary).reindex(["f1_macro", "f1_micro", "f1_weighted", "mae", "max_ae", "min_ae", "n", "n_classes"])


def build_credigrain_detailed_metrics(y_test: dict, pred_dict: dict, rel_bin_weight: float, rel_cts_weight: float, thresholds: dict[str, np.ndarray] | None=None) -> pd.DataFrame:
    rows=[]
    label_maps=y_test.get("label_maps", {})

    for stream_name in get_classification_streams(y_test):
        stream_mask=get_stream_mask(y_test, stream_name)
        y_true_mat=y_test[stream_name]
        stream_thresholds=np.full(y_true_mat.shape[1], 0.5) if thresholds is None else np.asarray(thresholds[stream_name])
        y_pred_mat=_threshold_predictions(pred_dict[stream_name], stream_name, thresholds)
        class_names=label_maps.get(stream_name, [f"class_{idx}" for idx in range(y_true_mat.shape[1])])

        for class_idx, class_name in enumerate(class_names):
            class_mask=stream_mask[:,class_idx]
            y_true_col=y_true_mat[class_mask,class_idx]
            y_pred_col=y_pred_mat[class_mask,class_idx]
            support=int(np.sum(y_true_col))
            if support>0:
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_true_col,
                    y_pred_col,
                    average="binary",
                    zero_division=0,
                )
            else:
                precision=recall=f1=float("nan")
            rows.append(
                {
                    "stream": stream_name,
                    "label": class_name,
                    "metric_scope": "per_label",
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "support": support,
                    "evaluated_count": int(np.sum(class_mask)),
                    "decision_threshold": float(stream_thresholds[class_idx]),
                }
            )

    cts_mask=get_stream_mask(y_test, "epistemic_reliability.cts")
    score_mask=get_stream_row_mask(y_test, "epistemic_reliability.bin") & cts_mask
    cts_mae=float("nan")
    cts_min_ae=float("nan")
    cts_max_ae=float("nan")
    rel_mae=float("nan")
    rel_min_ae=float("nan")
    rel_max_ae=float("nan")
    if np.any(cts_mask):
        cts_mae,cts_min_ae,cts_max_ae=absolute_error_summary(
            y_test["epistemic_reliability.cts"][cts_mask],
            pred_dict["epistemic_reliability.cts"][cts_mask],
        )
    if np.any(score_mask):
        rel_bin_true,rel_bin_pred=get_credibility_values(y_test, pred_dict)
        reliability_score_true=rel_bin_weight * rel_bin_true + rel_cts_weight * y_test["epistemic_reliability.cts"]
        reliability_score_pred=rel_bin_weight * rel_bin_pred + rel_cts_weight * pred_dict["epistemic_reliability.cts"]
        rel_mae,rel_min_ae,rel_max_ae=absolute_error_summary(
            reliability_score_true[score_mask],
            reliability_score_pred[score_mask],
        )

    rows.append(
        {
            "stream": "epistemic_reliability.cts",
            "label": "__regression__",
            "metric_scope": "stream_summary",
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "support": int(np.sum(cts_mask)),
            "evaluated_count": int(np.sum(cts_mask)),
            "mae": float(cts_mae),
            "min_ae": float(cts_min_ae),
            "max_ae": float(cts_max_ae),
        }
    )
    rows.append(
        {
            "stream": "reliability_score",
            "label": "__weighted_bin_cts__",
            "metric_scope": "stream_summary",
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "support": int(np.sum(score_mask)),
            "evaluated_count": int(np.sum(score_mask)),
            "mae": float(rel_mae),
            "min_ae": float(rel_min_ae),
            "max_ae": float(rel_max_ae),
        }
    )
    return pd.DataFrame(rows)


def mlp_classifier_credigrain(args) -> None:
    run_t0=time.perf_counter()
    now = datetime.now()
    iso_compact = now.strftime("%Y%m%dT%H%M%S")
    run_file_name=f"CrediGrain_{args.month}_{args.embed_type}_{args.emb_model}_{args.fusion_mode}_{args.classification_loss_mode}_K{args.class_support_cutoff}_{iso_compact}"
    setup_logging(f"{args.logs_out_path}/{run_file_name}.log")
    logging.info(f"args={args}")
    run_file_name=f"{args.plots_out_path}/{run_file_name}"

    loader_t0=time.perf_counter()
    X_train, y_train, X_valid, y_valid, X_test, y_test, X_train_feat, X_valid_feat, X_test_feat = CrediGrain.load_run_embeddings(args)
    loader_wall_s=round(time.perf_counter()-loader_t0, 3)
    write_testset_emb(run_file_name, X_test, X_test_feat)
    rel_bin_weight, rel_cts_weight = resolve_reliability_component_weights(
        args.loss_weight_reliability_bin,
        args.loss_weight_reliability_cts,
    )

    head_dims={stream_name:y_train[stream_name].shape[1] for stream_name in get_classification_streams(y_train)}
    model=CrediGrainMultiTaskMLP(input_dim=len(X_train_feat[0]), head_dims=head_dims, hidden_dims=[max(128, len(X_train_feat[0])//2), max(64, len(X_train_feat[0])//4)])
    train_t0=time.perf_counter()
    model, history = train_credigrain_multitask(
        model=model,
        X_train_feat=X_train_feat,
        y_train=y_train,
        X_valid_feat=X_valid_feat,
        y_valid=y_valid,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        classification_loss_mode=args.classification_loss_mode,
        head_weights={
            "functional_category":args.loss_weight_functional,
            "cybersecurity":args.loss_weight_cybersecurity,
            "epistemic_reliability":args.loss_weight_epistemic,
            "reliability_score":args.loss_weight_reliability_score,
            "epistemic_reliability.bin":rel_bin_weight,
            "epistemic_reliability.cts":rel_cts_weight,
        },
    )
    train_wall_s=round(time.perf_counter()-train_t0, 3)

    with open(f"{run_file_name}_credibench_MLP_Model.pkl", 'wb') as file:
        pickle.dump(model, file)

    eval_t0=time.perf_counter()
    pred_valid=model.predict(X_valid_feat)
    pred_dict=model.predict(X_test_feat)
    thresholds=tune_credigrain_thresholds(y_valid, pred_valid)
    metrics=compute_credigrain_metrics(y_test, pred_dict, rel_bin_weight, rel_cts_weight, thresholds)

    logging.info(f"CrediGrain metrics={metrics}")
    plot_loss(history["train_total"], history["valid_total"], None, None, run_file_name+"_loss_total.pdf", ylabel="Multi-task Loss")
    plot_loss(history["train_reliability_score"], history["valid_reliability_score"], None, None, run_file_name+"_loss_reliability_score.pdf", ylabel="Reliability Loss")

    metrics_df=build_credigrain_summary_metrics(y_test, metrics)
    metrics_df.to_csv(f"{run_file_name}_credigrain_results.csv", index_label="metric")
    detailed_df=build_credigrain_detailed_metrics(y_test, pred_dict, rel_bin_weight, rel_cts_weight, thresholds)
    detailed_df.to_csv(f"{run_file_name}_credigrain_detailed.csv", index=False)

    eval_wall_s=round(time.perf_counter()-eval_t0, 3)
    timing_rows=[]
    loader_stage_times=getattr(args, "_credigrain_loader_stage_times", {})
    for stage_name, stage_value in loader_stage_times.items():
        timing_rows.append({"stage": stage_name, "seconds": stage_value})
    timing_rows.extend([
        {"stage": "loader_wall_s", "seconds": loader_wall_s},
        {"stage": "train_wall_s", "seconds": train_wall_s},
        {"stage": "eval_wall_s", "seconds": eval_wall_s},
        {"stage": "total_run_s", "seconds": round(time.perf_counter()-run_t0, 3)},
    ])
    timing_df=pd.DataFrame(timing_rows)
    timing_df.to_csv(f"{run_file_name}_timings.csv", index=False)


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
    parser.add_argument("--task_profile", type=str, default="domainrel", choices=["domainrel", "credigrain"], help="training profile")
    parser.add_argument("--domainRel_target", type=str, default="weak_label", choices=["weak_label"], help="the credability target")
    parser.add_argument("--domainRel_text_emb_path", type=str, default=str(root + "/data/weaksupervision") ,help="emb files path")
    parser.add_argument("--domainRel_gnn_emb_path", type=str, default=str(root + "/data/weaksupervision") ,help="emb files path")
    parser.add_argument("--domainRel_path", type=str, default=str(root + "/data/weaksupervision"),help="dqr dataset path")
    parser.add_argument("--crediGrain_text_emb_path", type=str, default=str(root + "/data/credigrain"), help="local fallback path for CrediGrain artifacts")
    parser.add_argument("--crediGrain_gnn_emb_path", type=str, default=str(root + "/data/credigrain"), help="local gnn emb path for CrediGrain artifacts")
    parser.add_argument("--crediGrain_path", type=str, default="credi-net/CrediGrain", help="HuggingFace dataset repo id for CrediGrain split data")
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
    parser.add_argument("--loss_weight_functional", type=float, default=1.0, help="loss weight for functional category head")
    parser.add_argument("--loss_weight_cybersecurity", type=float, default=1.0, help="loss weight for cybersecurity head")
    parser.add_argument("--loss_weight_epistemic", type=float, default=1.0, help="loss weight for epistemic categorical head")
    parser.add_argument("--loss_weight_reliability_score", type=float, default=1.0, help="loss weight for reliability score combined loss")
    parser.add_argument("--loss_weight_reliability_bin", type=float, default=0.5, help="loss component weight for epistemic reliability bin head")
    parser.add_argument("--loss_weight_reliability_cts", type=float, default=0.5, help="loss component weight for epistemic reliability cts head")
    parser.add_argument("--classification_loss_mode", type=str, default="positive_weights", choices=["bce", "positive_weights", "focal", "balanced_softmax"], help="classification loss for CrediGrain classification heads")
    parser.add_argument("--class_support_cutoff", type=int, default=100, help="minimum training positives for functional/cybersecurity classes; 0 disables cutoff")
    parser.add_argument("--seed", type=int, default=42, help="random seed used for deterministic training")
    args = parser.parse_args()
    validate_task_profile_args(args, parser)
    configure_determinism(args.seed)
    if args.task_profile=="credigrain":
        mlp_classifier_credigrain(args)
    else:
        mlp_classifier(args)
    