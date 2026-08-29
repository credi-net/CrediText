import numpy as np
import pandas as pd
import torch
from types import SimpleNamespace

from creditext.experiments.mlp_experiments import dataset_loader
from creditext.experiments.mlp_experiments import mlp_train_classifier
from creditext.experiments.mlp_experiments.mlp_modules import CrediGrainMultiTaskMLP, credigrain_classification_loss, masked_row_mean, train_credigrain_multitask
from creditext.experiments.mlp_experiments.utils import absolute_error_summary


def test_absolute_error_summary_reports_mean_min_and_max():
    mae,min_ae,max_ae=absolute_error_summary([1.0, 3.0], [[0.0], [5.0]])

    assert mae == 1.5
    assert min_ae == 1.0
    assert max_ae == 2.0


def test_masked_row_mean_ignores_unlabelled_rows():
    losses=torch.tensor([[1.0, 3.0], [100.0, 200.0]], requires_grad=True)

    masked_loss=masked_row_mean(losses, torch.tensor([1.0, 0.0]))

    assert masked_loss.item() == 2.0
    masked_loss.backward()
    np.testing.assert_array_equal(losses.grad.numpy(), [[0.5, 0.5], [0.0, 0.0]])


def test_credigrain_classification_loss_modes_handle_rare_positives():
    logits=torch.zeros((1, 2))
    targets=torch.ones((1, 2))
    positive_counts=torch.tensor([1.0, 2.0])
    negative_counts=torch.tensor([3.0, 2.0])

    weighted=credigrain_classification_loss(logits, targets, "positive_weights", positive_counts, negative_counts)
    focal=credigrain_classification_loss(logits, targets, "focal", positive_counts, negative_counts)
    balanced=credigrain_classification_loss(logits, targets, "balanced_softmax", positive_counts, negative_counts)

    base_loss=torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    torch.testing.assert_close(weighted, base_loss*torch.tensor([3.0, 1.0]))
    torch.testing.assert_close(focal, base_loss*0.25)
    assert balanced[0, 0] > balanced[0, 1]


def test_credigrain_continuous_predictions_are_bounded():
    model=CrediGrainMultiTaskMLP(
        input_dim=2,
        head_dims={
            "functional_category":1,
            "cybersecurity":1,
            "epistemic_reliability":1,
            "epistemic_reliability.bin":1,
        },
        hidden_dims=[2],
    )
    with torch.no_grad():
        model.epistemic_cts_head.weight.fill_(1000.0)
        model.epistemic_cts_head.bias.fill_(-1000.0)

    predictions=model(torch.tensor([[0.0, 0.0], [1.0, 1.0]]))["epistemic_reliability.cts"]

    assert torch.all(predictions >= 0.0)
    assert torch.all(predictions <= 1.0)


def test_credigrain_metrics_ignore_unlabelled_rows():
    y_test={
        "functional_category":np.array([[1.0], [0.0]], dtype=np.float32),
        "functional_category_mask":np.array([[1.0], [0.0]], dtype=np.float32),
        "cybersecurity":np.array([[1.0], [0.0]], dtype=np.float32),
        "cybersecurity_mask":np.array([[1.0], [0.0]], dtype=np.float32),
        "epistemic_reliability":np.array([[1.0], [0.0]], dtype=np.float32),
        "epistemic_reliability_mask":np.array([[1.0], [0.0]], dtype=np.float32),
        "epistemic_reliability.bin":np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32),
        "epistemic_reliability.bin_mask":np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.float32),
        "epistemic_reliability.cts":np.array([0.8, np.nan], dtype=np.float32),
        "epistemic_reliability.cts_mask":np.array([1.0, 0.0], dtype=np.float32),
        "label_maps":{"epistemic_reliability.bin":["reliable", "unreliable"]},
    }
    pred_dict={
        "functional_category":np.array([[0.9], [0.9]], dtype=np.float32),
        "cybersecurity":np.array([[0.9], [0.9]], dtype=np.float32),
        "epistemic_reliability":np.array([[0.9], [0.9]], dtype=np.float32),
        "epistemic_reliability.bin":np.array([[0.9, 0.1], [0.9, 0.9]], dtype=np.float32),
        "epistemic_reliability.cts":np.array([0.7, 100.0], dtype=np.float32),
    }

    metrics=mlp_train_classifier.compute_credigrain_metrics(y_test, pred_dict, 0.5, 0.5)
    summary=mlp_train_classifier.build_credigrain_summary_metrics(y_test, metrics)
    detailed=mlp_train_classifier.build_credigrain_detailed_metrics(y_test, pred_dict, 0.5, 0.5)

    assert metrics["functional_category_f1_micro"] == 1.0
    assert np.isclose(metrics["epistemic_reliability_cts_mae"], 0.1)
    assert np.isclose(metrics["epistemic_reliability_cts_min_ae"], 0.1)
    assert np.isclose(metrics["epistemic_reliability_cts_max_ae"], 0.1)
    assert np.isclose(metrics["reliability_score_mae"], 0.1)
    assert np.isclose(metrics["reliability_score_min_ae"], 0.1)
    assert np.isclose(metrics["reliability_score_max_ae"], 0.1)
    assert list(summary.index) == ["f1_macro", "f1_micro", "mae", "max_ae", "min_ae", "n", "n_classes"]
    assert list(summary.columns) == [
        "functional_category",
        "cybersecurity",
        "epistemic_reliability",
        "epistemic_reliability.bin",
        "epistemic_reliability.cts",
    ]
    np.testing.assert_array_equal(summary.loc["n"], [1, 1, 1, 1, 1])
    np.testing.assert_array_equal(summary.loc["n_classes"], [1, 1, 1, 2, 1])
    assert np.isclose(summary.loc["mae", "epistemic_reliability.cts"], 0.1)
    assert set(detailed["evaluated_count"]) == {1}
    regression_rows=detailed[detailed["metric_scope"] == "stream_summary"]
    np.testing.assert_allclose(regression_rows["min_ae"], [0.1, 0.1], atol=1e-7)
    np.testing.assert_allclose(regression_rows["max_ae"], [0.1, 0.1], atol=1e-7)


def test_credigrain_train_entrypoint_smoke_with_mocked_hf(tmp_path, monkeypatch):
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.DataFrame(
        {
            "domain": ["a.example", "b.example"],
            "functional_category": ["news:wikipedia_miscellaneous", "shopping:webshrinker"],
            "cybersecurity": ["phishing:phish_db", "malicious:benign_malicious_urls"],
            "epistemic_reliability": [
                "reliability.bin=unreliable:wikipedia_miscellaneous|type.unreliable=fake news:wikipedia_miscellaneous|reliability.cts=0.2:wikipedia_miscellaneous",
                "reliability.bin=reliable:benign_malicious_urls|type.reliable=fact-checker:benign_malicious_urls|reliability.cts=0.8:benign_malicious_urls",
            ],
        }
    )
    val_df = pd.DataFrame(
        {
            "domain": ["c.example"],
            "functional_category": ["news:other_source"],
            "cybersecurity": ["phishing:other_source"],
            "epistemic_reliability": ["reliability.bin=unreliable:other_source|type.unreliable=fake news:other_source|reliability.cts=0.1:other_source"],
        }
    )
    test_df = pd.DataFrame(
        {
            "domain": ["d.example"],
            "functional_category": ["shopping:other_source"],
            "cybersecurity": ["malicious:other_source"],
            "epistemic_reliability": ["reliability.bin=reliable:other_source|type.reliable=fact-checker:other_source|reliability.cts=0.9:other_source"],
        }
    )

    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"
    test_csv = splits_dir / "test.csv"
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    def fake_hf_hub_download(repo_id, repo_type, filename, token=None):
        if filename in {
            "split/balanced/credible-non/train/credi-grain.csv",
            "split/balanced/train/credi-grain.csv",
            "split/train/credi-grain.csv",
            "train/credi-grain.csv",
        }:
            return str(train_csv)
        if filename in {
            "split/balanced/credible-non/val/credi-grain.csv",
            "split/balanced/val/credi-grain.csv",
            "split/val/credi-grain.csv",
            "val/credi-grain.csv",
        }:
            return str(val_csv)
        if filename in {
            "split/balanced/credible-non/test/credi-grain.csv",
            "split/balanced/test/credi-grain.csv",
            "split/test/credi-grain.csv",
            "test/credi-grain.csv",
        }:
            return str(test_csv)
        raise ValueError(f"Unexpected filename requested in smoke test: {filename}")

    def fake_load_emb_dict_from_parquet(*args, **kwargs):
        return {
            "a.example": [0.1, 0.2, 0.3],
            "b.example": [0.2, 0.3, 0.4],
            "c.example": [0.3, 0.4, 0.5],
            "d.example": [0.4, 0.5, 0.6],
        }

    class FakeModel:
        def __init__(self, y_template):
            self.y_template = y_template

        def predict(self, X_test_feat):
            n = len(X_test_feat)
            return {
                "functional_category": np.full((n, self.y_template["functional_category"].shape[1]), 0.6, dtype=np.float32),
                "cybersecurity": np.full((n, self.y_template["cybersecurity"].shape[1]), 0.6, dtype=np.float32),
                "epistemic_reliability": np.full((n, self.y_template["epistemic_reliability"].shape[1]), 0.6, dtype=np.float32),
                "epistemic_reliability.bin": np.full((n, self.y_template["epistemic_reliability.bin"].shape[1]), 0.6, dtype=np.float32),
                "epistemic_reliability.cts": np.zeros((n,), dtype=np.float32),
            }

    def fake_train(*, y_train, y_valid, **kwargs):
        assert kwargs["classification_loss_mode"] == "focal"
        assert y_train["label_maps"] == {
            "functional_category": ["news", "shopping"],
            "cybersecurity": ["malicious", "phishing"],
            "epistemic_reliability": ["type.reliable=fact-checker", "type.unreliable=fake news"],
            "epistemic_reliability.bin": ["reliable", "unreliable"],
        }
        np.testing.assert_array_equal(y_train["functional_category"], [[1.0, 0.0], [0.0, 1.0]])
        np.testing.assert_array_equal(y_train["cybersecurity"], [[0.0, 1.0], [1.0, 0.0]])
        np.testing.assert_array_equal(y_train["epistemic_reliability"], [[0.0, 1.0], [1.0, 0.0]])
        np.testing.assert_array_equal(y_train["epistemic_reliability.bin"], [[0.0, 1.0], [1.0, 0.0]])
        np.testing.assert_allclose(y_train["epistemic_reliability.cts"], [0.2, 0.8])
        np.testing.assert_array_equal(y_valid["cybersecurity"], [[0.0, 1.0]])
        np.testing.assert_array_equal(y_valid["epistemic_reliability.bin"], [[0.0, 1.0]])
        history = {
            "train_total": [1.0],
            "valid_total": [1.0],
            "train_reliability_score": [1.0],
            "valid_reliability_score": [1.0],
        }
        return FakeModel(y_train), history

    monkeypatch.setattr(dataset_loader, "hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr(dataset_loader.CrediGrain, "load_emb_dict_from_parquet", staticmethod(fake_load_emb_dict_from_parquet))
    monkeypatch.setattr(dataset_loader.CrediGrain, "load_emb_dict", staticmethod(lambda *args, **kwargs: {}))
    monkeypatch.setattr(mlp_train_classifier, "setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(mlp_train_classifier, "plot_loss", lambda *args, **kwargs: None)
    monkeypatch.setattr(mlp_train_classifier, "train_credigrain_multitask", fake_train)
    monkeypatch.setattr(mlp_train_classifier.pickle, "dump", lambda obj, file_obj: None)

    plots_dir = tmp_path / "plots"
    logs_dir = tmp_path / "logs"
    plots_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    args = SimpleNamespace(
        task_profile="credigrain",
        split_mode="balanced",
        test_mode="credible-non",
        crediGrain_text_emb_path=str(tmp_path),
        crediGrain_gnn_emb_path=str(tmp_path),
        crediGrain_path="dummy/repo",
        embed_type="text",
        emb_model="embeddinggemma-300m",
        agg_text_emb=False,
        month="dec",
        emb_dim=3,
        original_emb_dim=3,
        keep_content="all",
        keep_content_count=1,
        gnn_encoder="RNI",
        agg_function="avg",
        use_gnn_emb=False,
        use_FQDN=False,
        agg_month_emb=False,
        fusion_mode="cat",
        lr=0.001,
        epochs=1,
        batch_size=2,
        loss_weight_functional=1.0,
        loss_weight_cybersecurity=1.0,
        loss_weight_epistemic=1.0,
        loss_weight_reliability_score=1.0,
        loss_weight_reliability_bin=0.5,
        loss_weight_reliability_cts=0.5,
        classification_loss_mode="focal",
        plots_out_path=str(plots_dir),
        logs_out_path=str(logs_dir),
    )

    mlp_train_classifier.mlp_classifier_credigrain(args)

    result_files = list(plots_dir.glob("*credigrain_results.csv"))
    assert len(result_files) == 1
    summary_df=pd.read_csv(result_files[0])
    assert list(summary_df.columns) == [
        "metric",
        "functional_category",
        "cybersecurity",
        "epistemic_reliability",
        "epistemic_reliability.bin",
        "epistemic_reliability.cts",
    ]
    assert summary_df["metric"].tolist() == ["f1_macro", "f1_micro", "mae", "max_ae", "min_ae", "n", "n_classes"]
    detailed_files = list(plots_dir.glob("*credigrain_detailed.csv"))
    assert len(detailed_files) == 1
    timings_files = list(plots_dir.glob("*timings.csv"))
    assert len(timings_files) == 1


def test_real_credigrain_trainer_masks_missing_continuous_targets():
    features=np.array([[0.0, 0.2], [0.4, 0.1], [0.8, 0.6], [1.0, 0.9]], dtype=np.float32)
    source_df=pd.DataFrame(
        {
            "domain":["a.test", "b.test", "c.test", "d.test"],
            "functional_category":["news:source_a", "shopping:source_b", "news:source_a", "shopping:source_b"],
            "cybersecurity":["phishing:phish_db", "malicious:benign_malicious_urls", "phishing:phish_db", "malicious:benign_malicious_urls"],
            "epistemic_reliability":[
                "reliability.bin=credible:source_a|reliability.cts=0.8:source_a|type.reliable=fact-checker:source_a",
                "reliability.bin=unreliable:source_b|type.unreliable=fake news:source_b",
                "reliability.bin=credible:source_a|reliability.cts=0.6:source_a|type.reliable=fact-checker:source_a",
                "reliability.bin=unreliable:source_b|type.unreliable=fake news:source_b",
            ],
        }
    )
    prepared=dataset_loader.CrediGrain._prepare_target_streams(source_df)
    class_maps=dataset_loader.CrediGrain._fit_stream_class_maps(prepared)
    targets=dataset_loader.CrediGrain._encode_target_streams(prepared, class_maps)
    model=CrediGrainMultiTaskMLP(
        input_dim=2,
        head_dims={
            "functional_category":targets["functional_category"].shape[1],
            "cybersecurity":targets["cybersecurity"].shape[1],
            "epistemic_reliability":targets["epistemic_reliability"].shape[1],
            "epistemic_reliability.bin":targets["epistemic_reliability.bin"].shape[1],
        },
        hidden_dims=[4],
    )

    _,history=train_credigrain_multitask(
        model,
        features,
        targets,
        features,
        targets,
        epochs=2,
        batch_size=2,
        lr=0.01,
    )

    assert set(history)=={"train_total", "valid_total", "train_reliability_score", "valid_reliability_score"}
    assert all(np.isfinite(values).all() for values in history.values())


def test_get_credibility_values_selects_credible_class():
    y_dict={
        "epistemic_reliability.bin":np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        "label_maps":{"epistemic_reliability.bin":["credible", "non_credible"]},
    }
    pred_dict={"epistemic_reliability.bin":np.array([[0.8, 0.2], [0.1, 0.9]], dtype=np.float32)}

    y_true,y_pred=mlp_train_classifier.get_credibility_values(y_dict, pred_dict)

    np.testing.assert_array_equal(y_true, np.array([1.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(y_pred, np.array([0.8, 0.1], dtype=np.float32))
