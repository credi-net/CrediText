import os

import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace
import pyarrow as pa
import pyarrow.parquet as pq

from creditext.experiments.mlp_experiments.dataset_loader import CrediGrain


def test_get_hf_token_loads_dotenv_from_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    (tmp_path / ".env").write_text("HF_TOKEN=test-token\n")

    try:
        assert CrediGrain._get_hf_token() == "test-token"
    finally:
        os.environ.pop("HF_TOKEN", None)


def test_fit_and_encode_stream_targets_are_deterministic():
    df = pd.DataFrame(
        {
            "functional_category_stream": [["malware", "phishing"], ["phishing"]],
            "cybersecurity_stream": [["spam"], []],
            "epistemic_reliability_stream": [["source_type"], ["claim_quality"]],
            "epistemic_reliability.bin_stream": [["credible"], ["non_credible"]],
            "epistemic_reliability.cts_stream": [0.25, np.nan],
            "functional_category_sourced_stream": [[("malware", "source_a"), ("phishing", "source_a")], [("phishing", "source_a")]],
            "cybersecurity_sourced_stream": [[("spam", "source_b")], []],
            "epistemic_reliability_sourced_stream": [[("source_type", "source_c")], [("claim_quality", "source_d")]],
            "epistemic_reliability.bin_sourced_stream": [[("credible", "source_c")], [("non_credible", "source_d")]],
            "popularity": ["rank=tranco_rank.3251913:tranco", "rank=iffy_rank.42:iffy"],
        }
    )

    class_maps = CrediGrain._fit_stream_class_maps(df)
    y_dict = CrediGrain._encode_target_streams(df, class_maps)

    assert class_maps["functional_category"] == ["malware", "phishing"]
    assert class_maps["epistemic_reliability.bin"] == ["credible", "non_credible"]

    np.testing.assert_array_equal(
        y_dict["functional_category"],
        np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        y_dict["cybersecurity_mask"],
        np.array([[1.0], [0.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        y_dict["epistemic_reliability.bin_mask"],
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        y_dict["epistemic_reliability.cts_mask"],
        np.array([1.0, 0.0], dtype=np.float32),
    )
    assert "popularity" not in y_dict
    assert "rank" not in y_dict


def test_class_support_cutoff_uses_training_positives_and_omits_epistemic_categories():
    df=pd.DataFrame(
        {
            "functional_category_stream":[["common"], ["common"], ["rare"]],
            "cybersecurity_stream":[["safe"], ["safe"], ["rare-threat"]],
        }
    )
    class_maps={
        "functional_category":["common", "rare"],
        "cybersecurity":["rare-threat", "safe"],
        "epistemic_reliability":["type.reliable=fact-checker"],
        "epistemic_reliability.bin":["reliable", "unreliable"],
    }

    filtered=CrediGrain._apply_class_support_cutoff(df, class_maps, min_positive_count=2)

    assert filtered == {
        "functional_category":["common"],
        "cybersecurity":["safe"],
        "epistemic_reliability.bin":["reliable", "unreliable"],
    }
    assert CrediGrain._apply_class_support_cutoff(df, class_maps, min_positive_count=0) is class_maps


def test_class_availability_uses_source_taxonomy():
    df=pd.DataFrame(
        {
            "functional_category_stream":[["alpha"], ["beta"], ["gamma"]],
            "cybersecurity_stream":[["safe"], ["safe"], ["safe"]],
            "epistemic_reliability_stream":[["known"], ["known"], ["known"]],
            "epistemic_reliability.bin_stream":[["reliable"], ["reliable"], ["reliable"]],
            "epistemic_reliability.cts_stream":[np.nan, np.nan, np.nan],
            "functional_category_sourced_stream":[[("alpha", "source_a")], [("beta", "source_a")], [("gamma", "source_b")]],
            "cybersecurity_sourced_stream":[[("safe", "source_c")]]*3,
            "epistemic_reliability_sourced_stream":[[("known", "source_d")]]*3,
            "epistemic_reliability.bin_sourced_stream":[[("reliable", "source_d")]]*3,
        }
    )
    class_maps=CrediGrain._fit_stream_class_maps(df)
    source_maps=CrediGrain._fit_stream_class_source_maps(df, class_maps)

    targets=CrediGrain._encode_target_streams(df, class_maps, source_maps)

    np.testing.assert_array_equal(
        targets["functional_category_mask"],
        np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=np.float32),
    )


def test_prepare_target_streams_routes_channels_and_discards_datasources():
    df=pd.DataFrame(
        {
            "domain":["example.test"],
            "functional_category":["news:wikipedia_miscellaneous"],
            "cybersecurity":["phishing:phish_db|malicious:benign_malicious_urls"],
            "epistemic_reliability":["reliability.bin=unreliable:wikipedia_miscellaneous|type.unreliable=fake news:wikipedia_miscellaneous"],
        }
    )

    prepared=CrediGrain._prepare_target_streams(df)

    assert prepared.loc[0, "functional_category_stream"] == ["news"]
    assert prepared.loc[0, "cybersecurity_stream"] == ["malicious", "phishing"]
    assert prepared.loc[0, "epistemic_reliability.bin_stream"] == ["unreliable"]
    assert prepared.loc[0, "epistemic_reliability_stream"] == ["type.unreliable=fake news"]
    assert np.isnan(prepared.loc[0, "epistemic_reliability.cts_stream"])


def test_prepare_target_streams_canonicalizes_legacy_reliability_labels():
    df=pd.DataFrame(
        {
            "domain":["credible.test", "non-credible.test"],
            "functional_category":["news:source", "news:source"],
            "cybersecurity":["phishing:source", "malicious:source"],
            "epistemic_reliability":[
                "reliability.bin=credible:source|type.reliable=fact-checker:source",
                "reliability.bin=non_credible:source|type.unreliable=fake news:source",
            ],
        }
    )

    prepared=CrediGrain._prepare_target_streams(df)

    assert prepared["epistemic_reliability.bin_stream"].tolist() == [["reliable"], ["unreliable"]]


def test_prepare_target_streams_rejects_unknown_reliability_binary_label():
    df=pd.DataFrame(
        {
            "domain":["example.test"],
            "functional_category":["news:source"],
            "cybersecurity":["phishing:source"],
            "epistemic_reliability":["reliability.bin=maybe:source|type.unreliable=fake news:source"],
        }
    )

    with pytest.raises(ValueError, match="Unsupported CrediGrain reliability.bin labels"):
        CrediGrain._prepare_target_streams(df)


def test_prepare_target_streams_rejects_ambiguous_epistemic_labels():
    df=pd.DataFrame(
        {
            "domain":["example.test"],
            "functional_category":["news:source"],
            "cybersecurity":["phishing:source"],
            "epistemic_reliability":["fake news:source"],
        }
    )

    with pytest.raises(ValueError, match="unsupported epistemic_reliability labels"):
        CrediGrain._prepare_target_streams(df)


def test_aggregate_domain_pages_sorts_pages_and_averages():
    raw = {
        "example.com": [
            {"page": "https://b", "emb": [2.0, 2.0, 2.0]},
            {"page": "https://a", "emb": [0.0, 0.0, 0.0]},
            {"page": "https://c", "emb": [4.0, 4.0, 4.0]},
        ]
    }

    agg = CrediGrain._aggregate_domain_pages(raw, emb_dim=3, keep_content="all", keep_count=3)
    np.testing.assert_allclose(agg["example.com"], np.array([2.5, 2.5, 2.5]), atol=1e-6)


def test_validate_split_contract_sorts_domains_and_requires_columns():
    split_df = pd.DataFrame(
        {
            "domain": ["B.example", None, "a.example"],
            "functional_category": ["fc1", "fc2", "fc3"],
            "cybersecurity": ["cy1", "cy2", "cy3"],
            "epistemic_reliability": ["er1", "er2", "er3"],
        }
    )

    validated = CrediGrain._validate_split_contract(split_df, "train")

    assert validated["domain"].tolist() == ["a.example", "b.example"]


def test_validate_disjoint_splits_rejects_domain_leakage():
    train_df=pd.DataFrame({"domain":["shared.example", "train.example"]})
    valid_df=pd.DataFrame({"domain":["valid.example"]})
    test_df=pd.DataFrame({"domain":["shared.example"]})

    with pytest.raises(ValueError, match="train/test splits overlap"):
        CrediGrain._validate_disjoint_splits(train_df, valid_df, test_df)


def test_validate_stream_classes_filters_unseen_evaluation_labels():
    train_df=pd.DataFrame(
        {
            "functional_category_stream":[["malware"]],
            "cybersecurity_stream":[["spam"]],
            "epistemic_reliability_stream":[["source_type"]],
            "epistemic_reliability.bin_stream":[["credible"]],
        }
    )
    test_df=train_df.copy()
    test_df["functional_category_stream"]=[["phishing"]]
    class_maps=CrediGrain._fit_stream_class_maps(train_df)

    unseen = CrediGrain._validate_stream_classes(test_df, class_maps, "test")
    assert unseen["functional_category"] == ["phishing"]

    filtered = CrediGrain._filter_unseen_stream_classes(test_df, class_maps)
    assert filtered["functional_category_stream"].tolist() == [[]]


def test_group_domains_by_chunk_is_sorted_and_normalizes_domains():
    grouped = CrediGrain._group_domains_by_chunk(
        {
            "B.example": "chunk_2",
            "a.example": "chunk_1",
            "C.example": "chunk_1",
        }
    )

    assert grouped == [("chunk_1", ["a.example", "c.example"]), ("chunk_2", ["b.example"])]


def test_load_run_embeddings_filters_domains_missing_text_embeddings(monkeypatch):
    def make_split_df(domains):
        return pd.DataFrame(
        {
            "domain": domains,
            "functional_category": ["fc1", "fc2"],
            "cybersecurity": ["cy1", "cy2"],
            "epistemic_reliability": [
                "reliability.bin=credible|reliability.cts=0.5:src",
                "reliability.bin=non_credible|reliability.cts=0.7:src",
            ],
        }
    )

    def fake_load_splits(dataset_repo, split_mode="balanced", test_mode="credible-non"):
        return (
            make_split_df(["a.example", "b.example"]),
            make_split_df(["c.example", "d.example"]),
            make_split_df(["e.example", "f.example"]),
        )

    def fake_load_emb_dict_from_parquet(*args, **kwargs):
        return {
            "a.example": [0.0, 1.0, 2.0],
            "c.example": [1.0, 2.0, 3.0],
            "e.example": [2.0, 3.0, 4.0],
        }

    monkeypatch.setattr(CrediGrain, "load_splits", staticmethod(fake_load_splits))
    monkeypatch.setattr(CrediGrain, "load_emb_dict_from_parquet", staticmethod(fake_load_emb_dict_from_parquet))
    monkeypatch.setattr(CrediGrain, "load_emb_dict", staticmethod(lambda *args, **kwargs: {}))

    args = SimpleNamespace(
        crediGrain_text_emb_path="/tmp",
        crediGrain_gnn_emb_path="/tmp",
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
        agg_month_emb=False,
        fusion_mode="cat",
        use_gnn_emb=False,
        use_FQDN=False,
    )

    X_train, _, X_valid, _, X_test, _, X_train_feat, X_valid_feat, X_test_feat = CrediGrain.load_run_embeddings(args)

    assert X_train["domain"].tolist() == ["a.example"]
    assert X_valid["domain"].tolist() == ["c.example"]
    assert X_test["domain"].tolist() == ["e.example"]
    assert len(X_train_feat) == len(X_valid_feat) == len(X_test_feat) == 1


def test_load_hf_gnn_embeddings_uses_domain_index_shards(tmp_path, monkeypatch):
    index_path = tmp_path / "domain_index.parquet"
    shard0_path = tmp_path / "shard_0.parquet"
    shard1_path = tmp_path / "shard_1.parquet"

    index_df = pd.DataFrame(
        {
            "domain": ["a.example", "org.b"],
            "shard": ["shard_0.parquet", "shard_1.parquet"],
        }
    )
    index_df.to_parquet(index_path, index=False)

    shard0_table = pa.Table.from_pydict(
        {
            "domain": ["a.example"],
            "emb": [[0.1, 0.2, 0.3]],
        }
    )
    shard1_table = pa.Table.from_pydict(
        {
            "domain": ["org.b"],
            "emb": [[0.4, 0.5, 0.6]],
        }
    )
    pq.write_table(shard0_table, shard0_path)
    pq.write_table(shard1_table, shard1_path)

    file_map = {
        "gnn-embeddings/dec-2024/rni/rni_updated/domain_index.parquet": str(index_path),
        "gnn-embeddings/dec-2024/rni/rni_updated/shard_0.parquet": str(shard0_path),
        "gnn-embeddings/dec-2024/rni/rni_updated/shard_1.parquet": str(shard1_path),
    }

    def fake_hf_hub_download(repo_id, repo_type, filename, token=None):
        if filename in file_map:
            return file_map[filename]
        raise FileNotFoundError(filename)

    monkeypatch.setattr("creditext.experiments.mlp_experiments.dataset_loader.hf_hub_download", fake_hf_hub_download)

    emb_dict = CrediGrain._load_hf_gnn_embeddings(
        dataset_repo="dummy/repo",
        month="dec",
        gnn_encoder="RNI",
        q_domains=["a.example", "b.org"],
    )

    assert "a.example" in emb_dict
    assert "org.b" in emb_dict
    np.testing.assert_allclose(np.asarray(emb_dict["a.example"], dtype=np.float32), np.array([0.1, 0.2, 0.3], dtype=np.float32))
    np.testing.assert_allclose(np.asarray(emb_dict["org.b"], dtype=np.float32), np.array([0.4, 0.5, 0.6], dtype=np.float32))


def test_load_run_embeddings_uses_feature_cache_and_split_scoped_text(monkeypatch, tmp_path):
    def make_split_df(domain):
        return pd.DataFrame(
            {
                "domain":[domain],
                "functional_category":["fc1"],
                "cybersecurity":["cy1"],
                "epistemic_reliability":["reliability.bin=credible|reliability.cts=0.5:src"],
            }
        )

    text_calls = {"count": 0}

    def fake_load_splits(dataset_repo, split_mode="balanced", test_mode="credible-non"):
        return make_split_df("a.example"), make_split_df("b.example"), make_split_df("c.example")

    def fake_text_loader(*args, **kwargs):
        text_calls["count"] += 1
        q_domains = kwargs.get("q_domains", [])
        assert set(q_domains) == {"a.example", "b.example", "c.example"}
        return {
            "a.example":[0.1, 0.2, 0.3],
            "b.example":[0.2, 0.3, 0.4],
            "c.example":[0.3, 0.4, 0.5],
        }

    monkeypatch.setattr(CrediGrain, "load_splits", staticmethod(fake_load_splits))
    monkeypatch.setattr(CrediGrain, "load_emb_dict_from_parquet", staticmethod(fake_text_loader))
    monkeypatch.setattr(CrediGrain, "load_emb_dict", staticmethod(lambda *args, **kwargs: {}))
    monkeypatch.setenv("CREDIGRAIN_FEATURE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("CREDIGRAIN_DISABLE_FEATURE_CACHE", raising=False)

    args = SimpleNamespace(
        crediGrain_text_emb_path="/tmp",
        crediGrain_gnn_emb_path="/tmp",
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
        split_mode="balanced",
        test_mode="credible-non",
    )

    first = CrediGrain.load_run_embeddings(args)
    second = CrediGrain.load_run_embeddings(args)

    assert text_calls["count"] == 1
    assert len(first[6]) == len(second[6])
    assert len(first[6]) > 0
