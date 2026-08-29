from creditext.experiments.mlp_experiments.utils import fuse_1d_emb, search_parquet_duckdb,normalize_embeddings,train_valid_test_split,resize_and_fuse_emb
import pandas as pd
import numpy as np
import pickle
import logging
import os
import time
import json
import hashlib
import tempfile
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from typing import Any
agg_months_dict={"oct":["oct"],
                "nov":["oct","nov"],
                "dec":["oct","nov","dec"]}


class CrediGrain(object):
    TARGET_SCHEMA_VERSION=7
    MONTH_FOLDERS={"oct":"oct2024","nov":"nov2024","dec":"dec2024"}
    GNN_MONTH_FOLDERS={"oct":"oct-2024","nov":"nov-2024","dec":"dec-2024"}
    REQUIRED_SPLIT_COLUMNS=("domain", "functional_category", "cybersecurity", "epistemic_reliability")
    SUPPORTED_SPLIT_MODE="balanced"
    SUPPORTED_TEST_MODE="credible-non"

    @staticmethod
    def _io_max_workers(default_workers: int=4):
        try:
            parsed=int(os.getenv("CREDIGRAIN_IO_WORKERS", str(default_workers)))
            return max(1, parsed)
        except Exception:
            return default_workers

    @staticmethod
    def _feature_cache_dir():
        default_root=os.path.join(os.getenv("SCRATCH", "/tmp"), "creditext-cache", "credigrain")
        return os.getenv("CREDIGRAIN_FEATURE_CACHE_DIR", default_root)

    @staticmethod
    def _feature_cache_enabled():
        return os.getenv("CREDIGRAIN_DISABLE_FEATURE_CACHE", "0") != "1"

    @staticmethod
    def _split_signature(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame):
        signature_payload={
            "train": train_df["domain"].astype(str).tolist(),
            "val": valid_df["domain"].astype(str).tolist(),
            "test": test_df["domain"].astype(str).tolist(),
        }
        return hashlib.sha256(json.dumps(signature_payload, sort_keys=True).encode("utf-8")).hexdigest()[0:16]

    @staticmethod
    def _build_feature_cache_key(args: Any, text_repo: str, gnn_repo: str, split_sig: str):
        cache_spec={
            "task_profile": "credigrain",
            "target_schema_version": CrediGrain.TARGET_SCHEMA_VERSION,
            "month": getattr(args, "month", "dec"),
            "embed_type": getattr(args, "embed_type", "text"),
            "emb_model": getattr(args, "emb_model", "embeddinggemma-300m"),
            "emb_dim": int(getattr(args, "emb_dim", 256)),
            "original_emb_dim": int(getattr(args, "original_emb_dim", 768)),
            "fusion_mode": getattr(args, "fusion_mode", "cat"),
            "keep_content": getattr(args, "keep_content", "all"),
            "keep_content_count": int(getattr(args, "keep_content_count", 2)),
            "gnn_encoder": getattr(args, "gnn_encoder", "RNI"),
            "agg_text_emb": bool(getattr(args, "agg_text_emb", False)),
            "agg_month_emb": bool(getattr(args, "agg_month_emb", False)),
            "agg_function": getattr(args, "agg_function", "cat"),
            "use_gnn_emb": bool(getattr(args, "use_gnn_emb", True)),
            "use_FQDN": bool(getattr(args, "use_FQDN", False)),
            "split_mode": getattr(args, "split_mode", CrediGrain.SUPPORTED_SPLIT_MODE),
            "test_mode": getattr(args, "test_mode", CrediGrain.SUPPORTED_TEST_MODE),
            "class_support_cutoff": int(getattr(args, "class_support_cutoff", 0)),
            "split_repo": getattr(args, "crediGrain_path", None),
            "text_repo": text_repo,
            "gnn_repo": gnn_repo,
            "split_sig": split_sig,
        }
        cache_json=json.dumps(cache_spec, sort_keys=True)
        return hashlib.sha256(cache_json.encode("utf-8")).hexdigest()

    @staticmethod
    def _cache_file_path(cache_key: str):
        return os.path.join(CrediGrain._feature_cache_dir(), f"{cache_key}.pkl")

    @staticmethod
    def _write_cache_atomic(cache_path: str, payload: dict):
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="wb", delete=False, dir=os.path.dirname(cache_path), suffix=".tmp") as tmp_file:
            pickle.dump(payload, tmp_file)
            tmp_name=tmp_file.name
        os.replace(tmp_name, cache_path)

    @staticmethod
    def _filter_domain_index_by_queries(domain_index: dict, q_domains: list[str]):
        if q_domains is None or len(q_domains) == 0:
            return domain_index

        requested=set()
        for raw_domain in q_domains:
            norm_domain=CrediGrain._normalize_domain(raw_domain)
            if norm_domain is None:
                continue
            requested.update(CrediGrain._domain_reverse_candidates(norm_domain))

        return {domain: chunk_ref for domain, chunk_ref in domain_index.items() if domain in requested}

    @staticmethod
    def _normalize_domain(domain: Any):
        if domain is None or pd.isna(domain):
            return None
        return str(domain).strip().lower()

    @staticmethod
    def _reverse_domain(domain: str):
        chunks=[c for c in str(domain).split(".") if c]
        return ".".join(chunks[::-1]) if len(chunks)>1 else domain

    @staticmethod
    def _domain_reverse_candidates(domain: str):
        reversed_domain=CrediGrain._reverse_domain(domain)
        if reversed_domain==domain:
            return [domain]
        return [domain,reversed_domain]

    @staticmethod
    def _get_hf_token():
        load_dotenv(".env", override=False)
        return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

    @staticmethod
    def _model_dir_candidates(model_name: str):
        candidates=[model_name]
        if "gemma" in model_name:
            candidates.append(model_name.replace("gemma","gema"))
        return list(dict.fromkeys(candidates))

    @staticmethod
    def _download_first_available(dataset_repo: str, rel_paths: list[str], hf_token: str=None):
        for rel_path in rel_paths:
            try:
                return hf_hub_download(repo_id=dataset_repo, repo_type="dataset", filename=rel_path, token=hf_token)
            except Exception:
                continue
        raise FileNotFoundError(f"Could not find any candidate file in {dataset_repo}: {rel_paths}")

    @staticmethod
    def _download_first_available_with_relpath(dataset_repo: str, rel_paths: list[str], hf_token: str=None):
        for rel_path in rel_paths:
            try:
                local_path=hf_hub_download(repo_id=dataset_repo, repo_type="dataset", filename=rel_path, token=hf_token)
                return local_path, rel_path
            except Exception:
                continue
        raise FileNotFoundError(f"Could not find any candidate file in {dataset_repo}: {rel_paths}")

    @staticmethod
    def _validate_split_contract(split_df: pd.DataFrame, split_name: str):
        missing_columns=[col for col in CrediGrain.REQUIRED_SPLIT_COLUMNS if col not in split_df.columns]
        if missing_columns:
            raise ValueError(f"CrediGrain {split_name} split is missing required columns: {missing_columns}")

        split_df=split_df.dropna(subset=["domain"]).copy()
        split_df["domain"]=split_df["domain"].apply(CrediGrain._normalize_domain)
        split_df=split_df.dropna(subset=["domain"]).sort_values("domain").reset_index(drop=True)
        return split_df

    @staticmethod
    def _validate_disjoint_splits(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame):
        split_domains={
            "train":set(train_df["domain"]),
            "validation":set(valid_df["domain"]),
            "test":set(test_df["domain"]),
        }
        for left_name,right_name in [("train", "validation"), ("train", "test"), ("validation", "test")]:
            overlap=sorted(split_domains[left_name] & split_domains[right_name])
            if overlap:
                raise ValueError(f"CrediGrain {left_name}/{right_name} splits overlap on {len(overlap)} domains; examples={overlap[:5]}")

    @staticmethod
    def _group_domains_by_chunk(domain_index: dict):
        chunk_to_domains={}
        for domain, chunk_ref in domain_index.items():
            norm_domain=CrediGrain._normalize_domain(domain)
            if norm_domain is None:
                continue
            chunk_to_domains.setdefault(chunk_ref, []).append(norm_domain)

        grouped=[]
        for chunk_ref in sorted(chunk_to_domains.keys(), key=lambda value: str(value)):
            grouped.append((chunk_ref, sorted(set(chunk_to_domains[chunk_ref]))))
        return grouped

    @staticmethod
    def _assert_embeddings_cover_domains(df: pd.DataFrame, emb_dict: dict, source_name: str):
        missing_domains=sorted(set(df["domain"].tolist()) - set(emb_dict.keys()))
        if missing_domains:
            preview=missing_domains[:10]
            raise ValueError(f"CrediGrain {source_name} embeddings are missing {len(missing_domains)} domains; examples={preview}")

    @staticmethod
    def _filter_to_embedding_domains(df: pd.DataFrame, emb_dict: dict, source_name: str):
        available_domains=set(emb_dict.keys())
        filtered_df=df[df["domain"].isin(available_domains)].copy().reset_index(drop=True)
        dropped_count=len(df)-len(filtered_df)
        logging.info(
            f"CrediGrain {source_name} coverage: kept={len(filtered_df)} "
            f"dropped={dropped_count} total={len(df)}"
        )
        if filtered_df.empty:
            raise ValueError(f"CrediGrain {source_name} embeddings have no overlap with requested split domains")
        return filtered_df

    @staticmethod
    def _month_folder(month: str):
        month_key=str(month).strip().lower()
        if month_key not in CrediGrain.MONTH_FOLDERS:
            raise ValueError(f"Unsupported month={month}. Supported months: {list(CrediGrain.MONTH_FOLDERS.keys())}")
        return CrediGrain.MONTH_FOLDERS[month_key]

    @staticmethod
    def _gnn_month_folder(month: str):
        month_key=str(month).strip().lower()
        if month_key not in CrediGrain.GNN_MONTH_FOLDERS:
            raise ValueError(f"Unsupported month={month}. Supported months: {list(CrediGrain.GNN_MONTH_FOLDERS.keys())}")
        return CrediGrain.GNN_MONTH_FOLDERS[month_key]

    @staticmethod
    def _candidate_gnn_dirs(month: str, gnn_encoder: str):
        month_folder=CrediGrain._gnn_month_folder(month)
        root_dir=f"gnn-embeddings/{month_folder}"
        if gnn_encoder == "RNI":
            return [
                f"{root_dir}/rni/rni_updated",
                f"{root_dir}/rni",
                root_dir,
            ]
        return [root_dir, f"{root_dir}/rni", f"{root_dir}/rni/rni_updated"]

    @staticmethod
    def _infer_domain_col(columns: list[str]):
        lower_to_original={str(col).lower():str(col) for col in columns}
        for candidate in ["domain", "node", "host"]:
            if candidate in lower_to_original:
                return lower_to_original[candidate]
        for col in columns:
            if "domain" in str(col).lower():
                return str(col)
        raise ValueError(f"Could not infer domain column from columns={columns}")

    @staticmethod
    def _infer_shard_col(columns: list[str], domain_col: str):
        candidates=["shard", "shard_id", "shard_idx", "chunk", "chunk_id", "file", "parquet_file", "parquet"]
        lower_to_original={str(col).lower():str(col) for col in columns}
        for candidate in candidates:
            if candidate in lower_to_original and lower_to_original[candidate] != domain_col:
                return lower_to_original[candidate]
        for col in columns:
            col_name=str(col)
            if col_name != domain_col:
                return col_name
        raise ValueError(f"Could not infer shard column from columns={columns}")

    @staticmethod
    def _infer_embedding_col(columns: list[str], domain_col: str):
        candidates=["emb", "embedding", "embeddings", "vector", "feat", "features"]
        lower_to_original={str(col).lower():str(col) for col in columns}
        for candidate in candidates:
            if candidate in lower_to_original and lower_to_original[candidate] != domain_col:
                return lower_to_original[candidate]
        for col in columns:
            col_name=str(col)
            if col_name != domain_col:
                return col_name
        raise ValueError(f"Could not infer embedding column from columns={columns}")

    @staticmethod
    def _normalize_shard_ref(shard_ref: Any):
        if shard_ref is None or pd.isna(shard_ref):
            return None
        shard_text=str(shard_ref).strip()
        if shard_text == "":
            return None
        if shard_text.endswith(".parquet"):
            return shard_text
        if shard_text.startswith("shard_"):
            return f"{shard_text}.parquet"
        if shard_text.isdigit():
            return f"shard_{shard_text}.parquet"
        return shard_text

    @staticmethod
    def _load_hf_gnn_embeddings(dataset_repo: str, month: str, gnn_encoder: str, q_domains: list[str], max_memory: str="8GB"):
        if not q_domains:
            raise ValueError("HF GNN loading requires non-empty query domains.")

        hf_token=CrediGrain._get_hf_token()
        gnn_dirs=CrediGrain._candidate_gnn_dirs(month=month, gnn_encoder=gnn_encoder)
        index_candidates=[f"{gnn_dir}/domain_index.parquet" for gnn_dir in gnn_dirs]
        index_local_path,index_rel_path=CrediGrain._download_first_available_with_relpath(dataset_repo, index_candidates, hf_token=hf_token)

        index_df=pd.read_parquet(index_local_path)
        if index_df.empty:
            return {}

        domain_col=CrediGrain._infer_domain_col(index_df.columns.tolist())
        shard_col=CrediGrain._infer_shard_col(index_df.columns.tolist(), domain_col=domain_col)

        domain_to_shard={}
        for _, row in index_df[[domain_col, shard_col]].dropna().iterrows():
            norm_domain=CrediGrain._normalize_domain(row[domain_col])
            shard_name=CrediGrain._normalize_shard_ref(row[shard_col])
            if norm_domain is None or shard_name is None:
                continue
            domain_to_shard[norm_domain]=shard_name

        shard_to_domains={}
        requested_domains={CrediGrain._normalize_domain(domain) for domain in q_domains}
        requested_domains={domain for domain in requested_domains if domain is not None}
        for domain in requested_domains:
            shard_name=None
            for candidate_domain in CrediGrain._domain_reverse_candidates(domain):
                if candidate_domain in domain_to_shard:
                    shard_name=domain_to_shard[candidate_domain]
                    break
            if shard_name is None:
                continue
            shard_to_domains.setdefault(shard_name, set()).update(CrediGrain._domain_reverse_candidates(domain))

        if len(shard_to_domains) == 0:
            return {}

        index_dir=os.path.dirname(index_rel_path)

        def load_single_shard(shard_name: str):
            shard_candidates=[]
            if "/" in shard_name:
                shard_candidates.append(shard_name)
            shard_candidates.append(f"{index_dir}/{shard_name}")
            for gnn_dir in gnn_dirs:
                shard_candidates.append(f"{gnn_dir}/{shard_name}")
            shard_candidates=list(dict.fromkeys(shard_candidates))

            shard_local_path,_=CrediGrain._download_first_available_with_relpath(dataset_repo, shard_candidates, hf_token=hf_token)
            shard_columns=pq.read_schema(shard_local_path).names
            shard_domain_col=CrediGrain._infer_domain_col(shard_columns)
            emb_col=CrediGrain._infer_embedding_col(shard_columns, domain_col=shard_domain_col)
            raw_shard_emb_dict=search_parquet_duckdb(
                shard_local_path,
                col=shard_domain_col,
                q_domains=sorted(shard_to_domains[shard_name]),
                max_memory=max_memory,
                schema={"key": shard_domain_col, "val": emb_col},
            )
            shard_emb_dict={}
            for domain_key, emb_val in raw_shard_emb_dict.items():
                norm_key=CrediGrain._normalize_domain(domain_key)
                if norm_key is None:
                    continue
                shard_emb_dict[norm_key]=emb_val
            return shard_emb_dict

        gnn_emb_dict={}
        shard_names=sorted(shard_to_domains.keys())
        max_workers=min(CrediGrain._io_max_workers(), max(1, len(shard_names)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures={executor.submit(load_single_shard, shard_name): shard_name for shard_name in shard_names}
            for future in as_completed(futures):
                shard_emb_dict=future.result()
                gnn_emb_dict.update(shard_emb_dict)

        return gnn_emb_dict

    @staticmethod
    def _load_month_content_index(dataset_repo: str, month: str, model_name: str):
        hf_token=CrediGrain._get_hf_token()
        month_folder=CrediGrain._month_folder(month)
        index_stem=month_folder
        index_names=[
            f"{index_stem}_wetcontent_domains_index.pkl",
            f"{index_stem}_webcontent_domains_index.pkl",
            f"{index_stem}_content_domains_index.pkl"
        ]

        file_candidates=[]
        for model_dir in CrediGrain._model_dir_candidates(model_name):
            for idx_name in index_names:
                file_candidates.append(f"content_embedding/{month_folder}/{model_dir}/{idx_name}")
                file_candidates.append(f"{month_folder}/{model_dir}/{idx_name}")

        index_file_path=CrediGrain._download_first_available(dataset_repo, file_candidates, hf_token=hf_token)
        with open(index_file_path, "rb") as f:
            index_dict=pickle.load(f)

        normalized_index={}
        for domain,chunk_ref in index_dict.items():
            norm_domain=CrediGrain._normalize_domain(domain)
            if norm_domain is None:
                continue
            normalized_index[norm_domain]=chunk_ref
        return normalized_index

    @staticmethod
    def _chunk_file_candidates(month_folder: str, model_name: str, chunk_ref: Any):
        chunk_ref_str=str(chunk_ref)
        chunk_stem=chunk_ref_str.rsplit(".", 1)[0] if "." in chunk_ref_str else chunk_ref_str
        # Some indexes store extensionless chunk refs; try both raw and common suffixes.
        chunk_variants=[chunk_ref_str]
        if not chunk_ref_str.endswith(".pkl"):
            chunk_variants.append(f"{chunk_stem}.pkl")
        if not chunk_ref_str.endswith(".parquet"):
            chunk_variants.append(f"{chunk_stem}.parquet")

        candidates=[]
        for chunk_name in chunk_variants:
            candidates.append(chunk_name)
        for model_dir in CrediGrain._model_dir_candidates(model_name):
            for chunk_name in chunk_variants:
                candidates.append(f"content_embedding/{month_folder}/{model_dir}/{chunk_name}")
                candidates.append(f"{month_folder}/{model_dir}/{chunk_name}")
        return list(dict.fromkeys(candidates))

    @staticmethod
    def _load_hf_text_chunk(chunk_path: str, chunk_domains: list[str]):
        # CrediBench text chunks may be stored as .pkl (domain -> page embeddings) or parquet.
        if str(chunk_path).lower().endswith(".pkl"):
            with open(chunk_path, "rb") as f:
                raw_chunk_dict=pickle.load(f)

            normalized_chunk={}
            for raw_domain, emb_val in raw_chunk_dict.items():
                norm_domain=CrediGrain._normalize_domain(raw_domain)
                if norm_domain is None:
                    continue
                normalized_chunk[norm_domain]=emb_val

            out={}
            for domain in chunk_domains:
                norm_domain=CrediGrain._normalize_domain(domain)
                if norm_domain is None:
                    continue
                for candidate_domain in CrediGrain._domain_reverse_candidates(norm_domain):
                    if candidate_domain in normalized_chunk:
                        out[norm_domain]=normalized_chunk[candidate_domain]
                        break
            return out

        chunk_embd=search_parquet_duckdb(chunk_path, col="domain", q_domains=chunk_domains, max_memory="8GB", schema={'key':'domain','val':'embeddings'})
        out={}
        for key,val in chunk_embd.items():
            out[CrediGrain._normalize_domain(key)]=val
        return out

    @staticmethod
    def _extract_page_embeddings(raw_page_list: Any, emb_dim: int):
        pairs=[]
        if raw_page_list is None:
            return pairs
        for entry in raw_page_list:
            page_key=""
            vector=None
            if isinstance(entry, dict):
                page_key=str(entry.get("page", ""))
                vector=entry.get("emb")
            elif isinstance(entry, list) and len(entry)>=2 and isinstance(entry[0], str):
                page_key=str(entry[0])
                vector=entry[1]

            if vector is None:
                continue
            vector_arr=np.asarray(vector, dtype=np.float32)
            if vector_arr.ndim!=1 or vector_arr.size==0:
                continue
            pairs.append((page_key, vector_arr[0:emb_dim].tolist()))

        pairs.sort(key=lambda x: x[0])
        return pairs

    @staticmethod
    def _aggregate_domain_pages(raw_domain_pages: dict, emb_dim: int, keep_content: str="all", keep_count: int=3, month_lengths_dict: dict=None):
        aggregated={}
        for domain in sorted(raw_domain_pages.keys()):
            page_pairs=CrediGrain._extract_page_embeddings(raw_domain_pages[domain], emb_dim=emb_dim)
            if not page_pairs:
                continue

            if keep_content!="all":
                if month_lengths_dict is None:
                    raise ValueError("keep_content longest/shortest requires month_lengths_dict.")
                scored=[]
                for page, emb in page_pairs:
                    if page in month_lengths_dict:
                        scored.append((page, emb, month_lengths_dict[page]))
                if not scored:
                    continue
                scored=sorted(scored, key=lambda x: (x[2],x[0]), reverse=(keep_content=="longest"))
                page_pairs=[(page, emb) for page, emb, _ in scored[0:keep_count]]

            fused=page_pairs[0][1]
            for _, emb in page_pairs[1:]:
                fused=fuse_1d_emb(fused, emb, fusion_mode="avg")
            aggregated[domain]=fused[0:emb_dim]
        return aggregated

    @staticmethod
    def _merge_possible_reversed_domains(embd_dict: dict):
        merged={}
        for raw_domain, emb in embd_dict.items():
            norm_domain=CrediGrain._normalize_domain(raw_domain)
            if norm_domain is None:
                continue
            merged[norm_domain]=emb
            rev_domain=CrediGrain._reverse_domain(norm_domain)
            if rev_domain not in merged:
                merged[rev_domain]=emb
        return merged

    @staticmethod
    def _fit_stream_class_maps(df: pd.DataFrame):
        stream_cols={
            "functional_category":"functional_category_stream",
            "cybersecurity":"cybersecurity_stream",
            "epistemic_reliability":"epistemic_reliability_stream",
            "epistemic_reliability.bin":"epistemic_reliability.bin_stream",
        }
        class_maps={}
        for target_name,col_name in stream_cols.items():
            labels=sorted({label for sample_labels in df[col_name].tolist() for label in sample_labels})
            class_maps[target_name]=labels
        return class_maps

    @staticmethod
    def _apply_class_support_cutoff(df: pd.DataFrame, class_maps: dict, min_positive_count: int):
        if min_positive_count <= 0:
            return class_maps

        filtered_maps={}
        stream_cols={
            "functional_category":"functional_category_stream",
            "cybersecurity":"cybersecurity_stream",
        }
        for target_name,col_name in stream_cols.items():
            positive_counts={label:0 for label in class_maps[target_name]}
            for sample_labels in df[col_name]:
                for label in sample_labels:
                    if label in positive_counts:
                        positive_counts[label] += 1
            filtered_maps[target_name]=[
                label for label in class_maps[target_name]
                if positive_counts[label] >= min_positive_count
            ]
            if not filtered_maps[target_name]:
                raise ValueError(
                    f"CrediGrain cutoff={min_positive_count} removes every {target_name} class."
                )

        filtered_maps["epistemic_reliability.bin"]=class_maps["epistemic_reliability.bin"]
        return filtered_maps

    @staticmethod
    def _fit_stream_class_source_maps(df: pd.DataFrame, class_maps: dict):
        stream_cols={
            "functional_category":"functional_category_sourced_stream",
            "cybersecurity":"cybersecurity_sourced_stream",
            "epistemic_reliability":"epistemic_reliability_sourced_stream",
            "epistemic_reliability.bin":"epistemic_reliability.bin_sourced_stream",
        }
        class_source_maps={}
        for target_name in class_maps:
            col_name=stream_cols[target_name]
            sources_by_class={label:set() for label in class_maps[target_name]}
            for annotations in df[col_name].tolist():
                for label,source in annotations:
                    if label in sources_by_class:
                        sources_by_class[label].add(source)
            class_source_maps[target_name]={label:sorted(sources) for label,sources in sources_by_class.items()}
        return class_source_maps

    @staticmethod
    def _validate_stream_classes(df: pd.DataFrame, class_maps: dict, split_name: str):
        stream_cols={
            "functional_category":"functional_category_stream",
            "cybersecurity":"cybersecurity_stream",
            "epistemic_reliability":"epistemic_reliability_stream",
            "epistemic_reliability.bin":"epistemic_reliability.bin_stream",
        }
        unseen_by_target={}
        for target_name in class_maps:
            col_name=stream_cols[target_name]
            observed={label for sample_labels in df[col_name].tolist() for label in sample_labels}
            unseen=sorted(observed-set(class_maps[target_name]))
            unseen_by_target[target_name]=unseen
            if unseen:
                logging.warning(
                    "CrediGrain %s split has labels absent from training for %s: %s. Filtering them from evaluation.",
                    split_name,
                    target_name,
                    unseen,
                )
        return unseen_by_target

    @staticmethod
    def _filter_unseen_stream_classes(df: pd.DataFrame, class_maps: dict):
        filtered=df.copy()
        stream_cols={
            "functional_category":"functional_category_stream",
            "cybersecurity":"cybersecurity_stream",
            "epistemic_reliability":"epistemic_reliability_stream",
            "epistemic_reliability.bin":"epistemic_reliability.bin_stream",
        }
        for target_name in class_maps:
            col_name=stream_cols[target_name]
            allowed=set(class_maps[target_name])
            filtered[col_name]=filtered[col_name].apply(lambda labels: [label for label in labels if label in allowed])
        return filtered

    @staticmethod
    def _encode_multilabel_stream(sample_labels_lst: list[list[str]], classes: list[str]):
        idx_map={label:idx for idx,label in enumerate(classes)}
        encoded=np.zeros((len(sample_labels_lst), len(classes)), dtype=np.float32)
        for row_idx,sample_labels in enumerate(sample_labels_lst):
            for label in sample_labels:
                if label in idx_map:
                    encoded[row_idx, idx_map[label]]=1.0
        return encoded

    @staticmethod
    def _encode_class_availability(sample_annotations: list[list[tuple[str, str]]], classes: list[str], class_source_map: dict):
        encoded=np.zeros((len(sample_annotations), len(classes)), dtype=np.float32)
        for row_idx,annotations in enumerate(sample_annotations):
            row_labels={label for label,_ in annotations}
            row_sources={source for _,source in annotations}
            for class_idx,class_name in enumerate(classes):
                class_sources=set(class_source_map.get(class_name, []))
                if class_name in row_labels or not row_sources.isdisjoint(class_sources):
                    encoded[row_idx,class_idx]=1.0
        return encoded

    @staticmethod
    def _encode_target_streams(df: pd.DataFrame, class_maps: dict, class_source_maps: dict=None):
        if class_source_maps is None:
            class_source_maps=CrediGrain._fit_stream_class_source_maps(df, class_maps)
        y_dict={}
        for target_name,classes in class_maps.items():
            y_dict[target_name]=CrediGrain._encode_multilabel_stream(
                df[f"{target_name}_stream"].tolist(), classes
            )
            y_dict[f"{target_name}_mask"]=CrediGrain._encode_class_availability(
                df[f"{target_name}_sourced_stream"].tolist(), classes, class_source_maps[target_name]
            )
        y_dict.update({
            "epistemic_reliability.cts":df["epistemic_reliability.cts_stream"].astype(float).fillna(np.nan).to_numpy(dtype=np.float32),
            "epistemic_reliability.cts_mask":(~df["epistemic_reliability.cts_stream"].isna()).to_numpy(dtype=np.float32),
            "label_maps":class_maps,
            "label_source_maps":class_source_maps,
        })
        return y_dict

    @staticmethod
    def _split_items(raw_val):
        if pd.isna(raw_val):
            return []
        return [elem.strip() for elem in str(raw_val).split("|") if str(elem).strip()]

    @staticmethod
    def _extract_multilabel(raw_val, prefix:str=None):
        labels=[]
        for item in CrediGrain._split_items(raw_val):
            if prefix is not None and not item.startswith(prefix):
                continue
            if prefix is not None:
                item=item[len(prefix):]
            label=item.split(":",1)[0].strip()
            if "=" in label:
                label=label.split("=",1)[1].strip()
            if label:
                labels.append(label)
        return sorted(list(set(labels)))

    @staticmethod
    def _extract_epistemic_categorical(raw_val):
        labels=[]
        for item in CrediGrain._split_items(raw_val):
            if not item.startswith("type."):
                continue
            label=item.split(":",1)[0].strip()
            if label:
                labels.append(label)
        return sorted(list(set(labels)))

    @staticmethod
    def _extract_reliability_binary(raw_val):
        aliases={
            "credible":"reliable",
            "non_credible":"unreliable",
            "reliable":"reliable",
            "unreliable":"unreliable",
        }
        raw_labels=CrediGrain._extract_multilabel(raw_val, prefix="reliability.bin=")
        unknown=sorted(set(raw_labels)-set(aliases))
        if unknown:
            raise ValueError(f"Unsupported CrediGrain reliability.bin labels: {unknown}")
        return sorted({aliases[label] for label in raw_labels})

    @staticmethod
    def _extract_sourced_labels(raw_val, extractor):
        annotations=[]
        for item in CrediGrain._split_items(raw_val):
            source_parts=item.split(":",1)
            source=source_parts[1].strip() if len(source_parts)>1 else "unknown"
            for label in extractor(item):
                annotations.append((label,source))
        return sorted(set(annotations))

    @staticmethod
    def _validate_epistemic_channel(raw_val, domain: str):
        allowed_prefixes=("reliability.bin=", "reliability.cts=", "type.")
        unknown=[item for item in CrediGrain._split_items(raw_val) if not item.startswith(allowed_prefixes)]
        if unknown:
            raise ValueError(f"CrediGrain domain {domain} has unsupported epistemic_reliability labels: {unknown}")

    @staticmethod
    def _extract_epistemic_cts(raw_val):
        cts_by_source={}
        for item in CrediGrain._split_items(raw_val):
            if not item.startswith("reliability.cts="):
                continue
            payload=item[len("reliability.cts="):]
            payload_parts=payload.split(":",1)
            cts_str=payload_parts[0].strip()
            source=payload_parts[1].strip() if len(payload_parts)>1 else "unknown"
            try:
                # Latest-by-source policy: when a source appears multiple times, keep the last observed value.
                cts_by_source[source]=float(cts_str)
            except (ValueError, TypeError):
                continue
        if len(cts_by_source)==0:
            return np.nan
        return float(np.mean(list(cts_by_source.values())))

    @staticmethod
    def _postprocess_emb_dict(embd_dict: dict, emb_dim: int=256, normalize: bool=False):
        if embd_dict is None or len(embd_dict)==0:
            return {}
        sample_val=embd_dict[list(embd_dict.keys())[0]]
        if sample_val is None or len(sample_val)==0:
            return {}
        if isinstance(sample_val[0], dict): #list of dicts per domain pages (parquet format)
            embd_dict={ k:v[0]['emb'][0:emb_dim] for k,v in embd_dict.items()}
        elif isinstance(sample_val[0], list) and isinstance(sample_val[0][0], str): #list of lists per domain pages (parquet format)
            embd_dict={ k:v[0][1][0:emb_dim] for k,v in embd_dict.items()}
        if normalize:
            embd_dict=normalize_embeddings(embd_dict)
        return embd_dict

    @staticmethod
    def load_agg_Nmonth_emb_dict(
        embed_type: str,
        path: str ="../../../data",
        model_name: str ="embeddinggemma-300m",
        month_lst: list[str]=["dec", "nov", "oct"],
        agg: str ="avg",
        gnn_encoder: str="text",
        normalize: bool=False,
        emb_dim: int=256,
        original_emb_dim: int=256,
        dataset_repo: str=None,
        q_domains: list[str]=None,
        ):
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
                embd_dict=CrediGrain.load_emb_dict(embed_type, path=path, month=month, gnn_encoder=gnn_encoder, emb_dim=emb_dim, normalize=False, dataset_repo=dataset_repo, q_domains=q_domains)
            elif embed_type == "text":
                embd_dict=CrediGrain.load_emb_dict_from_parquet(embed_type, path=path, model_name=model_name, month=month,normalize=False,emb_dim=emb_dim, original_emb_dim=original_emb_dim, dataset_repo=dataset_repo, q_domains=q_domains)
            else:
                embd_dict=CrediGrain.load_emb_dict(embed_type, path=path, model_name=model_name, month=month, emb_dim=emb_dim, normalize=False, gnn_encoder=gnn_encoder)
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
    def load_emb_dict(embed_type: str, path:str="../../../data",pickle_name:str=None, model_name: str="embeddinggemma-300m", month: str="dec", target: str="epistemic_reliability.cts", emb_dim: int=8192,normalize: bool=False,gnn_encoder: str="RNI", dataset_repo: str=None, q_domains: list[str]=None):
        embd_dict=None
        if pickle_name:
            with open(f'{path}/{pickle_name}', 'rb') as f:
                embd_dict = pickle.load(f)
        elif embed_type == "text":
            embd_dict=CrediGrain.load_emb_dict_from_parquet(embed_type, path=path, model_name=model_name, month=month, emb_dim=emb_dim, normalize=normalize, original_emb_dim=emb_dim, dataset_repo=dataset_repo)
            return embd_dict
        elif embed_type == "domainName":
            with open(f'{path}/credigrain_domainName_embeddingQwen3-0.6B_1024.pkl', 'rb') as f:
                embd_dict = pickle.load(f)
        elif embed_type == "GNN_GAT":
            gnn_repo=dataset_repo or os.getenv("CREDIGRAIN_GNN_REPO")
            if gnn_repo:
                embd_dict=CrediGrain._load_hf_gnn_embeddings(gnn_repo, month=month, gnn_encoder=gnn_encoder, q_domains=q_domains)
            elif gnn_encoder=="RNI":
                file_path=f"{path}/gnn_embedding/{gnn_encoder}/{month}_credigrain_gat-RNI_emb.parquet"
                logging.info(f"GNN emb file path={file_path}")
                embd_dict=search_parquet_duckdb(file_path, col=None,q_domains=None,max_memory="8GB",schema={'key':'domain','val':'emb'})
            elif gnn_encoder=="text":
                file_path=f"{path}/gnn_embedding/{gnn_encoder}/{month}_credigrain_from_text_embeddings.parquet"
                logging.info(f"GNN emb file path={file_path}")
                embd_dict=search_parquet_duckdb(file_path, col=None,q_domains=None,max_memory="8GB",schema={'key':'domain','val':'embeddings'})
        elif embed_type == "FQDN":
            fqdn_file_name="credigrain_fqdn_features.pkl"
            with open(f'{path}/{fqdn_file_name}', 'rb') as f:
                embd_dict = pickle.load(f)

        embd_dict=CrediGrain._postprocess_emb_dict(embd_dict, emb_dim=emb_dim, normalize=normalize)
        return embd_dict

    @staticmethod
    def load_emb_dict_from_parquet(embed_type: str, path:str="../../../data", model_name:str="embeddinggemma-300m", month:str="dec", target:str="epistemic_reliability.cts", emb_dim:int=8192,normalize:bool=False,original_emb_dim:int=1024,keep_content:str="all",keep_count:int=3,dataset_repo: str=None, q_domains: list[str]=None):
        embd_dict=None
        if embed_type == "text":
            if dataset_repo is not None:
                domain_index=CrediGrain._load_month_content_index(dataset_repo, month, model_name)
                domain_index=CrediGrain._filter_domain_index_by_queries(domain_index, q_domains=q_domains)
                chunk_to_domains=CrediGrain._group_domains_by_chunk(domain_index)
                embd_dict={}
                hf_token=CrediGrain._get_hf_token()
                month_folder=CrediGrain._month_folder(month)

                def load_single_chunk(chunk_ref: Any, chunk_domains: list[str]):
                    chunk_candidates=CrediGrain._chunk_file_candidates(month_folder, model_name, chunk_ref)
                    chunk_path=CrediGrain._download_first_available(dataset_repo, chunk_candidates, hf_token=hf_token)
                    return CrediGrain._load_hf_text_chunk(chunk_path, chunk_domains)

                if len(chunk_to_domains) > 0:
                    max_workers=min(CrediGrain._io_max_workers(), max(1, len(chunk_to_domains)))
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures=[executor.submit(load_single_chunk, chunk_ref, chunk_domains) for chunk_ref, chunk_domains in chunk_to_domains]
                        for future in as_completed(futures):
                            embd_dict.update(future.result())
            else:
                file_path=f'{path}/credigrain_content_emb_{month}_{model_name}_{original_emb_dim}.parquet'
                embd_dict=search_parquet_duckdb(file_path, col="domain",q_domains=q_domains,max_memory="8GB",schema={'key':'domain','val':'embeddings'})
        elif embed_type == "GNN_GAT":
            with open(f'{path}/{month}_{target}_credigrain_rni_embeddings.pkl', 'rb') as f:
                embd_dict = pickle.load(f)

        if embd_dict is None or len(embd_dict)==0:
            return {}

        month_lengths_dict=None
        if keep_content!="all" and embed_type=="text" and dataset_repo is None:
            month_lengths_dict=pickle.load(open(f'{path}/credigrain_pages_length_dict.pkl', 'rb'))

        if embed_type=="text":
            embd_dict=CrediGrain._aggregate_domain_pages(embd_dict, emb_dim=emb_dim, keep_content=keep_content, keep_count=keep_count, month_lengths_dict=month_lengths_dict)
        elif embed_type!="text" or keep_content=="all":
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
        if normalize:
            embd_dict=normalize_embeddings(embd_dict)
        return embd_dict

    @staticmethod
    def load_splits(dataset_repo:str,split_mode:str="balanced", test_mode:str="credible-non"):
        if dataset_repo is None:
            raise ValueError("Missing args.crediGrain_path for CrediGrain HF dataset repo id")
        if split_mode != CrediGrain.SUPPORTED_SPLIT_MODE:
            raise ValueError(f"CrediGrain only supports split_mode={CrediGrain.SUPPORTED_SPLIT_MODE}; got {split_mode}")
        if test_mode != CrediGrain.SUPPORTED_TEST_MODE:
            raise ValueError(f"CrediGrain only supports test_mode={CrediGrain.SUPPORTED_TEST_MODE}; got {test_mode}")
        hf_token=CrediGrain._get_hf_token()
        split_files={}
        split_roots=[
            f"split/{split_mode}/{test_mode}",
            f"split/{split_mode}",
            "split",
        ]
        for split in ["train","val","test"]:
            candidates=[]
            for root in split_roots:
                candidates.append(f"{root}/{split}/credi-grain.csv")
                candidates.append(f"{root}/{split}.csv")
                candidates.append(f"{root}/{split}_domains.csv")

            # Backward-compatibility fallback for pre-split-folder layouts.
            candidates.extend([
                f"{split}/credi-grain.csv",
                f"{split}.csv",
            ])

            split_files[split]=CrediGrain._download_first_available(dataset_repo, candidates, hf_token=hf_token)

        splits_lst=[]
        for split in ["train","val","test"]:
            split_df=pd.read_csv(split_files[split])
            split_df=CrediGrain._validate_split_contract(split_df, split)
            splits_lst.append(split_df)
        return splits_lst[0],splits_lst[1],splits_lst[2]

    @staticmethod
    def _prepare_target_streams(df: pd.DataFrame):
        df=df.copy()
        for row in df[["domain", "epistemic_reliability"]].itertuples(index=False):
            CrediGrain._validate_epistemic_channel(row.epistemic_reliability, row.domain)
        df["functional_category_stream"]=df["functional_category"].apply(lambda x: CrediGrain._extract_multilabel(x))
        df["cybersecurity_stream"]=df["cybersecurity"].apply(lambda x: CrediGrain._extract_multilabel(x))
        df["epistemic_reliability_stream"]=df["epistemic_reliability"].apply(lambda x: CrediGrain._extract_epistemic_categorical(x))
        df["epistemic_reliability.bin_stream"]=df["epistemic_reliability"].apply(CrediGrain._extract_reliability_binary)
        df["epistemic_reliability.cts_stream"]=df["epistemic_reliability"].apply(lambda x: CrediGrain._extract_epistemic_cts(x))
        df["functional_category_sourced_stream"]=df["functional_category"].apply(
            lambda x: CrediGrain._extract_sourced_labels(x, CrediGrain._extract_multilabel)
        )
        df["cybersecurity_sourced_stream"]=df["cybersecurity"].apply(
            lambda x: CrediGrain._extract_sourced_labels(x, CrediGrain._extract_multilabel)
        )
        df["epistemic_reliability_sourced_stream"]=df["epistemic_reliability"].apply(
            lambda x: CrediGrain._extract_sourced_labels(x, CrediGrain._extract_epistemic_categorical)
        )
        df["epistemic_reliability.bin_sourced_stream"]=df["epistemic_reliability"].apply(
            lambda x: CrediGrain._extract_sourced_labels(x, CrediGrain._extract_reliability_binary)
        )
        return df

    @staticmethod
    def _extract_stream_targets(df: pd.DataFrame):
        return {
            "functional_category":df["functional_category_stream"].tolist(),
            "cybersecurity":df["cybersecurity_stream"].tolist(),
            "epistemic_reliability":df["epistemic_reliability_stream"].tolist(),
            "epistemic_reliability.bin":df["epistemic_reliability.bin_stream"].tolist(),
            "epistemic_reliability.cts":df["epistemic_reliability.cts_stream"].tolist()
        }

    @staticmethod
    def load_run_embeddings(args: dict):
        global agg_months_dict
        t0=time.perf_counter()
        stage_times={}

        crediGrain_text_emb_path=getattr(args,"crediGrain_text_emb_path","../../../data")
        crediGrain_gnn_emb_path=getattr(args,"crediGrain_gnn_emb_path",crediGrain_text_emb_path)
        crediGrain_path=getattr(args,"crediGrain_path",None)
        crediGrain_text_repo=os.getenv("CREDIGRAIN_TEXT_REPO", crediGrain_path)
        crediGrain_gnn_repo=os.getenv("CREDIGRAIN_GNN_REPO")
        split_mode=getattr(args, "split_mode", CrediGrain.SUPPORTED_SPLIT_MODE)
        test_mode=getattr(args, "test_mode", CrediGrain.SUPPORTED_TEST_MODE)
        if crediGrain_path is None:
            raise ValueError("Missing args.crediGrain_path (HuggingFace dataset repository id).")

        split_t0=time.perf_counter()
        train_df,valid_df,test_df=CrediGrain.load_splits(crediGrain_path, split_mode=split_mode, test_mode=test_mode)
        CrediGrain._validate_disjoint_splits(train_df, valid_df, test_df)
        stage_times["split_load_s"]=round(time.perf_counter()-split_t0, 3)
        train_df["split"]="train"
        valid_df["split"]="val"
        test_df["split"]="test"
        credigrain_df = pd.concat([train_df,valid_df,test_df],ignore_index=True)
        credigrain_df=credigrain_df.sort_values(["split", "domain"]).reset_index(drop=True)
        gnn_query_domains=credigrain_df["domain"].tolist()

        split_sig=CrediGrain._split_signature(train_df, valid_df, test_df)
        cache_key=CrediGrain._build_feature_cache_key(args, text_repo=crediGrain_text_repo, gnn_repo=crediGrain_gnn_repo, split_sig=split_sig)
        cache_path=CrediGrain._cache_file_path(cache_key)
        if CrediGrain._feature_cache_enabled() and os.path.exists(cache_path):
            cache_read_t0=time.perf_counter()
            try:
                with open(cache_path, "rb") as f:
                    payload=pickle.load(f)
                stage_times["feature_cache_read_s"]=round(time.perf_counter()-cache_read_t0, 3)
                stage_times["total_loader_s"]=round(time.perf_counter()-t0, 3)
                setattr(args, "_credigrain_loader_stage_times", stage_times)
                logging.info(f"CrediGrain feature cache hit: {cache_path}")
                return payload["X_train"], payload["y_train"], payload["X_valid"], payload["y_valid"], payload["X_test"], payload["y_test"], payload["X_train_feat"], payload["X_valid_feat"], payload["X_test_feat"]
            except Exception as cache_err:
                logging.warning(f"CrediGrain cache read failed for {cache_path}; recomputing. err={cache_err}")
                try:
                    os.remove(cache_path)
                except OSError:
                    pass

        ############## Load text embeddings ###############
        text_t0=time.perf_counter()
        if args.embed_type=="FQDN":
            month_emb_dict =CrediGrain.load_emb_dict(args.embed_type, crediGrain_text_emb_path, pickle_name=None,emb_dim=args.emb_dim,normalize=True)
        else:
            if args.emb_model=="embeddingTE3L":
                args.emb_model="embeddinggemma-300m"
            if args.agg_text_emb:
                month_emb_dict = CrediGrain.load_agg_Nmonth_emb_dict("text", crediGrain_text_emb_path, model_name=args.emb_model, agg=args.agg_function, gnn_encoder=args.gnn_encoder, month_lst=agg_months_dict[args.month], emb_dim=args.emb_dim, original_emb_dim=args.original_emb_dim, dataset_repo=crediGrain_text_repo, q_domains=gnn_query_domains)
            else:
                month_emb_dict = CrediGrain.load_emb_dict_from_parquet(args.embed_type, crediGrain_text_emb_path, args.emb_model, args.month, normalize=False, emb_dim=args.emb_dim, original_emb_dim=args.original_emb_dim, keep_content=args.keep_content, keep_count=args.keep_content_count, dataset_repo=crediGrain_text_repo, q_domains=gnn_query_domains)
        stage_times["text_load_s"]=round(time.perf_counter()-text_t0, 3)

        credigrain_df=CrediGrain._filter_to_embedding_domains(credigrain_df, month_emb_dict, "text")
        credigrain_df=CrediGrain._prepare_target_streams(credigrain_df)

        text_emb_dict=month_emb_dict
        ############### filter by the GNN graph node splits ###################
        gnn_t0=time.perf_counter()
        gnn_query_domains=credigrain_df["domain"].tolist()
        gnn_emb_dict = CrediGrain.load_emb_dict("GNN_GAT", crediGrain_gnn_emb_path,month=args.month,gnn_encoder=args.gnn_encoder, dataset_repo=crediGrain_gnn_repo, q_domains=gnn_query_domains)
        if gnn_emb_dict is not None and len(gnn_emb_dict)>0:
            gnn_emb_dict=CrediGrain._merge_possible_reversed_domains(gnn_emb_dict)

        ################### GNN Embedding #############
        features_emb_dict = None
        if args.use_gnn_emb:
            if args.agg_month_emb:
                gnn_emb_dict = CrediGrain.load_agg_Nmonth_emb_dict("GNN_GAT",crediGrain_gnn_emb_path, agg=args.agg_function,gnn_encoder=args.gnn_encoder,month_lst=agg_months_dict[args.month], dataset_repo=crediGrain_gnn_repo, q_domains=gnn_query_domains)
                gnn_emb_dict=CrediGrain._merge_possible_reversed_domains(gnn_emb_dict)
            elif gnn_emb_dict is None or len(gnn_emb_dict)==0:
                gnn_emb_dict = CrediGrain.load_emb_dict("GNN_GAT", crediGrain_gnn_emb_path,month=args.month,gnn_encoder=args.gnn_encoder, dataset_repo=crediGrain_gnn_repo, q_domains=gnn_query_domains)
                gnn_emb_dict=CrediGrain._merge_possible_reversed_domains(gnn_emb_dict)
            credigrain_df=CrediGrain._filter_to_embedding_domains(credigrain_df, gnn_emb_dict, "gnn")
        else:
            gnn_emb_dict=None
        stage_times["gnn_load_s"]=round(time.perf_counter()-gnn_t0, 3)

        if args.use_FQDN:
            features_emb_dict =CrediGrain.load_emb_dict("FQDN", crediGrain_text_emb_path, pickle_name=None,model_name=None,emb_dim=args.emb_dim,normalize=True)

        train_domains_set=set(train_df["domain"])
        valid_domains_set=set(valid_df["domain"])
        test_domains_set=set(test_df["domain"])

        train_df=credigrain_df[credigrain_df["domain"].isin(train_domains_set)].reset_index(drop=True)
        valid_df=credigrain_df[credigrain_df["domain"].isin(valid_domains_set)].reset_index(drop=True)
        test_df=credigrain_df[credigrain_df["domain"].isin(test_domains_set)].reset_index(drop=True)

        X_train=train_df[["domain","split"]]
        X_valid=valid_df[["domain","split"]]
        X_test=test_df[["domain","split"]]
        class_maps=CrediGrain._fit_stream_class_maps(train_df)
        class_maps=CrediGrain._apply_class_support_cutoff(
            train_df,
            class_maps,
            int(getattr(args, "class_support_cutoff", 0)),
        )
        class_source_maps=CrediGrain._fit_stream_class_source_maps(train_df, class_maps)
        CrediGrain._validate_stream_classes(valid_df, class_maps, "validation")
        CrediGrain._validate_stream_classes(test_df, class_maps, "test")
        valid_df=CrediGrain._filter_unseen_stream_classes(valid_df, class_maps)
        test_df=CrediGrain._filter_unseen_stream_classes(test_df, class_maps)
        y_train=CrediGrain._encode_target_streams(train_df, class_maps, class_source_maps)
        y_valid=CrediGrain._encode_target_streams(valid_df, class_maps, class_source_maps)
        y_test=CrediGrain._encode_target_streams(test_df, class_maps, class_source_maps)

        logging.info(f"len(X_train)={len(X_train)}\tlen(X_valid)={len(X_valid)}\tlen(X_test)={len(X_test)}\t")
        fuse_t0=time.perf_counter()
        X_train_feat, X_valid_feat, X_test_feat = resize_and_fuse_emb(text_emb_dict, "functional_category", X_train, X_valid,X_test, gnn_emb=gnn_emb_dict,topic_emb=features_emb_dict,trim_to=args.emb_dim,fusion_mode=args.fusion_mode)
        stage_times["feature_fusion_s"]=round(time.perf_counter()-fuse_t0, 3)
        logging.info(f"X_train_feat.shape={len(X_train_feat[0]) if len(X_train_feat)>0 and type(X_train_feat[0]) == list else X_train_feat[0].shape if len(X_train_feat)>0 else 0}")

        if CrediGrain._feature_cache_enabled():
            cache_write_t0=time.perf_counter()
            payload={
                "X_train": X_train,
                "y_train": y_train,
                "X_valid": X_valid,
                "y_valid": y_valid,
                "X_test": X_test,
                "y_test": y_test,
                "X_train_feat": X_train_feat,
                "X_valid_feat": X_valid_feat,
                "X_test_feat": X_test_feat,
            }
            CrediGrain._write_cache_atomic(cache_path, payload)
            stage_times["feature_cache_write_s"]=round(time.perf_counter()-cache_write_t0, 3)

        stage_times["total_loader_s"]=round(time.perf_counter()-t0, 3)
        setattr(args, "_credigrain_loader_stage_times", stage_times)
        logging.info(f"CrediGrain loader stage times: {stage_times}")
        return X_train, y_train, X_valid, y_valid, X_test, y_test,X_train_feat, X_valid_feat, X_test_feat

    @staticmethod
    def get_domains_lst(dataset_repo:str):
        train_df,_,_=CrediGrain.load_splits(dataset_repo=dataset_repo)
        return train_df["domain"].tolist()



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
            test_counts=weaklabeles_df[weaklabeles_df["domain"].isin(test_domains_df["domain"])]["weak_label"].value_counts()
            valid_counts=weaklabeles_df[weaklabeles_df["domain"].isin(valid_domains_df["domain"])]["weak_label"].value_counts()
            train_counts=weaklabeles_df[weaklabeles_df["domain"].isin(train_domains_df["domain"])]["weak_label"].value_counts()
            logging.info(f"test set labels count ={test_counts}")
            logging.info(f"valid set labels count ={valid_counts}")
            logging.info(f"train set labels count ={train_counts}")
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
        target_norm_col=f"{args.dqr_target}_norm"
        logging.info(f"test set lables count ={len(filtered_test_df),filtered_test_df[target_norm_col].value_counts()}")

        valid_domains_df=search_parquet_duckdb(f'{args.dqr_path}/splits/val_regression_domains.parquet', col=None,q_domains=None,max_memory="8GB",schema=None)
        valid_domains_df['domain']=valid_domains_df['domain'].apply(lambda x: '.'.join(str(x).split('.')[::-1]))
        filtered_val_df=labeled_11k_df[labeled_11k_df["domain"].isin(valid_domains_df['domain'])]
        logging.info(f"val set lables count ={len(filtered_val_df),filtered_val_df[target_norm_col].value_counts()}")

        train_domains_df=search_parquet_duckdb(f'{args.dqr_path}/splits/train_regression_domains.parquet', col=None,q_domains=None,max_memory="8GB",schema=None)
        train_domains_df['domain']=train_domains_df['domain'].apply(lambda x: '.'.join(str(x).split('.')[::-1]))
        filtered_train_df=labeled_11k_df[labeled_11k_df["domain"].isin(train_domains_df['domain'])]
        logging.info(f"train set lables count ={len(filtered_train_df),filtered_train_df[target_norm_col].value_counts()}")

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
        
