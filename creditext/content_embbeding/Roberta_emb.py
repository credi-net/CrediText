"""
RoBERTa embeddings for texts longer than 512 tokens.
Strategies: truncate | chunks (mean) | sliding window
"""

"""
High-throughput RoBERTa inference optimized for large VRAM GPUs.
Strategies: large batch sizes, torch.compile, fp16, multi-instance.
"""

import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm
import time
import pyarrow.parquet as pq
import pickle
import argparse
from  creditext.experiments.mlp_experiments.utils import write_domain_emb_parquet
class FastRobertaEmbedder:
    MAX_TOKENS = 512

    def __init__(
        self,
        model_name: str = "roberta-base",
        device: str = "cuda",
        precision: str = "fp16",       # "fp32" | "fp16" | "bf16"
        compile_model: bool = True,     # torch.compile (PyTorch 2.0+)
        batch_size: int = 512,          # documents per GPU batch
    ):
        self.device = device
        self.batch_size = batch_size

        dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
        self.dtype = dtype_map[precision]

        # print(f"Loading {model_name} | precision={precision} | compile={compile_model}")
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(
            model_name,
            add_pooling_layer=False,
            torch_dtype=self.dtype,
        ).to(device)
        self.model.eval()

        if compile_model:
            print("Compiling model with torch.compile (first batch will be slow)...")
            self.model = torch.compile(self.model, mode="max-autotune")

    # ── Public API ──────────────────────────────────────────────────────────

    def embed_documents(
        self,
        documents: list[str],
        strategy: str = "sliding_window",
        window_size: int = 512,
        stride: int = 256,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Embed a list of documents at maximum GPU throughput.
        Returns np.ndarray of shape (N, 768).
        """
        # 1. Tokenize all documents upfront (CPU, parallelizable)
        # print("Tokenizing...")
        all_tokens = [
            self.tokenizer.encode(doc, add_special_tokens=False)
            for doc in documents
        ]

        # 2. Expand long documents into chunks
        # print("Building chunks...")
        chunks, doc_indices = self._build_chunks(all_tokens, strategy, window_size, stride)
        # print(f"  {len(documents)} documents → {len(chunks)} chunks")

        # 3. Encode all chunks in large GPU batches
        # print("Running GPU inference...")
        chunk_embeddings = self._encode_chunks_batched(chunks, show_progress)

        # 4. Aggregate chunks back into per-document embeddings
        # print("Aggregating...")
        return self._aggregate(chunk_embeddings, doc_indices, len(documents))

    # ── Chunking ────────────────────────────────────────────────────────────

    def _build_chunks(
        self,
        all_tokens: list[list[int]],
        strategy: str,
        window_size: int,
        stride: int,
    ) -> tuple[list[list[int]], list[int]]:
        """
        Flatten all documents into a list of ≤510-token chunks.
        Returns:
            chunks:      flat list of token lists
            doc_indices: which document each chunk belongs to
        """
        max_content = min(window_size, self.MAX_TOKENS) - 2
        stride = min(stride, max_content)

        chunks = []
        doc_indices = []

        for doc_idx, tokens in enumerate(all_tokens):
            if len(tokens) <= max_content:
                chunks.append(tokens)
                doc_indices.append(doc_idx)
                continue

            if strategy == "truncate":
                chunks.append(tokens[:max_content])
                doc_indices.append(doc_idx)

            elif strategy == "chunks":
                for i in range(0, len(tokens), max_content):
                    chunks.append(tokens[i: i + max_content])
                    doc_indices.append(doc_idx)

            elif strategy == "sliding_window":
                starts = list(range(0, len(tokens), stride))
                if starts[-1] + max_content < len(tokens):
                    starts.append(len(tokens) - max_content)
                for start in starts:
                    chunks.append(tokens[start: start + max_content])
                    doc_indices.append(doc_idx)

        return chunks, doc_indices

    # ── GPU batched inference ────────────────────────────────────────────────

    def _encode_chunks_batched(
        self, chunks: list[list[int]], show_progress: bool
    ) -> np.ndarray:
        """
        Encode all chunks in large batches.
        Pads each batch to the longest sequence in that batch (not global max).
        Returns array of shape (num_chunks, hidden_size).
        """
        all_embeddings = []

        for i in  range(0, len(chunks), self.batch_size):
            batch_tokens = chunks[i: i + self.batch_size]

            # Add special tokens
            batch_ids = [
                [self.tokenizer.bos_token_id] + t + [self.tokenizer.eos_token_id]
                for t in batch_tokens
            ]

            # Pad to longest in THIS batch (not 512) — saves compute
            max_len = max(len(ids) for ids in batch_ids)
            pad_id = self.tokenizer.pad_token_id

            input_ids = torch.tensor(
                [ids + [pad_id] * (max_len - len(ids)) for ids in batch_ids],
                dtype=torch.long,
                device=self.device,
            )
            attention_mask = (input_ids != pad_id).long()

            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=self.dtype):
                out = self.model(input_ids=input_ids, attention_mask=attention_mask)

            # Mean pool (ignore padding and special tokens)
            hidden = out.last_hidden_state          # (B, T, H)
            mask = attention_mask.unsqueeze(-1)      # (B, T, 1)

            # Exclude first (<s>) and last (</s>) tokens from pooling
            content_mask = mask.clone()
            content_mask[:, 0, :] = 0               # mask <s>
            for b_idx, ids in enumerate(batch_ids):
                content_mask[b_idx, len(ids) - 1, :] = 0   # mask </s>

            sum_hidden = (hidden * content_mask).sum(dim=1)
            count = content_mask.sum(dim=1).clamp(min=1e-9)
            embeddings = (sum_hidden / count).float().cpu().numpy()

            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)

    # ── Aggregation ─────────────────────────────────────────────────────────

    def _aggregate(
        self,
        chunk_embeddings: np.ndarray,
        doc_indices: list[int],
        n_docs: int,
    ) -> np.ndarray:
        """Average chunk embeddings back into per-document embeddings."""
        hidden_size = chunk_embeddings.shape[1]
        doc_sum = np.zeros((n_docs, hidden_size), dtype=np.float64)
        doc_count = np.zeros(n_docs, dtype=np.float64)

        for chunk_idx, doc_idx in enumerate(doc_indices):
            doc_sum[doc_idx] += chunk_embeddings[chunk_idx]
            doc_count[doc_idx] += 1

        return (doc_sum / doc_count[:, None]).astype(np.float32)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on a dataset.")
    parser.add_argument("--content_path", type=str, default="/home/mila/a/abdallah/scratch/hsh_projects/CrediText/data/weaksupervision/weaksupervision_content_Dec2024.parquet", help="Path to the JSON dataset")
    parser.add_argument("--start_idx", type=int, default=0, help="start doc idx")
    parser.add_argument("--end_idx", type=int, default=100, help="end doc idx")
    parser.add_argument("--parquet_batch_size", type=int, default=int(1e4), help="parquet_batch_size")
    parser.add_argument("--emb_batch_size", type=int, default=int(1e2), help="emb_batch_size")
    
    
    args = parser.parse_args()
    print(f"args={args}")
    parquet_file = pq.ParquetFile(args.content_path)
    parquet_batch_size=args.parquet_batch_size
    emb_batch_size=args.emb_batch_size
    embedder = FastRobertaEmbedder(
        model_name="roberta-base",
        precision="fp16",
        compile_model=True,
        batch_size=512,
    ) 
    ################ emb Dec Month Content ########################
    print(f"batches_count={1+int(parquet_file.metadata.num_rows /parquet_batch_size)}")
    for parquet_batch_idx, batch in tqdm(enumerate(parquet_file.iter_batches(batch_size=parquet_batch_size)),
                            total=1 + int(parquet_file.metadata.num_rows / parquet_batch_size)):
        print(f"parquet_batch_idx={parquet_batch_idx}")
        if parquet_batch_idx<args.start_idx or parquet_batch_idx>=args.end_idx:
            print(f"skip parquet_batch_idx:{parquet_batch_idx}")
            continue
        
        df_chunk = batch.to_pandas()
        domains_lst = df_chunk["domain"].tolist()
        domains_txt = df_chunk["pages"].tolist()
        embds = {}        
        for emb_batch_idx in tqdm(range(0, len(domains_lst), emb_batch_size)):
            batch_docs_lst,batch_page_url_lst,batch_domains_lst=[],[],[]
            for idx in range(emb_batch_idx,min(emb_batch_idx + emb_batch_size, len(domains_lst))):
                for page_dict in domains_txt[idx]:
                    batch_docs_lst.append(page_dict['wet_record_txt'])
                    batch_page_url_lst.append(page_dict['WARC_Target_URI'])
                    batch_domains_lst.append(domains_lst[idx])           

            # print("\n=== Strategy: truncate ===")
            # batch_emb = embedder.embed_documents(batch_docs_lst, strategy="truncate")

            # print("\n=== Strategy: chunks ===")
            # batch_emb = embedder.embed_documents(batch_docs_lst, strategy="chunks", window_size=512)

            # print("\n=== Strategy: sliding_window ===")
            batch_emb = embedder.embed_documents(batch_docs_lst, strategy="sliding_window", window_size=512, stride=64)            
            for emb_idx in range(len(batch_emb)):
                if batch_domains_lst[emb_idx] not in embds:
                    embds[batch_domains_lst[emb_idx]] = []
                embds[batch_domains_lst[emb_idx]].append([batch_page_url_lst[emb_idx], batch_emb[emb_idx]])
            
        emb_file_name = f'/home/mila/a/abdallah/scratch/hsh_projects/CrediText/data/weaksupervision/weaksupervision_RoBERTa_{parquet_batch_idx}.pkl'
        print(f"emb_file_name={emb_file_name}")
        with open(emb_file_name, 'wb') as emb_file:
            pickle.dump(embds, emb_file)

    ######################Merging all the generated embeddings into one parquet file ########################
    # emb_dict={}
    # for i in range(0,6):
    #     emb_file_name = f'/home/mila/a/abdallah/scratch/hsh_projects/CrediText/data/weaksupervision/weaksupervision_RoBERTa_{i}.pkl'
    #     with open(emb_file_name, 'rb') as emb_file:
    #         embds = pickle.load(emb_file)
    #         emb_dict.update(embds)

    # emb_val_lst=[]
    # emb_key_lst=[]        
    # for k in emb_dict:            
    #     emb_elem_lst=[]
    #     for elem in emb_dict[k]:
    #         if isinstance(elem,list) and len(elem)>1:
    #             emb_elem_lst.append({'page':elem[0],'emb':elem[1]})
    #     if len(emb_elem_lst)>0:
    #         emb_key_lst.append(k) 
    #         emb_val_lst.append(emb_elem_lst)
    # del emb_dict
    # weaksupervision_RoBERTa_dict={'domain':emb_key_lst, 'embeddings':emb_val_lst}
    # write_domain_emb_parquet(weaksupervision_RoBERTa_dict,
    #                          f'/home/mila/a/abdallah/scratch/hsh_projects/CrediText/data/weaksupervision/',
    #                          f"weaksupervision_RoBERTa_dec_emb_{len(emb_val_lst[0][1][1])}.parquet")    

            
     

    