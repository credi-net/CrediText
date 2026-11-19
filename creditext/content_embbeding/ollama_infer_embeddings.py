import ollama
# MODEL = "qwen3:8b"
texts = [
    "Graph neural networks for knowledge graphs",
    "LLM-based retrieval augmented generation",
    "Scalable inference on heterogeneous GPUs"
]
response = ollama.embed(model="hf.co/Qwen/Qwen3-Embedding-8B-GGUF:Q4_K_M",input=texts)
embeddings = response["embedding"]
print(f"Generated {len(embeddings)} embeddings")
print(f"Embedding dimension: {len(embeddings[0])}")
