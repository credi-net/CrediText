from llama_cpp import Llama

# llm = Llama.from_pretrained(
# 	model_path="unsloth/embeddinggemma-2-GGUF",
# 	filename="embeddinggemma-2-UD-Q4_K_XL.gguf")

llm = Llama(
	model_path="/home/mila/a/abdallah/scratch/hsh_projects/content_emb/embeddinggemma-2-UD-Q4_K_XL.gguf",
    # n_gpu_layers=-1, # Uncomment to use GPU acceleration
    seed=1337, # Uncomment to set a specific seed
    n_ctx=8192, # Uncomment to increase the context window
)
c=llm.create_chat_completion(
	messages = "\"Today is a sunny day and I will get some ice cream.\""
)
print(c)