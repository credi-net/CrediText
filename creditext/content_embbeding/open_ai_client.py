from openai import OpenAI
def chat():
    print("Create OpenAI client pointing to llama.cpp")
    client = OpenAI(
        base_url="http://0.0.0.0:8080/v1",
        api_key="dummy"   # llama.cpp ignores the key but OpenAI client requires one
    )
    print("Client Created")
    response = client.chat.completions.create(
        model="local-model",  # name is ignored by llama.cpp
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain what tokenization means in NLP."}
        ],
        temperature=0.7,
        max_tokens=256
    )
    print("reponse returned")
    print(f"response={response.choices[0].message}")
def text_emb():
        print("Create OpenAI text embedding via llama.cpp")
        client = OpenAI(
            base_url="http://0.0.0.0:8080/v1",
            api_key="llama-cpp"  # dummy value; not validated by llama.cpp
        )

        response = client.embeddings.create(
            model="local-embedding-model",
            input=["Graph neural networks enable relational inductive bias in structured data. 1",
                   "Graph neural networks enable relational inductive bias in structured data. 2",
                   "Graph neural networks enable relational inductive bias in structured data. 3"]
        )
        for emb in response.data:
            embedding = emb.embedding
            # print(f"Embedding dimension: {len(embedding)}\t embedding={embedding}")
            print(f"Embedding dimension: {len(embedding)}")

text_emb()