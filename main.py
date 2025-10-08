from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

import os
import warnings
from openai import OpenAI
from urllib3.exceptions import NotOpenSSLWarning

warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Step 1: Load your reviews
reviews = [
    {"id": 1, "restaurant": "Luigi's", "text": "The pasta was amazing but service was slow."},
    {"id": 2, "restaurant": "Luigi's", "text": "Terrible service, long wait time, but the pizza was good."},
    {"id": 3, "restaurant": "Green Bites", "text": "Great vegan options and friendly staff."},
    {"id": 4, "restaurant": "Green Bites", "text": "Healthy bowls, affordable price, quick service."},
    {"id": 5, "restaurant": "Ocean Grill", "text": "Fresh seafood, but a bit overpriced."},
    {"id": 6, "restaurant": "Ocean Grill", "text": "Fantastic oysters, loved the view of the sea."}
]

# Step 2: Create embeddings locally using sentence-transformers
embedder = SentenceTransformer('all-MiniLM-L6-v2')


def preprocess_text(r):
    return f"Restaurant: {r['restaurant']}. Review: {r['text']}"

texts_for_embedding = [preprocess_text(r) for r in reviews]
embeddings = np.array(embedder.encode(texts_for_embedding)).astype('float32')
#print("Embeddings shape:", embeddings.shape)
dim = embeddings.shape[1]


# Step 3: Store in FAISS
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

print("🔎 Ask questions about restaurant reviews!")
print("Type 'exit' or 'quit' to stop.\n")

while True:
    query = input("📝 Your question: ")
    if query.lower() in ["exit", "quit"]:
        print("👋 Exiting. Thank you!")
        break

    # Step 4: Query embedding and retrieval
    #query = "What do people say about the service at Green Bites?"
    #query = "Please suggest veggan option restaurant?"
    query_embedding=np.array(embedder.encode([query])).astype('float32')
    #print("Embedding Query shape:", query_embedding.shape)
    D, I = index.search(query_embedding, k=3)
    retrieved = [preprocess_text(reviews[i]) for i in I[0]]

    # Step 5: Use local LLM via Docker
    # OpenAI-compatible local API setup
    client = OpenAI(
    base_url="http://localhost:12434/engines/llama.cpp/v1",
    api_key="test-key"
    )

    # Build prompt
    context = "\n".join(retrieved)
    prompt = f"""You are a helpful assistant that answers questions using restaurant reviews.

    Question: {query}

    Relevant reviews:
    {context}

    Answer:"""

    # Call local LLM for completion
    completion = client.chat.completions.create(
    model="ai/qwen3:0.6B-Q4_0",  # or whatever model name your server exposes
    messages=[{"role": "user", "content": prompt}],
    temperature=0.3
    )



    # Output
    print("💬 Answer:", completion.choices[0].message.content)
    print("\n" + "-"*50 + "\n")


