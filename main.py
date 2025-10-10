import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from urllib3.exceptions import NotOpenSSLWarning
import gc

# ---- SAFETY SETTINGS ----
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

INDEX_FILE = "reviews_hnsw.index"
META_FILE = "reviews_metadata.json"
DATA_FILE = "/Users/in45860399/Downloads/reviews_data.json" 

embedder = SentenceTransformer('all-MiniLM-L6-v2')

def preprocess_text(entry):
    return f"Restaurant: {entry['business_name']} ({entry['stars']}★). Review: {entry['review']}"

# ---- Load review data from file ----
def load_reviews(file_path):
    reviews = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            reviews.append({
                "business_name": obj.get("business_name", ""),
                "categories": obj.get("categories", ""),
                "stars": float(obj.get("stars", 0)),
                "review": obj.get("review", "")
            })
    return reviews

reviews = load_reviews(DATA_FILE)

# ---- Build or Load FAISS Index ----
if os.path.exists(INDEX_FILE) and os.path.exists(META_FILE):
    print(f"📂 Loading FAISS index from '{INDEX_FILE}'")
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "r") as f:
        id_to_review = json.load(f)
    texts = [preprocess_text(id_to_review[str(i)]) for i in range(len(id_to_review))]
    embeddings = np.array(embedder.encode(texts)).astype('float32')

else:
    print("🛠 Building FAISS HNSW index from scratch...")
    texts_for_embedding = [preprocess_text(r) for r in reviews]
    embeddings = np.array(embedder.encode(texts_for_embedding)).astype('float32')

    dim = embeddings.shape[1]
    M = 32
    index = faiss.IndexHNSWFlat(dim, M)
    index.hnsw.efConstruction = 64
    index.hnsw.efSearch = 64
    index.add(embeddings)
    print(f"✅ Added {index.ntotal} vectors")

    faiss.write_index(index, INDEX_FILE)
    print(f"💾 Index saved as '{INDEX_FILE}'")

    id_to_review = {str(i): reviews[i] for i in range(len(reviews))}
    with open(META_FILE, "w") as f:
        json.dump(id_to_review, f)
    print(f"💾 Metadata saved as '{META_FILE}'")

# ---- Filtered search function ----
def search_with_filter(query_text, category_filter=None, min_stars=None, top_k=3, top_n_print=10):
    query_embedding = np.array(embedder.encode([query_text])).astype('float32')

    # Apply filters
    filtered_ids = list(range(len(id_to_review)))

    if category_filter:
        filtered_ids = [
            i for i in filtered_ids
            if category_filter.lower() in id_to_review[str(i)]["categories"].lower()
        ]

    if min_stars is not None:
        filtered_ids = [
            i for i in filtered_ids
            if id_to_review[str(i)]["stars"] >= min_stars
        ]

    if not filtered_ids:
        print(f"⚠ No reviews found for category '{category_filter}' with stars ≥ {min_stars}")
        return [], None

    # Cosine similarity for filtered subset
    filtered_embeddings = embeddings[filtered_ids, :]
    cosine_scores = cosine_similarity(query_embedding, filtered_embeddings)[0]

    # Sort scores and get top 10 highest matches
    top_n = 10
    top_indices = np.argsort(cosine_scores)[::-1][:top_n]
    top_labels = [
        f"R{filtered_ids[idx]} ({id_to_review[str(filtered_ids[idx])]['business_name']} - {id_to_review[str(filtered_ids[idx])]['stars']}★)"
        for idx in top_indices
    ]
    top_scores = [cosine_scores[idx] for idx in top_indices]

    df_query_sim = pd.DataFrame([top_scores], index=["Query"], columns=top_labels)

    # Print DataFrame
    print(f"\n📊 Top {top_n_print} cosine similarity matches "
          f"(Category: {category_filter or 'all'}, Stars ≥ {min_stars or 'any'})")
    print(df_query_sim.round(3))

   # Step 6: Plot clean heatmap
    plt.figure(figsize=(1.5 * top_n_print, 2))
    sns.heatmap(df_query_sim, annot=True, cmap="YlGnBu", fmt=".2f", cbar=False)
    plt.title(f"Top {top_n_print} Similarity Matches\n(Category: {category_filter or 'all'}, Stars ≥ {min_stars or 'any'})")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # Create a temporary FAISS index with the filtered subset
    temp_index = faiss.IndexHNSWFlat(filtered_embeddings.shape[1], 32)
    temp_index.add(filtered_embeddings)

    D, I = temp_index.search(query_embedding, top_k)
    matched_texts = [
        preprocess_text(id_to_review[str(filtered_ids[idx])]) for idx in I[0]
    ]

    return matched_texts, df_query_sim

# ---- Interactive Loop ----
print("\n🔎 Ask questions about restaurant reviews from file!")
print("You can filter by category keyword (e.g. Vegan, Italian, Seafood) and minimum star rating.")
print("Type 'exit' to stop.\n")

while True:
    query = input("📝 Your question: ")
    if query.lower() in ["exit", "quit"]:
        print("👋 Exiting.")
        break

    category = input("📂 Category filter keyword (leave empty for all): ").strip()
    try:
        stars_input = input("⭐ Minimum stars (leave empty for any): ").strip()
        min_stars = float(stars_input) if stars_input else None
    except ValueError:
        min_stars = None

    matched_reviews, df_sim = search_with_filter(
        query,
        category_filter=category or None,
        min_stars=min_stars
    )

    if matched_reviews:
        client = OpenAI(
            base_url="http://localhost:12434/engines/llama.cpp/v1",
            api_key="test-key"
        )
        context = "\n".join(matched_reviews)
        prompt = f"""You are a helpful assistant that answers questions using restaurant reviews.

        Question: {query}
        Relevant reviews (category: {category or 'all'}, stars ≥ {min_stars or 'any'}):
        {context}

        Answer:"""

        completion = client.chat.completions.create(
            model="ai/qwen3:0.6B-Q4_0",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        print("\n💬 Answer:", completion.choices[0].message.content)
        print("\n" + "-"*50 + "\n")

# ---- Cleanup ----
del index
del embeddings
gc.collect()