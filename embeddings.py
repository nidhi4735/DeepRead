# ---------------------------------------------------------
# 🧮 Embeddings Generator - HuggingFace se vector banata hai
# ---------------------------------------------------------
import numpy as np
from huggingface_hub import InferenceClient
from sklearn.metrics.pairwise import cosine_similarity
from setup.config import HF_TOKEN, EMBED_MODEL

# client global hi bana lo ek baar
client = InferenceClient(EMBED_MODEL, token=HF_TOKEN)

def get_embeddings(texts):
    print("🔢 Generating embeddings...")
    embeddings = [np.array(client.feature_extraction(t)[0]) for t in texts]
    print(f"✅ Created {len(embeddings)} embeddings.")
    return np.vstack(embeddings)

def find_similar(query, texts, embeddings, top_k=3):
    query_vec = np.array(client.feature_extraction(query)[0]).reshape(1, -1)
    sims = cosine_similarity(query_vec, embeddings)[0]
    top_indices = np.argsort(sims)[::-1][:top_k]
    return [(sims[i], texts[i]) for i in top_indices]
