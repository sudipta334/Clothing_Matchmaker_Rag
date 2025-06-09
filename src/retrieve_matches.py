import numpy as np
import faiss
from src.utils import client, embedding_model
import json, os

def embed_query(text):
    resp = client.embeddings.create(model=embedding_model, input=[text])
    return np.array(resp.data[0].embedding).astype('float32')

def retrieve_similar(query_text, k=5):
    query_vec = embed_query(query_text)
    index = faiss.read_index("data/vectorstore/faiss.index")
    D, I = index.search(np.array([query_vec]), k)
    return I[0]

# Load filenames to map index back
def load_filenames(metadata_dir):
    return sorted(os.listdir(metadata_dir))

# Usage:
# retrieved_indices = retrieve_similar("type: Shirt color: Blue style: Casual", 5)
# filenames = load_filenames("data/extracted_metadata")
# for i in retrieved_indices:
#     print(filenames[i])




