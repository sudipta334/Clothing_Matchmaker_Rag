import os, json
import faiss
import numpy as np
from src.utils import client, embedding_model

def load_metadata(metadata_dir):
    all_docs = []
    filenames = []
    for file in os.listdir(metadata_dir):
        with open(os.path.join(metadata_dir, file), "r") as f:
            data = json.load(f)
            doc = " ".join(f"{k}: {v}" for k, v in data.items())
            all_docs.append(doc)
            filenames.append(file)
    return all_docs, filenames

def build_embeddings(texts):
    embeddings = []
    for text in texts:
        resp = client.embeddings.create(
    model=embedding_model,
    input=[text]
)
        vector = resp.data[0].embedding
        embeddings.append(vector)
    return embeddings

def build_faiss_index(embeddings):
    if not embeddings:
        raise ValueError("No embeddings provided to build FAISS index. Please verify ingestion.")
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype('float32'))
    return index


def save_faiss(index, path):
    faiss.write_index(index, path)

# Usage:
# docs, files = load_metadata("data/extracted_metadata")
# embeddings = build_embeddings(docs)
# index = build_faiss_index(embeddings)
# save_faiss(index, "data/vectorstore/faiss.index")




