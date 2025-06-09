import os
from src import ingest_images, build_vector_store

def main():
    # Paths
    input_dir = "data/raw_images"
    output_dir = "data/extracted_metadata"
    vectorstore_path = "data/vectorstore/faiss.index"

    # Step - Extract metadata
    print("🚀 Starting metadata extraction...")
    ingest_images.process_folder(input_dir, output_dir)
    print("Metadata extraction complete.")

    # Step  - Load metadata
    docs, files = build_vector_store.load_metadata(output_dir)
    print(f"Total metadata files loaded: {len(docs)}")

    # Defensive check for empty metadata
    if not docs:
        print("No valid metadata found. Aborting ingestion.")
        return

    # Step - Build embeddings
    print("Building embeddings...")
    embeddings = build_vector_store.build_embeddings(docs)

    # Defensive check for empty embeddings
    if not embeddings:
        print("No embeddings generated. Aborting ingestion.")
        return

    # Step - Build FAISS index
    print("Building FAISS index...")
    index = build_vector_store.build_faiss_index(embeddings)

    # Step - Save FAISS index
    os.makedirs(os.path.dirname(vectorstore_path), exist_ok=True)
    build_vector_store.save_faiss(index, vectorstore_path)
    print("FAISS index saved successfully.")

    print("Ingestion pipeline completed successfully!")

if __name__ == "__main__":
    main()

