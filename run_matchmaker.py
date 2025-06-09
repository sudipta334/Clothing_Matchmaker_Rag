from src import ingest_images, retrieve_matches, matchmaker_agent, build_vector_store
import json
import os

def main():
    # STEP 1: Provide your query image path
    query_image_path = "data/raw_images/query_sample.jpg"

    # STEP 2: Extract metadata from query image
    print("Extracting metadata for query image...")
    query_metadata = ingest_images.extract_metadata(query_image_path)
    print("Metadata extracted:", query_metadata)

    # STEP 3: Convert metadata into text string for embedding
    query_text = " ".join(f"{k}: {v}" for k, v in query_metadata.items())

    # STEP 4: Retrieve similar items
    print("Retrieving similar items...")
    retrieved_indices = retrieve_matches.retrieve_similar(query_text, k=5)
    filenames = retrieve_matches.load_filenames("data/extracted_metadata")

    retrieved_items_metadata = []
    for idx in retrieved_indices:
        fname = filenames[idx]
        with open(os.path.join("data/extracted_metadata", fname)) as f:
            metadata = json.load(f)
        retrieved_items_metadata.append(metadata)

    # STEP 5: Pass to GPT-4o RAG reasoning engine
    print("Generating match recommendations...")
    result = matchmaker_agent.matchmaker_agent(query_metadata, retrieved_items_metadata)

    print("\n GPT-4o Recommendations")
    print(result)

if __name__ == "__main__":
    main()
