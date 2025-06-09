


# Clothing Matchmaker Assistant using RAG + GPT-4 Vision

This project implements an end-to-end **Clothing Matchmaker Assistant** by combining image ingestion, metadata extraction, vector similarity search, and Retrieval Augmented Generation (RAG) using GPT-4 Vision models. It allows users to upload clothing images and receive intelligent recommendations based on visual features, extracted metadata, and semantic similarity.

---

## Solution Architecture

* **Image Ingestion**
  Uploads and parses images from the dataset folder.

* **Metadata Extraction (Vision Model)**
  Extracts structured metadata from images using GPT-4 Vision models (via OpenAI API).

* **Vector Store Creation**
  Builds an FAISS vector store by embedding the extracted metadata into vector space for efficient similarity search.

* **Semantic Search (RAG Flow)**
  Retrieves similar images using embeddings and generates rich descriptions and recommendations using GPT-4.

* **Interactive Agent**
  A matchmaker agent handles user queries and orchestrates retrieval + generation.

---

## Project Structure

```
clothing_matchmaker_rag/
│
├── data/
│   ├── raw_images/             # Raw input images
│   └── vectorstore/            # FAISS vector index storage
│
├── src/
│   ├── ingest_images.py        # Image ingestion & metadata extraction logic
│   ├── build_vector_store.py   # Builds FAISS vector store from extracted metadata
│   ├── retrieve_matches.py     # Similarity search & RAG-based retrieval
│   ├── matchmaker_agent.py     # Orchestration agent for recommendations
│   ├── utils.py                # Common helper functions
│
├── run_ingestion.py            # Entry point for full ingestion pipeline
├── query_sample1.jpg           # Sample image file for querying
├── .env                        # Environment variables (API Keys, etc.)
└── venv/                       # Virtual environment files
```

---

## Technology Stack

* **Python 3.10.18**
* **OpenAI GPT-4 Vision API**
* **FAISS (Facebook AI Similarity Search)**
* **RAG (Retrieval Augmented Generation)**
* **Langchain (optional for expansion)**
* **VS Code (development environment)**

---

##  Setup Instructions

### Clone Repository

git clone <your-repo-url>
cd clothing_matchmaker_rag


### Setup Virtual Environment


python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows


### Install Dependencies


pip install -r requirements.txt


You may need to manually include packages like `openai`, `faiss-cpu`, `Pillow` etc.

###  Configure Environment

Create `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key
VISION_MODEL=gpt-4o-vision
VISION_PROMPT=your_custom_prompt_if_any
```

### Run Full Ingestion Pipeline


python run_ingestion.py


This will:

* Process images in `data/raw_images/`
* Extract metadata via GPT-4 Vision
* Store metadata & embeddings into FAISS index

### Query Matching Results

Example:


python retrieve_matches.py --query_path data/raw_images/query_sample1.jpg


This will:

* Extract features from query image
* Search similar items from FAISS
* Generate AI-powered recommendations

---

## Key Features

*  Visual similarity powered by embeddings
*  GPT-4 Vision for robust metadata extraction
*  RAG-based semantic retrieval
*  Extensible for e-commerce and fashion catalog systems
*  Modular, easy-to-extend architecture

---

##  Future Enhancements

* Web UI for interactive querying
* Support for multiple clothing categories & styles
* Fine-tuning prompts for more accurate metadata extraction
* Hybrid search: combining image embeddings & text metadata
* Integration with e-commerce product catalog

---

##  Authors

* Developed by **Sudipta Mukhopadhyay** (Tredence Assignment 5)

---








