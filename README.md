# LLM Project (FAISS + Ollama)

---

## Project Overview

This project implements a **local Retrieval-Augmented Generation (RAG) system** for querying structured information extracted from multiple PDF documents (e.g., aluminum alloy datasheets).

The system extracts table data from PDFs, converts each row into semantic embeddings using a pretrained sentence transformer, and builds a FAISS vector index for fast similarity search. A local LLM (Ollama) is then used to generate natural language answers based on retrieved table context.

This project demonstrates a complete **end-to-end AI pipeline**:
PDF parsing → data structuring → vector database → semantic retrieval → LLM-based question answering.



## File Description

### extract_tables.py
Preprocessing script for extracting table data from PDFs.

- Reads multiple PDF files from the dataset folder  
- Extracts tabular data using Camelot  
- Converts table rows into structured format  
- Saves processed data into `output/table_rows.pkl`  



### build_table_index.py
Builds the vector database for semantic search.

- Loads extracted table rows from `table_rows.pkl`  
- Uses SentenceTransformer (`all-MiniLM-L6-v2`) to generate embeddings  
- Builds FAISS index for fast similarity search  
- Saves index as `output/table_index.faiss`  



### ask_table.py
Main interactive question-answering system.

- Loads FAISS index and table data  
- Converts user queries into embeddings  
- Retrieves top-k most relevant table rows  
- Sends retrieved context to Ollama LLM  
- Generates final natural language answer  
- Supports interactive CLI chat mode  



## Features

- PDF table extraction and preprocessing  
- Semantic embedding using SentenceTransformer  
- Fast similarity search using FAISS  
- Local LLM inference using Ollama (no API required)  
- Retrieval-Augmented Generation (RAG) pipeline  
- Fully offline and privacy-preserving system  
- Interactive command-line question answering  



## Usage Instructions

### 1. Install dependencies
```bash
pip install numpy faiss-cpu sentence-transformers ollama camelot-py[cv] pandas
