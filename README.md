# PDF Table RAG: High-Precision Table Extraction & QA System

## Project Overview
This project is a RAG (Retrieval-Augmented Generation) system specifically engineered for handling complex tables within PDF documents.

By implementing a robust "Extraction - Indexing - Retrieval - Generation" pipeline, the system addresses common challenges in traditional RAG, such as row-column misalignment, missing headers, and merged cell interference, ensuring high-accuracy parsing and questioning of PDF table data.

## File Description

### 1. extract_tables.py
The core extraction and cleaning script responsible for converting raw PDF tables into structured text records.

- **Multi-Mode Extraction**: Integrates Camelot (Lattice & Stream) and pdfplumber extraction modes.
- **Structural Completion**: Implements `ffill` (forward-fill) logic to handle merged cells.
- **Semantic Enhancement**: Within the `row_to_text` process, each row of data is explicitly aligned with its corresponding headers to generate context-aware text chunks.
- **Quality Logging**: Records the `parse_score` metric provided by the extraction engine into `documents.jsonl` to measure extraction confidence.

### 2. build_table_index.py
The indexing script responsible for vectorizing and storing the extracted text.

- **Noise Filtering**: Automatically filters out empty lines, blank strings, and distracting "garbage" lines before building the index.
- **Retrieval Optimization**: Utilizes the `all-MiniLM-L6-v2` model with `normalize_embeddings` enabled to improve the stability of cosine similarity search.
- **Efficient Retrieval**: Constructs a vector library based on FAISS `IndexFlatIP` (Inner Product index), which is better suited for text semantic matching than traditional L2 distance.

### 3. ask_table.py
The inference script for context retrieval and LLM response generation.

- **Semantic Retrieval**: Retrieves the Top-K most relevant table rows (Row Chunks) from the FAISS index based on the user's query.
- **Model Upgrade**: Upgraded from the original 1.5B model to `qwen2.5:7b`, significantly enhancing the understanding of table numerical logic.
- **Strict Grounding**: Uses a prompt that mandates the model answer ONLY based on the provided `Table Data`; if no answer is found, it must respond with "I don't know."

## Key Issues Solved

1.  **Header-Row "Decoupling"**:
    In `extract_tables.py`, by forcibly combining each row with its headers, the system solves the issue where the AI finds a data row but cannot identify field meanings (e.g., distinguishing "Purity" from "Price").
2.  **"Noise & Horizontal Merging" Interference**:
    Uses the `fill_merged_cells` algorithm to fill merged cells and cleans redundant empty delimiters, preventing large amounts of blank characters from interfering with embedding similarity calculations.
3.  **Enhanced Embedding Numerical Sensitivity**:
    Achieved through `all-MiniLM-L6-v2` combined with normalization. **Note: The project is currently in the accuracy verification phase**, utilizing the `parse_score` recorded in the code to evaluate extraction quality and refine retrieval precision.
4.  **Limited Reasoning in Small Models**:
    Replaced the generation model with `qwen2.5:7b` and redesigned the restrictive prompt to eliminate "hallucinated answers" caused by irrelevant lines (like directories or page numbers) in the context.

## Usage Instructions

### 1. Install Dependencies
```bash
pip install camelot-py[cv] pdfplumber pandas faiss-cpu sentence-transformers ollama
