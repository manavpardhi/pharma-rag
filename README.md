# 💊 Pharma-Aware RAG Pipeline

> [!IMPORTANT]
> **API Key Required**: This project requires an `OPENAI_API_KEY` for both Summary Generation (GPT-4o-mini) and Retrieval (Embeddings).

A high-precision Retrieval-Augmented Generation (RAG) pipeline specifically engineered for pharmaceutical data. This system transforms flat, unstructured medical PDFs/JSONs into a compliance-grade knowledge base by preserving medical context.

## 🚀 The Core Engineering "Twists"

Standard RAG often fails in medical contexts because character-based chunking separates crucial warnings from their drug subjects. This pipeline solves that using:

1.  **Semantic Chunking**: Instead of character counts, we split text by **Medical Sections** (e.g., ADVERSE REACTIONS, CONTRAINDICATIONS).
2.  **Parent-Child Retrieval (Summary Vectors)**:
    -   We generate concise, strict summaries for every chunk.
    -   We **Index the Summary** for high-precision semantic search (avoiding "concept bleeding" between sections).
    -   We **Retrieve the Raw Content** from metadata to ensure the LLM generates answers based on the full clinical text.

## 🛠️ Tech Stack

-   **Data**: OpenFDA API (Drug Labels)
-   **Vector DB**: ChromaDB (with OpenAI Embeddings)
-   **Orchestration**: Python (Strictly raw implementation, no generic frameworks like LangChain)
-   **LLM**: OpenAI (GPT-4o-mini)
-   **UI**: Gradio
-   **Environment**: `uv` for lightning-fast dependency management

## 📂 Project Structure

```text
rx-research/
├── data/
│   ├── raw/             # Downloaded OpenFDA JSONs
│   ├── processed/       # Extracted medical chunks
│   └── chroma_db/       # Persistent Vector Database
├── src/
│   ├── ingestion/       # data_loader.py, ingest.py
│   ├── processing/      # chunker.py, summarizer.py
│   ├── retrieval/       # vector_store.py
│   └── app.py           # Gradio Interface
├── benchmark.py         # 100% Accuracy Validation script
└── pyproject.toml       # Managed by uv
```

## ⚙️ Setup & Installation

1.  **Install uv** (if not already installed):
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Environment Setup**:
    ```bash
    uv sync
    ```

3.  **API Keys**:
    Create a `.env` file in the root directory:
    ```text
    OPENAI_API_KEY=sk-your-key-here
    ```

## 🏃 Usage

### 1. Data Ingestion
Download recent drug labels and build the vector index with summary vectors:
```bash
# Optional: Clear old DB
rm -rf data/chroma_db

# Run full ingestion pipeline
uv run python -m src.ingestion.ingest
```

### 2. Validation (The "Show-Off" Part)
Verify retrieval accuracy against 5 "hard" pharmacist questions:
```bash
uv run python -m benchmark
```
*Expected: 100% (5/5) Passing.*

### 3. Launch the UI
Start the interactive pharmacist assistant:
```bash
uv run python -m src.app
```
Access the UI at `http://localhost:7861`.

## 🧪 Ground Truth Validation
Current benchmark questions cover:
- Detailed Dosing for Mekinist
- Differentiating Indications vs. Contraindications
- Boxed Warnings for Naproxen
- Specific allergen contraindications (Sulfonamides)
- Cardiovascular event incidence in Varenicline trials

---
*Disclaimer: Developed as a Compliance-Grade Engineering Demo. Always verify medical information with professional sources.*
