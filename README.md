## RAG Tutorial – Router Agent + RAG Generator + LLM Judge

This project is an end‑to‑end **Retrieval‑Augmented Generation (RAG)** tutorial that combines:

- **Router Agent** – decides whether a user query should use RAG or be answered directly.
- **RAG Generator** – retrieves relevant chunks from Qdrant and generates answers with Gemini.
- **LLM Judge** – evaluates generated answers against ground‑truth responses.
- **Qdrant Populator** – ingests PDF documents, chunks them, embeds them with Ollama, and stores them in Qdrant.

The main entry point is `services/main.py`, which exposes:

- A **batch mode** for running test queries with evaluation and statistics.
- An **interactive CLI** for ad‑hoc querying and stats.

---

## 1. Project structure

Key files and directories:

- `services/main.py` – orchestration of the full RAG system (Router → RAG/Direct → Judge, CSV logging).
- `services/router_agent.py` – Gemini‑powered routing agent that returns `"rag"` or `"direct"` decisions.
- `services/rag_generator.py` – RAG pipeline (embedding, Qdrant retrieval, answer generation).
- `services/llm_judge.py` – Gemini‑based semantic equivalence judge for responses vs. ground truth.
- `services/embedding_manager.py` – Ollama‑based embedding helper (`nomic-embed-text` by default).
- `services/qdrant_manager.py` – thin wrapper around `qdrant_client` for collection and point management.
- `services/preprocessing.py` – PDF → text chunking utilities.
- `services/populate_qdrant.py` – CLI tool to populate and inspect the Qdrant collection.
- `qdrant_data/` – local Qdrant storage (if you run Qdrant in local/embedded mode or mount here).
- `results/` – generated **CSV** files with run statistics and detailed per‑query logs.

---

## 2. Prerequisites

- **Python**: 3.10+ (recommended)
- **Ollama** installed and running locally, with the `nomic-embed-text` model pulled:

  ```bash
  ollama pull nomic-embed-text
  ```

- **Qdrant** running and accessible at `http://localhost:6333` (default in `qdrant_manager.py`).
  - You can run Qdrant via Docker, for example:

  ```bash
  docker run -p 6333:6333 -v qdrant_data:/qdrant/storage qdrant/qdrant
  ```

- **Google Gemini API key** with access to the specified model (default: `gemini-2.5-flash-lite`).

---

## 3. Installation

1. **Clone the repository** (or open it in Cursor/PyCharm).
2. (Recommended) Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # on Windows
   # or
   source .venv/bin/activate  # on macOS/Linux
   ```

3. **Install dependencies** (example, adapt to your existing `requirements.txt` if present):

   ```bash
   pip install qdrant-client ollama numpy PyPDF2 python-dotenv pydantic-ai google-generativeai
   ```

4. **Set environment variables**:

   Create a `.env` file in the project root with:

   ```env
   GEMINI_API_KEY=your_google_gemini_api_key_here
   ```

---

## 4. Preparing the knowledge base (Qdrant population)

1. Place your PDF files in the `data/` directory (relative to the project root). The default in
   `populate_qdrant.py` is `../data` from within `services/`, which resolves to `<project_root>/data`.

2. Run the Qdrant population CLI:

   ```bash
   cd services
   python populate_qdrant.py
   ```

3. Use the interactive menu:

   - **Option 1** – Populate Qdrant with PDFs (append to existing collection).
   - **Option 2** – Populate Qdrant with PDFs and recreate the collection.
   - **Option 3** – Run sample search queries.
   - **Option 4** – Interactive search mode.
   - **Option 5** – Show collection info.

The populator uses:

- `preprocessing.pdf2chunks` to convert PDFs into overlapping text chunks.
- `embedding_manager.Embedder` (Ollama `nomic-embed-text`) to embed chunks.
- `qdrant_manager.QdrantManager` to store them in the `pdf_documents` collection.

---

## 5. Running the RAG system

The main demo/CLI lives in `services/main.py`.

From the project root:

```bash
cd services
python main.py
```

You will see a menu:

- **1. Process test queries** – runs a predefined set of queries with ground‑truth responses, then:
  - routes each query through the Router Agent,
  - uses RAG or direct generation,
  - evaluates via LLM Judge,
  - prints and **saves statistics + detailed results as CSV** under `results/`.

- **2. Interactive mode** – free‑form loop:
  - type a query to get an answer (with routing and, if applicable, RAG),
  - type `stats` to see aggregated statistics for the session,
  - type `quit` / `exit` / `q` to leave interactive mode.

- **3. Single query test** – process one query with an optional true response, and print the result.

- **4. Exit** – on exit, the system will (if any results were collected):
  - compute statistics,
  - save them to a CSV file,
  - save detailed per‑query results to another CSV file.

---

## 6. CSV outputs

All run logs and statistics are saved in the `results/` directory as timestamped CSV files:

- **Statistics CSV** (`stats_YYYYMMDD_HHMMSS.csv`):
  - flattened metrics such as total queries, number/percentage of RAG vs direct, judgment accuracy,
    and performance metrics (average time, average contexts retrieved).

- **Detailed results CSV** (`results_YYYYMMDD_HHMMSS.csv`):
  - one row per query, including:
    - timestamp, query text, routing decision/reasoning,
    - generated response,
    - number of contexts used,
    - true response (if provided),
    - judge decision, confidence, explanation,
    - elapsed processing time.

These files can be opened directly in Excel, Google Sheets, or any data analysis tool.

---

