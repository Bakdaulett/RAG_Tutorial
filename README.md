# RAG Tutorial – Router Agent + RAG Generator + LLM Judge

A complete **Retrieval-Augmented Generation (RAG)** system implementation with intelligent routing, document retrieval, and automated evaluation.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Results & Accuracy Reporting](#results--accuracy-reporting)
- [Statistics](#statistics)
- [Troubleshooting](#troubleshooting)

---

## Overview

This project implements a production-ready RAG system with three main components:

1. **Router Agent** (Gemini) – Intelligently decides whether a query requires RAG retrieval or can be answered directly
2. **RAG Generator** (Gemini + Qdrant) – Retrieves relevant document chunks and generates context-aware answers
3. **LLM Judge** (Ollama) – Evaluates answer quality by comparing generated responses to ground-truth answers

### Key Features

- ✅ **Intelligent Routing** – Automatically determines when to use RAG vs. direct LLM responses
- ✅ **Document Retrieval** – Semantic search over PDF documents using Qdrant vector database
- ✅ **Automated Evaluation** – LLM-as-judge assessment with accuracy metrics
- ✅ **API Key Rotation** – Handles quota limits with automatic key/model switching
- ✅ **Checkpoint & Resume** – Saves progress and can resume interrupted evaluations
- ✅ **Comprehensive Reporting** – CSV statistics and TXT accuracy summaries

---

## Architecture

```
User Query
    ↓
Router Agent (Gemini)
    ├─→ Decision: "rag" → RAG Generator (Gemini + Qdrant)
    └─→ Decision: "direct" → Direct LLM (Gemini)
    ↓
Generated Answer
    ↓
LLM Judge (Ollama) ← Ground Truth Answer
    ↓
Evaluation Results (Accuracy, Confidence, Explanation)
```

### Component Details

- **Router Agent** (`services/router_agent.py`)
  - Uses Gemini to analyze query intent
  - Returns routing decision (`"rag"` or `"direct"`) with reasoning

- **RAG Generator** (`services/rag_generator.py`)
  - Embeds queries using Ollama (`nomic-embed-text`)
  - Retrieves top-k relevant chunks from Qdrant
  - Generates answers using Gemini with retrieved contexts

- **LLM Judge** (`services/llm_judge.py`)
  - Uses Ollama (e.g., `llama3.1:8b-instruct`) for semantic comparison
  - Evaluates if generated answer matches ground truth
  - Returns judgment (True/False), confidence score, and explanation

---

## Prerequisites

- **Python**: 3.10 or higher
- **Ollama**: Installed and running locally
  ```bash
  ollama pull nomic-embed-text
  ollama pull llama3.1:8b-instruct-q4_0  # Recommended for judge
  ```
- **Qdrant**: Running on `http://localhost:6333`
  ```bash
  docker run -p 6333:6333 -v qdrant_data:/qdrant/storage qdrant/qdrant
  ```
- **Google Gemini API Key**: Free tier supports 20 requests/day per model

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd RAG_Tutorial
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # or
   source .venv/bin/activate  # macOS/Linux
   ```

3. **Install dependencies**:
   ```bash
   pip install qdrant-client ollama numpy PyPDF2 python-dotenv pydantic-ai google-generativeai pandas openpyxl
   ```

4. **Configure environment variables**:
   
   Create `.env` file in project root:
   ```env
   # Main Gemini API key (for interactive chat)
   GEMINI_API_KEY=your_main_api_key_here
   PDF_DOCUMENTS=pdf_documents
   GEMINI_MODEL_NAME=gemini-2.5-flash-lite
   
   # Multiple API keys for evaluation (optional)
   GEMINI_API_KEY_1=your_key_1
   GEMINI_MODELS_1=gemini-2.5-flash-lite,gemini-2.5-flash
   GEMINI_API_KEY_2=your_key_2
   GEMINI_MODELS_2=gemini-2.5-pro
   # ... add more as needed
   ```

---

## Quick Start

### 1. Populate Qdrant with Documents

Place PDF files in `data/` directory, then:

```bash
cd services
python populate_qdrant.py
```

Choose option **1** or **2** to populate the collection.

### 2. Run Interactive Chat

```bash
cd services
python main.py
```

Type your questions and get answers with automatic RAG routing!

### 3. Evaluate on Question-Answer Pairs

**With Ollama (local, no quota limits)**:
```bash
cd services
python evaluate_ollama.py
```

**With Gemini (requires API keys)**:
```bash
cd services
python evaluate_gemini.py
```

---

## Usage

### Interactive Chat (`main.py`)

Simple chat interface with automatic routing:

```bash
python services/main.py
```

**Commands**:
- Type your question → Get answer (with RAG or direct)
- Type `stats` → See current session statistics
- Type `quit` / `exit` → End chat and save results

**Features**:
- Router decides RAG vs direct automatically
- Optional LLM judge evaluation (paste reference answer)
- Statistics saved on exit

### Ollama Evaluation (`evaluate_ollama.py`)

Evaluates all 60 question-answer pairs using **local Ollama** (no API limits):

```bash
python services/evaluate_ollama.py
```

**What it does**:
- Loads questions from `RAG Documents.xlsx` (columns C & D)
- Uses Ollama for generation + judge (no Gemini)
- Saves TXT summary: `Right answer: X/60, accuracy: Y%`

### Gemini Evaluation (`evaluate_gemini.py`)

Evaluates using **Gemini** with API key rotation:

```bash
python services/evaluate_gemini.py
```

**Features**:
- Rotates through multiple API keys/models when quota hit
- Saves checkpoint after each question
- Can resume from last processed question
- Saves intermediate results per API key/model

**Configuration** (in `.env`):
```env
GEMINI_API_KEY_1=key1
GEMINI_MODELS_1=gemini-2.5-flash-lite,gemini-2.5-flash
GEMINI_API_KEY_2=key2
GEMINI_MODELS_2=gemini-2.5-pro
```

---

## Results & Accuracy Reporting

All results are saved in `services/results/` directory.

### Evaluation Results

#### Ollama Model Performance

**Evaluation Date**: February 13, 2026  
**Model**: Local Ollama (llama3.1:8b-instruct-q4_0)  
**Dataset**: 60 question-answer pairs from `RAG Documents.xlsx`

**Results**:
```
Right answer: 43/60, accuracy: 72%
```

- **Correct Answers**: 43 out of 60
- **Accuracy**: 72%
- **Evaluation Method**: LLM-as-judge (Ollama) semantic comparison

#### Gemini Model Performance

**Evaluation Date**: February 13, 2026  
**Model**: Google Gemini (gemini-2.5-flash-lite / gemini-2.5-flash)  
**Dataset**: 47 question-answer pairs processed (out of 60 total)

**Results**:
```
Right answer: 29/47, accuracy: 62%
```

- **Correct Answers**: 29 out of 47 processed
- **Accuracy**: 62%
- **Evaluation Method**: LLM-as-judge (Ollama) semantic comparison
- **Note**: Evaluation stopped at question 47 due to API quota limits

#### Model Comparison Summary

| Model | Correct | Total | Accuracy | Notes |
|-------|---------|-------|----------|-------|
| **Ollama** (llama3.1:8b-instruct) | 43 | 60 | **72%** | Complete evaluation, no API limits |
| **Gemini** (2.5-flash-lite/flash) | 29 | 47 | **62%** | Partial evaluation due to quota |

**Key Observations**:
- Ollama achieved **72% accuracy** on the full 60-question dataset
- Gemini achieved **62% accuracy** on 47 questions (78% of dataset)
- Both models use the same LLM-as-judge (Ollama) for evaluation consistency
- Ollama evaluation completed without quota constraints
- Gemini evaluation demonstrates API key rotation capability but requires multiple keys for full dataset

### Output Files

#### 1. **Accuracy Summary** (TXT)
- **File**: `evaluation_result_YYYYMMDD_HHMMSS.txt` (Ollama) or `gemini_evaluation_final_YYYYMMDD_HHMMSS.txt` (Gemini)
- **Format**: `Right answer: X/Y, accuracy: Z%`
- **Examples**:
  ```
  Right answer: 43/60, accuracy: 72%  # Ollama evaluation
  Right answer: 29/47, accuracy: 62%   # Gemini evaluation
  ```

#### 2. **Statistics CSV**
- **File**: `stats_YYYYMMDD_HHMMSS.csv`
- **Contains**:
  - Total queries processed
  - RAG vs Direct routing counts and percentages
  - Judgment statistics (total judged, correct, incorrect, accuracy)
  - Performance metrics (avg time, avg contexts retrieved)

#### 3. **Detailed Results CSV**
- **File**: `results_YYYYMMDD_HHMMSS.csv`
- **Contains** (one row per question):
  - Timestamp
  - Query text
  - Routing decision & reasoning
  - Generated response
  - Number of contexts used
  - True response (if provided)
  - Judge decision (True/False)
  - Judge confidence score
  - Judge explanation
  - Elapsed time

#### 4. **Checkpoint File** (JSON)
- **File**: `gemini_eval_checkpoint.json`
- **Purpose**: Allows resuming interrupted evaluations
- **Contains**: Last processed index + all accumulated results

#### 5. **Intermediate Results** (CSV)
- **Files**: `gemini_results_api1_MODEL_TIMESTAMP.csv`
- **Purpose**: Results saved when switching API keys/models
- **Useful**: For tracking progress across multiple runs

---

## Statistics

### Accuracy Metrics

The system calculates accuracy based on **LLM-as-judge** evaluations:

```
Accuracy = (Correct Judgments / Total Judged) × 100%
```

**Example**:
- Total questions: 60
- Judged: 60 (all had ground-truth answers)
- Correct: 51
- **Accuracy: 85%**

### Routing Statistics

Tracks how often RAG vs Direct routing was used:

- **RAG Percentage**: `(RAG decisions / Total queries) × 100%`
- **Direct Percentage**: `(Direct decisions / Total queries) × 100%`

**Example**:
- Total queries: 60
- RAG decisions: 45
- Direct decisions: 15
- **RAG Percentage: 75%**

### Performance Metrics

- **Average Processing Time**: Mean time per query (seconds)
- **Average Contexts Retrieved**: Mean number of document chunks used per RAG query

### Sample Statistics Output

```json
{
  "timestamp": "2026-02-13T18:30:00",
  "total_queries": 60,
  "routing": {
    "rag": 45,
    "direct": 15,
    "rag_percentage": 75.0
  },
  "judgment": {
    "total_judged": 60,
    "correct": 51,
    "incorrect": 9,
    "accuracy": 85.0
  },
  "performance": {
    "avg_time_seconds": 3.2,
    "avg_contexts_retrieved": 4.8
  }
}
```

---

## Troubleshooting

### Ollama Model Not Found

**Error**: `Model 'nomic-embed-text' not found`

**Solution**:
```bash
ollama pull nomic-embed-text
ollama pull llama3.1:8b-instruct-q4_0
```

### Qdrant Connection Error

**Error**: `Connection refused` or `Could not connect to Qdrant`

**Solution**:
1. Ensure Qdrant is running:
   ```bash
   docker ps | grep qdrant
   ```
2. Start Qdrant if not running:
   ```bash
   docker run -p 6333:6333 -v qdrant_data:/qdrant/storage qdrant/qdrant
   ```

### Gemini API Quota Exceeded

**Error**: `429 Quota exceeded` or `RESOURCE_EXHAUSTED`

**Solutions**:
1. **Use Ollama evaluation** (`evaluate_ollama.py`) – no quota limits
2. **Add more API keys** to `.env` for rotation
3. **Wait for quota reset** (usually daily)
4. **Use checkpoint resume** – script will continue from last question

### API Key Expired

**Error**: `API key expired` or `API_KEY_INVALID`

**Solution**:
1. Generate new API key from [Google AI Studio](https://aistudio.google.com/)
2. Update `.env` file with new key
3. Restart the script

### Excel File Not Found

**Error**: `Excel file not found: RAG Documents.xlsx`

**Solution**:
1. Ensure `RAG Documents.xlsx` is in project root directory
2. Check that columns C (questions) and D (answers) contain data
3. Verify file is not open in Excel

### Checkpoint Resume Issues

**Problem**: Script starts from beginning instead of resuming

**Solution**:
1. Check `services/results/gemini_eval_checkpoint.json` exists
2. Verify it contains valid `last_index` and `results`
3. Delete checkpoint file if you want to start fresh

---

## Project Structure

```
RAG_Tutorial/
├── services/
│   ├── main.py                 # Interactive chat with RAG system
│   ├── evaluate_ollama.py      # Ollama-based evaluation (60 Q&A pairs)
│   ├── evaluate_gemini.py      # Gemini-based evaluation with key rotation
│   ├── router_agent.py         # Gemini router (RAG vs Direct)
│   ├── rag_generator.py        # RAG pipeline (Gemini + Qdrant)
│   ├── llm_judge.py            # Ollama-based judge
│   ├── embedding_manager.py    # Ollama embeddings
│   ├── qdrant_manager.py       # Qdrant operations
│   ├── preprocessing.py         # PDF chunking
│   ├── populate_qdrant.py      # Populate vector DB
│   └── results/                # Output directory
│       ├── *.csv               # Statistics & detailed results
│       ├── *.txt               # Accuracy summaries
│       └── *.json              # Checkpoints
├── data/                       # PDF documents directory
├── qdrant_data/                # Qdrant storage
├── .env                        # Environment variables (API keys)
├── RAG Documents.xlsx          # Question-answer pairs (60 rows)
└── README.md                   # This file
```

---

## License

This project is intended for educational and research purposes.

---

## Acknowledgments

- **Qdrant** – Vector database for semantic search
- **Ollama** – Local LLM inference
- **Google Gemini** – Cloud LLM API
- **Pydantic AI** – LLM framework

---

## Contact & Support

For issues or questions, please open an issue in the repository.
