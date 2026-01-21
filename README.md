#  CEI 2025 Election Chat — PDF → DB → Chat Agent

Chat with the official CEI 2025 legislative election results, extracted from the PDF, cleaned, indexed, and queried through a natural language interface.

# Project Overview

This application allows users to ask questions about the 2025 CEI election dataset using natural language.
Under the hood:
1. We extract raw tables from the PDF
2. Normalize & clean the data
3. Load structured results into DuckDB
4. Build a semantic search index (RAG)
5. Query via:
    - Text-to-SQL agent for analytics
    - RAG agent for fuzzy lookup & narratives
6. Display results + charts in a Streamlit UI

# Repository Structure

# Repository Structure

├── data/
│   ├── raw/                  ← Original CEI PDF
│   ├── processed/            ← Cleaned CSV / Parquet outputs
│   └── rag/                  ← Vector embeddings index
│
├── src/
│   ├── ingestion/            ← PDF parsing, normalization, cleaning
│   ├── db/                   ← DuckDB initialization & data loading
│   ├── agent/                ← Unified agent (SQL + RAG + clarify + guardrails)
│   ├── rag/                  ← RAG index builder & retriever
│   └── app/
│       └── streamlit_app.py  ← Main Streamlit chat UI
│
├── eval/
│   ├── datasets/             ← Offline eval questions (JSONL)
│   ├── assertions.py         ← Eval assertions (engine, status, sources, content)
│   ├── run_eval.py           ← Offline evaluation runner
│   ├── metrics.py            ← Metrics computation
│   └── run_metrics.py        ← Metrics CLI
│
├── .env.example              ← Environment variables template
├── requirements.txt          ← Python dependencies
└── README.md                 ← Project documentation


## 1. Setup Environment

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

## 2. Download + Inspect PDF

python src/ingestion/inspect_pdf.py

This prints page/table summaries so you can visually confirm structure.

## 3. Parse & Normalize PDF Data

python src/ingestion/parse_pdf.py


## 4. Load Into DuckDB

python src/db/load_to_duckdb.py

Creates:
    resultats (raw detailed results)
    dim_candidat, dim_parti, dim_circonscription
Views:
    vw_elus
    vw_participation_region
    vw_parti_summary
    vw_ranking

## 5. Build RAG Index

python src/rag/build_index.py

Creates vector embeddings for semantic lookup located in:
data/rag/rag_index.parquet

## 6. Launch Web App


python -m streamlit run src/app/streamlit_app.py

## 7. Run Offline Evaluation

python eval/run_eval.py
   ---> Executes the offline evaluation suite and produces: eval/report.json

## 8.Compute Evaluation Metrics
python eval/run_metrics.py

# Agent Decision Flow

QUESTION
│
├─ malicious ? ─────────────► REFUSED
│
├─ out_of_scope ? ───────────► NOT_FOUND
│
├─ normalize + tokenize
│
├─ fuzzy match against dataset entities
│
├─ aggregate ?
│   ├─ YES → SQL (safe aggregation)
│   │        └─ validate intent vs result
│   │
│   └─ NO
│
├─ local / narrative lookup ?
│   ├─ YES → RAG (entity-filtered)
│   │        └─ validate answerability
│
├─ ambiguous ?
│   └─ YES → CLARIFY
│
└─ NOT_FOUND (fallback)
