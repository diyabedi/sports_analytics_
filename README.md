# Apollo MIS — RAG Sports Performance System

A working end-to-end pipeline that takes a natural language question
about athlete performance and returns a structured answer and chart.

---

## What this system does

```
Your question (text)
      ↓
[KPI Retrieval]   — finds the most relevant metric definition using
                    sentence embeddings + cosine similarity
      ↓
[SQL Generation]  — Claude (via Anthropic API) writes a SQL query
                    using your question + retrieved KPI context
      ↓
[Validation]      — deterministic safety check (SELECT-only, no
                    dangerous keywords, only known tables)
      ↓
[Execution]       — query runs against a local SQLite database
                    built from the provided CSV files
      ↓
[Visualization]   — auto-picks bar / line / scatter chart
                    based on what columns came back
```

---

## Project structure

```
apollo_rag_system/
├── demo.py            ← MAIN ENTRY POINT — run this
├── database.py        ← loads CSVs into SQLite, run_query() helper
├── kpi_retrieval.py   ← KPI catalog + embedding-based search
├── sql_generator.py   ← LLM text-to-SQL with validation
├── visualizer.py      ← auto chart selection and rendering
├── requirements.txt
├── data/              ← CSV files go here
│   ├── athletes.csv
│   ├── sessions.csv
│   ├── gps_metrics.csv
│   ├── wellness.csv
│   └── KPIs.csv
└── charts/            ← generated charts saved here (auto-created)
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your Anthropic/GROQ API key
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```
Get a key at: https://console.anthropic.com

### 3. Make sure the CSV files are in the `data/` folder
The files needed are: `athletes.csv`, `sessions.csv`,
`gps_metrics.csv`, `wellness.csv`, `KPIs.csv`

---

## Running the demo

### Interactive mode (ask your own questions)
```bash
python demo.py
```

### Auto mode (runs 5 built-in demo questions)
```bash
python demo.py --auto
```

### Rebuild the database from scratch
```bash
python demo.py --rebuild-db
```

---

## Example questions you can ask

- "Which athletes had the highest workload last week?"
- "Show average sprint distance by position over the last 30 days"
- "Who is trending below their baseline performance?"
- "What is the average fatigue score per athlete?"
- "Show total distance for each athlete"
- "Who has the most high intensity efforts overall?"

---

## Running individual modules

Each module can also be run on its own to test just that component:

```bash
# Test database setup
python database.py

# Test KPI retrieval
python kpi_retrieval.py

# Test SQL generation (needs API key)
python sql_generator.py

# Test visualization
python visualizer.py
```

---

## How the RAG part works (Task 3)

The KPI catalog (`KPIs.csv`) contains 10 metric definitions.
Each is embedded as a dense vector using `sentence-transformers/all-MiniLM-L6-v2`.

When you ask a question:
1. Your question is embedded with the same model.
2. Cosine similarity is computed between your question and every KPI vector.
3. The top 2 most similar KPIs are retrieved.
4. Their descriptions are injected into the SQL generation prompt as context.

This means when you ask "who is getting tired?", the system knows to
look for `fatigue_trend` and write SQL against the `wellness` table —
without you having to name the column.

---

## Safety

- Only `SELECT` queries are allowed — any other statement is blocked before execution.
- Generated column names are checked against the known schema.
- Only whitelisted tables (`athletes`, `sessions`, `gps_metrics`, `wellness`) can be queried.
- On SQL execution error, the model automatically retries with the error appended.
