"""
demo.py
-------
Apollo MIS — RAG Internship Assessment
End-to-end runnable demo of the full pipeline:

  Natural language question
        ↓
  [KPI Retrieval]  — find the relevant metric using embeddings
        ↓
  [SQL Generation] — Claude turns the question into a SQL query
        ↓
  [SQL Execution]  — run against the SQLite database
        ↓
  [Visualization]  — auto-render the best chart for the result

Usage:
  # Interactive mode (ask your own questions):
  python demo.py

  # Run the built-in demo questions automatically:
  python demo.py --auto

Requirements:
  pip install anthropic sentence-transformers pandas matplotlib numpy

  Set your Anthropic API key:
  export ANTHROPIC_API_KEY=sk-ant-...
"""

import argparse
import os
import sys
import textwrap
import subprocess
from dotenv import load_dotenv
load_dotenv()
from database import build_database, run_query
from kpi_retrieval import KPIRetriever
from sql_generator import SQLGenerator
from visualizer import visualize


DEMO_QUESTIONS = [
    "Which athletes had the highest workload last week?",
    "Show average sprint distance by position over the last 30 days",
    "Who has the highest number of high intensity efforts overall?",
    "What is the average fatigue score per athlete?",
    "Show total distance for each athlete",
]

BANNER = """
╔══════════════════════════════════════════════════════════╗
║         Apollo MIS — RAG Sports Performance System       ║
║              Natural Language → SQL → Chart              ║
╚══════════════════════════════════════════════════════════╝
"""


def print_divider(label: str = "") -> None:
    width = 60
    if label:
        pad = (width - len(label) - 2) // 2
        print(f"\n{'─' * pad} {label} {'─' * pad}")
    else:
        print("─" * width)


def open_image(path) -> None:
    """Try to open the chart image (works on Mac/Linux)."""
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        elif sys.platform.startswith("linux"):
            subprocess.Popen(["xdg-open", str(path)])
    except Exception:
        pass


def run_pipeline(
    question: str,
    retriever: KPIRetriever,
    generator: SQLGenerator,
    conn,
    show_chart: bool = True,
) -> None:
    """
    Execute the full RAG pipeline for one question and print results.
    """
    print_divider("QUESTION")
    print(f"  {question}")

    # Step 1: KPI Retrieval 
    print_divider("STEP 1 — KPI Retrieval")
    matches = retriever.retrieve(question, top_k=2)
    kpi_context = retriever.format_for_prompt(matches)
    if matches:
        for m in matches:
            print(f"  ✓ {m.kpi_name}  (similarity={m.score:.2f})")
            print(f"    \"{m.description}\"")
    else:
        print("  (no matching KPI found — proceeding without context)")

    #Step 2: SQL Generation
    print_divider("STEP 2 — SQL Generation")
    sql, df = generator.generate_with_retry(question, kpi_context, conn)

    if not sql:
        print("  ✗ Could not generate valid SQL.")
        return

    # Pretty-print the SQL
    for line in sql.splitlines():
        print(f"  {line}")

    # Step 3: Results
    print_divider("STEP 3 — Query Results")
    if df is None or df.empty:
        print("  No results returned.")
        return

    print(df.to_string(index=False))
    print(f"\n  → {len(df)} row(s) returned")

    # Step 4: Visualization
    if show_chart:
        print_divider("STEP 4 — Visualization")
        chart_path = visualize(df, question)
        if chart_path:
            print(f"  ✓ Chart saved: {chart_path}")
            open_image(chart_path)
        else:
            print("  (skipped — not enough columns for a chart)")

    print_divider()


def interactive_mode(retriever, generator, conn):
    """REPL loop for asking custom questions."""
    print(BANNER)
    print("Type a question about athlete performance, or 'quit' to exit.\n")
    print("Example questions:")
    for q in DEMO_QUESTIONS:
        print(f"  • {q}")
    print()

    while True:
        try:
            question = input("Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        run_pipeline(question, retriever, generator, conn)
        print()


def auto_mode(retriever, generator, conn):
    """Run all built-in demo questions automatically."""
    print(BANNER)
    print(f"Running {len(DEMO_QUESTIONS)} demo questions...\n")

    for i, question in enumerate(DEMO_QUESTIONS, 1):
        print(f"\n[{i}/{len(DEMO_QUESTIONS)}]")
        run_pipeline(question, retriever, generator, conn, show_chart=True)


def main():
    parser = argparse.ArgumentParser(description="Apollo MIS RAG Demo")
    parser.add_argument("--auto", action="store_true",
                        help="Run built-in demo questions automatically")
    parser.add_argument("--rebuild-db", action="store_true",
                        help="Force rebuild the SQLite database")
    parser.add_argument("--no-charts", action="store_true",
                        help="Skip chart generation")
    args = parser.parse_args()

    # Check API key 
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    # Initialise components
    print("[INIT] Setting up database...")
    conn = build_database(force_rebuild=args.rebuild_db)

    print("[INIT] Loading KPI retriever...")
    retriever = KPIRetriever()

    print("[INIT] Connecting to Anthropic API...")
    generator = SQLGenerator()

    print("[INIT] All systems ready.\n")

    # Run the demo
    if args.auto:
        auto_mode(retriever, generator, conn)
    else:
        interactive_mode(retriever, generator, conn)


if __name__ == "__main__":
    main()
