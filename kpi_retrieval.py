"""
kpi_retrieval.py
----------------
Task 3 — RAG-based KPI retrieval.

Given a natural-language user query, find the most relevant KPI(s)
from the catalog using vector search (TF-IDF embeddings + cosine
similarity).

How it works:
  1. At startup, each KPI description is vectorized with sklearn's
     TF-IDF vectorizer.  TF-IDF works entirely offline and is a solid
     baseline for short, domain-specific text like metric definitions.
  2. At query time, the user question is vectorized with the same
     fitted vectorizer.
  3. Cosine similarity is computed between the query vector and every
     KPI vector.  Top-k results above a confidence threshold are returned.
  4. The matched KPIs are injected into the SQL-generation prompt so
     the LLM knows exactly which metric to compute.

Upgrade path:
  When running in an environment with internet access, swap the
  TF-IDF backend for sentence-transformers:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    self.embeddings = model.encode(documents, normalize_embeddings=True)
  The rest of the retrieval logic (cosine similarity, thresholding)
  is identical — dense embeddings just give better semantic matching.
"""

import csv
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).parent / "data"
SIMILARITY_THRESHOLD = 0.05           # TF-IDF scores are lower than dense; 0.05 is a good cutoff


@dataclass
class KPIMatch:
    kpi_name: str
    description: str
    score: float # cosine similarity 0–1

    def __str__(self):
        return f"{self.kpi_name} (score={self.score:.2f}): {self.description}"


class KPIRetriever:
    """
    Loads the KPI catalog, builds a TF-IDF index, then answers semantic
    lookup queries in milliseconds.
    """

    def __init__(self, kpi_csv: Path = DATA_DIR / "KPIs.csv"):
        # Load KPI definitions from CSV and store in self.kpis
        self.kpis: list[dict] = []
        with open(kpi_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.kpis.append({
                    "kpi_name":    row["kpi_name"].strip(),
                    "description": row["description"].strip(),
                })

        # ---- Build TF-IDF index ----
        # Concatenate kpi_name + description so both the short identifier
        # ("fatigue_trend") and the plain English ("change in fatigue score
        # over time") are captured in the same vector.
        documents = [
            f"{k['kpi_name'].replace('_', ' ')} {k['description']}"
            for k in self.kpis
        ]
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),     # unigrams + bigrams for better phrase matching
            stop_words="english",
            sublinear_tf=True,      # log-scale TF to reduce dominance of common terms
        )
        self.embeddings = self.vectorizer.fit_transform(documents)
        print(f"[KPI] TF-IDF index ready — {len(self.kpis)} KPIs indexed.")

    # Public API for retrieval and prompt formatting
    def retrieve(self, query: str, top_k: int = 3) -> list[KPIMatch]:
        """
        Return the top-k KPIs most relevant to the query.
        Results below SIMILARITY_THRESHOLD are filtered out.
        """
        # Vectorize the query with the same fitted TF-IDF
        query_vec = self.vectorizer.transform([query])

        # Cosine similarity between query and all KPI vectors
        scores = cosine_similarity(query_vec, self.embeddings).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score >= SIMILARITY_THRESHOLD:
                results.append(KPIMatch(
                    kpi_name    = self.kpis[idx]["kpi_name"],
                    description = self.kpis[idx]["description"],
                    score       = score,
                ))
        return results

    def retrieve_top1(self, query: str) -> KPIMatch | None:
        """Convenience: return only the single best match, or None."""
        results = self.retrieve(query, top_k=1)
        return results[0] if results else None

    def format_for_prompt(self, matches: list[KPIMatch]) -> str:
        """
        Format retrieved KPIs as a string to inject into the LLM prompt.
        """
        if not matches:
            return "(No matching KPI found)"
        lines = ["Relevant KPI definitions:"]
        for m in matches:
            lines.append(f"  - {m.kpi_name}: {m.description}  [similarity={m.score:.2f}]")
        return "\n".join(lines)


# Demo
if __name__ == "__main__":
    retriever = KPIRetriever()

    test_queries = [
        "Which athletes are getting more tired over time?",
        "Who ran the most sprints last week?",
        "Show me sleep quality trends",
        "Compare match performance vs training",
        "Who has the highest workload per minute?",
        "What is the team's overall running load?",
    ]

    print("\n" + "=" * 60)
    print("KPI RETRIEVAL DEMO")
    print("=" * 60)
    for q in test_queries:
        print(f"\nQuery: \"{q}\"")
        matches = retriever.retrieve(q, top_k=2)
        if matches:
            for m in matches:
                print(f"  → {m}")
        else:
            print("  → No match above threshold")
