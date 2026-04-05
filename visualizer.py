"""
visualizer.py
-------------
Task 4 — Automatic visualization of query results.

Given a query result (DataFrame) and metadata about the query intent,
this module picks the most appropriate chart type and renders it.

Chart selection rules (deterministic logic):
  - One categorical + one numeric column → horizontal bar chart
  - One time/date column + one numeric → line chart
  - Two numeric columns + optional categorical → scatter plot
  - Multiple groups over time → multi-series line chart
  - Ranking / top-N → ranked horizontal bar
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path
from datetime import datetime

OUTPUT_DIR = Path(__file__).parent / "charts"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Style constants ────────────────────────────────────────────────────────
COLORS = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#44BBA4", "#E94F37"]
BG = "#F4F6F9"
GRAY_TEXT = "#374151"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 130,
})


# ── Chart type detection ───
def _is_date_col(col: pd.Series) -> bool:
    try:
        pd.to_datetime(col)
        return True
    except Exception:
        return False


def detect_chart_type(df: pd.DataFrame) -> str:
    """
    Inspect the DataFrame columns and return the best chart type:
      'bar', 'line', 'scatter', 'multiline'
    """
    cols = df.columns.tolist()
    n_numeric = sum(pd.api.types.is_numeric_dtype(df[c]) for c in cols)
    n_text = sum(df[c].dtype == object for c in cols)
    date_cols = [c for c in cols if "date" in c.lower() or "week" in c.lower()]

    # Multiple numeric cols + a group col → likely comparison → bar
    if n_text == 1 and n_numeric >= 2:
        return "bar"
    # Date/time col present + one group + one numeric → multiline
    if date_cols and n_text == 1 and n_numeric == 1:
        return "multiline"
    # Date col + no groups → single line
    if date_cols and n_numeric == 1:
        return "line"
    # Two numerics → scatter
    if n_numeric == 2 and n_text <= 1:
        return "scatter"
    # Fallback: bar chart
    return "bar"


# ── Individual chart renderers ────

def _bar_chart(df: pd.DataFrame, title: str, save_path: Path) -> None:
    """Horizontal bar chart — best for rankings and comparisons."""
    cat_col = next((c for c in df.columns if df[c].dtype == object), df.columns[0])
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    if not num_cols:
        print("[VIZ] No numeric columns found for bar chart")
        return

    # If multiple numeric cols, take the first meaningful one
    val_col = num_cols[0]
    df_sorted = df.sort_values(val_col, ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(9, max(4, len(df_sorted) * 0.5 + 1)))
    fig.patch.set_facecolor("white")

    bars = ax.barh(df_sorted[cat_col].astype(str), df_sorted[val_col],
                   color=COLORS[0], edgecolor="white", height=0.6)

    # Value labels on bars
    for bar in bars:
        w = bar.get_width()
        ax.text(w + max(df_sorted[val_col]) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{w:,.1f}", va="center", fontsize=9, color=GRAY_TEXT)

    ax.set_facecolor(BG)
    ax.set_title(title, fontsize=13, fontweight="bold", color=GRAY_TEXT, pad=12)
    ax.set_xlabel(val_col.replace("_", " ").title(), fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def _line_chart(df: pd.DataFrame, title: str, save_path: Path) -> None:
    """Single-series line chart — best for one metric over time."""
    date_col = next((c for c in df.columns if "date" in c.lower() or "week" in c.lower()), df.columns[0])
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    val_col = num_cols[0]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")
    ax.plot(df[date_col].astype(str), df[val_col], marker="o", linewidth=2.5,
            color=COLORS[0], markersize=8)
    ax.fill_between(range(len(df)), df[val_col], alpha=0.08, color=COLORS[0])
    ax.set_facecolor(BG)
    ax.set_title(title, fontsize=13, fontweight="bold", color=GRAY_TEXT, pad=12)
    ax.set_xlabel(date_col.replace("_", " ").title(), fontsize=10)
    ax.set_ylabel(val_col.replace("_", " ").title(), fontsize=10)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def _multiline_chart(df: pd.DataFrame, title: str, save_path: Path) -> None:
    """Multi-series line chart — best for comparing athletes/groups over time."""
    group_col = next((c for c in df.columns if df[c].dtype == object), None)
    date_col = next((c for c in df.columns if "date" in c.lower() or "week" in c.lower()), None)
    if not date_col:
        date_col = df.columns[0]
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    val_col = num_cols[0]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor("white")

    if group_col:
        for i, (grp_name, grp_df) in enumerate(df.groupby(group_col)):
            grp_sorted = grp_df.sort_values(date_col)
            ax.plot(grp_sorted[date_col].astype(str), grp_sorted[val_col],
                    marker="o", linewidth=2.5, label=str(grp_name),
                    color=COLORS[i % len(COLORS)], markersize=7)
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    else:
        ax.plot(df[date_col].astype(str), df[val_col], marker="o",
                color=COLORS[0], linewidth=2.5)

    ax.set_facecolor(BG)
    ax.set_title(title, fontsize=13, fontweight="bold", color=GRAY_TEXT, pad=12)
    ax.set_xlabel(date_col.replace("_", " ").title(), fontsize=10)
    ax.set_ylabel(val_col.replace("_", " ").title(), fontsize=10)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def _scatter_chart(df: pd.DataFrame, title: str, save_path: Path) -> None:
    """Scatter plot — best for correlation between two numeric metrics."""
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_col = next((c for c in df.columns if df[c].dtype == object), None)
    x_col, y_col = num_cols[0], num_cols[1]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor("white")

    if cat_col:
        groups = df[cat_col].unique()
        for i, grp in enumerate(groups):
            sub = df[df[cat_col] == grp]
            ax.scatter(sub[x_col], sub[y_col], label=str(grp),
                       color=COLORS[i % len(COLORS)], s=80, alpha=0.8, edgecolors="white")
        ax.legend(fontsize=9)
    else:
        ax.scatter(df[x_col], df[y_col], color=COLORS[0], s=80, alpha=0.8)

    # Trend line
    z = np.polyfit(df[x_col], df[y_col], 1)
    xs = np.linspace(df[x_col].min(), df[x_col].max(), 100)
    ax.plot(xs, np.poly1d(z)(xs), "k--", linewidth=1, alpha=0.4)

    ax.set_facecolor(BG)
    ax.set_title(title, fontsize=13, fontweight="bold", color=GRAY_TEXT, pad=12)
    ax.set_xlabel(x_col.replace("_", " ").title(), fontsize=10)
    ax.set_ylabel(y_col.replace("_", " ").title(), fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


# ── Public API ───
def visualize(df: pd.DataFrame, query: str, chart_type: str | None = None) -> Path | None:
    """
    Create the best visualization for the given DataFrame and query.

    Args:
        df:          Query result as a DataFrame.
        query:       Original user question (used for the chart title).
        chart_type:  Override auto-detection: 'bar', 'line', 'scatter', 'multiline'.

    Returns:
        Path to the saved PNG, or None if visualization was not possible.
    """
    if df is None or df.empty:
        print("[VIZ] Empty DataFrame — skipping visualization.")
        return None

    if len(df.columns) < 2:
        print("[VIZ] Need at least 2 columns to visualize.")
        return None

    chart = chart_type or detect_chart_type(df)
    title = query[:70] + ("..." if len(query) > 70 else "")
    ts = datetime.now().strftime("%H%M%S")
    save_path = OUTPUT_DIR / f"chart_{ts}.png"

    print(f"[VIZ] Drawing {chart} chart → {save_path.name}")

    dispatch = {
        "bar":       _bar_chart,
        "line":      _line_chart,
        "multiline": _multiline_chart,
        "scatter":   _scatter_chart,
    }
    renderer = dispatch.get(chart, _bar_chart)
    renderer(df, title, save_path)
    return save_path


# ── Demo ───
if __name__ == "__main__":
    import pandas as pd
    from database import build_database, run_query

    conn = build_database()

    demos = [
        ("Total distance by athlete",
         "SELECT a.name, SUM(g.total_distance) AS total_distance FROM athletes a JOIN sessions s ON a.athlete_id = s.athlete_id JOIN gps_metrics g ON s.session_id = g.session_id GROUP BY a.name ORDER BY total_distance DESC"),
        ("Sprint distance by position",
         "SELECT a.position, ROUND(AVG(g.sprint_distance),1) AS avg_sprint FROM athletes a JOIN sessions s ON a.athlete_id = s.athlete_id JOIN gps_metrics g ON s.session_id = g.session_id GROUP BY a.position"),
    ]

    for title, sql in demos:
        df = run_query(conn, sql)
        path = visualize(df, title)
        print(f"Saved: {path}")
