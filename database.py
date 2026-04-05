"""
database.py
-----------
Loads the Apollo CSV files into a SQLite database and provides a
query helper used by the rest of the system.

Tables created:
  athletes    — athlete_id, name, position, team
  sessions    — session_id, athlete_id, session_date, duration_minutes, session_type
  gps_metrics — session_id, total_distance, sprint_distance, high_intensity_efforts
  wellness    — athlete_id, date, sleep_score, fatigue_score
"""

import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path(__file__).parent / "apollo.db"
DATA_DIR = Path(__file__).parent / "data"


def build_database(force_rebuild: bool = False) -> sqlite3.Connection:
    """
    Create (or reuse) the SQLite database from the CSV files.
    Returns an open connection.
    """
    if DB_PATH.exists() and not force_rebuild:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        print(f"[DB] Using existing database at {DB_PATH}")
        return conn

    print("[DB] Building database from CSVs...")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # athletes
    athletes = pd.read_csv(DATA_DIR / "athletes.csv")
    athletes.to_sql("athletes", conn, if_exists="replace", index=False)

    # sessions
    sessions = pd.read_csv(DATA_DIR / "sessions.csv")
    sessions["session_date"] = pd.to_datetime(sessions["session_date"]).dt.strftime("%Y-%m-%d")
    sessions.to_sql("sessions", conn, if_exists="replace", index=False)

    # gps_metrics
    gps = pd.read_csv(DATA_DIR / "gps_metrics.csv")
    gps.to_sql("gps_metrics", conn, if_exists="replace", index=False)

    # wellness
    wellness = pd.read_csv(DATA_DIR / "wellness.csv")
    wellness["date"] = pd.to_datetime(wellness["date"]).dt.strftime("%Y-%m-%d")
    wellness.to_sql("wellness", conn, if_exists="replace", index=False)

    conn.commit()
    print(f"[DB] Database ready at {DB_PATH}")
    _print_summary(conn)
    return conn


def _print_summary(conn: sqlite3.Connection) -> None:
    for table in ["athletes", "sessions", "gps_metrics", "wellness"]:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table:15s} → {count} rows")


def run_query(conn: sqlite3.Connection, sql: str) -> pd.DataFrame:
    """
    Execute a SELECT query and return results as a DataFrame.
    Raises ValueError if the query is not a SELECT (safety guard).
    """
    first_word = sql.strip().split()[0].upper()
    if first_word != "SELECT":
        raise ValueError(
            f"Only SELECT queries are allowed. Got: '{first_word}'"
        )
    return pd.read_sql_query(sql, conn)


# For reference, the database schema (not used programmatically, but helpful for prompt construction):
SCHEMA = """
TABLE athletes
  athlete_id  INTEGER  PRIMARY KEY
  name        TEXT
  position    TEXT     (values: 'Forward', 'Midfielder', 'Defender')
  team        TEXT     (values: 'A', 'B')

TABLE sessions
  session_id      INTEGER  PRIMARY KEY
  athlete_id      INTEGER  FK → athletes.athlete_id
  session_date    DATE     (format: YYYY-MM-DD)
  duration_minutes INTEGER
  session_type    TEXT     (values: 'Training', 'Match')

TABLE gps_metrics
  session_id            INTEGER  FK → sessions.session_id
  total_distance        REAL     (metres)
  sprint_distance       REAL     (metres)
  high_intensity_efforts INTEGER

TABLE wellness
  athlete_id   INTEGER  FK → athletes.athlete_id
  date         DATE     (format: YYYY-MM-DD)
  sleep_score  INTEGER  (0–100, higher = better)
  fatigue_score INTEGER (0–100, higher = more fatigued)
""".strip()


if __name__ == "__main__":
    conn = build_database(force_rebuild=True)
    print("\nQuick test:")
    df = run_query(conn, "SELECT a.name, a.position, COUNT(s.session_id) AS sessions FROM athletes a JOIN sessions s ON a.athlete_id = s.athlete_id GROUP BY a.athlete_id")
    print(df.to_string(index=False))
