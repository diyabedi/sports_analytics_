"""
sql_generator.py
----------------
Task 2 — LLM-powered text-to-SQL.

Given a natural-language query (and optionally retrieved KPI context),
asks Claude to generate a safe, read-only SQL query against the Apollo
schema. The generated SQL is validated before execution.

Safety layers:
  1. System prompt instructs the LLM to only produce read-only queries.
  2. Deterministic validator allows SELECT and WITH (CTEs) as valid entry
     points. WITH is required for multi-step analytical queries that use
     Common Table Expressions (e.g. baseline vs recent comparison).
  3. For WITH queries, the validator confirms the final statement after
     all CTE definitions is a SELECT, blocking constructs like:
       WITH x AS (...) DELETE FROM athletes
  4. Dangerous write/destructive keywords (DROP, DELETE, UPDATE, INSERT,
     CREATE, ALTER, TRUNCATE) are blocked anywhere in the query body.
  5. Only whitelisted table names are allowed. CTE alias names are
     extracted and excluded from the whitelist check to avoid false
     positives on CTE-internal references.
  6. On execution error the LLM is asked to self-correct (one retry).

Why INSERT / UPDATE / DELETE are blocked:
  This is a read-only analytics interface. Its purpose is answering
  questions about performance data, not modifying it. Allowing writes
  through a conversational UI without a confirmation step, audit logging,
  and role-based access control risks silent data corruption with no undo.
  If write operations were ever required, the correct approach is a
  dedicated confirmation + audit layer, not removing this block.
"""

import os
import re
from groq import Groq

from database import SCHEMA

ALLOWED_TABLES = {"athletes", "sessions", "gps_metrics", "wellness"}

# System prompt 

SYSTEM_PROMPT = f"""You are a SQL expert for a sports performance database.
You generate safe, read-only SQLite queries based on the user's question.

DATABASE SCHEMA:
{SCHEMA}

RULES:
- Only generate SELECT statements or WITH ... SELECT (CTEs).
- NEVER use DROP, DELETE, UPDATE, INSERT, CREATE, or ALTER anywhere.
- Only reference tables listed in the schema above.
- Use date('now') for "today" and date('now','-N days') for relative windows in SQLite.
- Always JOIN on the correct foreign keys shown in the schema.
- Return ONLY the raw SQL query — no explanation, no markdown, no code fences.
- If the question is ambiguous, make a reasonable assumption and write the query.
- For multi-step calculations (e.g. comparing recent vs historical averages),
  use CTEs (WITH clauses) to keep the query readable.

EXAMPLES:

  User: top athletes by total distance last week
  SQL:
  SELECT a.name, SUM(g.total_distance) AS total_distance
  FROM athletes a
  JOIN sessions s    ON a.athlete_id = s.athlete_id
  JOIN gps_metrics g ON s.session_id = g.session_id
  WHERE s.session_date >= date('now', '-7 days')
  GROUP BY a.athlete_id, a.name
  ORDER BY total_distance DESC
  LIMIT 10;

  User: average sprint distance by position last 30 days
  SQL:
  SELECT a.position,
         ROUND(AVG(g.sprint_distance), 1) AS avg_sprint_distance,
         COUNT(DISTINCT a.athlete_id)      AS athlete_count
  FROM athletes a
  JOIN sessions s    ON a.athlete_id = s.athlete_id
  JOIN gps_metrics g ON s.session_id = g.session_id
  WHERE s.session_date >= date('now', '-30 days')
  GROUP BY a.position
  ORDER BY avg_sprint_distance DESC;

  User: who is below their baseline performance
  SQL:
  WITH baseline AS (
      SELECT a.athlete_id, a.name,
             AVG(g.total_distance) AS baseline_avg
      FROM athletes a
      JOIN sessions s    ON a.athlete_id = s.athlete_id
      JOIN gps_metrics g ON s.session_id = g.session_id
      WHERE s.session_date <= date((SELECT MIN(session_date) FROM sessions), '+30 days')
      GROUP BY a.athlete_id, a.name
  ),
  recent AS (
      SELECT a.athlete_id,
             AVG(g.total_distance) AS recent_avg
      FROM athletes a
      JOIN sessions s    ON a.athlete_id = s.athlete_id
      JOIN gps_metrics g ON s.session_id = g.session_id
      WHERE s.session_date >= date('now', '-7 days')
      GROUP BY a.athlete_id
  )
  SELECT b.name,
         ROUND(b.baseline_avg, 1)                        AS baseline_avg,
         ROUND(r.recent_avg, 1)                          AS recent_avg,
         ROUND((r.recent_avg / b.baseline_avg) * 100, 1) AS pct_of_baseline
  FROM baseline b
  JOIN recent r ON b.athlete_id = r.athlete_id
  WHERE r.recent_avg < b.baseline_avg * 0.85
  ORDER BY pct_of_baseline ASC;
"""


#  Validation 

class SQLValidationError(Exception):
    pass


def validate_sql(sql: str) -> str:
    """
    Deterministic safety check on the generated SQL.
    Returns the cleaned SQL string, or raises SQLValidationError.

    Allowed entry points:
      SELECT ...
      WITH cte_name AS (...) SELECT ...   <- CTEs for multi-step queries

    Blocked everywhere in the query body:
      DROP, DELETE, UPDATE, INSERT, CREATE, ALTER, TRUNCATE
    """
    # Strip markdown fences if the model accidentally included them
    sql = re.sub(r"```sql|```", "", sql).strip()

    first_word = sql.strip().split()[0].upper()

    # ── Entry point check ───
    # Allow SELECT (standard queries) and WITH (CTEs).
    # WITH is needed for multi-step analytical queries like baseline comparison.
    # Everything else — UPDATE, DELETE, DROP, INSERT — is rejected immediately.
    if first_word not in ("SELECT", "WITH"):
        raise SQLValidationError(
            f"Query must start with SELECT or WITH, got '{first_word}'. "
            f"This system only permits read-only queries."
        )

    # ── CTE safety check ───
    # For WITH queries, verify the final statement after all CTE definitions
    # is a SELECT. This blocks: WITH x AS (SELECT 1) DELETE FROM athletes
    if first_word == "WITH":
        last_paren_pos = sql.rfind(")")
        if last_paren_pos == -1:
            raise SQLValidationError(
                "WITH clause has no closing parenthesis — malformed CTE."
            )
        remainder = sql[last_paren_pos:].upper()
        if "SELECT" not in remainder:
            raise SQLValidationError(
                "WITH clause must be followed by a SELECT statement. "
                "No SELECT found after the final CTE definition."
            )

    # ── Forbidden keyword check ───
    # Block dangerous write/destructive keywords anywhere in the query.
    # Second line of defence after the entry point check above.
    forbidden = {
        "DROP", "DELETE", "UPDATE", "INSERT",
        "CREATE", "ALTER", "TRUNCATE"
    }
    tokens = set(re.findall(r"\b[A-Z]+\b", sql.upper()))
    found = forbidden & tokens
    if found:
        raise SQLValidationError(
            f"Query contains forbidden keyword(s): {found}. "
            f"Only read-only SELECT queries are permitted."
        )

    # ── Table whitelist check ── 
    # Extract every table name referenced after FROM or JOIN.
    table_refs = re.findall(r"\b(?:FROM|JOIN)\s+(\w+)", sql, re.IGNORECASE)
    referenced = {t.lower() for t in table_refs}

    # Extract CTE alias names (e.g. WITH baseline AS ..., recent AS ...)
    # and exclude them — they are not real tables, just named subqueries.
    cte_aliases = {
        a.lower()
        for a in re.findall(r"\b(\w+)\s+AS\s*\(", sql, re.IGNORECASE)
    }

    unknown = referenced - ALLOWED_TABLES - cte_aliases
    if unknown:
        raise SQLValidationError(
            f"Query references unknown table(s): {unknown}. "
            f"Allowed tables: {ALLOWED_TABLES}."
        )

    return sql


# Generator 

class SQLGenerator:
    """
    Wraps the Anthropic API to turn natural-language questions into
    validated SQL queries, with optional KPI context injection.
    """

    def __init__(self, api_key: str | None = None):
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise EnvironmentError("Set GROQ_API_KEY environment variable")
        self.client = Groq(api_key=key)

    def generate(
        self,
        user_query: str,
        kpi_context: str = "",
        retry_with_error: str = "",
    ) -> str:
        """
        Generate and validate SQL for the given user query.

        Args:
            user_query:       The natural language question.
            kpi_context:      Retrieved KPI definitions to inject (optional).
            retry_with_error: Previous SQL error message for self-correction.

        Returns:
            A validated SQL string ready for execution.
        """
        user_parts = []
        if kpi_context:
            user_parts.append(kpi_context)
        user_parts.append(f"Question: {user_query}")
        if retry_with_error:
            user_parts.append(
                f"\nThe previous query failed with this error:\n{retry_with_error}\n"
                "Please fix the SQL and try again."
            )
        user_message = "\n\n".join(user_parts)

        print(f"[SQL] Calling Claude for: \"{user_query}\"")
        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=1000,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": user_message}],
        )

        raw_sql = response.choices[0].message.content.strip()
        print(f"[SQL] Raw response:\n{raw_sql}\n")

        validated_sql = validate_sql(raw_sql)
        return validated_sql

    def generate_with_retry(
        self,
        user_query: str,
        kpi_context: str,
        conn,
    ) -> tuple[str, object]:
        """
        Generate SQL, execute it, and retry once on error.

        Returns:
            (sql_string, pandas_DataFrame_or_None)
        """
        from database import run_query

        # First attempt
        try:
            sql = self.generate(user_query, kpi_context)
        except SQLValidationError as e:
            print(f"[SQL] Validation failed: {e}")
            return "", None

        # Execute
        try:
            df = run_query(conn, sql)
            return sql, df
        except Exception as exec_err:
            print(f"[SQL] Execution error: {exec_err}  — retrying...")
            try:
                sql2 = self.generate(
                    user_query, kpi_context,
                    retry_with_error=str(exec_err)
                )
                df2 = run_query(conn, sql2)
                return sql2, df2
            except Exception as e2:
                print(f"[SQL] Retry also failed: {e2}")
                return sql, None


# Demo 
if __name__ == "__main__":
    from database import build_database

    conn = build_database()
    gen  = SQLGenerator()

    questions = [
        "Which athletes had the highest workload last week?",
        "Show average sprint distance by position over the last 30 days",
        "Who is trending below their baseline performance?",
    ]

    for q in questions:
        print("\n" + "=" * 60)
        print(f"QUESTION: {q}")
        sql, df = gen.generate_with_retry(q, kpi_context="", conn=conn)
        print(f"SQL:\n{sql}")
        if df is not None:
            print(f"\nRESULT ({len(df)} rows):")
            print(df.to_string(index=False))
        else:
            print("No result returned.")