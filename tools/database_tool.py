import json
import sqlite3
from typing import Any, Dict, List, Optional
import sqlparse
from langchain.tools import Tool

class DatabaseTool:
    """SQLite database execution tool with schema introspection and safety checks.

    - Default is read-only (SELECT-only)
    - Provides helpers to introspect schema for LLM grounding
    """
    def __init__(self, db_path: str = "company.db", allow_write: bool = False) -> None:
        self.db_path = db_path
        self.allow_write = allow_write

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    # --------------------- Schema Introspection ---------------------
    def get_schema(self) -> Dict[str, List[Dict[str, Any]]]:
        """Return DB schema as {table: [{name, type, notnull, pk, default}], ...}."""
        schema: Dict[str, List[Dict[str, Any]]] = {}
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")
            tables = [r[0] for r in cur.fetchall()]
            for t in tables:
                cur.execute(f"PRAGMA table_info({t})")
                cols = cur.fetchall()
                schema[t] = [
                    {
                        "name": c[1],
                        "type": c[2],
                        "notnull": bool(c[3]),
                        "default": c[4],
                        "pk": bool(c[5]),
                    }
                    for c in cols
                ]
        return schema

    def get_schema_markdown(self, max_cols: int = 12) -> str:
        """Compact markdown of schema for tool description (truncated; use sql_schema for full)."""
        schema = self.get_schema()
        lines: List[str] = ["(Columns may be truncated below; call sql_schema for full schema.)"]
        for table, cols in schema.items():
            col_list = ", ".join([c["name"] for c in cols[:max_cols]])
            more = " ..." if len(cols) > max_cols else ""
            lines.append(f"- {table}({col_list}{more})")
        return "\n".join(lines)

    # --------------------- SQL Validation ---------------------
    def _validate_sql(self, sql: str) -> None:
        """Validate SQL: single statement and SELECT-only unless writes allowed."""
        sql_stripped = sql.strip().strip(";")
        parsed = sqlparse.parse(sql_stripped)
        if len(parsed) != 1:
            raise ValueError("Only a single SQL statement is allowed.")
        stmt = parsed[0]
        first_token = next((t for t in stmt.tokens if not t.is_whitespace and t.ttype != sqlparse.tokens.Comment), None)
        if first_token is None:
            raise ValueError("Empty SQL statement.")
        is_select = first_token.ttype is sqlparse.tokens.DML and first_token.value.upper() == "SELECT"
        if not self.allow_write and not is_select:
            raise ValueError("Only SELECT statements are allowed by this tool.")
        # Basic guardrails against dangerous pragmas or attaches in read mode
        dangerous = ("ATTACH", "DETACH", "PRAGMA", "VACUUM", "ALTER", "DROP")
        if not self.allow_write:
            upper = sql_stripped.upper()
            if any(word in upper for word in dangerous):
                raise ValueError("Disallowed statement for read-only execution.")

    def _explain_plan_ok(self, sql: str) -> bool:
        """Optional: run EXPLAIN to ensure SQLite can parse; returns True/False."""
        try:
            with self._connect() as conn:
                cur = conn.cursor()
                cur.execute(f"EXPLAIN {sql}")
                _ = cur.fetchall()
                return True
        except Exception:
            return False

    # --------------------- Execution ---------------------
    def execute_sql(self, sql: str, parameters: Optional[tuple] = None) -> Dict[str, Any]:
        """Execute SQL and return a structured result.

        - Restricts to SELECT statements unless allow_write=True
        - Ensures single-statement execution and rejects dangerous keywords
        - Supports parameterized queries via parameters tuple
        """
        self._validate_sql(sql)
        if not self._explain_plan_ok(sql):
            raise ValueError("SQL failed EXPLAIN validation. Please check syntax.")
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, parameters or ())
            # Determine if SELECT
            sql_up = sql.lstrip().upper()
            if sql_up.startswith("SELECT"):
                rows = [dict(r) for r in cursor.fetchall()]
                return {
                    "rows": rows,
                    "count": len(rows),
                    "message": "No rows matched" if not rows else "OK",
                }
            return {"rows": [], "count": cursor.rowcount, "message": "Non-SELECT executed"}

    # --------------------- LangChain Tools ---------------------
    def as_langchain_tool(self) -> Tool:
        schema_md = self.get_schema_markdown()
        def _run(sql: str) -> str:
            result = self.execute_sql(sql)
            return json.dumps(result, default=str)
        description = (
            "Execute read-only SQL (SELECT ...) against the internal SQLite database.\n"
            "Schema (tables and columns):\n"
            f"{schema_md}\n\n"
            "Guidelines: Use valid SQLite syntax. Only a single SELECT statement is allowed."
        )
        return Tool(name="sql_database", func=_run, description=description)

    def as_schema_tool(self) -> Tool:
        def _run(_: str = "") -> str:
            return json.dumps(self.get_schema(), default=str)
        return Tool(
            name="sql_schema",
            func=_run,
            description=(
                "Return the current SQLite schema as JSON with tables and their columns. "
                "Use this before writing SQL to avoid hallucinated columns."
            ),
        )

if __name__ == "__main__":
    tool = DatabaseTool()
    print("Schema:")
    print(tool.get_schema_markdown())
    sample = tool.execute_sql("SELECT name, department FROM employees LIMIT 5")
    print(json.dumps(sample, indent=2))