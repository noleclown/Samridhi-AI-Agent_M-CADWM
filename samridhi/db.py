"""
samridhi/db.py
==============
Database abstraction layer.

Provides a single `get_connection()` factory that returns either a
SQLite connection or a PostgreSQL connection based on config.yaml.

All cache classes (FeedbackCache, WebCache, ExpansionStore) use the
`DbAdapter` wrapper instead of calling sqlite3 directly.  Switching
backends is a single config.yaml change — no code changes anywhere.

SWITCHING TO POSTGRESQL
-----------------------
1. Install the driver:
       pip install psycopg2-binary

2. Add to config.yaml:
       database:
         backend: postgres
         postgres_url: "postgresql://user:password@host:5432/samridhi"

3. Restart the app.  That's it.

SQLITE (default — zero config needed)
--------------------------------------
       database:
         backend: sqlite          # this is the default if section is absent

DESIGN
------
DbAdapter wraps a connection object and normalises the two key differences
between SQLite and PostgreSQL:

  1. Placeholder style:   SQLite uses ?    PostgreSQL uses %s
  2. Upsert syntax:       SQLite uses INSERT OR REPLACE
                          PostgreSQL uses INSERT ... ON CONFLICT DO UPDATE

All SQL in cache.py is written in SQLite style.  DbAdapter translates
on the fly when the backend is postgres.

The adapter is intentionally minimal — it only exposes the methods
cache.py actually calls:
  .execute(sql, params)     → cursor
  .executemany(sql, rows)   → cursor
  .commit()
  .fetchone(sql, params)    → row | None
  .fetchall(sql, params)    → list[row]
  .close()
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

from samridhi.config import cfg
from samridhi.logger import get_logger

log = get_logger()

# ── optional psycopg2 ─────────────────────────────────────────
try:
    import psycopg2        # type: ignore
    import psycopg2.extras # type: ignore
    _HAS_PSYCOPG2 = True
except ImportError:
    _HAS_PSYCOPG2 = False


# ══════════════════════════════════════════════════════════════
# SQL TRANSLATION  (SQLite → PostgreSQL)
# ══════════════════════════════════════════════════════════════

def _to_pg(sql: str) -> str:
    """
    Translate SQLite-style SQL to PostgreSQL-compatible SQL.

    Transformations:
      1. ? → %s                            (placeholder style)
      2. INTEGER → INTEGER                 (no change — compatible)
      3. TEXT → TEXT                       (no change — compatible)
      4. PRAGMA journal_mode=WAL → ''      (SQLite-only, ignored)
      5. INSERT OR REPLACE INTO x → INSERT INTO x ... ON CONFLICT(...) DO UPDATE SET ...
         (handled separately in DbAdapter.execute — see _translate_upsert)
    """
    # Placeholder: ? → %s
    sql = re.sub(r'\?', '%s', sql)
    # Remove SQLite-only PRAGMAs
    sql = re.sub(r'PRAGMA\s+\S+\s*=\s*\S+', '', sql, flags=re.IGNORECASE).strip()
    return sql


def _translate_upsert(sql: str, table: str) -> str:
    """
    Convert  INSERT OR REPLACE INTO <table> (cols...) VALUES (...)
    to       INSERT INTO <table> (cols...) VALUES (...)
             ON CONFLICT(<first_col>) DO UPDATE SET col=EXCLUDED.col, ...

    This is called by DbAdapter.execute() only when INSERT OR REPLACE is detected.
    """
    # Extract column list
    m = re.search(
        r'INSERT\s+OR\s+REPLACE\s+INTO\s+\w+\s*\(([^)]+)\)',
        sql, re.IGNORECASE
    )
    if not m:
        # Can't parse — fall back to plain INSERT (may fail on conflict)
        return re.sub(r'INSERT\s+OR\s+REPLACE', 'INSERT', sql, flags=re.IGNORECASE)

    cols     = [c.strip() for c in m.group(1).split(',')]
    pk       = cols[0]                           # first column = primary key
    updates  = ', '.join(
        f"{c}=EXCLUDED.{c}" for c in cols[1:]
    )
    base_sql = re.sub(
        r'INSERT\s+OR\s+REPLACE', 'INSERT', sql, flags=re.IGNORECASE
    )
    return f"{base_sql} ON CONFLICT({pk}) DO UPDATE SET {updates}"


# ══════════════════════════════════════════════════════════════
# DB ADAPTER
# ══════════════════════════════════════════════════════════════

class DbAdapter:
    """
    Thin adapter over sqlite3 or psycopg2.

    Normalises:
      - placeholder style (? vs %s)
      - upsert syntax (INSERT OR REPLACE vs ON CONFLICT)
      - WAL pragma (SQLite only, ignored on Postgres)

    Usage — identical regardless of backend:
        db = DbAdapter.create("feedback")
        db.execute("CREATE TABLE IF NOT EXISTS ...")
        db.execute("INSERT OR REPLACE INTO t VALUES (?,?)", (k, v))
        row = db.fetchone("SELECT * FROM t WHERE key=?", (k,))
        db.commit()
    """

    def __init__(self, conn, backend: str):
        self._conn    = conn
        self._backend = backend   # "sqlite" or "postgres"

    # ── factory ──────────────────────────────────────────────

    @classmethod
    def create(cls, db_name: str, db_path: Path | None = None) -> "DbAdapter":
        """
        Create a DbAdapter for the configured backend.

        db_name  — logical name used only for logging ("feedback", "web", "expansions")
        db_path  — SQLite file path (ignored for postgres)
        """
        backend = cfg.get("database", {}).get("backend", "sqlite").lower()

        if backend == "postgres":
            if not _HAS_PSYCOPG2:
                log.warning(
                    "database.backend=postgres but psycopg2 is not installed. "
                    "Falling back to SQLite.  Run: pip install psycopg2-binary"
                )
                backend = "sqlite"
            else:
                url = cfg.get("database", {}).get("postgres_url", "")
                if not url:
                    log.warning(
                        "database.backend=postgres but postgres_url is not set in config.yaml. "
                        "Falling back to SQLite."
                    )
                    backend = "sqlite"
                else:
                    try:
                        conn = psycopg2.connect(url)
                        conn.autocommit = False
                        log.info(f"DbAdapter[{db_name}]: connected to PostgreSQL")
                        return cls(conn, "postgres")
                    except Exception as e:
                        log.error(
                            f"DbAdapter[{db_name}]: PostgreSQL connection failed ({e}). "
                            "Falling back to SQLite."
                        )
                        backend = "sqlite"

        # SQLite path
        if db_path is None:
            from samridhi.config import BASE_DIR
            db_path = BASE_DIR / f"{db_name}.db"
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        log.info(f"DbAdapter[{db_name}]: connected to SQLite at {db_path.name}")
        return cls(conn, "sqlite")

    # ── query helpers ─────────────────────────────────────────

    def _prepare(self, sql: str) -> str:
        """Translate SQL for the current backend."""
        if self._backend == "sqlite":
            return sql
        # PostgreSQL: handle upsert first, then general translation
        if re.search(r'INSERT\s+OR\s+REPLACE', sql, re.IGNORECASE):
            m = re.search(r'INSERT\s+OR\s+REPLACE\s+INTO\s+(\w+)', sql, re.IGNORECASE)
            table = m.group(1) if m else ""
            sql   = _translate_upsert(sql, table)
        return _to_pg(sql)

    def execute(self, sql: str, params: tuple = ()) -> Any:
        """Execute a statement. Returns cursor."""
        try:
            cur = self._conn.cursor()
            cur.execute(self._prepare(sql), params)
            return cur
        except Exception as e:
            log.warning(f"DbAdapter.execute failed: {e}\nSQL: {sql[:120]}")
            raise

    def executemany(self, sql: str, rows: list) -> Any:
        """Execute a statement for multiple rows. Returns cursor."""
        try:
            cur = self._conn.cursor()
            cur.executemany(self._prepare(sql), rows)
            return cur
        except Exception as e:
            log.warning(f"DbAdapter.executemany failed: {e}")
            raise

    def fetchone(self, sql: str, params: tuple = ()) -> tuple | None:
        """Execute and return first row or None."""
        try:
            cur = self._conn.cursor()
            cur.execute(self._prepare(sql), params)
            return cur.fetchone()
        except Exception as e:
            log.warning(f"DbAdapter.fetchone failed: {e}")
            return None

    def fetchall(self, sql: str, params: tuple = ()) -> list:
        """Execute and return all rows."""
        try:
            cur = self._conn.cursor()
            cur.execute(self._prepare(sql), params)
            return cur.fetchall()
        except Exception as e:
            log.warning(f"DbAdapter.fetchall failed: {e}")
            return []

    def commit(self):
        try:
            self._conn.commit()
        except Exception as e:
            log.warning(f"DbAdapter.commit failed: {e}")

    def close(self):
        try:
            self._conn.close()
        except Exception:
            pass

    @property
    def backend(self) -> str:
        return self._backend


# ══════════════════════════════════════════════════════════════
# CONVENIENCE FACTORY FUNCTIONS
# Used by cache.py to get the right adapter for each DB.
# ══════════════════════════════════════════════════════════════

def feedback_db_adapter() -> DbAdapter:
    from samridhi.config import FEEDBACK_DB
    return DbAdapter.create("feedback", FEEDBACK_DB)

def web_db_adapter() -> DbAdapter:
    from samridhi.config import WEB_DB_FILE
    return DbAdapter.create("web", WEB_DB_FILE)

def expansions_db_adapter() -> DbAdapter:
    from samridhi.config import EXPANSIONS_DB
    return DbAdapter.create("expansions", EXPANSIONS_DB)
