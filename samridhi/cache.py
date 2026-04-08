"""
samridhi/cache.py
=================
All persistent and in-memory caching.

  FeedbackCache  — user thumbs-up/down, language-isolated, versioned
  WebCache       — scraped cadwm.gov.in pages, TTL-based
  ExpansionStore — admin-editable acronym expansion table
  AnalyticsLog   — rotating JSONL append-only query log

All DB classes use DbAdapter (samridhi/db.py) instead of sqlite3 directly.
Switching from SQLite → PostgreSQL is a single config.yaml change:

    database:
      backend: postgres
      postgres_url: "postgresql://user:password@host:5432/samridhi"

No code changes needed in this file or anywhere else.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

from samridhi.config import (
    ANALYTICS_FILE, BASE_DIR, EXPANSIONS_DB, FEEDBACK_DB,
    FEEDBACK_SCHEMA_VERSION, WEB_DB_FILE, cfg,
)
from samridhi.db import DbAdapter, expansions_db_adapter, feedback_db_adapter, web_db_adapter
from samridhi.logger import get_logger

log = get_logger()


# ══════════════════════════════════════════════════════════════
# KEY HELPERS
# ══════════════════════════════════════════════════════════════

def make_cache_key(q: str, lang: str) -> str:
    """
    Language-isolated key: MD5("en::normalised_query").
    Same question in EN and HI → different hashes → zero cross-language leakage.
    O(1).
    """
    return hashlib.md5(f"{lang}::{q.lower().strip()}".encode()).hexdigest()


def make_web_key(q: str) -> str:
    """Web cache key — language-agnostic (page content is independent of UI language)."""
    return hashlib.md5(q.lower().strip().encode()).hexdigest()


# ══════════════════════════════════════════════════════════════
# FEEDBACK CACHE
# ══════════════════════════════════════════════════════════════

class FeedbackCache:
    """
    Persistent store for user feedback (thumbs-up / thumbs-down).

    Schema (v4.0):
        key          TEXT PRIMARY KEY   — MD5(lang::query)
        schema_ver   TEXT
        lang         TEXT               — "en" or "hi"
        question     TEXT
        answer       TEXT
        thumbs_up    INTEGER DEFAULT 0
        thumbs_down  INTEGER DEFAULT 0
        last_updated TEXT               — ISO-8601
        reingested   INTEGER DEFAULT 0  — 1 after FAISS re-ingest

    Backend: SQLite (default) or PostgreSQL — controlled by config.yaml.
    Compaction removes stale and low-vote entries on init.
    """

    def __init__(self, db_path: Path = FEEDBACK_DB):
        self._db = feedback_db_adapter()
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                key          TEXT PRIMARY KEY,
                schema_ver   TEXT,
                lang         TEXT,
                question     TEXT,
                answer       TEXT,
                thumbs_up    INTEGER DEFAULT 0,
                thumbs_down  INTEGER DEFAULT 0,
                last_updated TEXT,
                reingested   INTEGER DEFAULT 0
            )
        """)
        # Index on lang for fast language-filtered queries
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_lang ON feedback(lang)"
        )
        self._db.commit()
        self._compact()
        log.info(f"FeedbackCache ready [{self._db.backend}]")

    # ── public API ───────────────────────────────────────────

    def get(self, q: str, lang: str) -> str | None:
        """Return cached answer if thumbs_up > thumbs_down, else None. O(1)."""
        key = make_cache_key(q, lang)
        row = self._db.fetchone(
            "SELECT answer, thumbs_up, thumbs_down FROM feedback WHERE key=?", (key,)
        )
        if row and row[1] > row[2]:
            return row[0]
        return None

    def record(self, q: str, answer: str, vote: str, lang: str):
        """
        Record a thumbs-up or thumbs-down vote.
        Upserts: creates entry on first vote, increments counter on subsequent votes.
        """
        key = make_cache_key(q, lang)
        now = datetime.now().isoformat()
        existing = self._db.fetchone(
            "SELECT thumbs_up, thumbs_down FROM feedback WHERE key=?", (key,)
        )
        try:
            if existing:
                col = "thumbs_up" if vote == "up" else "thumbs_down"
                self._db.execute(
                    f"UPDATE feedback SET {col}={col}+1, answer=?, last_updated=? WHERE key=?",
                    (answer, now, key)
                )
            else:
                self._db.execute(
                    "INSERT OR REPLACE INTO feedback VALUES (?,?,?,?,?,?,?,?,?)",
                    (key, FEEDBACK_SCHEMA_VERSION, lang, q, answer,
                     1 if vote == "up" else 0,
                     0 if vote == "up" else 1,
                     now, 0)
                )
            self._db.commit()
            log.info(f"Feedback: {vote} | key={key[:8]} | lang={lang}")
        except Exception as e:
            log.warning(f"FeedbackCache.record failed: {e}")

    def mark_reingested(self, q: str, lang: str):
        """Mark an entry as already re-ingested into FAISS (prevents double injection)."""
        key = make_cache_key(q, lang)
        try:
            self._db.execute(
                "UPDATE feedback SET reingested=1 WHERE key=?", (key,)
            )
            self._db.commit()
        except Exception as e:
            log.warning(f"FeedbackCache.mark_reingested failed: {e}")

    def get_reingest_candidates(self, min_votes: int, lang: str) -> list[dict]:
        """Return entries eligible for FAISS re-ingest: thumbs_up >= min_votes, not yet done."""
        rows = self._db.fetchall(
            "SELECT key, question, answer, lang FROM feedback "
            "WHERE thumbs_up >= ? AND reingested=0 AND lang=?",
            (min_votes, lang)
        )
        return [{"key": r[0], "question": r[1], "answer": r[2], "lang": r[3]}
                for r in rows]

    def stats(self) -> dict:
        """Return basic stats for health check display."""
        total = self._db.fetchone("SELECT COUNT(*) FROM feedback")
        pos   = self._db.fetchone(
            "SELECT COUNT(*) FROM feedback WHERE thumbs_up > thumbs_down"
        )
        return {
            "total":    total[0] if total else 0,
            "positive": pos[0]   if pos   else 0,
            "backend":  self._db.backend,
        }

    # ── private ──────────────────────────────────────────────

    def _compact(self):
        """
        Remove entries that are stale (> max_age_days) or
        have zero engagement (thumbs_up < min AND thumbs_down = 0).
        Runs at init — keeps DB lean over time.
        """
        max_age = cfg["cache"]["feedback_max_age_days"]
        min_up  = cfg["cache"]["feedback_keep_min_thumbsup"]
        cutoff  = (datetime.now() - timedelta(days=max_age)).isoformat()
        try:
            self._db.execute(
                "DELETE FROM feedback "
                "WHERE (thumbs_up < ? AND thumbs_down = 0) OR last_updated < ?",
                (min_up, cutoff)
            )
            self._db.commit()
            log.info("FeedbackCache compacted.")
        except Exception as e:
            log.warning(f"FeedbackCache._compact failed: {e}")


# ══════════════════════════════════════════════════════════════
# WEB CACHE
# ══════════════════════════════════════════════════════════════

class WebCache:
    """
    Two-layer web scrape cache:
      Layer 1 — in-memory dict           (session_web_ttl_seconds — avoids DB hit)
      Layer 2 — SQLite or PostgreSQL DB  (web_ttl_seconds = 24 hr, survives restart)

    Stale DB rows purged on init.
    Backend controlled by config.yaml (same as FeedbackCache).
    """

    def __init__(self, db_path: Path = WEB_DB_FILE):
        self._db  = web_db_adapter()
        self._mem: dict[str, dict] = {}   # {key: {"text": str, "ts": float}}
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS web_cache "
            "(key TEXT PRIMARY KEY, text TEXT, ts REAL)"
        )
        self._db.commit()
        self._purge_stale()
        log.info(f"WebCache ready [{self._db.backend}]")

    def get(self, query: str) -> str | None:
        """Return cached text or None. Checks memory first (O(1)), then DB."""
        key = make_web_key(query)
        now = time.time()

        # L1 — in-memory guard
        entry = self._mem.get(key)
        if entry and (now - entry["ts"]) < cfg["cache"]["session_web_ttl_seconds"]:
            return entry["text"]

        # L2 — DB
        row = self._db.fetchone(
            "SELECT text, ts FROM web_cache WHERE key=?", (key,)
        )
        if row and (now - row[1]) < cfg["cache"]["web_ttl_seconds"]:
            self._mem[key] = {"text": row[0], "ts": row[1]}
            return row[0]
        return None

    def set(self, query: str, text: str | None):
        """Store scraped text in both memory and DB layers."""
        key = make_web_key(query)
        ts  = time.time()
        self._mem[key] = {"text": text, "ts": ts}
        try:
            self._db.execute(
                "INSERT OR REPLACE INTO web_cache VALUES (?,?,?)", (key, text, ts)
            )
            self._db.commit()
        except Exception as e:
            log.warning(f"WebCache.set failed: {e}")

    def stats(self) -> dict:
        row = self._db.fetchone("SELECT COUNT(*) FROM web_cache")
        return {
            "total_entries": row[0] if row else 0,
            "backend": self._db.backend,
        }

    def _purge_stale(self):
        cutoff = time.time() - cfg["cache"]["web_ttl_seconds"]
        try:
            self._db.execute("DELETE FROM web_cache WHERE ts < ?", (cutoff,))
            self._db.commit()
            log.info("WebCache stale rows purged.")
        except Exception as e:
            log.warning(f"WebCache._purge_stale failed: {e}")


# ══════════════════════════════════════════════════════════════
# EXPANSION STORE
# ══════════════════════════════════════════════════════════════

# Seed data — loaded once on first run if the table is empty.
_EXPANSION_SEEDS: list[tuple[str, str, str]] = [
    ("wua",    "water users association WUA",
     "Water Users Association — farmer collective managing irrigation infrastructure"),
    ("wus",    "water users society WUS",
     "Water Users Society — registered body under WUA"),
    ("cad",    "command area development CAD",
     "Command Area Development — area served by an irrigation project"),
    ("o&m",    "operations and maintenance O&M",
     "Operations and Maintenance — upkeep of irrigation infrastructure"),
    ("pmksy",  "pradhan mantri krishi sinchayee yojana PMKSY",
     "Pradhan Mantri Krishi Sinchayee Yojana — national irrigation scheme"),
    ("imti",   "irrigation management transfer institute IMTI",
     "Irrigation Management Transfer Institute — training body"),
    ("smis",   "scheme monitoring information system SMIS",
     "Scheme Monitoring Information System — online portal for M-CADWM"),
    ("mcadwm", "M-CADWM modernisation command area development water management scheme CADWM CAD",
     "M-CADWM — all variant spellings: MCADWM, M-CADWM, MCAD, CAD-WM all refer to this scheme"),
]


class ExpansionStore:
    """
    Persistent store for query expansion acronyms.

    Schema:
        acronym    TEXT PRIMARY KEY  — lowercase (e.g. "wua")
        expansion  TEXT              — full-form appended to query
        notes      TEXT              — human-readable description
        enabled    INTEGER DEFAULT 1 — 0 to disable without deleting
        updated_at TEXT              — ISO-8601

    Seeded with _EXPANSION_SEEDS on first run.
    Editable at runtime via add/remove/toggle — no restart needed.
    In-memory dict cache keeps retrieval at O(1) after initial load.
    Backend controlled by config.yaml.
    """

    def __init__(self, db_path: Path = EXPANSIONS_DB):
        self._db = expansions_db_adapter()
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS expansions (
                acronym    TEXT PRIMARY KEY,
                expansion  TEXT NOT NULL,
                notes      TEXT DEFAULT '',
                enabled    INTEGER DEFAULT 1,
                updated_at TEXT DEFAULT ''
            )
        """)
        self._db.commit()
        self._seed_if_empty()
        self._cache: dict[str, str] = self._load_cache()
        log.info(
            f"ExpansionStore ready [{self._db.backend}] — "
            f"{len(self._cache)} active expansions"
        )

    # ── public API ───────────────────────────────────────────

    def get_all(self) -> dict[str, str]:
        """Return {acronym: expansion} for all enabled entries. Refreshes from DB each call."""
        self._cache = self._load_cache()
        return self._cache

    def add(self, acronym: str, expansion: str, notes: str = "") -> bool:
        """Insert or replace an expansion. Returns True on success."""
        acronym = acronym.lower().strip()
        now     = datetime.now().isoformat()
        try:
            self._db.execute(
                "INSERT OR REPLACE INTO expansions "
                "(acronym, expansion, notes, enabled, updated_at) VALUES (?,?,?,1,?)",
                (acronym, expansion, notes, now)
            )
            self._db.commit()
            self._cache = self._load_cache()
            log.info(f"ExpansionStore: added/updated '{acronym}'")
            return True
        except Exception as e:
            log.warning(f"ExpansionStore.add failed: {e}")
            return False

    def remove(self, acronym: str) -> bool:
        """Delete an entry permanently. Use toggle(False) to disable without deleting."""
        try:
            self._db.execute(
                "DELETE FROM expansions WHERE acronym=?", (acronym.lower().strip(),)
            )
            self._db.commit()
            self._cache = self._load_cache()
            log.info(f"ExpansionStore: removed '{acronym}'")
            return True
        except Exception as e:
            log.warning(f"ExpansionStore.remove failed: {e}")
            return False

    def toggle(self, acronym: str, enabled: bool) -> bool:
        """Enable or disable an entry without deleting it."""
        now = datetime.now().isoformat()
        try:
            self._db.execute(
                "UPDATE expansions SET enabled=?, updated_at=? WHERE acronym=?",
                (1 if enabled else 0, now, acronym.lower().strip())
            )
            self._db.commit()
            self._cache = self._load_cache()
            log.info(f"ExpansionStore: toggled '{acronym}' → {'on' if enabled else 'off'}")
            return True
        except Exception as e:
            log.warning(f"ExpansionStore.toggle failed: {e}")
            return False

    def all_rows(self) -> list[dict]:
        """Return all rows including disabled ones — for admin/health UI display."""
        rows = self._db.fetchall(
            "SELECT acronym, expansion, notes, enabled, updated_at "
            "FROM expansions ORDER BY acronym"
        )
        return [
            {"acronym": r[0], "expansion": r[1], "notes": r[2],
             "enabled": bool(r[3]), "updated_at": r[4]}
            for r in rows
        ]

    def stats(self) -> dict:
        total   = self._db.fetchone("SELECT COUNT(*) FROM expansions")
        enabled = self._db.fetchone(
            "SELECT COUNT(*) FROM expansions WHERE enabled=1"
        )
        return {
            "total":   total[0]   if total   else 0,
            "enabled": enabled[0] if enabled else 0,
            "backend": self._db.backend,
        }

    # ── private ──────────────────────────────────────────────

    def _load_cache(self) -> dict[str, str]:
        rows = self._db.fetchall(
            "SELECT acronym, expansion FROM expansions WHERE enabled=1"
        )
        return {r[0]: r[1] for r in rows}

    def _seed_if_empty(self):
        """Seed the table with defaults on first run only."""
        row = self._db.fetchone("SELECT COUNT(*) FROM expansions")
        if row and row[0] == 0:
            now = datetime.now().isoformat()
            try:
                self._db.executemany(
                    "INSERT OR REPLACE INTO expansions "
                    "(acronym, expansion, notes, enabled, updated_at) VALUES (?,?,?,1,?)",
                    [(a, e, n, now) for a, e, n in _EXPANSION_SEEDS]
                )
                self._db.commit()
                log.info(f"ExpansionStore seeded with {len(_EXPANSION_SEEDS)} entries.")
            except Exception as e:
                log.warning(f"ExpansionStore._seed_if_empty failed: {e}")


# ══════════════════════════════════════════════════════════════
# SESSION STORE  (SQLite WAL — persists conversations across reloads)
# ══════════════════════════════════════════════════════════════

class SessionStore:
    """
    Persists named conversation sessions so users can restore
    a previous chat after clicking "New conversation".

    Schema:
        session_id   TEXT PRIMARY KEY  — UUID string
        title        TEXT              — first user message (truncated)
        lang         TEXT              — "en" or "hi"
        messages     TEXT              — JSON-serialised message list
        created_at   TEXT              — ISO-8601
        updated_at   TEXT              — ISO-8601

    Keeps the most recent MAX_SESSIONS sessions per language.
    Older sessions are pruned automatically on save.

    Uses the same DbAdapter as the other caches — inherits
    SQLite/PostgreSQL backend from config.yaml.
    """

    MAX_SESSIONS = 20   # max stored sessions per language
    TITLE_LEN    = 60   # chars to use from first user message as title

    def __init__(self):
        self._db = DbAdapter.create("sessions", BASE_DIR / "sessions.db")
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                title      TEXT,
                lang       TEXT,
                messages   TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_sess_lang ON sessions(lang, updated_at)"
        )
        self._db.commit()
        log.info("SessionStore ready")

    # ── public API ───────────────────────────────────────────

    def save(self, session_id: str, messages: list, lang: str):
        """
        Save or update a session.
        Title is derived from the first user message.
        Prunes oldest sessions if MAX_SESSIONS is exceeded.
        """
        # Extract title from first user message
        title = next(
            (m["content"][:self.TITLE_LEN] for m in messages if m.get("role") == "user"),
            "Untitled session"
        )
        now  = datetime.now().isoformat()
        data = json.dumps(
            [{"role": m["role"], "content": m["content"]}
             for m in messages],   # strip follow_ups — not needed in history
            ensure_ascii=False
        )
        try:
            existing = self._db.fetchone(
                "SELECT session_id FROM sessions WHERE session_id=?", (session_id,)
            )
            if existing:
                self._db.execute(
                    "UPDATE sessions SET title=?, messages=?, updated_at=? WHERE session_id=?",
                    (title, data, now, session_id)
                )
            else:
                self._db.execute(
                    "INSERT INTO sessions VALUES (?,?,?,?,?,?)",
                    (session_id, title, lang, data, now, now)
                )
            self._db.commit()
            self._prune(lang)
            log.debug(f"SessionStore: saved session {session_id[:8]} ({lang})")
        except Exception as e:
            log.warning(f"SessionStore.save failed: {e}")

    def list_sessions(self, lang: str) -> list[dict]:
        """
        Return all sessions for a language, newest first.
        Each entry: {session_id, title, created_at, updated_at}
        """
        rows = self._db.fetchall(
            "SELECT session_id, title, created_at, updated_at "
            "FROM sessions WHERE lang=? ORDER BY updated_at DESC",
            (lang,)
        )
        return [
            {"session_id": r[0], "title": r[1],
             "created_at": r[2], "updated_at": r[3]}
            for r in rows
        ]

    def load(self, session_id: str) -> list[dict] | None:
        """
        Load messages for a session.
        Returns list of {role, content} dicts, or None if not found.
        """
        row = self._db.fetchone(
            "SELECT messages FROM sessions WHERE session_id=?", (session_id,)
        )
        if not row:
            return None
        try:
            msgs = json.loads(row[0])
            # Restore follow_ups field so render code doesn't error
            for m in msgs:
                m.setdefault("follow_ups", [])
            return msgs
        except Exception as e:
            log.warning(f"SessionStore.load failed: {e}")
            return None

    def delete(self, session_id: str):
        """Delete a session permanently."""
        try:
            self._db.execute(
                "DELETE FROM sessions WHERE session_id=?", (session_id,)
            )
            self._db.commit()
        except Exception as e:
            log.warning(f"SessionStore.delete failed: {e}")

    def stats(self) -> dict:
        row = self._db.fetchone("SELECT COUNT(*) FROM sessions")
        return {"total": row[0] if row else 0}

    # ── private ──────────────────────────────────────────────

    def _prune(self, lang: str):
        """Remove oldest sessions beyond MAX_SESSIONS for this language."""
        try:
            rows = self._db.fetchall(
                "SELECT session_id FROM sessions WHERE lang=? "
                "ORDER BY updated_at DESC",
                (lang,)
            )
            if len(rows) > self.MAX_SESSIONS:
                to_delete = [r[0] for r in rows[self.MAX_SESSIONS:]]
                for sid in to_delete:
                    self._db.execute(
                        "DELETE FROM sessions WHERE session_id=?", (sid,)
                    )
                self._db.commit()
                log.debug(f"SessionStore: pruned {len(to_delete)} old sessions")
        except Exception as e:
            log.warning(f"SessionStore._prune failed: {e}")


# ══════════════════════════════════════════════════════════════
# ANALYTICS LOG  (rotating JSONL — file-based, no DB backend)
# ══════════════════════════════════════════════════════════════

class AnalyticsLog:
    """
    Append-only JSONL query log.

    Each record: ts, lang, query, intent, layer, confidence, response_ms

    Rotates file when size exceeds analytics.max_file_mb (default 50 MB).
    File-based — not affected by DB backend choice.
    All writes are non-blocking (failures silently swallowed).
    """

    def __init__(self, path: Path = ANALYTICS_FILE):
        self._path = path

    def log(
        self,
        query: str,
        lang: str,
        layer: str,
        confidence: float,
        response_ms: float,
        intent: str = "",
    ):
        if not cfg["analytics"]["enabled"]:
            return
        self._maybe_rotate()
        record = {
            "ts":          datetime.utcnow().isoformat(),
            "lang":        lang,
            "query":       query,
            "intent":      intent,
            "layer":       layer,
            "confidence":  round(confidence, 4),
            "response_ms": round(response_ms, 1),
        }
        try:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def recent(self, n: int = 20) -> list[dict]:
        """Return the last n records (for health-check display)."""
        records = []
        try:
            lines = self._path.read_text(encoding="utf-8").splitlines()
            for line in lines[-n:]:
                if line.strip():
                    records.append(json.loads(line))
        except Exception:
            pass
        return records

    def _maybe_rotate(self):
        try:
            max_bytes = cfg["analytics"]["max_file_mb"] * 1024 * 1024
            if self._path.exists() and self._path.stat().st_size > max_bytes:
                rotated = self._path.with_suffix(f".{int(time.time())}.jsonl")
                os.replace(str(self._path), str(rotated))
                log.info(f"Analytics log rotated to {rotated.name}")
        except Exception:
            pass
