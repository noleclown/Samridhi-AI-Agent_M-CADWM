"""
samridhi/api.py
===============
Session management and chat handler for the FastAPI layer.

No Streamlit dependencies. Each HTTP session gets an isolated
bucket (rate limiter) and message history stored in memory.
Sessions expire after SESSION_TTL_SECONDS of inactivity.

Exports:
    SessionManager   — in-memory session store with TTL eviction
    ChatSession      — per-session state (history + rate bucket)
    handle_chat()    — runs the pipeline for one turn, returns ChatResponse
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from samridhi.config import cfg
from samridhi.logger import get_logger
from samridhi.pipeline import Pipeline, PipelineResult, RateLimiter
from samridhi.ui_strings import UI

log = get_logger()

SESSION_TTL_SECONDS = 3600   # 1 hour inactivity → session evicted
MAX_SESSIONS        = 5000   # hard cap — evict oldest when exceeded


# ══════════════════════════════════════════════════════════════
# SESSION
# ══════════════════════════════════════════════════════════════

@dataclass
class ChatSession:
    session_id:  str
    lang:        str = "en"
    messages:    list[dict] = field(default_factory=list)
    rate_bucket: dict = field(default_factory=RateLimiter.make_bucket)
    created_at:  float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    def touch(self):
        self.last_active = time.time()

    def ui(self) -> dict:
        return UI.get(self.lang, UI["en"])

    def add_user(self, content: str):
        self.messages.append(
            {"role": "user", "content": content, "follow_ups": [], "layer": ""}
        )

    def add_assistant(self, result: PipelineResult):
        self.messages.append({
            "role":       "assistant",
            "content":    result.answer,
            "follow_ups": result.follow_ups,
            "layer":      result.layer,
        })


# ══════════════════════════════════════════════════════════════
# SESSION MANAGER
# ══════════════════════════════════════════════════════════════

class SessionManager:
    """
    In-memory session store with TTL eviction.

    Thread-safe for concurrent FastAPI requests via asyncio
    (single-process — no locks needed since Python GIL protects dict ops).

    For multi-process deployments (multiple Gunicorn workers),
    replace _sessions with a Redis-backed store.
    """

    def __init__(self):
        self._sessions: dict[str, ChatSession] = {}

    def create(self, lang: str = "en") -> ChatSession:
        self._evict_stale()
        session = ChatSession(session_id=str(uuid.uuid4()), lang=lang)
        self._sessions[session.session_id] = session
        log.info(f"Session created: {session.session_id[:8]} lang={lang}")
        return session

    def get(self, session_id: str) -> Optional[ChatSession]:
        session = self._sessions.get(session_id)
        if session is None:
            return None
        if time.time() - session.last_active > SESSION_TTL_SECONDS:
            del self._sessions[session_id]
            log.info(f"Session expired: {session_id[:8]}")
            return None
        session.touch()
        return session

    def get_or_create(self, session_id: Optional[str], lang: str = "en") -> ChatSession:
        if session_id:
            session = self.get(session_id)
            if session:
                return session
        return self.create(lang)

    def delete(self, session_id: str):
        self._sessions.pop(session_id, None)

    def stats(self) -> dict:
        now = time.time()
        active = sum(
            1 for s in self._sessions.values()
            if now - s.last_active < SESSION_TTL_SECONDS
        )
        return {"total": len(self._sessions), "active": active}

    def _evict_stale(self):
        now = time.time()
        stale = [
            sid for sid, s in self._sessions.items()
            if now - s.last_active > SESSION_TTL_SECONDS
        ]
        for sid in stale:
            del self._sessions[sid]
        # Hard cap — evict oldest if still over limit
        if len(self._sessions) >= MAX_SESSIONS:
            oldest = sorted(self._sessions.values(), key=lambda s: s.last_active)
            for s in oldest[:len(self._sessions) - MAX_SESSIONS + 1]:
                del self._sessions[s.session_id]
            log.warning(f"Session cap reached — evicted {len(oldest)} sessions")


# ══════════════════════════════════════════════════════════════
# RESPONSE SCHEMA
# ══════════════════════════════════════════════════════════════

@dataclass
class ChatResponse:
    session_id:  str
    answer:      str
    layer:       str
    follow_ups:  list[str]
    confidence:  float
    intent:      str
    response_ms: float
    lang:        str

    def to_dict(self) -> dict:
        return {
            "session_id":  self.session_id,
            "answer":      self.answer,
            "layer":       self.layer,
            "follow_ups":  self.follow_ups,
            "confidence":  float(self.confidence),   # cast numpy float32 → Python float
            "intent":      self.intent,
            "response_ms": float(self.response_ms),  # cast numpy float32 → Python float
            "lang":        self.lang,
        }


# ══════════════════════════════════════════════════════════════
# CHAT HANDLER
# ══════════════════════════════════════════════════════════════

def handle_chat(
    question:  str,
    session:   ChatSession,
    pipeline:  Pipeline,
) -> ChatResponse:
    """
    Run one chat turn through the pipeline.

    1. Add user message to session history
    2. Run pipeline (uses session history for conversation context)
    3. Add assistant response to session history
    4. Return ChatResponse

    Stateless from the pipeline's perspective — all state lives in ChatSession.
    """
    session.add_user(question)

    result: PipelineResult = pipeline.run(
        question,
        session.lang,
        session.messages,
        session.rate_bucket,
        session.ui(),
    )

    session.add_assistant(result)
    pipeline.maybe_reingest(session.lang, set())

    return ChatResponse(
        session_id  = session.session_id,
        answer      = result.answer,
        layer       = result.layer,
        follow_ups  = result.follow_ups,
        confidence  = result.confidence,
        intent      = result.intent,
        response_ms = result.response_ms,
        lang        = session.lang,
    )
