"""
samridhi/resources.py
=====================
Process-level singletons loaded once via @st.cache_resource.

Exports:
    get_vector_db()   → FAISS index
    get_llm()         → ChatGroq client
    get_feedback_db() → FeedbackCache instance
    get_web_cache()   → WebCache instance
    get_analytics()   → AnalyticsLog instance

All heavy resources are loaded exactly once per Streamlit server process
and shared across all reruns and users.
"""

from __future__ import annotations

import streamlit as st

from samridhi.cache import AnalyticsLog, ExpansionStore, FeedbackCache, SessionStore, WebCache
from samridhi.config import FAISS_PATH, BASE_DIR, cfg
from samridhi.logger import get_logger

log = get_logger()


@st.cache_resource(show_spinner=False)
def get_vector_db():
    """Load FAISS index + embedding model once. Cached at process level."""
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings

    emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder=str(BASE_DIR / "models"),
    )
    db = FAISS.load_local(FAISS_PATH, emb, allow_dangerous_deserialization=True)
    log.info("FAISS index loaded.")
    return db


@st.cache_resource(show_spinner=False)
def get_llm():
    """Create ChatGroq client once. Cached at process level."""
    import os
    from langchain_groq import ChatGroq

    key = os.getenv("GROQ_API_KEY")
    if not key:
        try:
            key = st.secrets["GROQ_API_KEY"]
        except Exception:
            key = None

    client = ChatGroq(
        groq_api_key=key,
        model_name=cfg["llm"]["model"],
        temperature=cfg["llm"]["temperature"],
        max_tokens=cfg["llm"]["max_tokens"],
    )
    log.info(f"LLM client created: {cfg['llm']['model']}")
    return client


@st.cache_resource(show_spinner=False)
def get_feedback_db() -> FeedbackCache:
    """Shared FeedbackCache instance (SQLite WAL, compacted on init)."""
    return FeedbackCache()


@st.cache_resource(show_spinner=False)
def get_web_cache() -> WebCache:
    """Shared WebCache instance (two-layer: memory + SQLite WAL)."""
    return WebCache()


@st.cache_resource(show_spinner=False)
def get_analytics() -> AnalyticsLog:
    """Shared AnalyticsLog instance (rotating JSONL)."""
    return AnalyticsLog()


@st.cache_resource(show_spinner=False)
def get_expansions() -> ExpansionStore:
    """
    Shared ExpansionStore instance (SQLite WAL).
    Seeded with default acronyms on first run.
    Admin-editable at runtime — no restart needed.
    """
    return ExpansionStore()


@st.cache_resource(show_spinner=False)
def get_session_store() -> SessionStore:
    """
    Shared SessionStore instance.
    Persists named conversation sessions across page reloads.
    """
    return SessionStore()
