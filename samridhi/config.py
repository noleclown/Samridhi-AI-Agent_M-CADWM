"""
samridhi/config.py
==================
Single source of truth for all runtime constants.

Load order:
  1. _CFG_DEFAULTS  (hardcoded, always present)
  2. config.yaml    (optional, deep-merged on top; no code change needed to tune)

All other modules import `cfg` and `BASE_DIR` from here.
Nothing in this module imports from other samridhi modules (no circular deps).
"""

from __future__ import annotations

import os
import logging
from pathlib import Path

# ── paths ────────────────────────────────────────────────────
BASE_DIR       = Path(os.path.dirname(os.path.abspath(__file__))).parent
FAISS_PATH     = str(BASE_DIR / "faiss_index")
LOGO_PATH      = BASE_DIR / "logo.png"
FEEDBACK_DB    = BASE_DIR / "feedback.db"
WEB_DB_FILE    = BASE_DIR / "web_cache.db"
EXPANSIONS_DB  = BASE_DIR / "expansions.db"   # admin-editable query expansion store
ANALYTICS_FILE = BASE_DIR / "analytics.jsonl"
CONFIG_FILE    = BASE_DIR / "config.yaml"
LOG_FILE       = BASE_DIR / "samridhi.log"

# ── web ──────────────────────────────────────────────────────
CADWM_BASE  = "http://cadwm.gov.in/"
WEB_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; SamridhiBot/4.0)"}

# ── schema ───────────────────────────────────────────────────
FEEDBACK_SCHEMA_VERSION = "4.0"
INTENTS = frozenset({"definition", "procedure", "data", "smis", "general"})

# ── hardcoded defaults ────────────────────────────────────────
_CFG_DEFAULTS: dict = {
    "retrieval": {
        "faiss_confidence_threshold": 0.35,
        "faiss_k": 10,
        "faiss_max_dist": 1.4,
        "faiss_semantic_weight": 0.60,
        "faiss_keyword_weight": 0.40,
        "mmr_lambda": 0.5,
        "mmr_fetch_k": 20,
        "context_turns": 2,
        "max_input_chars": 500,
    },
    "cache": {
        "web_ttl_seconds": 86400,
        "session_web_ttl_seconds": 300,
        "feedback_reingest_min_votes": 3,
        "feedback_max_age_days": 180,
        "feedback_keep_min_thumbsup": 1,
    },
    "llm": {
        "model": "llama-3.3-70b-versatile",
        "temperature": 0.15,
        "max_tokens": 1024,
        "typo_skip_words": 4,
        "rate_capacity": 10,
        "rate_refill_seconds": 60,
    },
    "tts": {
        "en_voice": "en-IN-NeerjaNeural",
        "hi_voice": "hi-IN-SwaraNeural",
        "en_rate": "+10%",
        "hi_rate": "+5%",
    },
    "web_scrape": {
        "max_pages": 3,
        "per_page_chars": 3000,
        "timeout_seconds": 8,
    },
    "analytics": {
        "enabled": True,
        "max_file_mb": 50,
    },
    "database": {
        "backend":      "sqlite",    # "sqlite" or "postgres"
        "postgres_url": "",          # e.g. "postgresql://user:pass@host:5432/samridhi"
    },
    "ui": {
        "starter_questions_en": [
            "What is the M-CADWM scheme?",
            "How do I register on the SMIS portal?",
            "What are the eligibility criteria for WUA formation?",
            "What documents are required for M-CADWM beneficiaries?",
            "How does M-CADWM funding work?",
            "What is the role of a Water Users Association?",
        ],
        "starter_questions_hi": [
            "M-CADWM योजना क्या है?",
            "SMIS पोर्टल पर पंजीकरण कैसे करें?",
            "WUA गठन के लिए पात्रता मानदंड क्या हैं?",
            "M-CADWM लाभार्थियों के लिए कौन से दस्तावेज़ आवश्यक हैं?",
            "M-CADWM में वित्त पोषण कैसे कार्य करता है?",
            "जल उपयोगकर्ता संघ की क्या भूमिका है?",
        ],
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Returns new dict; base is never mutated."""
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config() -> dict:
    """
    Load config.yaml if present and merge over defaults.
    Returns the merged config dict.
    Falls back silently to defaults if yaml is missing or uninstalled.
    """
    try:
        import yaml  # type: ignore
        if CONFIG_FILE.exists():
            raw = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8")) or {}
            merged = _deep_merge(_CFG_DEFAULTS, raw)
            logging.getLogger("samridhi").info("config.yaml loaded and merged.")
            return merged
    except Exception as e:
        logging.getLogger("samridhi").warning(f"config.yaml load failed ({e}); using defaults.")
    return dict(_CFG_DEFAULTS)


# Module-level singleton — imported by all other modules
cfg: dict = load_config()
