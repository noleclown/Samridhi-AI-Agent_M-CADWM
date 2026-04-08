"""
test_samridhi.py — Unit tests for Samridhi pure functions
==========================================================
Run with:  pytest test_samridhi.py -v

These tests cover all functions that do NOT require Streamlit,
FAISS, or the Groq LLM — i.e. everything that can be tested
without the full runtime stack.
"""

import hashlib
import re
import time
import pytest

# ── Inline reimplementations of pure functions ─────────────────
# We duplicate the logic here so tests can run without importing
# the full app (which would try to connect to Streamlit).

_GREETING_SET = frozenset({
    "hi", "hello", "hey", "good morning", "good evening", "good afternoon",
    "namaste", "namaskar", "howdy", "greetings", "helo", "hii", "hiya",
    "नमस्ते", "हेलो", "हाय",
})

def is_greeting(text: str) -> bool:
    t = text.lower().strip()
    if len(t.split()) > 5:
        return False
    return t in _GREETING_SET or any(t.startswith(g) for g in _GREETING_SET)

def _cache_key(q: str, language: str) -> str:
    return hashlib.md5(f"{language}::{q.lower().strip()}".encode()).hexdigest()

_STOPWORDS = frozenset({
    "a","an","the","is","are","in","on","at","to","of","and","or",
    "for","what","tell","me","how","do","can","please","give","about",
})

def _keywords(query: str) -> list:
    return [w for w in re.findall(r'\w+', query.lower())
            if w not in _STOPWORDS and len(w) > 2]

def _keyword_score(text: str, kws: list) -> float:
    if not kws:
        return 0.0
    tl = text.lower()
    return sum(1 for w in kws if w in tl) / len(kws)

_RELATED_KW = frozenset({
    "water","irrigation","agriculture","smis","cadwm","mcadwm","m-cadwm",
    "scheme","wua","farmer","kisan","yojana","subsidy","registration",
})

def is_related_topic(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in _RELATED_KW)

_INTENT_KW = {
    "definition": frozenset({"what is","what are","define","definition","meaning","explain"}),
    "procedure":  frozenset({"how to","how do","steps","process","register","apply","submit"}),
    "data":       frozenset({"how many","statistics","figure","budget","coverage","total","fund"}),
    "smis":       frozenset({"smis","portal","login","password","dashboard","data entry","report"}),
}
INTENTS = frozenset({"definition","procedure","data","smis","general"})

def classify_intent_kw(query: str) -> str:
    ql = query.lower()
    for intent, kws in _INTENT_KW.items():
        if any(kw in ql for kw in kws):
            return intent
    return "general"

def _compaction_eligible(entry: dict, max_age_days: int, min_thumbsup: int) -> bool:
    """Return True if entry should be DROPPED during compaction."""
    if entry.get("thumbs_up", 0) < min_thumbsup:
        return True
    last = entry.get("last_updated", "")
    if last:
        try:
            from datetime import datetime, timedelta
            age = datetime.now() - datetime.fromisoformat(last)
            if age > timedelta(days=max_age_days):
                return True
        except Exception:
            pass
    return False

TTS_ABBR = {
    "M-CADWM": "M-CAD-WM",
    "SMIS":    "S-MIS",
    "&":       "and",
    "%":       "percent",
}

def clean_tts_en(text: str) -> str:
    text = re.sub(r"#{1,6}\s*", "", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"^\s*[-•]\s*", ", ", text, flags=re.MULTILINE)
    for abbr, exp in TTS_ABBR.items():
        text = text.replace(abbr, exp)
    text = re.sub(r"\n{2,}", ". ", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


# ══════════════════════════════════════════════════════════════
# TEST CLASSES
# ══════════════════════════════════════════════════════════════

class TestGreeting:
    def test_basic_greetings(self):
        assert is_greeting("hello") is True
        assert is_greeting("hi") is True
        assert is_greeting("namaste") is True
        assert is_greeting("नमस्ते") is True

    def test_greeting_with_name(self):
        assert is_greeting("hello there") is True

    def test_greeting_too_long(self):
        assert is_greeting("hello can you help me with smis registration") is False

    def test_not_greeting(self):
        assert is_greeting("what is WUA") is False
        assert is_greeting("how to register on SMIS") is False

    def test_case_insensitive(self):
        assert is_greeting("HELLO") is True
        assert is_greeting("Hello") is True


class TestCacheKey:
    def test_language_isolation(self):
        en_key = _cache_key("what is WUA", "en")
        hi_key = _cache_key("what is WUA", "hi")
        assert en_key != hi_key, "EN and HI keys must differ"

    def test_normalisation(self):
        k1 = _cache_key("What is WUA", "en")
        k2 = _cache_key("  what is wua  ", "en")
        assert k1 == k2, "Should normalise case and whitespace"

    def test_different_questions(self):
        k1 = _cache_key("what is WUA", "en")
        k2 = _cache_key("what is SMIS", "en")
        assert k1 != k2


class TestKeywords:
    def test_removes_stopwords(self):
        kws = _keywords("what is the SMIS portal")
        assert "what" not in kws
        assert "the" not in kws
        assert "smis" in kws

    def test_short_words_removed(self):
        kws = _keywords("WUA is a key body")
        assert "is" not in kws
        assert "a" not in kws
        assert "wua" in kws

    def test_empty(self):
        assert _keywords("") == []

    def test_keyword_score_perfect(self):
        score = _keyword_score("SMIS portal registration form", ["smis", "portal", "registration"])
        assert score == 1.0

    def test_keyword_score_zero(self):
        score = _keyword_score("unrelated text about nothing", ["smis", "portal"])
        assert score == 0.0

    def test_keyword_score_partial(self):
        score = _keyword_score("SMIS portal login", ["smis", "portal", "registration"])
        assert 0.0 < score < 1.0


class TestIntentClassifier:
    def test_definition_intent(self):
        assert classify_intent_kw("what is WUA") == "definition"
        assert classify_intent_kw("define M-CADWM") == "definition"
        assert classify_intent_kw("explain the scheme") == "definition"

    def test_procedure_intent(self):
        assert classify_intent_kw("how to register on SMIS") == "procedure"
        assert classify_intent_kw("steps to apply for the scheme") == "procedure"

    def test_data_intent(self):
        assert classify_intent_kw("how many beneficiaries") == "data"
        assert classify_intent_kw("total fund allocation") == "data"

    def test_smis_intent(self):
        assert classify_intent_kw("SMIS portal login issue") == "smis"
        assert classify_intent_kw("data entry on portal") == "smis"

    def test_general_fallback(self):
        assert classify_intent_kw("cadwm objectives") == "general"


class TestRelatedTopic:
    def test_related(self):
        assert is_related_topic("water users association eligibility") is True
        assert is_related_topic("SMIS portal registration") is True
        assert is_related_topic("irrigation scheme guidelines") is True

    def test_unrelated(self):
        assert is_related_topic("cricket match score") is False
        assert is_related_topic("best restaurants in Delhi") is False


class TestCompaction:
    def test_drops_zero_thumbsup(self):
        entry = {"thumbs_up": 0, "thumbs_down": 2, "last_updated": ""}
        assert _compaction_eligible(entry, 180, 1) is True

    def test_keeps_good_entry(self):
        from datetime import datetime
        entry = {
            "thumbs_up": 3,
            "thumbs_down": 0,
            "last_updated": datetime.now().isoformat()
        }
        assert _compaction_eligible(entry, 180, 1) is False

    def test_drops_old_entry(self):
        from datetime import datetime, timedelta
        old_date = (datetime.now() - timedelta(days=200)).isoformat()
        entry = {"thumbs_up": 5, "thumbs_down": 0, "last_updated": old_date}
        assert _compaction_eligible(entry, 180, 1) is True


class TestTTSCleaner:
    def test_strips_markdown_headers(self):
        result = clean_tts_en("## Introduction\nSome text")
        assert "##" not in result

    def test_strips_bold(self):
        result = clean_tts_en("This is **important** text")
        assert "**" not in result
        assert "important" in result

    def test_expands_abbreviations(self):
        result = clean_tts_en("M-CADWM scheme")
        assert "M-CAD-WM" in result

    def test_collapses_whitespace(self):
        result = clean_tts_en("word   word")
        assert "  " not in result

    def test_merges_newlines(self):
        result = clean_tts_en("line one\n\nline two")
        assert "\n" not in result
