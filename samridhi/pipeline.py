"""
samridhi/pipeline.py
====================
Core RAG pipeline — pure Python, no Streamlit imports.

Exports:
    Pipeline  — stateless class wrapping the 7-layer retrieval flow
    RateLimiter — token-bucket, per-session

Pipeline.run(question, lang, history, session_web_cache) → PipelineResult

PipelineResult:
    answer      str
    layer       str   ("greeting"|"cache"|"faiss"|"live"|"fallback"|"error"|"rate_limited")
    follow_ups  list[str]
    confidence  float
    intent      str
    response_ms float
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from samridhi.config import INTENTS, cfg
from samridhi.logger import get_logger
from samridhi.prompts import (
    build_faiss_messages,
    build_fallback_messages,
    build_followup_prompt,
    build_intent_prompt,
    build_live_messages,
    build_typo_prompt,
)
from samridhi.retrieval import (
    classify_intent,
    correct_typos,
    fetch_live_context,
    is_greeting,
    is_related_topic,
    normalise_query,
    retrieve_docs,
)

log = get_logger()


# ══════════════════════════════════════════════════════════════
# RATE LIMITER  (token-bucket, per-session dict)
# ══════════════════════════════════════════════════════════════

class RateLimiter:
    """
    Token-bucket rate limiter.
    State is a plain dict so it can live in st.session_state.

    Capacity and refill period are read from cfg at call time,
    so changes to config.yaml take effect on next refill cycle.
    """

    @staticmethod
    def make_bucket() -> dict:
        capacity = cfg["llm"]["rate_capacity"]
        return {"tokens": float(capacity), "last_refill": time.time()}

    @staticmethod
    def consume(bucket: dict) -> bool:
        """
        Refill tokens proportionally to elapsed time, then consume one.
        Returns True if call is allowed, False if throttled.
        """
        capacity = cfg["llm"]["rate_capacity"]
        refill_s = cfg["llm"]["rate_refill_seconds"]
        now      = time.time()
        elapsed  = now - bucket["last_refill"]
        bucket["tokens"] = min(
            float(capacity),
            bucket["tokens"] + elapsed * (capacity / refill_s),
        )
        bucket["last_refill"] = now
        if bucket["tokens"] >= 1.0:
            bucket["tokens"] -= 1.0
            return True
        return False


# ══════════════════════════════════════════════════════════════
# PIPELINE RESULT
# ══════════════════════════════════════════════════════════════

@dataclass
class PipelineResult:
    answer:      str
    layer:       str
    follow_ups:  list[str] = field(default_factory=list)
    confidence:  float = 0.0
    intent:      str = "general"
    response_ms: float = 0.0


# ══════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════

class Pipeline:
    """
    Stateless 7-layer RAG pipeline.

    All resources (llm, vector_db, feedback_db, web_cache) are injected
    at construction time so this class is fully testable without Streamlit.

    Layer order:
      L0  Greeting           O(1) frozenset
      L1  Feedback cache     O(1) SQLite get
      L2  Typo correction    O(1) LLM  (skipped ≤ typo_skip_words)
      L3  Intent classify    O(n) keyword → O(1) LLM fallback
      L4  FAISS + MMR        O(k log n) hybrid retrieval
      L5  Live scrape        O(p) TTL-cached network
      L6  Scope fallback     O(1) LLM
    """

    def __init__(self, llm, vector_db, feedback_db, web_cache, analytics, expansion_store=None):
        self._llm             = llm
        self._vector_db       = vector_db
        self._feedback_db     = feedback_db
        self._web_cache       = web_cache
        self._analytics       = analytics
        self._expansion_store = expansion_store  # ExpansionStore | None
        self._auth_retried    = False             # guard against infinite retry loop

    # ── LLM wrapper with API key rotation ───────────────────

    def _invoke(self, messages: list, bucket: dict) -> str:
        """
        Rate-limited LLM call.
        Raises RuntimeError('rate_limited') if throttled.

        API key rotation handling:
          If Groq returns an AuthenticationError (e.g. key rotated / revoked),
          the LLM client is re-initialised once from the current environment.
          A second consecutive AuthenticationError propagates normally so the
          app surfaces the error rather than looping forever.
        """
        if not RateLimiter.consume(bucket):
            raise RuntimeError("rate_limited")
        try:
            result = self._llm.invoke(messages).content
            self._auth_retried = False  # reset on success
            return result
        except Exception as e:
            err_str = str(e).lower()
            # Detect authentication / invalid key errors from Groq
            if any(sig in err_str for sig in
                   ("authentication", "invalid api key", "401", "api key")):
                if not self._auth_retried:
                    log.warning("AuthenticationError detected — re-initialising LLM client.")
                    self._auth_retried = True
                    self._llm = self._reinit_llm()
                    # Retry the call once with the fresh client
                    return self._llm.invoke(messages).content
                else:
                    log.error(
                        "AuthenticationError on retry — GROQ_API_KEY may be invalid. "
                        "Update the key in .env or Streamlit secrets and restart."
                    )
            raise  # re-raise for caller to handle

    def _reinit_llm(self):
        """
        Re-create the LLM client from the current environment.
        Called automatically when an AuthenticationError is detected.
        Reads the key fresh from os.environ / st.secrets so a rotated
        key takes effect without restarting the server process.
        """
        import os
        from langchain_groq import ChatGroq
        key = os.getenv("GROQ_API_KEY", "")
        if not key:
            try:
                import streamlit as st
                key = st.secrets.get("GROQ_API_KEY", "")
            except Exception:
                pass
        return ChatGroq(
            groq_api_key=key,
            model_name=cfg["llm"]["model"],
            temperature=cfg["llm"]["temperature"],
            max_tokens=cfg["llm"]["max_tokens"],
        )

    # ── follow-up generator ──────────────────────────────────

    def _follow_ups(self, query: str, answer: str, lang: str, bucket: dict) -> list[str]:
        try:
            msgs = build_followup_prompt(query, answer, lang)
            raw  = self._invoke(msgs, bucket)
            lines = [
                l.strip().lstrip("•-123456789. ")
                for l in raw.splitlines() if l.strip()
            ]
            return [l for l in lines if len(l) > 5][:3]
        except Exception:
            return []

    # ── FAISS re-ingest ──────────────────────────────────────

    def maybe_reingest(self, lang: str, reingest_done: set):
        """
        Check feedback DB for highly-rated entries and inject them
        into FAISS as synthetic documents. Saves index to disk after each injection.
        """
        from langchain_core.documents import Document

        min_votes = cfg["cache"]["feedback_reingest_min_votes"]
        candidates = self._feedback_db.get_reingest_candidates(min_votes, lang)
        for entry in candidates:
            key = entry["key"]
            if key in reingest_done:
                continue
            try:
                doc = Document(
                    page_content=f"Q: {entry['question']}\nA: {entry['answer']}",
                    metadata={"source": "feedback", "lang": entry["lang"]},
                )
                self._vector_db.add_documents([doc])
                from samridhi.config import FAISS_PATH
                self._vector_db.save_local(FAISS_PATH)
                self._feedback_db.mark_reingested(entry["question"], lang)
                reingest_done.add(key)
                log.info(f"Re-ingested feedback key {key[:8]} into FAISS.")
            except Exception as e:
                log.warning(f"Re-ingest failed for key {key[:8]}: {e}")

    # ── main entry point ─────────────────────────────────────

    def run(
        self,
        question: str,
        lang: str,
        history: list[dict],
        bucket: dict,
        ui: dict,
    ) -> PipelineResult:
        """
        Execute the 7-layer pipeline and return a PipelineResult.

        Parameters
        ----------
        question  : raw user input
        lang      : "en" or "hi"
        history   : st.session_state.messages (for conversation context)
        bucket    : st.session_state.rate_bucket (mutable token-bucket dict)
        ui        : UI[lang] string dict (for localised messages)
        """
        t0 = time.time()

        def _ms():
            return round((time.time() - t0) * 1000, 1)

        def _log(layer, conf=0.0, intent=""):
            self._analytics.log(question, lang, layer, conf, _ms(), intent)

        # ── input length guard ───────────────────────────────
        # Queries over max_input_chars risk exceeding embedding model limits
        # and producing unpredictable FAISS results.
        # We truncate and prepend a visible warning so the user knows.
        max_chars    = cfg["retrieval"]["max_input_chars"]
        input_note   = ""
        if len(question) > max_chars:
            question   = question[:max_chars]
            input_note = ui.get("input_truncated", "").format(n=max_chars) + "\n\n"
            log.info(f"Input truncated to {max_chars} chars.")

        # ── query normalisation ──────────────────────────────
        # Maps all domain-term variants to canonical form BEFORE
        # anything else runs.  Prevents "MCADWM" / "M-CADWM" / "MCAD"
        # from being treated as different terms by embeddings or the
        # typo corrector.
        question = normalise_query(question)

        # L0 — Greeting
        if is_greeting(question):
            _log("greeting")
            return PipelineResult(
                answer=input_note + ui["welcome"], layer="greeting", response_ms=_ms()
            )

        # L1 — Language-isolated feedback cache
        cached = self._feedback_db.get(question, lang)
        if cached:
            _log("cache", 1.0)
            return PipelineResult(
                answer=input_note + ui["cached_note"] + cached,
                layer="cache",
                confidence=1.0,
                response_ms=_ms(),
            )

        # L2 — Typo correction
        # Skip if the question exactly matches a follow-up suggestion from
        # the previous assistant turn — agent-generated text has no typos.
        recent_fups = set()
        for m in history[-4:]:
            if m.get("role") == "assistant":
                for fup in m.get("follow_ups", []):
                    recent_fups.add(fup.strip().lower())

        is_followup  = question.strip().lower() in recent_fups
        corrected_q  = question if is_followup else \
                       correct_typos(question, lambda msgs: self._invoke(msgs, bucket))
        typo_note    = ""
        if not is_followup and corrected_q.lower() != question.lower():
            typo_note = ui["typo_note"].format(q=corrected_q) + "\n\n"
        # Combine both prefix notes — input_note always first
        prefix   = input_note + typo_note
        search_q = corrected_q

        # L3 — Intent classification
        intent = classify_intent(
            search_q,
            llm_fn=lambda msgs: self._invoke(msgs, bucket),
        )
        log.debug(f"Intent: {intent} | query: {search_q[:60]}")

        # L4 — FAISS + MMR
        docs, confidence = retrieve_docs(
            search_q, self._vector_db, intent,
            expansion_store=self._expansion_store,
        )
        threshold = cfg["retrieval"]["faiss_confidence_threshold"]
        if confidence >= threshold and docs:
            context = "\n\n---\n\n".join(d.page_content for d in docs)
            try:
                msgs   = build_faiss_messages(context, search_q, intent, history, lang)
                answer = self._invoke(msgs, bucket)
                fups   = self._follow_ups(search_q, answer, lang, bucket)
                _log("faiss", confidence, intent)
                return PipelineResult(
                    answer=prefix + answer + ui["src_faiss"],
                    layer="faiss",
                    follow_ups=fups,
                    confidence=confidence,
                    intent=intent,
                    response_ms=_ms(),
                )
            except RuntimeError:
                _log("rate_limited")
                return PipelineResult(answer=ui["rate_limited"], layer="rate_limited", response_ms=_ms())
            except Exception as e:
                log.warning(f"FAISS LLM call failed: {e}")

        # L5 — Live cadwm.gov.in
        live_ctx = fetch_live_context(search_q, self._web_cache)
        if live_ctx:
            try:
                msgs   = build_live_messages(live_ctx, search_q, intent, history, lang)
                answer = self._invoke(msgs, bucket)
                fups   = self._follow_ups(search_q, answer, lang, bucket)
                _log("live", 0.5, intent)
                return PipelineResult(
                    answer=prefix + answer + ui["src_live"],
                    layer="live",
                    follow_ups=fups,
                    confidence=0.5,
                    intent=intent,
                    response_ms=_ms(),
                )
            except RuntimeError:
                _log("rate_limited")
                return PipelineResult(answer=ui["rate_limited"], layer="rate_limited", response_ms=_ms())
            except Exception as e:
                log.warning(f"Live LLM call failed: {e}")

        # L6 — Scope-aware fallback
        try:
            msgs   = build_fallback_messages(corrected_q, is_related_topic(corrected_q), lang)
            answer = self._invoke(msgs, bucket)
            _log("fallback", 0.0, intent)
            return PipelineResult(
                answer=prefix + answer + ui["src_general"],
                layer="fallback",
                intent=intent,
                response_ms=_ms(),
            )
        except RuntimeError:
            _log("rate_limited")
            return PipelineResult(answer=ui["rate_limited"], layer="rate_limited", response_ms=_ms())
        except Exception as e:
            log.error(f"Fallback LLM call failed: {e}")
            _log("error")
            return PipelineResult(answer=ui["no_result"], layer="error", response_ms=_ms())
