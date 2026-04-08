"""
samridhi/retrieval.py
=====================
All retrieval logic — pure Python, no Streamlit imports.

Exports:
    classify_intent(query)                        → intent label string
    expand_query(query)                           → query with domain-acronym expansions appended
    retrieve_docs(question, vector_db, intent)    → (docs, best_score)
    fetch_live_context(question, web_cache)       → str | None
    is_related_topic(question)                    → bool
    is_greeting(text)                             → bool
    correct_typos(question, llm_fn)               → corrected string

MMR STRATEGY (three-tier, best available is used automatically):
  Tier 1 — LangChain native MMR  (max_marginal_relevance_search)
           Uses the FAISS index's own stored vectors for cosine-similarity
           diversity scoring. Most accurate — no extra embedding calls.
  Tier 2 — numpy dot-product MMR  (embed_query on the fly)
           Falls back to this if the FAISS index doesn't expose native MMR.
           Embeds each candidate once; computes pairwise cosine similarity
           with selected set. Accurate but one extra embed_query call.
  Tier 3 — Word-overlap MMR proxy
           Falls back if numpy is unavailable or embedding fails.
           Uses Jaccard word-overlap as a diversity proxy. No extra I/O.
           Least accurate but zero dependencies.
"""

from __future__ import annotations

import asyncio
import re
import time

import requests
from bs4 import BeautifulSoup

from samridhi.config import CADWM_BASE, INTENTS, WEB_HEADERS, cfg
from samridhi.logger import get_logger

log = get_logger()

# Optional: httpx for concurrent async scraping
try:
    import httpx
    _HAS_HTTPX = True
except ImportError:
    _HAS_HTTPX = False


# ══════════════════════════════════════════════════════════════
# GREETING DETECTION  (O(1) frozenset)
# ══════════════════════════════════════════════════════════════

_GREETING_SET: frozenset = frozenset({
    "hi", "hello", "hey", "good morning", "good evening", "good afternoon",
    "namaste", "namaskar", "howdy", "greetings", "helo", "hii", "hiya",
    "नमस्ते", "हेलो", "हाय",
})

def is_greeting(text: str) -> bool:
    t = text.lower().strip()
    return len(t.split()) <= 5 and (
        t in _GREETING_SET or any(t.startswith(g) for g in _GREETING_SET)
    )


# ══════════════════════════════════════════════════════════════
# QUERY NORMALISATION  (runs before typo correction)
# ══════════════════════════════════════════════════════════════
#
# Problem: "MCADWM", "M-CADWM", "MCAD", "m cadwm", "mcad wm" etc.
# all produce different embeddings and the LLM typo corrector
# was actively "fixing" correctly-written M-CADWM by removing the hyphen.
#
# Solution: map ALL known variants → canonical form BEFORE anything
# else runs.  This guarantees consistent embeddings and prevents
# the typo corrector from touching domain-specific terms.
#
# The normalisation table is case-insensitive and uses word-boundary
# matching to avoid false replacements (e.g. "mcad" inside "mcadwm"
# should not be replaced separately).
# ──────────────────────────────────────────────────────────────

# Each entry: (pattern_to_match, canonical_replacement)
# Patterns are matched case-insensitively on word boundaries.
# Order matters — longer/more-specific patterns first.
_NORMALISE_MAP: list[tuple[str, str]] = [
    # M-CADWM variants
    (r'\bm[\s\-_]*cadwm\b',      "M-CADWM"),
    (r'\bmcadwm\b',               "M-CADWM"),
    (r'\bm\.cadwm\b',             "M-CADWM"),
    # SMIS variants
    (r'\bs[\s\-_]*m[\s\-_]*i[\s\-_]*s\b', "SMIS"),
    # PMKSY variants
    (r'\bpmksy\b',                "PMKSY"),
    (r'\bpm[\s\-_]*ksy\b',        "PMKSY"),
    # WUA variants
    (r'\bw[\s\-_]*u[\s\-_]*a\b',  "WUA"),
    # CAD variants (only when standalone, not inside M-CADWM)
    (r'\bcad\b(?!wm)',             "CAD"),
    # IMTI variants
    (r'\bimti\b',                  "IMTI"),
    # O&M variants
    (r'\bo\s*&\s*m\b',             "O&M"),
    (r'\bo\s+and\s+m\b',          "O&M"),
]

# Pre-compile all patterns for O(1) per-query application
_NORMALISE_COMPILED = [
    (re.compile(pat, re.IGNORECASE), repl)
    for pat, repl in _NORMALISE_MAP
]


def normalise_query(query: str) -> str:
    """
    Map all known domain-term variants to their canonical forms.

    Runs BEFORE typo correction so:
      1. The LLM typo corrector never sees non-canonical variants
         (preventing it from "fixing" correctly-written M-CADWM)
      2. FAISS always receives the canonical form → consistent embeddings
      3. All variants ("MCADWM", "mcad wm", "m-cadwm") are treated identically

    O(k) where k = number of normalisation patterns.
    Pure function — no side effects, safe for unit testing.
    """
    for pattern, replacement in _NORMALISE_COMPILED:
        query = pattern.sub(replacement, query)
    return query


# ══════════════════════════════════════════════════════════════
# TYPO CORRECTION  (skipped for short queries)
# ══════════════════════════════════════════════════════════════

def correct_typos(question: str, llm_fn) -> str:
    """
    llm_fn is a callable(messages) → str (the rate-limited LLM wrapper).
    Skipped for queries ≤ typo_skip_words.
    Normalisation should already have been applied before this is called.

    The prompt explicitly instructs the LLM NOT to alter domain terms
    (M-CADWM, SMIS, WUA, PMKSY, IMTI, CAD, O&M) so they are never
    accidentally "corrected".
    """
    if len(question.split()) <= cfg["llm"]["typo_skip_words"]:
        return question
    try:
        from langchain_core.messages import HumanMessage
        prompt = (
            "Fix any spelling mistakes or typos in the following query. "
            "Return ONLY the corrected query, nothing else. "
            "Do not change meaning, language, or add words. "
            "IMPORTANT: Do NOT alter these domain terms — leave them exactly as written: "
            "M-CADWM, SMIS, WUA, PMKSY, IMTI, CAD, O&M, M-CADWM, cadwm.gov.in.\n\n"
            "Query: " + question
        )
        corrected = llm_fn([HumanMessage(content=prompt)]).strip()
        if len(corrected) > len(question) * 2 or len(corrected) < 2:
            return question
        return corrected
    except Exception:
        return question


# ══════════════════════════════════════════════════════════════
# QUERY EXPANSION  (SQLite-backed — admin-editable, no restart needed)
# ══════════════════════════════════════════════════════════════
#
# The expansion table lives in expansions.db (managed by ExpansionStore
# in cache.py). This function receives the store as a parameter so
# retrieval.py has no direct DB coupling — it's fully testable without
# a database.
#
# Fallback: if expansion_store is None (e.g. in unit tests), no expansion
# is applied and the original query is returned unchanged.
# ──────────────────────────────────────────────────────────────

def expand_query(query: str, expansion_store=None) -> str:
    """
    Append full-form expansions for recognised acronyms.

    expansion_store — ExpansionStore instance (injected from pipeline).
                      If None, returns query unchanged (safe for tests).

    Example:
        "how to register WUA" → "how to register WUA water users association WUA"
    Complexity: O(k) where k = number of enabled expansions.
    """
    if expansion_store is None:
        return query
    expansions = expansion_store.get_all()   # {acronym: expansion}, in-memory
    ql         = query.lower()
    additions  = [
        exp for acr, exp in expansions.items()
        if acr in ql and exp.lower() not in ql
    ]
    return query + (" " + " ".join(additions) if additions else "")


# ══════════════════════════════════════════════════════════════
# INTENT CLASSIFICATION  (keyword → LLM fallback)
# ══════════════════════════════════════════════════════════════

_INTENT_KW: dict[str, frozenset] = {
    "definition": frozenset({
        "what is", "what are", "define", "definition",
        "meaning", "explain", "describe", "elaborat",
    }),
    "procedure": frozenset({
        "how to", "how do", "steps", "process", "procedure",
        "register", "apply", "submit", "login", "upload",
        "fill", "download", "activate", "enrol",
    }),
    "data": frozenset({
        "how many", "statistics", "data", "figure", "budget",
        "coverage", "beneficiar", "number of", "total",
        "fund", "amount", "target", "achievement",
    }),
    "smis": frozenset({
        "smis", "portal", "login", "password", "dashboard",
        "data entry", "report", "module", "screen",
        "interface", "user", "account", "otp",
    }),
}

def classify_intent(query: str, llm_fn=None) -> str:
    """
    Two-pass classifier:
      Pass 1 — keyword scan O(n): returns immediately if match found.
      Pass 2 — LLM fallback O(1 LLM): only invoked for ambiguous queries.
    llm_fn is optional; if absent, "general" is returned for ambiguous cases.
    """
    ql = query.lower()
    for intent, kws in _INTENT_KW.items():
        if any(kw in ql for kw in kws):
            return intent
    if llm_fn is None:
        return "general"
    try:
        from langchain_core.messages import HumanMessage
        prompt = (
            "Classify the following query into exactly one category. "
            "Reply with ONE word only — no explanation:\n"
            "  definition  procedure  data  smis  general\n\n"
            f"Query: {query}\n\nCategory:"
        )
        result = llm_fn([HumanMessage(content=prompt)]).strip().lower().split()[0]
        return result if result in INTENTS else "general"
    except Exception:
        return "general"


# ══════════════════════════════════════════════════════════════
# HYBRID RETRIEVAL + VECTOR-SPACE MMR RE-RANKING
# FAISS O(k log n) + BM25-style keyword O(k*d) + MMR re-rank.
# ══════════════════════════════════════════════════════════════

_STOPWORDS: frozenset = frozenset({
    "a", "an", "the", "is", "are", "in", "on", "at", "to", "of", "and", "or",
    "for", "what", "tell", "me", "how", "do", "can", "please", "give", "about",
    "does", "did", "was", "were", "has", "have", "will", "would", "should",
    "kya", "hai", "mujhe", "batao", "ke", "ka", "ki", "aur", "se",
    "yeh", "woh", "kaise", "kaisa", "kaun", "kyun", "kab", "kahan",
})

def _keywords(query: str) -> list[str]:
    return [w for w in re.findall(r'\w+', query.lower())
            if w not in _STOPWORDS and len(w) > 2]

def _keyword_score(text: str, kws: list[str]) -> float:
    if not kws:
        return 0.0
    tl = text.lower()
    return sum(1 for w in kws if w in tl) / len(kws)


# ── MMR helpers ──────────────────────────────────────────────

def _cosine(a, b) -> float:
    """Cosine similarity between two numpy vectors. O(d)."""
    import numpy as np
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _mmr_native(vector_db, query: str, fetch_k: int, k: int, lam: float) -> list | None:
    """
    Tier 1 — LangChain native MMR via max_marginal_relevance_search.

    Uses the FAISS index's internally stored vectors directly —
    no extra embed_query call, most accurate and fastest.
    Returns None if the index doesn't support this method.
    """
    try:
        docs = vector_db.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lam,   # LangChain uses lambda_mult (same as our lam)
        )
        return docs if docs else None
    except Exception:
        return None


def _mmr_numpy(docs: list, query_vec, embedding_fn, lam: float, k: int) -> list | None:
    """
    Tier 2 — numpy dot-product MMR.

    Embeds each candidate document once (using the embedding model's
    embed_documents method) then runs the MMR greedy selection loop
    using cosine similarity in vector space.

    query_vec   — numpy array, already computed by caller
    embedding_fn — callable(list[str]) → list[list[float]]

    Returns None on any failure so Tier 3 can take over.
    """
    try:
        import numpy as np
        if len(docs) <= k:
            return docs
        texts     = [d.page_content for d in docs]
        doc_vecs  = np.array(embedding_fn(texts), dtype=float)
        # Relevance scores: cosine(query_vec, doc_vec)
        rel_scores = np.array([_cosine(query_vec, dv) for dv in doc_vecs])
        selected_idx: list[int] = []
        remaining_idx = list(range(len(docs)))
        while len(selected_idx) < k and remaining_idx:
            if not selected_idx:
                # Pick most relevant first
                best = max(remaining_idx, key=lambda i: rel_scores[i])
            else:
                # MMR score: lam * relevance - (1-lam) * max_sim_to_selected
                sel_vecs = doc_vecs[selected_idx]
                best, best_score = remaining_idx[0], -1e9
                for i in remaining_idx:
                    sim_to_sel = max(_cosine(doc_vecs[i], sv) for sv in sel_vecs)
                    score = lam * rel_scores[i] - (1 - lam) * sim_to_sel
                    if score > best_score:
                        best_score, best = score, i
            selected_idx.append(best)
            remaining_idx.remove(best)
        return [docs[i] for i in selected_idx]
    except Exception as e:
        log.debug(f"_mmr_numpy failed ({e}), falling back to Tier 3")
        return None


def _mmr_overlap(docs: list, lam: float, k: int) -> list:
    """
    Tier 3 — Word-overlap MMR proxy (zero dependencies).

    Uses Jaccard word-overlap as a diversity measure.
    Least accurate but always available as a safe fallback.
    """
    if len(docs) <= k:
        return docs
    selected: list = []
    remaining = list(docs)
    try:
        while len(selected) < k and remaining:
            if not selected:
                selected.append(remaining.pop(0))
                continue
            best_idx, best_score = 0, -1e9
            for i, doc in enumerate(remaining):
                rel       = 1.0 - i / len(remaining)
                sel_words = set()
                for s in selected:
                    sel_words |= set(s.page_content.split())
                doc_words = set(doc.page_content.split())
                overlap   = len(doc_words & sel_words) / max(len(doc_words), 1)
                score     = lam * rel - (1 - lam) * overlap
                if score > best_score:
                    best_score, best_idx = score, i
            selected.append(remaining.pop(best_idx))
        return selected
    except Exception:
        return docs[:k]


def _mmr_rerank(
    docs: list,
    lam: float,
    k: int,
    vector_db=None,
    query: str = "",
    embedding_model=None,
) -> list:
    """
    Three-tier MMR dispatcher. Tries each tier in order, uses first that succeeds.

    Tier 1 (native)  — vector_db.max_marginal_relevance_search  ← most accurate
    Tier 2 (numpy)   — embed + cosine pairwise                  ← accurate
    Tier 3 (overlap) — Jaccard word overlap proxy               ← always works

    Parameters
    ----------
    docs            : pre-scored candidate documents (already filtered by distance)
    lam             : MMR lambda (0=max diversity, 1=max relevance)
    k               : number of documents to return
    vector_db       : FAISS index (needed for Tier 1)
    query           : original query string (needed for Tier 1 and Tier 2)
    embedding_model : HuggingFaceEmbeddings instance (needed for Tier 2)
    """
    if len(docs) <= k:
        return docs

    # Tier 1 — LangChain native MMR (uses index's own stored vectors)
    if vector_db is not None and query:
        fetch_k = cfg["retrieval"]["mmr_fetch_k"]
        result  = _mmr_native(vector_db, query, fetch_k, k, lam)
        if result:
            log.debug(f"MMR Tier 1 (native) used — {len(result)} docs selected")
            return result

    # Tier 2 — numpy cosine MMR (embed candidates on the fly)
    if embedding_model is not None and query:
        try:
            import numpy as np
            query_vec = np.array(embedding_model.embed_query(query), dtype=float)
            result    = _mmr_numpy(
                docs, query_vec,
                embedding_model.embed_documents,
                lam, k,
            )
            if result:
                log.debug(f"MMR Tier 2 (numpy cosine) used — {len(result)} docs selected")
                return result
        except Exception as e:
            log.debug(f"MMR Tier 2 setup failed ({e}), falling back to Tier 3")

    # Tier 3 — word-overlap proxy (always succeeds)
    log.debug("MMR Tier 3 (word-overlap proxy) used")
    return _mmr_overlap(docs, lam, k)


def retrieve_docs(
    question: str,
    vector_db,
    intent: str = "general",
    expansion_store=None,
) -> tuple:
    """
    Hybrid FAISS + keyword retrieval with three-tier vector-space MMR re-ranking.

    Strategy:
      1. Expand query with domain acronyms (from ExpansionStore if provided)
      2. Fetch fetch_k candidates from FAISS (cosine distance)
      3. Score each with 60% semantic + 40% BM25-style keyword + optional intent boost
      4. MMR re-rank top fetch_k to final k (Tier 1 → 2 → 3 fallback)

    Returns (docs[:k], best_hybrid_score).
    All thresholds and weights read from cfg — no constants in this function.
    """
    k        = cfg["retrieval"]["faiss_k"]
    max_dist = cfg["retrieval"]["faiss_max_dist"]
    sem_w    = cfg["retrieval"]["faiss_semantic_weight"]
    kw_w     = cfg["retrieval"]["faiss_keyword_weight"]
    mmr_lam  = cfg["retrieval"]["mmr_lambda"]
    fetch_k  = cfg["retrieval"]["mmr_fetch_k"]

    expanded   = expand_query(question, expansion_store)
    candidates = vector_db.similarity_search_with_score(expanded, k=fetch_k)
    kws        = _keywords(question)

    scored = []
    for doc, dist in candidates:
        if dist >= max_dist:
            continue
        # Intent-based metadata boost:
        # smis-intent queries get +0.10 for docs tagged source=smis*
        meta_boost = (
            0.10
            if intent == "smis" and doc.metadata.get("source", "").startswith("smis")
            else 0.0
        )
        sem   = 1.0 / (1.0 + dist)
        kw    = _keyword_score(doc.page_content, kws)
        score = sem_w * sem + kw_w * kw + meta_boost
        scored.append((doc, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    best = scored[0][1] if scored else 0.0
    pre_mmr_docs = [d for d, _ in scored[:fetch_k]]

    # Extract the embedding model from the vector_db if available (needed for Tier 2)
    embedding_model = getattr(vector_db, "embedding_function", None)

    final = _mmr_rerank(
        pre_mmr_docs,
        lam=mmr_lam,
        k=k,
        vector_db=vector_db,
        query=question,          # pass original (unexpanded) query for MMR diversity
        embedding_model=embedding_model,
    )
    return final, best


# ══════════════════════════════════════════════════════════════
# LIVE WEB SCRAPE
# Three-source fallback chain — most reliable to least:
#
#   Source 1 — Google site:cadwm.gov.in search
#              Most targeted. Blocked by CAPTCHA if scraped too often.
#   Source 2 — DuckDuckGo site:cadwm.gov.in search
#              Alternative search engine; different CAPTCHA behaviour.
#   Source 3 — Direct cadwm.gov.in crawl
#              Fetches homepage + /sitemap.xml pages directly.
#              Always available; less query-targeted but never CAPTCHA-blocked.
#
# CAPTCHA detection: any search response that contains fewer than 3
# cadwm.gov.in links AND is longer than 5 KB is flagged as a likely
# CAPTCHA/block page and triggers the next source.
# ══════════════════════════════════════════════════════════════

# Direct pages to crawl when search engines are blocked
_CADWM_DIRECT_PAGES = [
    "http://cadwm.gov.in/",
    "http://cadwm.gov.in/about.html",
    "http://cadwm.gov.in/scheme.html",
    "http://cadwm.gov.in/smis.html",
    "http://cadwm.gov.in/guidelines.html",
]


async def _fetch_url_async(client, url: str, max_chars: int) -> str:
    """Fetch a single URL asynchronously and return cleaned text."""
    try:
        r = await client.get(
            url,
            timeout=cfg["web_scrape"]["timeout_seconds"],
            headers=WEB_HEADERS,
            follow_redirects=True,
        )
        s = BeautifulSoup(r.text, "html.parser")
        for tag in s(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = re.sub(r'\s{2,}', ' ', s.get_text(separator=" ", strip=True))
        return f"[Source: {url}]\n{text[:max_chars]}" if len(text) > 200 else ""
    except Exception:
        return ""


def _fetch_urls_sync(urls: list[str], max_chars: int) -> list[str]:
    """Fetch a list of URLs synchronously and return cleaned text list."""
    texts = []
    for url in urls:
        try:
            r = requests.get(
                url, headers=WEB_HEADERS,
                timeout=cfg["web_scrape"]["timeout_seconds"]
            )
            s = BeautifulSoup(r.text, "html.parser")
            for tag in s(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()
            text = re.sub(r'\s{2,}', ' ', s.get_text(separator=" ", strip=True))
            if len(text) > 200:
                texts.append(f"[Source: {url}]\n{text[:max_chars]}")
        except Exception:
            pass
    return texts


def _fetch_pages(urls: list[str], max_chars: int) -> list[str]:
    """
    Fetch pages using async httpx if available, else sync requests.
    Returns list of non-empty text strings.
    """
    if _HAS_HTTPX:
        async def _gather():
            async with httpx.AsyncClient() as client:
                tasks = [_fetch_url_async(client, u, max_chars) for u in urls]
                return await asyncio.gather(*tasks)
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(_gather())
            loop.close()
            return [t for t in results if t]
        except Exception:
            pass
    return _fetch_urls_sync(urls, max_chars)


def _is_captcha_page(resp_text: str, found_urls: list[str]) -> bool:
    """
    Heuristic CAPTCHA / block detection for search engine responses.
    Returns True if the response looks like a block page rather than results.

    Signals:
      - Response is large (>5 KB) but contains no cadwm.gov.in links
      - Response contains known CAPTCHA/block keywords
    """
    if found_urls:
        return False   # got results — definitely not blocked
    captcha_signals = [
        "unusual traffic", "captcha", "verify you're a human",
        "access denied", "403 forbidden", "sorry, we couldn't",
        "automated queries", "our systems have detected",
    ]
    text_lower = resp_text.lower()
    if any(sig in text_lower for sig in captcha_signals):
        return True
    # Large response with no results = likely block/CAPTCHA page
    if len(resp_text) > 5000:
        return True
    return False


def _search_google(question: str, max_pages: int, timeout: int) -> list[str]:
    """
    Source 1: Google site:cadwm.gov.in search.
    Returns list of cadwm.gov.in URLs, or [] if blocked/failed.
    """
    try:
        query      = f"site:cadwm.gov.in {question}"
        search_url = f"https://www.google.com/search?q={requests.utils.quote(query)}&num=5"
        resp = requests.get(search_url, timeout=timeout, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        soup = BeautifulSoup(resp.text, "html.parser")
        urls: list[str] = []
        for a in soup.select("a[href]"):
            href = a["href"]
            if ("/url?q=http://cadwm.gov.in" in href
                    or "/url?q=https://cadwm.gov.in" in href):
                actual = href.split("/url?q=")[1].split("&")[0]
                if actual not in urls:
                    urls.append(actual)
            if len(urls) >= max_pages:
                break
        if _is_captcha_page(resp.text, urls):
            log.warning("Google search: CAPTCHA/block detected, trying DuckDuckGo.")
            return []
        return urls
    except Exception as e:
        log.warning(f"Google search failed: {e}")
        return []


def _search_duckduckgo(question: str, max_pages: int, timeout: int) -> list[str]:
    """
    Source 2: DuckDuckGo site:cadwm.gov.in search.
    Uses DuckDuckGo HTML endpoint (no API key needed).
    Returns list of cadwm.gov.in URLs, or [] if blocked/failed.
    """
    try:
        query      = f"site:cadwm.gov.in {question}"
        search_url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
        resp = requests.get(search_url, timeout=timeout, headers={
            "User-Agent": "Mozilla/5.0 (compatible; SamridhiBot/4.0)",
            "Accept-Language": "en-US,en;q=0.9",
        })
        soup = BeautifulSoup(resp.text, "html.parser")
        urls: list[str] = []
        # DuckDuckGo HTML results use class "result__url" or links with cadwm.gov.in
        for a in soup.select("a.result__a, a[href]"):
            href = a.get("href", "")
            if "cadwm.gov.in" in href:
                # DuckDuckGo may wrap URLs in redirects — extract the real URL
                if "uddg=" in href:
                    from urllib.parse import unquote, parse_qs, urlparse
                    parsed = parse_qs(urlparse(href).query)
                    actual = unquote(parsed.get("uddg", [href])[0])
                else:
                    actual = href
                if actual not in urls and actual.startswith("http"):
                    urls.append(actual)
            if len(urls) >= max_pages:
                break
        if _is_captcha_page(resp.text, urls):
            log.warning("DuckDuckGo search: blocked, falling back to direct crawl.")
            return []
        log.info(f"DuckDuckGo search: found {len(urls)} URLs")
        return urls
    except Exception as e:
        log.warning(f"DuckDuckGo search failed: {e}")
        return []


def _crawl_direct(question: str, max_pages: int, max_chars: int) -> list[str]:
    """
    Source 3: Direct cadwm.gov.in crawl — always available, never CAPTCHA-blocked.
    Fetches known cadwm.gov.in pages and filters those most relevant to the query.
    Uses keyword matching to rank pages.
    """
    kws = _keywords(question)
    pages_to_fetch = _CADWM_DIRECT_PAGES[:max_pages]
    texts = _fetch_pages(pages_to_fetch, max_chars)
    if not texts:
        return []
    # Rank by keyword relevance so the most relevant pages bubble up
    ranked = sorted(texts, key=lambda t: _keyword_score(t, kws), reverse=True)
    log.info(f"Direct cadwm.gov.in crawl: fetched {len(ranked)} pages")
    return ranked


def fetch_live_context(question: str, web_cache) -> str | None:
    """
    Fetch live content from cadwm.gov.in using a three-source fallback chain.

    Source 1 — Google search  (most targeted; CAPTCHA-detected and bypassed)
    Source 2 — DuckDuckGo     (fallback if Google is blocked)
    Source 3 — Direct crawl   (always works; less query-targeted)

    Results are cached via WebCache (SQLite/Postgres, 24 hr TTL) so the
    network is only hit once per unique query per day.
    """
    # Cache check
    cached = web_cache.get(question)
    if cached is not None:
        return cached

    max_pages = cfg["web_scrape"]["max_pages"]
    max_chars = cfg["web_scrape"]["per_page_chars"]
    timeout   = cfg["web_scrape"]["timeout_seconds"]

    # ── Source 1: Google ─────────────────────────────────────
    urls = _search_google(question, max_pages, timeout)

    # ── Source 2: DuckDuckGo (if Google returned nothing) ────
    if not urls:
        urls = _search_duckduckgo(question, max_pages, timeout)

    # ── Fetch pages from search engine results ────────────────
    if urls:
        texts = _fetch_pages(urls[:max_pages], max_chars)
    else:
        texts = []

    # ── Source 3: Direct crawl (if search engines both failed) ─
    if not texts:
        log.info("Both search engines failed — falling back to direct cadwm.gov.in crawl")
        texts = _crawl_direct(question, max_pages, max_chars)

    result = "\n\n---\n\n".join(texts) if texts else None
    web_cache.set(question, result)
    return result


# ══════════════════════════════════════════════════════════════
# TOPIC SCOPE DETECTION  (O(n) frozenset scan)
# ══════════════════════════════════════════════════════════════

_RELATED_KW: frozenset = frozenset({
    "water", "irrigation", "agriculture", "farming", "crop", "dam", "canal",
    "river", "scheme", "government", "ministry", "portal", "smis", "cadwm",
    "mcadwm", "m-cadwm", "pmksy", "kisan", "farmer", "soil", "drainage", "flood",
    "watershed", "groundwater", "silt", "wus", "imti", "fund", "subsidy",
    "registration", "application", "form", "document", "certificate",
    "sarkar", "yojana", "command", "minor", "distributary", "outlet", "wua",
    "विभाग", "सिंचाई", "पानी", "कृषि", "किसान", "योजना",
})

def is_related_topic(question: str) -> bool:
    return any(kw in question.lower() for kw in _RELATED_KW)
