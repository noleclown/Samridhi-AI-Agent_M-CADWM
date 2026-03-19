"""
Samridhi — AI Assistant for M-CADWM & SMIS
============================================
Version: 2.0

RETRIEVAL ARCHITECTURE (6 layers):
  L0  Greeting detection          O(1)       frozenset lookup
  L1  Language-isolated cache     O(1)       MD5 hash + dict get
  L2  Typo correction             O(1) LLM   skipped for ≤4-word queries
  L3  Query intent classification O(1) LLM   definition / procedure / data / smis / general
  L4  Hybrid FAISS retrieval      O(k log n) 60% semantic + 40% BM25-style keyword
  L5  Live cadwm.gov.in scrape    O(p)       session TTL-cached (5 min)
  L6  Scope-aware fallback        O(1) LLM   topic-gated

NEW IN v2.0:
  - Query intent classification (L3): routes queries to intent-tuned prompts
  - SMIS-specific prompt path: login, data entry, reports handled separately
  - Persistent cross-session web cache: SQLite (24 hr TTL), survives tab close
  - Query analytics log: every query logged to analytics.jsonl for review
  - Confidence source badge: each answer tagged with its retrieval source
  - Suggested follow-up questions: generated post-answer for discoverability
  - Feedback → FAISS synthetic injection: highly-rated answers re-ingested
  - Web cache eviction: stale SQLite entries purged on startup
  - Logo cached in session state: avoids re-reading binary on every rerun
  - Feedback file written with atomic rename: no corruption on crash

DESIGN PRINCIPLES:
  - Language isolation   : cache key = lang::hash — zero cross-language leakage
  - Graceful degradation : every external call wrapped in try/except
  - Single source of truth: prompts in _prompt_*, UI strings in UI dict
  - Versioned schema     : feedback entries carry schema version for migration
  - Atomic file writes   : temp-file + os.replace() — no partial writes
  - Analytics observability: JSONL log enables future dashboarding
"""

import os
import base64
import json
import hashlib
import asyncio
import re
import time
import sqlite3
import requests
import streamlit as st
import edge_tts

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from datetime import datetime, timedelta
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage


# ══════════════════════════════════════════════════════════════
# 1. CONFIGURATION & CONSTANTS
# ══════════════════════════════════════════════════════════════

BASE_DIR       = Path(os.path.dirname(os.path.abspath(__file__)))
FAISS_PATH     = str(BASE_DIR / "faiss_index")
LOGO_PATH      = BASE_DIR / "logo.png"
FEEDBACK_FILE  = BASE_DIR / "feedback_cache.json"
AUDIO_FILE     = BASE_DIR / "response.mp3"
WEB_DB_FILE    = BASE_DIR / "web_cache.db"        # persistent cross-session web cache
ANALYTICS_FILE = BASE_DIR / "analytics.jsonl"     # append-only query log

CADWM_BASE  = "http://cadwm.gov.in/"
WEB_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; SamridhiBot/2.0)"}

# Retrieval
FAISS_CONFIDENCE_THRESHOLD = 0.35
FAISS_K                    = 10
FAISS_MAX_DIST             = 1.4

# Cache TTLs
WEB_CACHE_TTL_SECONDS   = 86400   # 24 hours (persistent DB)
SESSION_WEB_CACHE_TTL   = 300     # 5 min in-memory guard (avoids DB hit for same-session repeat)
FEEDBACK_REINGEST_MIN   = 3       # min thumbs_up before synthetic FAISS re-ingest

# Feedback schema
FEEDBACK_SCHEMA_VERSION = "2.0"

# Intent labels
INTENTS = frozenset({"definition", "procedure", "data", "smis", "general"})

# ── API key resolution ─────────────────────────────────────────
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        groq_api_key = None
if not groq_api_key:
    st.error("GROQ_API_KEY not found. Add it to Streamlit Cloud secrets or your .env file.")
    st.stop()

st.set_page_config(
    page_title="Samridhi – M-CADWM",
    page_icon="🏛️",
    layout="centered",
)


# ══════════════════════════════════════════════════════════════
# 2. UI STRINGS  (100% bilingual, zero cross-language text)
# ══════════════════════════════════════════════════════════════

UI: dict = {
    "en": {
        "title":        "Samridhi – M-CADWM",
        "subtitle":     "AI Assistant — M-CADWM & SMIS",
        "welcome": (
            "Hello. I am Samridhi, the AI Assistant for the M-CADWM scheme and the SMIS portal.\n\n"
            "I can assist with queries on M-CADWM official documents, the cadwm.gov.in website, "
            "the SMIS portal, scheme guidelines, registration, forms, and related processes.\n\n"
            "Please state your query."
        ),
        "placeholder":   "Enter your query or click 🎤 to speak...",
        "spinner":       "Searching M-CADWM documents...",
        "spinner_web":   "Checking cadwm.gov.in for the latest information...",
        "spinner_follow":"Generating follow-up suggestions...",
        "no_result":     "No relevant information was found in the M-CADWM documents. Please refine your query.",
        "cached_note":   "*(Retrieved from cache)*\n\n",
        "fb_up":         "Feedback recorded.",
        "fb_dn":         "Feedback recorded. Response will be reviewed.",
        "typo_note":     "Note: A typographical correction was applied. Response addresses: \"{q}\"",
        "src_faiss":     "\n\n---\n*Source: M-CADWM official documents*",
        "src_live":      "\n\n---\n*Source: cadwm.gov.in (live)*",
        "src_general":   "\n\n---\n*Note: Response is based on general knowledge, not M-CADWM official documents.*",
        "follow_header": "\n\n---\n**You may also ask:**",
        "btn_en":        "🇬🇧 EN",
        "btn_hi":        "🇮🇳 HI",
    },
    "hi": {
        "title":        "समृद्धि – M-CADWM",
        "subtitle":     "AI सहायक — M-CADWM और SMIS",
        "welcome": (
            "नमस्ते। मैं समृद्धि हूँ — M-CADWM योजना और SMIS पोर्टल के लिए AI सहायक।\n\n"
            "मैं M-CADWM आधिकारिक दस्तावेज़ों, cadwm.gov.in वेबसाइट, SMIS पोर्टल, "
            "योजना दिशानिर्देश, पंजीकरण, प्रपत्र और संबंधित प्रक्रियाओं से जुड़े "
            "प्रश्नों में सहायता कर सकती हूँ।\n\n"
            "कृपया अपना प्रश्न दर्ज करें।"
        ),
        "placeholder":   "प्रश्न दर्ज करें या 🎤 दबाकर बोलें...",
        "spinner":       "M-CADWM दस्तावेज़ खोज रहे हैं...",
        "spinner_web":   "cadwm.gov.in से नवीनतम जानकारी प्राप्त की जा रही है...",
        "spinner_follow":"अनुशंसित प्रश्न तैयार किए जा रहे हैं...",
        "no_result":     "M-CADWM दस्तावेज़ों में कोई प्रासंगिक जानकारी नहीं मिली। कृपया अपना प्रश्न पुनः दर्ज करें।",
        "cached_note":   "*(कैश से प्राप्त)*\n\n",
        "fb_up":         "फीडबैक दर्ज किया गया।",
        "fb_dn":         "फीडबैक दर्ज किया गया। उत्तर की समीक्षा की जाएगी।",
        "typo_note":     "सूचना: वर्तनी सुधार लागू किया गया। उत्तर इस प्रश्न के लिए है: \"{q}\"",
        "src_faiss":     "\n\n---\n*स्रोत: M-CADWM आधिकारिक दस्तावेज़*",
        "src_live":      "\n\n---\n*स्रोत: cadwm.gov.in (लाइव)*",
        "src_general":   "\n\n---\n*सूचना: यह उत्तर सामान्य ज्ञान पर आधारित है, M-CADWM आधिकारिक दस्तावेज़ों पर नहीं।*",
        "follow_header": "\n\n---\n**आप यह भी पूछ सकते हैं:**",
        "btn_en":        "🇬🇧 EN",
        "btn_hi":        "🇮🇳 HI",
    },
}


# ══════════════════════════════════════════════════════════════
# 3. CSS + VOICE INPUT JS
# ══════════════════════════════════════════════════════════════

st.markdown("""
<style>
#samridhi-mic {
    background:none;border:none;border-radius:50%;
    width:34px;height:34px;font-size:19px;color:#888;
    cursor:pointer;display:flex;align-items:center;
    justify-content:center;flex-shrink:0;padding:0;
    transition:color 0.2s,transform 0.15s;
}
#samridhi-mic:hover{color:#e05c2a;transform:scale(1.2);}
#samridhi-mic.listening{color:#e05c2a;animation:mic-pulse 0.8s infinite;}
@keyframes mic-pulse{
    0%,100%{text-shadow:0 0 0px #e05c2a;}
    50%{text-shadow:0 0 8px #e05c2a;}
}
#samridhi-mic-status{
    position:fixed;bottom:65px;left:50%;transform:translateX(-50%);
    z-index:99999;background:#1a1a1a;color:#e05c2a;font-size:12px;
    padding:5px 12px;border-radius:10px;display:none;max-width:280px;
    text-align:center;border:1px solid rgba(224,92,42,0.3);pointer-events:none;
}
</style>
<div id="samridhi-mic-status"></div>
""", unsafe_allow_html=True)

import streamlit.components.v1 as components
components.html("""
<script>
var _sr_active=false,_sr_obj=null,_sr_lang='en-IN';
var P=window.parent.document;
function getMicBtn(){return P.getElementById('samridhi-mic');}
function getStatusEl(){return P.getElementById('samridhi-mic-status');}
function injectMicIntoInputBar(){
    if(P.getElementById('samridhi-mic'))return;
    var iw=P.querySelector('[data-testid="stChatInput"]')||P.querySelector('.stChatInput');
    if(!iw)return;
    var btn=P.createElement('button');
    btn.id='samridhi-mic';btn.title='Click to speak your question';btn.innerHTML='🎤';
    btn.style.cssText='background:none;border:none;border-radius:50%;width:36px;height:36px;font-size:20px;color:#888;cursor:pointer;display:flex;align-items:center;justify-content:center;flex-shrink:0;padding:0;position:absolute;right:52px;bottom:50%;transform:translateY(50%);z-index:9999;transition:color 0.2s,transform 0.15s';
    iw.style.position='relative';iw.appendChild(btn);
    btn.onmouseenter=function(){if(!_sr_active){btn.style.color='#e05c2a';btn.style.transform='translateY(50%) scale(1.2)';}};
    btn.onmouseleave=function(){if(!_sr_active){btn.style.color='#888';btn.style.transform='translateY(50%)';}};
    btn.onclick=function(){if(_sr_active){_sr_obj&&_sr_obj.stop();}else{startMic();}};
}
injectMicIntoInputBar();
new MutationObserver(function(){injectMicIntoInputBar();}).observe(P.body,{childList:true,subtree:true});
function showStatus(m){var e=getStatusEl();if(e){e.innerText=m;e.style.display='block';}}
function hideStatus(){var e=getStatusEl();if(e)e.style.display='none';}
function resetBtn(){var b=getMicBtn();if(b){b.classList.remove('listening');b.innerText='🎤';b.style.color='#888';b.style.transform='translateY(50%)';}};
function injectAndSubmit(text){
    var ta=P.querySelector('textarea[data-testid="stChatInputTextArea"]')||P.querySelector('.stChatInput textarea')||P.querySelector('textarea');
    if(!ta){showStatus('Input not found: '+text);return;}
    var proto=Object.getOwnPropertyDescriptor(window.parent.HTMLTextAreaElement.prototype,'value');
    if(proto&&proto.set)proto.set.call(ta,text);else ta.value=text;
    ta.dispatchEvent(new window.parent.Event('input',{bubbles:true}));
    ta.dispatchEvent(new window.parent.Event('change',{bubbles:true}));
    ta.focus();
    setTimeout(function(){
        var btn=P.querySelector('button[data-testid="stChatInputSubmitButton"]')||P.querySelector('.stChatInput button[type="submit"]')||P.querySelector('button[aria-label="Send message"]');
        if(btn){btn.click();}else{['keydown','keypress','keyup'].forEach(function(t){ta.dispatchEvent(new window.parent.KeyboardEvent(t,{key:'Enter',code:'Enter',keyCode:13,which:13,bubbles:true,cancelable:true}));});}
        showStatus('Submitted: '+text);setTimeout(hideStatus,2500);
    },450);
}
function startMic(){
    var SR=window.parent.SpeechRecognition||window.parent.webkitSpeechRecognition||window.SpeechRecognition||window.webkitSpeechRecognition;
    if(!SR){showStatus('Use Chrome or Edge for voice input');return;}
    _sr_obj=new SR();_sr_obj.lang=_sr_lang;_sr_obj.continuous=false;_sr_obj.interimResults=true;_sr_obj.maxAlternatives=1;
    _sr_obj.onstart=function(){_sr_active=true;var b=getMicBtn();if(b){b.classList.add('listening');b.innerText='🔴';b.style.color='#e05c2a';}showStatus('Listening...');};
    _sr_obj.onresult=function(e){var interim='',final_t='';for(var i=e.resultIndex;i<e.results.length;i++){if(e.results[i].isFinal)final_t+=e.results[i][0].transcript;else interim+=e.results[i][0].transcript;}showStatus(interim||final_t);if(final_t)injectAndSubmit(final_t.trim());};
    _sr_obj.onerror=function(e){_sr_active=false;resetBtn();if(e.error!=='no-speech')showStatus('Error: '+e.error);else hideStatus();};
    _sr_obj.onend=function(){_sr_active=false;resetBtn();setTimeout(hideStatus,1800);};
    _sr_obj.start();
}
window.setMicLang=function(l){_sr_lang=l;};
window.parent.setMicLang=window.setMicLang;
</script>
""", height=0)


# ══════════════════════════════════════════════════════════════
# 4. SESSION STATE INITIALISATION  (idempotent)
# ══════════════════════════════════════════════════════════════

def _init_session():
    defaults = {
        "lang":              "en",
        "messages":          [],
        "pending_feedback":  {},
        "session_web_cache": {},   # {hash: {"text": str, "ts": float}} — in-memory guard
        "logo_b64":          None, # cached logo HTML — avoids re-reading binary each rerun
        "reingest_done":     set(),# set of cache keys already re-ingested into FAISS
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
    if "feedback_cache" not in st.session_state:
        st.session_state.feedback_cache = _load_feedback_file()

_init_session()


# ══════════════════════════════════════════════════════════════
# 5. LANGUAGE / TTS SETUP
# ══════════════════════════════════════════════════════════════

lang: str  = st.session_state.lang
ui:   dict = UI[lang]

speech_lang = "hi-IN"             if lang == "hi" else "en-IN"
tts_voice   = "hi-IN-SwaraNeural" if lang == "hi" else "en-IN-NeerjaNeural"
tts_rate    = "+5%"               if lang == "hi" else "+10%"


# ══════════════════════════════════════════════════════════════
# 6. LANGUAGE TOGGLE BUTTONS
# ══════════════════════════════════════════════════════════════

_, _ecol, _hicol = st.columns([5, 1, 1])
with _ecol:
    if st.button(UI["en"]["btn_en"],
                 type="primary" if lang == "en" else "secondary",
                 use_container_width=True):
        st.session_state.lang = "en"
        st.session_state.messages = [{"role": "assistant", "content": UI["en"]["welcome"]}]
        st.session_state.pending_feedback = {}
        st.rerun()
with _hicol:
    if st.button(UI["hi"]["btn_hi"],
                 type="primary" if lang == "hi" else "secondary",
                 use_container_width=True):
        st.session_state.lang = "hi"
        st.session_state.messages = [{"role": "assistant", "content": UI["hi"]["welcome"]}]
        st.session_state.pending_feedback = {}
        st.rerun()


# ══════════════════════════════════════════════════════════════
# 7. HEADER  (logo cached in session state — avoids binary re-read each rerun)
# ══════════════════════════════════════════════════════════════

if st.session_state.logo_b64 is None and LOGO_PATH.exists():
    try:
        st.session_state.logo_b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode()
    except Exception:
        st.session_state.logo_b64 = ""

_logo_html = (
    f'<img src="data:image/png;base64,{st.session_state.logo_b64}" width="110"><br>'
    if st.session_state.logo_b64 else ""
)

st.markdown(f"""
<div style="text-align:center;margin-top:-10px;">
    {_logo_html}
    <h1 style="margin-bottom:4px;">{ui["title"]}</h1>
    <p style="color:#aaa;margin-top:0;">{ui["subtitle"]}</p>
</div>
<hr style="margin:10px 0 18px 0;">
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# 8. RESOURCE LOADING  (process-level cache — loaded once)
# ══════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def _load_vector_db():
    emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder=str(BASE_DIR / "models"),
    )
    return FAISS.load_local(FAISS_PATH, emb, allow_dangerous_deserialization=True)

@st.cache_resource(show_spinner=False)
def _load_llm():
    return ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.15,
    )

try:
    vector_db = _load_vector_db()
except Exception as e:
    st.error(f"Failed to load FAISS index: {e}")
    st.stop()

llm = _load_llm()


# ══════════════════════════════════════════════════════════════
# 9. PERSISTENT WEB CACHE  (SQLite, 24 hr TTL)
#    Survives tab close / page reload — unlike session_state only.
#    Stale rows are purged on startup.
# ══════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def _init_web_db():
    """Create SQLite DB and purge stale rows. Returns connection."""
    conn = sqlite3.connect(str(WEB_DB_FILE), check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS web_cache (
            key  TEXT PRIMARY KEY,
            text TEXT,
            ts   REAL
        )
    """)
    conn.commit()
    # Purge rows older than TTL
    cutoff = time.time() - WEB_CACHE_TTL_SECONDS
    conn.execute("DELETE FROM web_cache WHERE ts < ?", (cutoff,))
    conn.commit()
    return conn

_web_db = _init_web_db()


def _web_key(query: str) -> str:
    return hashlib.md5(query.lower().strip().encode()).hexdigest()


def _get_web_cache(key: str):
    """Check in-memory guard first (O(1)), then SQLite."""
    hit = st.session_state.session_web_cache.get(key)
    if hit and (time.time() - hit["ts"]) < SESSION_WEB_CACHE_TTL:
        return hit["text"]
    try:
        row = _web_db.execute(
            "SELECT text, ts FROM web_cache WHERE key=?", (key,)
        ).fetchone()
        if row and (time.time() - row[1]) < WEB_CACHE_TTL_SECONDS:
            st.session_state.session_web_cache[key] = {"text": row[0], "ts": row[1]}
            return row[0]
    except Exception:
        pass
    return None


def _set_web_cache(key: str, text):
    ts = time.time()
    st.session_state.session_web_cache[key] = {"text": text, "ts": ts}
    try:
        _web_db.execute(
            "INSERT OR REPLACE INTO web_cache (key, text, ts) VALUES (?,?,?)",
            (key, text, ts)
        )
        _web_db.commit()
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════
# 10. ANALYTICS LOG  (append-only JSONL, non-blocking)
# ══════════════════════════════════════════════════════════════

def _log_query(query: str, layer: str, confidence: float, response_ms: float):
    """
    Append one record to analytics.jsonl.
    Fields: timestamp, lang, query, layer_used, confidence, response_ms.
    Non-blocking — failure is silently swallowed.
    """
    record = {
        "ts":          datetime.utcnow().isoformat(),
        "lang":        lang,
        "query":       query,
        "layer":       layer,         # "cache" / "faiss" / "live" / "fallback" / "greeting"
        "confidence":  round(confidence, 4),
        "response_ms": round(response_ms, 1),
    }
    try:
        with open(ANALYTICS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════
# 11. FEEDBACK CACHE  (language-isolated, versioned, atomic write)
# ══════════════════════════════════════════════════════════════

def _load_feedback_file() -> dict:
    if not FEEDBACK_FILE.exists():
        return {}
    try:
        data = json.loads(FEEDBACK_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data["_schema"] = FEEDBACK_SCHEMA_VERSION
        return data
    except Exception:
        return {}


def _save_feedback_file():
    """Atomic write: write to .tmp then os.replace() — prevents corruption on crash."""
    tmp = FEEDBACK_FILE.with_suffix(".tmp")
    try:
        tmp.write_text(
            json.dumps(st.session_state.feedback_cache, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        os.replace(str(tmp), str(FEEDBACK_FILE))
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def _cache_key(q: str, language: str) -> str:
    """
    Language-isolated key: MD5("en::question") ≠ MD5("hi::question").
    Prevents cross-language cache hits. O(1).
    """
    return hashlib.md5(f"{language}::{q.lower().strip()}".encode()).hexdigest()


def get_cached(q: str):
    key   = _cache_key(q, lang)
    entry = st.session_state.feedback_cache.get(key)
    if entry and entry.get("thumbs_up", 0) > entry.get("thumbs_down", 0):
        return entry.get("answer")
    return None


def record_feedback(q: str, ans: str, vote: str):
    key   = _cache_key(q, lang)
    cache = st.session_state.feedback_cache
    if key not in cache:
        cache[key] = {
            "schema":         FEEDBACK_SCHEMA_VERSION,
            "lang":           lang,
            "question":       q,
            "answer":         ans,
            "thumbs_up":      0,
            "thumbs_down":    0,
            "last_updated":   "",
            "reingested":     False,
        }
    cache[key].update({"answer": ans, "lang": lang})
    cache[key]["thumbs_up" if vote == "up" else "thumbs_down"] += 1
    cache[key]["last_updated"] = datetime.now().isoformat()
    _save_feedback_file()
    # Trigger synthetic FAISS re-ingest check
    _maybe_reingest(key, cache[key])


# ══════════════════════════════════════════════════════════════
# 12. FEEDBACK → FAISS SYNTHETIC RE-INGEST
#     Highly-rated answers (thumbs_up ≥ threshold) are summarised
#     and added back into the FAISS index as synthetic documents.
#     This improves retrieval over time without full re-indexing.
# ══════════════════════════════════════════════════════════════

def _maybe_reingest(key: str, entry: dict):
    """
    If thumbs_up >= FEEDBACK_REINGEST_MIN and not yet re-ingested,
    add a synthetic document to the FAISS index.
    Guard: reingest_done set in session state prevents duplicate injection.
    """
    if entry.get("reingested"):
        return
    if entry.get("thumbs_up", 0) < FEEDBACK_REINGEST_MIN:
        return
    if key in st.session_state.reingest_done:
        return
    try:
        from langchain_core.documents import Document
        q   = entry.get("question", "")
        ans = entry.get("answer", "")
        # Summarise Q+A into a compact synthetic document
        synthetic_text = f"Q: {q}\nA: {ans}"
        doc = Document(page_content=synthetic_text, metadata={"source": "feedback_cache", "lang": entry.get("lang","en")})
        vector_db.add_documents([doc])
        st.session_state.reingest_done.add(key)
        # Mark in persistent cache so we don't re-ingest after restart
        st.session_state.feedback_cache[key]["reingested"] = True
        _save_feedback_file()
    except Exception:
        pass  # FAISS write failure is non-fatal


# ══════════════════════════════════════════════════════════════
# 13. GREETING DETECTION  (O(1) frozenset)
# ══════════════════════════════════════════════════════════════

_GREETING_SET: frozenset = frozenset({
    "hi", "hello", "hey", "good morning", "good evening", "good afternoon",
    "namaste", "namaskar", "howdy", "greetings", "helo", "hii", "hiya",
    "नमस्ते", "हेलो", "हाय",
})

def is_greeting(text: str) -> bool:
    t = text.lower().strip()
    if len(t.split()) > 5:
        return False
    return t in _GREETING_SET or any(t.startswith(g) for g in _GREETING_SET)


# ══════════════════════════════════════════════════════════════
# 14. TYPO CORRECTION  (skipped for ≤4-word queries)
# ══════════════════════════════════════════════════════════════

def correct_typos(question: str) -> str:
    if len(question.split()) <= 4:
        return question
    try:
        prompt = (
            "Fix any spelling mistakes or typos in the following query. "
            "Return ONLY the corrected query, nothing else. "
            "Do not change meaning, language, or add words.\n\n"
            f"Query: {question}"
        )
        corrected = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        if len(corrected) > len(question) * 2 or len(corrected) < 2:
            return question
        return corrected
    except Exception:
        return question


# ══════════════════════════════════════════════════════════════
# 15. QUERY INTENT CLASSIFICATION
#     Classifies each query into one of five intents so the
#     downstream prompt can be tuned to the query type.
#
#     Intents:
#       definition  — "what is X", "explain X", "meaning of X"
#       procedure   — "how to", "steps to", "process of", "register", "apply"
#       data        — "how many", "statistics", "coverage", "budget", "figure"
#       smis        — queries about SMIS portal login, data entry, reports
#       general     — everything else
# ══════════════════════════════════════════════════════════════

# Fast keyword-based pre-classifier (O(n)) — avoids LLM call for obvious cases
_INTENT_KEYWORDS: dict = {
    "definition": frozenset({"what is","what are","define","definition","meaning","explain","describe","elaborat"}),
    "procedure":  frozenset({"how to","how do","steps","process","procedure","register","apply","submit","login","upload","fill","download","activate","enrol"}),
    "data":       frozenset({"how many","statistics","data","figure","budget","coverage","beneficiar","number of","total","fund","amount","target","achievement"}),
    "smis":       frozenset({"smis","portal","login","password","dashboard","data entry","report","module","screen","interface","user","account","otp"}),
}

def classify_intent(query: str) -> str:
    """
    Two-pass classifier:
      Pass 1: keyword scan — O(n), no LLM call.
      Pass 2 (fallback): single short LLM call if keywords are ambiguous.
    Returns one of: definition / procedure / data / smis / general
    """
    ql = query.lower()
    for intent, kws in _INTENT_KEYWORDS.items():
        if any(kw in ql for kw in kws):
            return intent
    # LLM fallback for ambiguous queries
    try:
        prompt = (
            "Classify the following query into exactly one category. "
            "Reply with ONE word only — no explanation:\n"
            "  definition  (asking what something is)\n"
            "  procedure   (asking how to do something)\n"
            "  data        (asking for numbers, statistics, or coverage)\n"
            "  smis        (about SMIS portal: login, data entry, reports)\n"
            "  general     (anything else)\n\n"
            f"Query: {query}\n\nCategory:"
        )
        result = llm.invoke([HumanMessage(content=prompt)]).content.strip().lower().split()[0]
        return result if result in INTENTS else "general"
    except Exception:
        return "general"


# ══════════════════════════════════════════════════════════════
# 16. HYBRID RETRIEVAL  (60% semantic FAISS + 40% BM25-style keyword)
# ══════════════════════════════════════════════════════════════

_STOPWORDS: frozenset = frozenset({
    "a","an","the","is","are","in","on","at","to","of","and","or",
    "for","what","tell","me","how","do","can","please","give","about",
    "does","did","was","were","has","have","will","would","should",
    "kya","hai","mujhe","batao","ke","ka","ki","aur","se","yeh","woh",
    "kaise","kaisa","kaun","kyun","kab","kahan",
})

def _keywords(query: str) -> list:
    return [w for w in re.findall(r'\w+', query.lower())
            if w not in _STOPWORDS and len(w) > 2]

def _keyword_score(text: str, kws: list) -> float:
    if not kws:
        return 0.0
    tl = text.lower()
    return sum(1 for w in kws if w in tl) / len(kws)

def retrieve_docs(question: str, k: int = FAISS_K, max_dist: float = FAISS_MAX_DIST):
    """
    Hybrid retrieval. O(k log n) FAISS + O(k * avg_doc_len) keyword re-rank.
    Returns (docs[:k], best_hybrid_score).
    """
    candidates = vector_db.similarity_search_with_score(question, k=k * 2)
    kws        = _keywords(question)
    scored     = []
    for doc, dist in candidates:
        if dist >= max_dist:
            continue
        sem = 1.0 / (1.0 + dist)
        kw  = _keyword_score(doc.page_content, kws)
        scored.append((doc, 0.60 * sem + 0.40 * kw))
    scored.sort(key=lambda x: x[1], reverse=True)
    best = scored[0][1] if scored else 0.0
    return [d for d, _ in scored[:k]], best


# ══════════════════════════════════════════════════════════════
# 17. LIVE WEB FETCH  (persistent SQLite + session-memory guard)
# ══════════════════════════════════════════════════════════════

def fetch_live_context(question: str, max_pages: int = 3):
    wk  = _web_key(question)
    hit = _get_web_cache(wk)
    if hit is not None:
        return hit

    try:
        query      = f"site:cadwm.gov.in {question}"
        search_url = f"https://www.google.com/search?q={requests.utils.quote(query)}&num=5"
        resp = requests.get(search_url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }, timeout=8)
        soup = BeautifulSoup(resp.text, "html.parser")

        urls = []
        for a in soup.select("a[href]"):
            href = a["href"]
            if "/url?q=http://cadwm.gov.in" in href or "/url?q=https://cadwm.gov.in" in href:
                actual = href.split("/url?q=")[1].split("&")[0]
                if actual not in urls:
                    urls.append(actual)
            if len(urls) >= max_pages:
                break
        if not urls:
            urls = [CADWM_BASE]

        texts = []
        for url in urls[:max_pages]:
            try:
                r = requests.get(url, headers=WEB_HEADERS, timeout=8)
                s = BeautifulSoup(r.text, "html.parser")
                for tag in s(["script", "style", "nav", "footer", "header", "aside"]):
                    tag.decompose()
                text = re.sub(r'\s{2,}', ' ', s.get_text(separator=" ", strip=True))
                if len(text) > 200:
                    texts.append(f"[Source: {url}]\n{text[:3000]}")
            except Exception:
                pass

        result = "\n\n---\n\n".join(texts) if texts else None
        _set_web_cache(wk, result)
        return result

    except Exception:
        return None


# ══════════════════════════════════════════════════════════════
# 18. TOPIC SCOPE DETECTION  (O(n) frozenset scan)
# ══════════════════════════════════════════════════════════════

_RELATED_KW: frozenset = frozenset({
    "water","irrigation","agriculture","farming","crop","dam","canal",
    "river","scheme","government","ministry","portal","smis","cadwm",
    "mcadwm","m-cadwm","pmksy","kisan","farmer","soil","drainage","flood",
    "watershed","groundwater","silt","wus","imti","fund","subsidy",
    "registration","application","form","document","certificate",
    "sarkar","yojana","command","minor","distributary","outlet","wua",
    "विभाग","सिंचाई","पानी","कृषि","किसान","योजना",
})

def is_related_topic(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in _RELATED_KW)


# ══════════════════════════════════════════════════════════════
# 19. PROMPT BUILDERS  (intent-aware, single source of truth)
# ══════════════════════════════════════════════════════════════

def _intent_instruction(intent: str) -> str:
    """Returns an intent-specific instruction line injected into FAISS/live prompts."""
    mapping = {
        "definition": (
            "Focus on providing a clear, precise definition or explanation. "
            "Cite the specific document or section where the definition appears.",
            "परिभाषा या व्याख्या स्पष्ट और संक्षिप्त रूप में प्रदान करें। "
            "उस दस्तावेज़ या अनुभाग का उल्लेख करें जहाँ यह परिभाषा उपलब्ध है।",
        ),
        "procedure": (
            "Provide a clear step-by-step procedure with numbered steps. "
            "Include any prerequisites, required documents, and official contact points.",
            "क्रमांकित चरणों में स्पष्ट प्रक्रिया प्रदान करें। "
            "पूर्वापेक्षाएँ, आवश्यक दस्तावेज़ और आधिकारिक संपर्क बिंदु शामिल करें।",
        ),
        "data": (
            "Present quantitative data clearly — use tables or structured lists where possible. "
            "Cite the source document and year for all figures.",
            "मात्रात्मक डेटा स्पष्ट रूप से प्रस्तुत करें — जहाँ संभव हो तालिका या सूची उपयोग करें। "
            "सभी आँकड़ों के लिए स्रोत दस्तावेज़ और वर्ष का उल्लेख करें।",
        ),
        "smis": (
            "This query is specifically about the SMIS portal. Provide step-by-step guidance "
            "on the portal workflow, screen names, and any known technical requirements or prerequisites.",
            "यह प्रश्न विशेष रूप से SMIS पोर्टल के बारे में है। पोर्टल वर्कफ्लो, स्क्रीन नाम "
            "और किसी भी तकनीकी आवश्यकता पर चरण-दर-चरण मार्गदर्शन प्रदान करें।",
        ),
        "general": (
            "Provide a comprehensive, well-structured response covering all relevant aspects.",
            "सभी प्रासंगिक पहलुओं को कवर करते हुए व्यापक और सुव्यवस्थित उत्तर प्रदान करें।",
        ),
    }
    idx  = 1 if lang == "hi" else 0
    pair = mapping.get(intent, mapping["general"])
    return pair[idx]


def _prompt_faiss(context: str, query: str, intent: str) -> str:
    ii = _intent_instruction(intent)
    if lang == "hi":
        return (
            "आप समृद्धि हैं — M-CADWM योजना, आधिकारिक दस्तावेज़ों, वेबसाइट और SMIS पोर्टल के लिए पेशेवर AI सहायक।\n\n"
            "निर्देश:\n"
            f"- {ii}\n"
            "- केवल नीचे दिए गए M-CADWM संदर्भ के आधार पर उत्तर दें।\n"
            "- ## शीर्षक, बुलेट पॉइंट और क्रमांकित सूचियाँ उचित स्थान पर उपयोग करें।\n"
            "- कोई emoji उपयोग न करें। औपचारिक सरकारी भाषा-शैली बनाए रखें।\n"
            "- व्यक्तिगत या आत्मीय भाषा का उपयोग न करें।\n"
            "- यदि संदर्भ में जानकारी अनुपलब्ध है, स्पष्ट रूप से सूचित करें।\n\n"
            f"संदर्भ (M-CADWM आधिकारिक दस्तावेज़):\n{context}\n\n"
            f"प्रश्न: {query}\n\nउत्तर:"
        )
    return (
        "You are Samridhi — the professional AI Assistant for the M-CADWM scheme, "
        "its official documents, website, and the SMIS portal.\n\n"
        "Instructions:\n"
        f"- {ii}\n"
        "- Answer strictly from the M-CADWM context provided below.\n"
        "- Use ## headings, bullet points, and numbered lists where appropriate.\n"
        "- Do NOT use emojis. Maintain a formal, government-appropriate tone.\n"
        "- Do NOT use informal or personable language. Do NOT begin with conversational openers.\n"
        "- Do NOT invent information absent from the context.\n"
        "- If the context does not contain sufficient information, state this clearly.\n\n"
        f"Context (M-CADWM official documents):\n{context}\n\n"
        f"Query: {query}\n\nResponse:"
    )


def _prompt_live(context: str, query: str, intent: str) -> str:
    ii = _intent_instruction(intent)
    if lang == "hi":
        return (
            "आप समृद्धि हैं — M-CADWM योजना और SMIS पोर्टल के लिए पेशेवर AI सहायक।\n\n"
            "निर्देश:\n"
            f"- {ii}\n"
            "- नीचे दिए cadwm.gov.in स्रोत के आधार पर विस्तृत और पेशेवर उत्तर हिंदी में प्रदान करें।\n"
            "- ## शीर्षक और बुलेट पॉइंट उपयोग करें।\n"
            "- कोई emoji उपयोग न करें। औपचारिक सरकारी भाषा-शैली बनाए रखें।\n\n"
            f"संदर्भ (cadwm.gov.in से):\n{context}\n\n"
            f"प्रश्न: {query}\n\nउत्तर:"
        )
    return (
        "You are Samridhi — the professional AI Assistant for the M-CADWM scheme and SMIS portal.\n\n"
        "Instructions:\n"
        f"- {ii}\n"
        "- Answer from the cadwm.gov.in content provided below.\n"
        "- Use ## headings and bullet points where appropriate.\n"
        "- Do NOT use emojis. Maintain a formal, government-appropriate tone.\n"
        "- Do NOT use informal or personable language.\n\n"
        f"Context (from cadwm.gov.in):\n{context}\n\n"
        f"Query: {query}\n\nResponse:"
    )


def _prompt_fallback(query: str, is_related: bool) -> str:
    if lang == "hi":
        scope = (
            "यदि प्रश्न M-CADWM या SMIS के व्यापक दायरे से संबंधित है, सामान्य ज्ञान से "
            "संक्षिप्त और तथ्यात्मक जानकारी प्रदान करें। स्पष्ट करें कि यह M-CADWM "
            "आधिकारिक दस्तावेज़ों से नहीं है।"
            if is_related else
            "विनम्रता से सूचित करें कि यह प्रश्न M-CADWM और SMIS के दायरे से बाहर है।"
        )
        return (
            "आप समृद्धि हैं — M-CADWM योजना और SMIS पोर्टल के लिए पेशेवर AI सहायक।\n\n"
            f"प्रश्न: \"{query}\"\n\n"
            "M-CADWM दस्तावेज़ों और वेबसाइट में यह जानकारी उपलब्ध नहीं है।\n\n"
            f"निर्देश:\n- {scope}\n"
            "- कोई emoji उपयोग न करें। औपचारिक भाषा-शैली बनाए रखें।\n"
            "- व्यक्तिगत भाषा उपयोग न करें। प्रश्न की श्रेणी का उल्लेख न करें।\n\nउत्तर:"
        )
    scope = (
        "Provide a brief, factual response from general knowledge if the query is broadly related "
        "to M-CADWM or SMIS. Clearly note it is not from official M-CADWM documents."
        if is_related else
        "Inform the user this query falls outside the scope of M-CADWM and SMIS, "
        "and direct them to submit M-CADWM/SMIS-specific queries."
    )
    return (
        "You are Samridhi — the professional AI Assistant for the M-CADWM scheme and SMIS portal.\n\n"
        f"Query: \"{query}\"\n\n"
        "This query is not covered in the M-CADWM official documents or website.\n\n"
        f"Instructions:\n- {scope}\n"
        "- Do NOT use emojis. Maintain a formal, government-appropriate tone.\n"
        "- Do NOT use informal language. Do NOT mention query categories.\n"
        "- Close by directing the user to M-CADWM/SMIS-specific queries.\n\nResponse:"
    )


# ══════════════════════════════════════════════════════════════
# 20. FOLLOW-UP QUESTION GENERATOR
#     Generates 2-3 contextually relevant follow-up questions
#     after a successful FAISS or live answer.
#     Displayed as clickable suggestions below the response.
# ══════════════════════════════════════════════════════════════

def generate_follow_ups(query: str, answer: str) -> list:
    """
    Ask the LLM to generate 2-3 short follow-up questions.
    Returns a list of strings. Returns [] on failure.
    Capped at one LLM call; non-blocking on error.
    """
    try:
        if lang == "hi":
            prompt = (
                "नीचे दिए गए प्रश्न और उत्तर के आधार पर, M-CADWM या SMIS से संबंधित "
                "2-3 संक्षिप्त और उपयोगी अनुवर्ती प्रश्न सुझाइए।\n"
                "प्रत्येक प्रश्न एक अलग पंक्ति पर लिखें। केवल प्रश्न लिखें, कोई संख्या या बुलेट नहीं।\n\n"
                f"प्रश्न: {query}\nउत्तर: {answer[:500]}\n\nअनुवर्ती प्रश्न:"
            )
        else:
            prompt = (
                "Based on the question and answer below, suggest 2-3 short, useful follow-up questions "
                "a user might want to ask about M-CADWM or SMIS.\n"
                "Write each question on a separate line. Write only the questions, no numbers or bullets.\n\n"
                f"Question: {query}\nAnswer: {answer[:500]}\n\nFollow-up questions:"
            )
        raw = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        lines = [l.strip().lstrip("•-123456789. ") for l in raw.splitlines() if l.strip()]
        return [l for l in lines if len(l) > 5][:3]
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════
# 21. CORE RAG PIPELINE
# ══════════════════════════════════════════════════════════════

def ask_rag(question: str) -> tuple:
    """
    Returns (answer: str, layer: str, follow_ups: list).

    6-layer pipeline:
      L0  Greeting       → static welcome text
      L1  Cache          → lang-isolated feedback cache
      L2  Typo fix       → skipped for ≤4-word queries
      L3  Intent         → keyword pre-classifier → LLM fallback
      L4  FAISS          → hybrid semantic + keyword retrieval
      L5  Live scrape    → cadwm.gov.in (persistent TTL cache)
      L6  Fallback       → scope-gated general knowledge
    """
    t0 = time.time()

    # L0 — Greeting
    if is_greeting(question):
        _log_query(question, "greeting", 0.0, (time.time()-t0)*1000)
        return ui["welcome"], "greeting", []

    # L1 — Language-isolated cache
    cached = get_cached(question)
    if cached:
        _log_query(question, "cache", 1.0, (time.time()-t0)*1000)
        return ui["cached_note"] + cached, "cache", []

    # L2 — Typo correction
    corrected_q = correct_typos(question)
    typo_note   = ""
    if corrected_q.lower() != question.lower():
        typo_note = ui["typo_note"].format(q=corrected_q) + "\n\n"
    search_q = corrected_q

    # L3 — Intent classification
    intent = classify_intent(search_q)

    # L4 — FAISS hybrid retrieval
    docs, confidence = retrieve_docs(search_q)
    if confidence >= FAISS_CONFIDENCE_THRESHOLD and docs:
        context = "\n\n---\n\n".join(d.page_content for d in docs)
        try:
            answer = llm.invoke(
                [HumanMessage(content=_prompt_faiss(context, search_q, intent))]
            ).content
            follow_ups = generate_follow_ups(search_q, answer)
            _log_query(question, "faiss", confidence, (time.time()-t0)*1000)
            return typo_note + answer + ui["src_faiss"], "faiss", follow_ups
        except Exception:
            pass

    # L5 — Live cadwm.gov.in
    with st.spinner(ui["spinner_web"]):
        live_ctx = fetch_live_context(search_q)
    if live_ctx:
        try:
            answer = llm.invoke(
                [HumanMessage(content=_prompt_live(live_ctx, search_q, intent))]
            ).content
            follow_ups = generate_follow_ups(search_q, answer)
            _log_query(question, "live", 0.5, (time.time()-t0)*1000)
            return typo_note + answer + ui["src_live"], "live", follow_ups
        except Exception:
            pass

    # L6 — Scope-aware fallback
    try:
        answer = llm.invoke(
            [HumanMessage(content=_prompt_fallback(corrected_q, is_related_topic(corrected_q)))]
        ).content
        _log_query(question, "fallback", 0.0, (time.time()-t0)*1000)
        return typo_note + answer + ui["src_general"], "fallback", []
    except Exception:
        _log_query(question, "error", 0.0, (time.time()-t0)*1000)
        return ui["no_result"], "error", []


# ══════════════════════════════════════════════════════════════
# 22. TTS
# ══════════════════════════════════════════════════════════════

_TTS_ABBR: dict = {
    "M-CADWM": "M-CAD-WM",
    "SMIS":    "S-MIS",
    "O&M":     "Operations and Maintenance",
    "&":       "and",
    "%":       "percent",
    "i.e.":    "that is",
    "e.g.":    "for example",
    "etc.":    "and so on",
    "Govt":    "Government",
    "Rs.":     "Rupees",
    "WUA":     "Water Users Association",
    "WUS":     "Water Users Society",
}

def clean_tts(text: str) -> str:
    text = re.sub(r"\*\(.*?\)\*\n\n", "", text)
    text = re.sub(r"#{1,6}\s*",        "", text)
    text = re.sub(r"\*\*(.*?)\*\*",   r"\1", text)
    text = re.sub(r"\*(.*?)\*",        r"\1", text)
    text = re.sub(r"`(.*?)`",          r"\1", text)
    text = re.sub(r"^\s*[-•]\s*",     ", ",  text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s*",    ", ",  text, flags=re.MULTILINE)
    if lang == "en":
        for abbr, exp in _TTS_ABBR.items():
            text = text.replace(abbr, exp)
    text = text.replace(":", " , ").replace(";", " , ")
    text = re.sub(r"\n{2,}", ". ", text)
    text = re.sub(r"\n",    " ",   text)
    text = re.sub(r"\s{2,}", " ",  text)
    text = re.sub(r"\.{2,}", ".",  text)
    return text.strip()

async def _gen_audio(text: str, voice: str, rate: str):
    await edge_tts.Communicate(text=text, voice=voice, rate=rate).save(str(AUDIO_FILE))

def speak(text: str):
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_gen_audio(clean_tts(text), tts_voice, tts_rate))
        loop.close()
    except Exception:
        pass

def autoplay_audio():
    if not AUDIO_FILE.exists():
        return
    try:
        b64 = base64.b64encode(AUDIO_FILE.read_bytes()).decode()
        st.markdown(
            f'<audio autoplay controls style="width:100%;margin-top:8px;border-radius:8px;">'
            f'<source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>',
            unsafe_allow_html=True,
        )
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════
# 23. WELCOME MESSAGE  (first load only)
# ══════════════════════════════════════════════════════════════

if not st.session_state.messages:
    st.session_state.messages = [{"role": "assistant", "content": ui["welcome"]}]


# ══════════════════════════════════════════════════════════════
# 24. CHAT HISTORY RENDER
# ══════════════════════════════════════════════════════════════

for _i, _msg in enumerate(st.session_state.messages):
    with st.chat_message(_msg["role"]):
        st.markdown(_msg["content"])
        # Follow-up suggestions stored in message metadata
        if _msg.get("follow_ups"):
            st.markdown(ui["follow_header"])
            for _fq in _msg["follow_ups"]:
                st.markdown(f"- {_fq}")
        # Feedback buttons
        if _msg["role"] == "assistant" and _i in st.session_state.pending_feedback:
            _pf = st.session_state.pending_feedback[_i]
            _rk = f"rated_{_i}"
            if _rk not in st.session_state:
                _c1, _c2, _ = st.columns([1, 1, 8])
                with _c1:
                    if st.button("👍", key=f"up_{_i}"):
                        record_feedback(_pf["q"], _pf["a"], "up")
                        st.session_state[_rk] = "up"
                        st.rerun()
                with _c2:
                    if st.button("👎", key=f"dn_{_i}"):
                        record_feedback(_pf["q"], _pf["a"], "down")
                        st.session_state[_rk] = "down"
                        st.rerun()
            else:
                st.caption(ui["fb_up"] if st.session_state[_rk] == "up" else ui["fb_dn"])


# ══════════════════════════════════════════════════════════════
# 25. USER INPUT HANDLER
# ══════════════════════════════════════════════════════════════

if question := st.chat_input(ui["placeholder"]):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner(ui["spinner"]):
            answer, layer, follow_ups = ask_rag(question)
        st.markdown(answer)

        # Render follow-up suggestions inline
        if follow_ups:
            st.markdown(ui["follow_header"])
            for fq in follow_ups:
                st.markdown(f"- {fq}")

        _idx = len(st.session_state.messages)
        st.session_state.pending_feedback[_idx] = {"q": question, "a": answer}

        _rk = f"rated_{_idx}"
        _c1, _c2, _ = st.columns([1, 1, 8])
        with _c1:
            if st.button("👍", key=f"up_{_idx}"):
                record_feedback(question, answer, "up")
                st.session_state[_rk] = "up"
                st.rerun()
        with _c2:
            if st.button("👎", key=f"dn_{_idx}"):
                record_feedback(question, answer, "down")
                st.session_state[_rk] = "down"
                st.rerun()

        # Store message with follow_ups for re-render on history scroll
        st.session_state.messages.append({
            "role":       "assistant",
            "content":    answer,
            "follow_ups": follow_ups,
        })
        speak(answer)
        autoplay_audio()