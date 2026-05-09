"""
app.py — Samridhi AI v2.0  (M-CADWM & SMIS)
============================================
Streamlit app styled to match index.html exactly.
Run:  streamlit run app.py
"""

from __future__ import annotations
import base64, os, re, time, threading
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from samridhi.config import BASE_DIR, LOGO_PATH, cfg
from samridhi.logger import get_logger
from samridhi.pipeline import Pipeline, RateLimiter
from samridhi.resources import (
    get_analytics, get_expansions, get_feedback_db,
    get_llm, get_vector_db, get_web_cache,
)
from samridhi.tts import autoplay_audio, speak
from samridhi.ui_strings import BRIDGE_JS, UI

log = get_logger()

_PENDING_FEEDBACK_MAX = 20
_FEEDBACK_LAYERS      = frozenset({"faiss", "live", "fallback", "cache"})

def _speak_bg(text: str, lang: str, result: list):
    try:
        result.append(speak(text, lang))
    except Exception:
        result.append("")

try:
    import yaml as _yaml;   _HAS_YAML = True
except ImportError:
    _HAS_YAML = False
try:
    import httpx as _httpx; _HAS_HTTPX = True
except ImportError:
    _HAS_HTTPX = False
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    _HAS_PDF = True
except ImportError:
    _HAS_PDF = False

# ── Logo ──────────────────────────────────────────────────────
_logo_b64 = ""
_logo_pil  = None
if LOGO_PATH.exists():
    try:
        _logo_b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode()
    except Exception:
        pass
    if _logo_b64:
        try:
            from PIL import Image as _PILImage
            _logo_pil = _PILImage.open(str(LOGO_PATH))
        except Exception:
            pass

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title = "Samridhi AI – M-CADWM",
    page_icon  = _logo_pil if _logo_pil else "🏛️",
    layout     = "wide",
)

# ── API key guard ─────────────────────────────────────────────
load_dotenv()
_groq_key = os.getenv("GROQ_API_KEY")
if not _groq_key:
    try:    _groq_key = st.secrets["GROQ_API_KEY"]
    except: _groq_key = None
if not _groq_key:
    st.error("GROQ_API_KEY not found. Add it to .env or Streamlit secrets.")
    st.stop()

# ── Resources ─────────────────────────────────────────────────
try:
    vector_db = get_vector_db()
except Exception as e:
    st.error(f"Failed to load FAISS index: {e}")
    st.stop()

llm         = get_llm()
feedback_db = get_feedback_db()
web_cache   = get_web_cache()
analytics   = get_analytics()
expansions  = get_expansions()
pipeline    = Pipeline(llm, vector_db, feedback_db, web_cache, analytics,
                       expansion_store=expansions)

# ── Session state ─────────────────────────────────────────────
def _init():
    for k, v in {
        "lang":             "en",
        "messages":         [],
        "pending_feedback": {},
        "reingest_done":    set(),
        "last_answer":      "",
        "rate_bucket":      RateLimiter.make_bucket(),
        "tts_enabled":      True,
        "followup_queue":   None,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v
_init()

lang: str = st.session_state.lang
ui:   dict = UI[lang]

# ══════════════════════════════════════════════════════════════
# GLOBAL CSS — replicates index.html exactly
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Hide Streamlit chrome ───────────────────────────────── */
#MainMenu, footer, header { visibility: hidden !important; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stStatusWidget"] { display: none !important; }

/* ── Root palette (matches index.html exactly) ───────────── */
:root {
  --bg:           #0E1117;
  --surface:      #1A1D2E;
  --surface2:     #12141F;
  --border:       rgba(255,255,255,0.08);
  --text:         #E8ECF4;
  --muted:        #8892A4;
  --accent:       #4A90D9;
  --accent-hover: #5BA3EC;
  --green:        #4ade80;
  --amber:        #fbbf24;
  --radius:       12px;
}

/* ── Full page background ────────────────────────────────── */
.stApp, [data-testid="stAppViewContainer"] {
  background-color: var(--bg) !important;
}

/* ── Main block container ────────────────────────────────── */
[data-testid="stMain"] {
  background-color: var(--bg) !important;
  padding: 0 !important;
}
.block-container {
  padding: 0 !important;
  max-width: 100% !important;
}

/* ── Sidebar — matches index.html .sidebar ───────────────── */
[data-testid="stSidebar"] {
  background-color: var(--surface2) !important;
  border-right: 1px solid var(--border) !important;
  min-width: 260px !important;
  max-width: 260px !important;
}
[data-testid="stSidebar"] > div {
  padding: 20px 14px !important;
  background-color: var(--surface2) !important;
}

/* ── Sidebar brand ───────────────────────────────────────── */
.sb-brand {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding-bottom: 16px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 12px;
}
.sb-brand img {
  width: 64px; height: 64px;
  object-fit: contain;
  border-radius: 8px;
  margin-bottom: 8px;
}
.sb-brand .name {
  font-weight: 700; font-size: 16px; color: var(--text);
}
.sb-brand .sub {
  font-size: 11px; color: var(--muted); margin-top: 2px;
}

/* ── Language toggle buttons ─────────────────────────────── */
.lang-row { display: flex; gap: 6px; margin-bottom: 8px; }
.lang-btn {
  flex: 1; padding: 7px 4px;
  border-radius: 7px;
  border: 1px solid var(--border);
  background: transparent;
  color: var(--muted);
  font-size: 12px; font-weight: 600;
  cursor: pointer; text-align: center;
  transition: all 0.15s;
}
.lang-btn.active {
  background: var(--accent);
  border-color: var(--accent);
  color: #fff;
}

/* ── Sidebar buttons ─────────────────────────────────────── */
[data-testid="stSidebar"] .stButton > button {
  background: transparent !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  border-radius: 8px !important;
  font-size: 13px !important;
  text-align: left !important;
  width: 100% !important;
  transition: background 0.15s !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
  background: rgba(255,255,255,0.05) !important;
  border-color: rgba(255,255,255,0.15) !important;
}

/* New conversation — primary blue button */
.new-conv-btn > button {
  background: var(--accent) !important;
  border-color: var(--accent) !important;
  color: #fff !important;
  font-weight: 600 !important;
  text-align: center !important;
  border-radius: 8px !important;
  width: 100% !important;
}
.new-conv-btn > button:hover {
  background: var(--accent-hover) !important;
}

/* ── Toggle ──────────────────────────────────────────────── */
[data-testid="stToggle"] label { color: var(--text) !important; font-size: 13px !important; }

/* ── Divider ─────────────────────────────────────────────── */
hr { border-color: var(--border) !important; margin: 10px 0 !important; }

/* ── Chat area ───────────────────────────────────────────── */
.chat-header {
  padding: 16px 24px 12px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: 12px;
  background: var(--bg);
}
.chat-header .title { font-size: 16px; font-weight: 700; color: var(--text); }
.chat-header .subtitle { font-size: 12px; color: var(--muted); }

/* ── Chat messages ───────────────────────────────────────── */
[data-testid="stChatMessage"] {
  background-color: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 16px !important;
  margin-bottom: 8px !important;
  color: var(--text) !important;
}

/* ── Chat input ──────────────────────────────────────────── */
[data-testid="stChatInput"] {
  background-color: var(--surface2) !important;
  border-top: 1px solid var(--border) !important;
}
[data-testid="stChatInputTextArea"] {
  background-color: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--text) !important;
  font-size: 15px !important;
}
[data-testid="stChatInput"] button {
  background: var(--accent) !important;
  border-radius: 8px !important;
}

/* ── Source badges ───────────────────────────────────────── */
.src-badge {
  display: inline-block;
  font-size: 12px; font-weight: 600;
  padding: 3px 10px; border-radius: 20px;
  margin-bottom: 10px;
}
.src-badge-green { background: rgba(74,222,128,0.12); color: var(--green); }
.src-badge-amber { background: rgba(251,191,36,0.12);  color: var(--amber); }
.src-badge-grey  { background: rgba(148,163,184,0.10); color: var(--muted); }

/* ── Copy button ─────────────────────────────────────────── */
.samridhi-copy-btn {
  background: none;
  border: 1px solid var(--border);
  border-radius: 5px;
  padding: 3px 10px;
  font-size: 12px;
  color: var(--muted);
  cursor: pointer;
  margin-top: 8px;
  transition: all 0.15s;
}
.samridhi-copy-btn:hover {
  background: var(--accent); color: #fff; border-color: var(--accent);
}

/* ── Follow-up buttons ───────────────────────────────────── */
.follow-label {
  font-size: 11px; font-weight: 600;
  color: var(--muted);
  text-transform: uppercase; letter-spacing: 0.5px;
  margin: 10px 0 4px 0;
}

/* ── Footer ──────────────────────────────────────────────── */
.samridhi-footer {
  text-align: center;
  font-size: 11px;
  color: var(--muted);
  padding: 10px 24px;
  border-top: 1px solid var(--border);
  background: var(--surface2);
}

/* ── Expander ────────────────────────────────────────────── */
[data-testid="stExpander"] {
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  background: transparent !important;
}
[data-testid="stExpander"] summary {
  color: var(--muted) !important; font-size: 13px !important;
}
</style>
""", unsafe_allow_html=True)

# ── JS bridge ─────────────────────────────────────────────────
components.html(BRIDGE_JS, height=0)

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    # Brand
    _sb_logo = _logo_b64
    logo_html = (
        f'<img src="data:image/png;base64,{_sb_logo}">'
        if _sb_logo else '🏛️'
    )
    st.markdown(
        f'<div class="sb-brand">'
        f'{logo_html}'
        f'<div class="name">Samridhi AI</div>'
        f'<div class="sub">M-CADWM &amp; SMIS</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Language toggle
    st.markdown('<div class="lang-row">', unsafe_allow_html=True)
    _c1, _c2 = st.columns(2)
    with _c1:
        if st.button("🇬🇧 EN", key="btn_en",
                     type="primary" if lang == "en" else "secondary",
                     use_container_width=True):
            st.session_state.lang = "en"
            st.session_state.messages = []
            st.session_state.pending_feedback = {}
            st.rerun()
    with _c2:
        if st.button("🇮🇳 HI", key="btn_hi",
                     type="primary" if lang == "hi" else "secondary",
                     use_container_width=True):
            st.session_state.lang = "hi"
            st.session_state.messages = []
            st.session_state.pending_feedback = {}
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # Voice toggle
    st.session_state.tts_enabled = st.toggle(
        ui["tts_toggle"], value=st.session_state.tts_enabled,
    )

    # Common queries
    _starters = (
        cfg["ui"]["starter_questions_hi"] if lang == "hi"
        else cfg["ui"]["starter_questions_en"]
    )
    with st.expander(f"❓ {ui['starter_label']}", expanded=False):
        for _sq in _starters:
            if st.button(_sq, key=f"sq_{hash(_sq)}", use_container_width=True):
                st.session_state.followup_queue = _sq
                st.rerun()

    st.divider()

    # New conversation
    st.markdown('<div class="new-conv-btn">', unsafe_allow_html=True)
    if st.button(f"+ {ui['clear_chat']}", use_container_width=True):
        st.session_state.messages         = []
        st.session_state.pending_feedback = {}
        st.session_state.last_answer      = ""
        st.session_state.followup_queue   = None
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # PDF download
    if _HAS_PDF and st.session_state.last_answer:
        if st.button(ui["pdf_label"], use_container_width=True):
            try:
                import io, datetime as _dt
                buf    = io.BytesIO()
                doc    = SimpleDocTemplate(buf, pagesize=A4)
                styles = getSampleStyleSheet()
                clean  = re.sub(r"#{1,6}\s*", "", st.session_state.last_answer)
                clean  = re.sub(r"\*\*(.*?)\*\*", r"\1", clean)
                clean  = re.sub(r"\*(.*?)\*",     r"\1", clean)
                story  = [
                    Paragraph("Samridhi AI — M-CADWM & SMIS", styles["Title"]),
                    Spacer(1, 12),
                    Paragraph(f"Generated: {_dt.datetime.now().strftime('%Y-%m-%d %H:%M')}",
                              styles["Normal"]),
                    Spacer(1, 12),
                ]
                for para in clean.split("\n\n"):
                    para = para.strip()
                    if para:
                        story.append(Paragraph(para.replace("\n", "<br/>"), styles["Normal"]))
                        story.append(Spacer(1, 8))
                doc.build(story)
                st.download_button(
                    "⬇ Download PDF", data=buf.getvalue(),
                    file_name=f"samridhi_{int(time.time())}.pdf",
                    mime="application/pdf",
                )
            except Exception as e:
                st.error(f"PDF failed: {e}")

    st.divider()

    # About
    with st.expander(ui["about_label"], expanded=False):
        st.markdown(
            '<div style="font-size:12px;line-height:1.8;color:var(--muted);">'
            '<b style="color:var(--text);">Samridhi AI</b> &nbsp;v2.0<br>'
            'AI Assistant for M-CADWM &amp; SMIS<br>'
            'Ministry of Jal Shakti<br>'
            'Government of India<br>'
            '<a href="https://cadwm.gov.in" target="_blank" '
            'style="color:var(--accent);">cadwm.gov.in</a>'
            '</div>',
            unsafe_allow_html=True,
        )

    # Operator panel
    _params = st.query_params
    if _params.get("operator") == "1":
        _op_pw = cfg.get("operator", {}).get("password", "samridhi-admin")
        if not st.session_state.get("op_unlocked"):
            _pwd = st.text_input("Password", type="password", key="op_pwd")
            if st.button("Unlock", key="op_btn"):
                if _pwd == _op_pw:
                    st.session_state.op_unlocked = True
                    st.rerun()
                else:
                    st.error("Incorrect password.")
        else:
            st.markdown("**System Status**")
            fb_s = feedback_db.stats(); wc_s = web_cache.stats(); ex_s = expansions.stats()
            ana_r = analytics.recent(5)
            st.markdown(
                f"- FAISS: {'✅' if (BASE_DIR/'faiss_index').exists() else '❌'}\n"
                f"- Feedback: {fb_s['total']} ({fb_s['positive']} positive)\n"
                f"- Web cache: {wc_s['total_entries']}\n"
                f"- Expansions: {ex_s['enabled']}/{ex_s['total']}"
            )
            if ana_r:
                st.markdown("**Recent queries**")
                for rec in reversed(ana_r):
                    st.caption(
                        f"`{rec.get('layer','?')}` "
                        f"{rec.get('confidence',0):.2f} "
                        f"{rec.get('response_ms',0):.0f}ms — "
                        f"{rec.get('query','')[:40]}"
                    )
            if st.button("Lock", key="op_lock"):
                st.session_state.op_unlocked = False
                st.rerun()

# ══════════════════════════════════════════════════════════════
# CHAT HEADER (matches index.html .chat-header)
# ══════════════════════════════════════════════════════════════
_hdr_logo = (
    f'<img src="data:image/png;base64,{_logo_b64}" '
    f'style="width:32px;height:32px;object-fit:contain;border-radius:6px;">'
    if _logo_b64 else "🏛️"
)
st.markdown(
    f'<div class="chat-header">'
    f'{_hdr_logo}'
    f'<div>'
    f'<div class="title">Samridhi AI</div>'
    f'<div class="subtitle">AI Assistant — M-CADWM &amp; SMIS</div>'
    f'</div>'
    f'</div>',
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════
_SRC_MARKERS = [
    UI["en"]["src_faiss"], UI["en"]["src_live"], UI["en"]["src_general"],
    UI["hi"]["src_faiss"], UI["hi"]["src_live"], UI["hi"]["src_general"],
]

def _strip_source(text: str) -> str:
    for marker in _SRC_MARKERS:
        if marker and text.rstrip().endswith(marker.strip()):
            return text[:len(text.rstrip()) - len(marker.strip())].rstrip()
    return text

def _source_badge(layer: str):
    badge_map = {
        "faiss":    ("src-badge-green", ui.get("badge_faiss",   "✦ M-CADWM Official Documents")),
        "live":     ("src-badge-amber", ui.get("badge_live",    "◉ cadwm.gov.in (live)")),
        "fallback": ("src-badge-grey",  ui.get("badge_general", "◈ General Knowledge")),
        "cache":    ("src-badge-grey",  ui.get("badge_cache",   "◇ Retrieved from cache")),
    }
    if layer not in badge_map:
        return
    css_cls, label = badge_map[layer]
    st.markdown(f'<span class="src-badge {css_cls}">{label}</span>', unsafe_allow_html=True)

def _follow_ups(fups: list, msg_idx: int):
    if not fups:
        return
    st.markdown('<div class="follow-label">You may also ask:</div>', unsafe_allow_html=True)
    for _fi, fq in enumerate(fups):
        if st.button(fq, key=f"fup_{msg_idx}_{_fi}", use_container_width=True):
            st.session_state.followup_queue = fq
            st.rerun()

def _copy_btn(content: str):
    b64 = base64.b64encode(content.encode("utf-8")).decode("utf-8")
    st.markdown(
        f'<button class="samridhi-copy-btn" data-copy="{b64}" '
        f'onclick="var t=atob(this.getAttribute(\'data-copy\'));'
        f'navigator.clipboard.writeText(t).then(()=>{{this.textContent=\'✓ Copied\';}});">'
        f'{ui["copy_label"]}</button>',
        unsafe_allow_html=True,
    )

def _feedback(i: int, layer: str = ""):
    if layer not in _FEEDBACK_LAYERS:
        return
    if i not in st.session_state.pending_feedback:
        return
    pf = st.session_state.pending_feedback[i]
    rk = f"rated_{i}"
    if rk not in st.session_state:
        c1, c2, _ = st.columns([1, 1, 8])
        with c1:
            if st.button("👍", key=f"up_{i}"):
                feedback_db.record(pf["q"], pf["a"], "up", lang)
                st.session_state[rk] = "up"
                st.rerun()
        with c2:
            if st.button("👎", key=f"dn_{i}"):
                feedback_db.record(pf["q"], pf["a"], "down", lang)
                st.session_state[rk] = "down"
                st.rerun()
    else:
        st.caption(ui["fb_up"] if st.session_state[rk] == "up" else ui["fb_dn"])

# ══════════════════════════════════════════════════════════════
# FOLLOW-UP QUEUE
# ══════════════════════════════════════════════════════════════
def _process_followup_queue():
    q = st.session_state.followup_queue
    if not q:
        return
    st.session_state.followup_queue = None
    st.session_state.messages.append(
        {"role": "user", "content": q, "follow_ups": [], "layer": ""}
    )
    r = pipeline.run(q, lang, st.session_state.messages,
                     st.session_state.rate_bucket, ui)
    pipeline.maybe_reingest(lang, st.session_state.reingest_done)
    st.session_state.last_answer = r.answer
    _idx = len(st.session_state.messages)
    st.session_state.pending_feedback[_idx] = {"q": q, "a": r.answer}
    if len(st.session_state.pending_feedback) > _PENDING_FEEDBACK_MAX:
        del st.session_state.pending_feedback[min(st.session_state.pending_feedback)]
    st.session_state.messages.append({
        "role": "assistant", "content": r.answer,
        "follow_ups": r.follow_ups, "layer": r.layer,
    })

_process_followup_queue()

# ══════════════════════════════════════════════════════════════
# WELCOME MESSAGE
# ══════════════════════════════════════════════════════════════
if not st.session_state.messages:
    st.session_state.messages = [
        {"role": "assistant", "content": ui["welcome"], "follow_ups": [], "layer": ""}
    ]

# ══════════════════════════════════════════════════════════════
# CHAT HISTORY
# ══════════════════════════════════════════════════════════════
for _i, _msg in enumerate(st.session_state.messages):
    with st.chat_message(_msg["role"]):
        _layer          = _msg.get("layer", "")
        _is_substantive = _layer in _FEEDBACK_LAYERS

        if _msg["role"] == "assistant" and _is_substantive:
            _source_badge(_layer)

        _display = (
            _strip_source(_msg["content"])
            if _msg["role"] == "assistant"
            else _msg["content"]
        )
        st.markdown(_display)
        _follow_ups(_msg.get("follow_ups", []), _i)

        if _msg["role"] == "assistant" and _is_substantive:
            _copy_btn(_msg["content"])
        if _msg["role"] == "assistant":
            _feedback(_i, _layer)

# ══════════════════════════════════════════════════════════════
# USER INPUT
# ══════════════════════════════════════════════════════════════
if question := st.chat_input(ui["placeholder"]):
    st.session_state.messages.append(
        {"role": "user", "content": question, "follow_ups": [], "layer": ""}
    )
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        _tts_result: list = []
        _tts_thread = None

        with st.spinner(ui["spinner"]):
            r = pipeline.run(
                question, lang,
                st.session_state.messages,
                st.session_state.rate_bucket, ui,
            )

        if st.session_state.tts_enabled and r.layer in _FEEDBACK_LAYERS:
            _tts_thread = threading.Thread(
                target=_speak_bg, args=(r.answer, lang, _tts_result), daemon=True
            )
            _tts_thread.start()

        _source_badge(r.layer)
        st.markdown(_strip_source(r.answer))
        st.session_state.last_answer = r.answer
        _follow_ups(r.follow_ups, len(st.session_state.messages))

        if r.layer in _FEEDBACK_LAYERS:
            _copy_btn(r.answer)

        _idx = len(st.session_state.messages)
        st.session_state.pending_feedback[_idx] = {"q": question, "a": r.answer}
        if len(st.session_state.pending_feedback) > _PENDING_FEEDBACK_MAX:
            del st.session_state.pending_feedback[min(st.session_state.pending_feedback)]

        _feedback(_idx, r.layer)
        st.session_state.messages.append({
            "role": "assistant", "content": r.answer,
            "follow_ups": r.follow_ups, "layer": r.layer,
        })
        pipeline.maybe_reingest(lang, st.session_state.reingest_done)

        if _tts_thread is not None:
            _tts_thread.join(timeout=15)
            if _tts_result:
                autoplay_audio(_tts_result[0], st)

# ══════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown(
    '<div class="samridhi-footer">'
    'Copyright &copy; 2025 &nbsp;·&nbsp; All Rights Reserved &nbsp;·&nbsp; '
    'CADWM Wing, Department of Water Resources, River Development &amp; '
    'Ganga Rejuvenation, Ministry of Jal Shakti, Government of India'
    '</div>',
    unsafe_allow_html=True,
)
