"""
app.py — Samridhi AI v1.0  (M-CADWM & SMIS)
============================================
Public-facing AI assistant. No login required.
Theme: dark — set in .streamlit/config.toml (framework level).

Run:  streamlit run app.py
Test: pytest test_samridhi.py -v
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

# ── constants ─────────────────────────────────────────────────
_PENDING_FEEDBACK_MAX = 20
_FEEDBACK_LAYERS      = frozenset({"faiss", "live", "fallback", "cache"})

# Palette — dark theme accent colours only
# Framework handles backgrounds; we only define accents + badges
_C = {
    "accent":           "#4A90D9",   # Streamlit primary
    "accent_hover":     "#6AABF0",
    "sidebar_border":   "#2A2D3E",
    "badge_green_bg":   "rgba(34,197,94,0.15)",
    "badge_green_fg":   "#4ade80",
    "badge_amber_bg":   "rgba(251,146,60,0.15)",
    "badge_amber_fg":   "#fbbf24",
    "badge_grey_bg":    "rgba(148,163,184,0.12)",
    "badge_grey_fg":    "#94a3b8",
    "follow_bg":        "rgba(74,144,217,0.10)",
    "follow_fg":        "#93c5fd",
    "follow_border":    "rgba(74,144,217,0.30)",
    "follow_hover_bg":  "#4A90D9",
    "follow_hover_fg":  "#ffffff",
    "divider":          "rgba(255,255,255,0.08)",
    "muted":            "#8892a4",
    "text":             "#E8ECF4",
    "surface":          "#1A1D2E",
    "surface2":         "#12141F",
}

# ── background TTS ────────────────────────────────────────────
def _speak_bg(text: str, lang: str, result: list):
    try:
        result.append(speak(text, lang))
    except Exception:
        result.append("")

# ── optional libs ─────────────────────────────────────────────
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
_logo_pil = None
_logo_b64  = ""
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

# ── page config ───────────────────────────────────────────────
st.set_page_config(
    page_title = "Samridhi AI – M-CADWM",
    page_icon  = _logo_pil if _logo_pil else "🏛️",
    layout     = "centered",
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

# ── resources ─────────────────────────────────────────────────
try:
    vector_db = get_vector_db()
except Exception as e:
    log.error(f"FAISS load failed: {e}")
    st.error(f"Failed to load FAISS index: {e}")
    st.stop()

llm         = get_llm()
feedback_db = get_feedback_db()
web_cache   = get_web_cache()
analytics   = get_analytics()
expansions  = get_expansions()
pipeline    = Pipeline(llm, vector_db, feedback_db, web_cache, analytics,
                       expansion_store=expansions)

# ── session state ─────────────────────────────────────────────
def _init():
    for k, v in {
        "lang":              "en",
        "messages":          [],
        "pending_feedback":  {},
        "logo_b64":          _logo_b64,
        "reingest_done":     set(),
        "last_answer":       "",
        "rate_bucket":       RateLimiter.make_bucket(),
        "tts_enabled":       True,
        "followup_queue":    None,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v
_init()

lang: str = st.session_state.lang
ui:   dict = UI[lang]

# ══════════════════════════════════════════════════════════════
# CSS — minimal layer on top of Streamlit dark theme
# Only customises surfaces the framework doesn't fully control.
# Never overrides text colour globally (Streamlit handles that).
# ══════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
/* ── Chat messages ───────────────────────────────────────── */
[data-testid="stChatMessage"] {{
    background-color: {_C["surface"]} !important;
    border: 1px solid {_C["divider"]} !important;
    border-radius: 12px !important;
    padding: 16px !important;
    margin-bottom: 8px !important;
}}

/* ── Chat input bar — full-width dark surface ────────────── */
[data-testid="stChatInput"],
[data-testid="stChatInput"] > div {{
    background-color: {_C["surface"]} !important;
    border-top: 1px solid {_C["divider"]} !important;
}}
[data-testid="stChatInputTextArea"] {{
    background-color: {_C["surface2"]} !important;
    border: 1px solid {_C["divider"]} !important;
    border-radius: 8px !important;
    color: {_C["text"]} !important;
    font-size: 15px !important;
}}

/* ── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {{
    border-right: 1px solid {_C["sidebar_border"]} !important;
}}

/* ── Source badges ───────────────────────────────────────── */
.src-badge {{
    display: inline-block;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.4px;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 10px;
}}
.src-badge-green {{
    background: {_C["badge_green_bg"]};
    color: {_C["badge_green_fg"]};
}}
.src-badge-amber {{
    background: {_C["badge_amber_bg"]};
    color: {_C["badge_amber_fg"]};
}}
.src-badge-grey {{
    background: {_C["badge_grey_bg"]};
    color: {_C["badge_grey_fg"]};
}}

/* ── Follow-up buttons ───────────────────────────────────── */
/* Follow-up suggestions use Streamlit st.button — styled via
   the framework primary colour. No custom HTML buttons. */

/* ── Copy button ─────────────────────────────────────────── */
.samridhi-copy-btn {{
    background: none;
    border: 1px solid {_C["divider"]};
    border-radius: 5px;
    padding: 2px 10px;
    font-size: 12px;
    color: {_C["muted"]};
    cursor: pointer;
    margin-top: 8px;
    transition: all 0.15s;
}}
.samridhi-copy-btn:hover {{
    background: {_C["accent"]};
    color: #fff;
    border-color: {_C["accent"]};
}}

/* ── Mic button ──────────────────────────────────────────── */
#samridhi-mic {{
    background: none; border: none; border-radius: 50%;
    width: 34px; height: 34px; font-size: 19px;
    color: {_C["muted"]};
    cursor: pointer; padding: 0;
    transition: color 0.2s, transform 0.15s;
}}
#samridhi-mic:hover {{ color: {_C["accent"]}; transform: scale(1.15); }}
#samridhi-mic.listening {{
    color: {_C["accent"]};
    animation: mic-pulse 0.8s infinite;
}}
@keyframes mic-pulse {{
    0%,100% {{ text-shadow: 0 0 0px {_C["accent"]}; }}
    50%      {{ text-shadow: 0 0 8px {_C["accent"]}; }}
}}
#samridhi-mic-status {{
    position: fixed; bottom: 68px; left: 50%; transform: translateX(-50%);
    z-index: 99999;
    background: {_C["surface"]};
    color: {_C["accent"]};
    font-size: 12px;
    padding: 5px 14px; border-radius: 10px; display: none;
    max-width: 300px; text-align: center;
    border: 1px solid {_C["follow_border"]};
    pointer-events: none;
}}

/* ── Divider ─────────────────────────────────────────────── */
hr {{
    border-color: {_C["divider"]} !important;
    margin: 12px 0 !important;
}}

/* ── Header HR ───────────────────────────────────────────── */
.samridhi-header-hr {{
    border: none;
    border-top: 1px solid {_C["divider"]};
    margin: 8px 0 18px 0;
}}
</style>
<div id="samridhi-mic-status"></div>
""", unsafe_allow_html=True)

# ── JS bridge ─────────────────────────────────────────────────
components.html(BRIDGE_JS, height=0)

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:

    # Logo + name
    _sb_logo = st.session_state.logo_b64 or _logo_b64
    if _sb_logo:
        st.markdown(
            f'<div style="text-align:center;padding:16px 0 6px 0;">'
            f'<img src="data:image/png;base64,{_sb_logo}" '
            f'style="max-width:80px;max-height:80px;object-fit:contain;'
            f'border-radius:8px;"></div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        f'<div style="text-align:center;font-weight:700;font-size:17px;'
        f'margin-bottom:2px;">Samridhi AI</div>'
        f'<div style="text-align:center;font-size:12px;color:{_C["muted"]};'
        f'margin-bottom:14px;">M-CADWM &amp; SMIS</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Voice toggle ──────────────────────────────────────────
    st.session_state.tts_enabled = st.toggle(
        ui["tts_toggle"],
        value=st.session_state.tts_enabled,
    )

    # ── Common queries ────────────────────────────────────────
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

    # ── New conversation ──────────────────────────────────────
    if st.button(ui["clear_chat"], use_container_width=True):
        st.session_state.messages         = []
        st.session_state.pending_feedback = {}
        st.session_state.last_answer      = ""
        st.session_state.followup_queue   = None
        st.rerun()

    # ── PDF download ──────────────────────────────────────────
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
                    Paragraph(
                        f"Generated: {_dt.datetime.now().strftime('%Y-%m-%d %H:%M')}",
                        styles["Normal"],
                    ),
                    Spacer(1, 12),
                ]
                for para in clean.split("\n\n"):
                    para = para.strip()
                    if para:
                        story.append(Paragraph(para.replace("\n", "<br/>"), styles["Normal"]))
                        story.append(Spacer(1, 8))
                doc.build(story)
                st.download_button(
                    "⬇ Download PDF",
                    data=buf.getvalue(),
                    file_name=f"samridhi_{int(time.time())}.pdf",
                    mime="application/pdf",
                )
            except Exception as e:
                st.error(f"PDF failed: {e}")

    st.divider()

    # ── About ─────────────────────────────────────────────────
    with st.expander(ui["about_label"], expanded=False):
        st.markdown(
            f'<div style="font-size:13px;line-height:1.7;color:{_C["muted"]};">'
            f'<b style="color:{_C["text"]};">Samridhi AI</b> &nbsp;v1.0<br>'
            f'AI Assistant for M-CADWM &amp; SMIS<br>'
            f'Ministry of Jal Shakti<br>'
            f'Government of India<br>'
            f'<a href="https://cadwm.gov.in" target="_blank" '
            f'style="color:{_C["accent"]};">cadwm.gov.in</a>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Operator panel — URL-param activated, invisible to users ─
    # Access via: ?operator=1  (append to URL, e.g. localhost:8080/?operator=1)
    # Then type the password when prompted.
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
                f"- YAML config: {'✅' if _HAS_YAML else '⚠'}\n"
                f"- httpx: {'✅' if _HAS_HTTPX else '⚠'}\n"
                f"- PDF: {'✅' if _HAS_PDF else '⚠'}\n"
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
# LANGUAGE TOGGLE
# ══════════════════════════════════════════════════════════════
_, _ec, _hc = st.columns([5, 1, 1])
with _ec:
    if st.button(UI["en"]["btn_en"],
                 type="primary" if lang == "en" else "secondary",
                 use_container_width=True):
        st.session_state.lang = "en"
        st.session_state.messages = []
        st.session_state.pending_feedback = {}
        st.rerun()
with _hc:
    if st.button(UI["hi"]["btn_hi"],
                 type="primary" if lang == "hi" else "secondary",
                 use_container_width=True):
        st.session_state.lang = "hi"
        st.session_state.messages = []
        st.session_state.pending_feedback = {}
        st.rerun()

# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════
_hdr_logo = (
    f'<img src="data:image/png;base64,{_logo_b64}" '
    f'style="max-height:64px;max-width:64px;object-fit:contain;'
    f'display:block;margin:0 auto 8px auto;border-radius:6px;">'
    if _logo_b64 else ""
)
st.markdown(
    f'<div style="text-align:center;margin-top:-10px;">'
    f'{_hdr_logo}'
    f'<h1 style="font-size:28px;font-weight:700;margin-bottom:2px;margin-top:0;">'
    f'Samridhi AI</h1>'
    f'<p style="font-size:14px;color:{_C["muted"]};margin-top:0;">'
    f'AI Assistant &mdash; M-CADWM &amp; SMIS</p>'
    f'</div>'
    f'<div class="samridhi-header-hr"></div>',
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════
# WELCOME MESSAGE INIT
# ══════════════════════════════════════════════════════════════
if not st.session_state.messages:
    st.session_state.messages = [
        {"role": "assistant", "content": ui["welcome"], "follow_ups": [], "layer": ""}
    ]

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
        "faiss":    ("src-badge-green", ui.get("badge_faiss", "M-CADWM Official Documents")),
        "live":     ("src-badge-amber", ui.get("badge_live",  "cadwm.gov.in (live)")),
        "fallback": ("src-badge-grey",  ui.get("badge_general", "General Knowledge")),
        "cache":    ("src-badge-grey",  ui.get("badge_cache",  "Retrieved from cache")),
    }
    if layer not in badge_map:
        return
    css_cls, label = badge_map[layer]
    st.markdown(
        f'<span class="src-badge {css_cls}">{label}</span>',
        unsafe_allow_html=True,
    )

def _follow_ups(fups: list, msg_idx: int):
    if not fups:
        return
    st.markdown(
        f'<div style="font-size:13px;font-weight:600;color:{_C["muted"]};'
        f'margin:10px 0 6px 0;">{ui["follow_header"]}</div>',
        unsafe_allow_html=True,
    )
    for _fi, fq in enumerate(fups):
        if st.button(fq, key=f"fup_{msg_idx}_{_fi}", use_container_width=True):
            st.session_state.followup_queue = fq
            st.rerun()

def _copy_btn(content: str):
    safe = content.replace("'", "\\'").replace("\n", "\\n")
    st.markdown(
        f'<button class="samridhi-copy-btn" '
        f'onclick="window.parent.samridhiCopy(\'{safe}\')">'
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
# FOLLOW-UP QUEUE PROCESSOR
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
# CHAT HISTORY RENDER
# ══════════════════════════════════════════════════════════════
for _i, _msg in enumerate(st.session_state.messages):
    with st.chat_message(_msg["role"]):
        _layer         = _msg.get("layer", "")
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
