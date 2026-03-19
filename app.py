import os
import base64
import json
import hashlib
import streamlit as st
import asyncio
import edge_tts
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from dotenv import load_dotenv
from datetime import datetime

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage


# ==============================
# CONFIG
# ==============================

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
FAISS_PATH    = os.path.join(BASE_DIR, "faiss_index")
LOGO_PATH     = os.path.join(BASE_DIR, "logo.png")
FEEDBACK_FILE = os.path.join(BASE_DIR, "feedback_cache.json")

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
    layout="centered"
)

# ==============================
# INJECT CSS + MIC JS
# ==============================

st.markdown("""
<style>
#samridhi-mic {
    background: none;
    border: none;
    border-radius: 50%;
    width: 34px;
    height: 34px;
    font-size: 19px;
    color: #888;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: color 0.2s, transform 0.15s;
    flex-shrink: 0;
    padding: 0;
    margin-right: 4px;
}
#samridhi-mic:hover { color: #e05c2a; transform: scale(1.2); }
#samridhi-mic.listening {
    color: #e05c2a;
    animation: mic-pulse 0.8s infinite;
}
@keyframes mic-pulse {
    0%,100% { text-shadow: 0 0 0px #e05c2a; }
    50%      { text-shadow: 0 0 8px #e05c2a; }
}
#samridhi-mic-status {
    position: fixed;
    bottom: 65px;
    left: 50%;
    transform: translateX(-50%);
    z-index: 99999;
    background: #1a1a1a;
    color: #e05c2a;
    font-size: 12px;
    padding: 5px 12px;
    border-radius: 10px;
    display: none;
    max-width: 230px;
    text-align: center;
    border: 1px solid rgba(224,92,42,0.3);
    pointer-events: none;
}
</style>
<div id="samridhi-mic-status"></div>
""", unsafe_allow_html=True)

import streamlit.components.v1 as components
components.html("""
<script>
var _sr_active = false;
var _sr_obj    = null;
var _sr_lang   = 'en-IN';

var P = window.parent.document;

function getMicBtn()   { return P.getElementById('samridhi-mic'); }
function getStatusEl() { return P.getElementById('samridhi-mic-status'); }

function injectMicIntoInputBar() {
    if (P.getElementById('samridhi-mic')) return;
    var inputWrapper = P.querySelector('[data-testid="stChatInput"]')
                    || P.querySelector('.stChatInput');
    if (!inputWrapper) return;

    var btn = P.createElement('button');
    btn.id = 'samridhi-mic';
    btn.title = 'Click to speak your question';
    btn.innerHTML = '🎤';
    btn.style.cssText = [
        'background:none','border:none','border-radius:50%',
        'width:36px','height:36px','font-size:20px',
        'color:#888','cursor:pointer',
        'display:flex','align-items:center','justify-content:center',
        'flex-shrink:0','padding:0','position:absolute',
        'right:52px','bottom:50%','transform:translateY(50%)',
        'z-index:9999','transition:color 0.2s, transform 0.15s'
    ].join(';');

    inputWrapper.style.position = 'relative';
    inputWrapper.appendChild(btn);

    btn.onmouseenter = function() { if (!_sr_active) { btn.style.color='#e05c2a'; btn.style.transform='translateY(50%) scale(1.2)'; } };
    btn.onmouseleave = function() { if (!_sr_active) { btn.style.color='#888'; btn.style.transform='translateY(50%)'; } };
    btn.onclick = function() {
        if (_sr_active) { _sr_obj && _sr_obj.stop(); }
        else { startMic(); }
    };
}

// Initial injection
injectMicIntoInputBar();

// Re-inject after every Streamlit rerun (DOM mutation)
new MutationObserver(function() {
    injectMicIntoInputBar();
}).observe(P.body, { childList: true, subtree: true });

function showStatus(msg) {
    var el = getStatusEl();
    if (el) { el.innerText = msg; el.style.display = 'block'; }
}
function hideStatus() {
    var el = getStatusEl();
    if (el) el.style.display = 'none';
}
function resetBtn() {
    var btn = getMicBtn();
    if (btn) { btn.classList.remove('listening'); btn.innerText = '🎤'; btn.style.color='#888'; btn.style.transform='translateY(50%)'; }
}

function injectAndSubmit(text) {
    var ta = P.querySelector('textarea[data-testid="stChatInputTextArea"]')
          || P.querySelector('.stChatInput textarea')
          || P.querySelector('textarea');
    if (!ta) { showStatus('Input not found — type: ' + text); return; }

    var proto = Object.getOwnPropertyDescriptor(window.parent.HTMLTextAreaElement.prototype, 'value');
    if (proto && proto.set) proto.set.call(ta, text);
    else ta.value = text;

    ta.dispatchEvent(new window.parent.Event('input',  {bubbles:true}));
    ta.dispatchEvent(new window.parent.Event('change', {bubbles:true}));
    ta.focus();

    setTimeout(function() {
        var btn = P.querySelector('button[data-testid="stChatInputSubmitButton"]')
               || P.querySelector('.stChatInput button[type="submit"]')
               || P.querySelector('button[aria-label="Send message"]');
        if (btn) {
            btn.click();
        } else {
            ['keydown','keypress','keyup'].forEach(function(t) {
                ta.dispatchEvent(new window.parent.KeyboardEvent(t, {
                    key:'Enter', code:'Enter', keyCode:13, which:13, bubbles:true, cancelable:true
                }));
            });
        }
        showStatus('✅ ' + text);
        setTimeout(hideStatus, 2500);
    }, 450);
}

function startMic() {
    var SR = window.parent.SpeechRecognition || window.parent.webkitSpeechRecognition
          || window.SpeechRecognition        || window.webkitSpeechRecognition;
    if (!SR) { showStatus('⚠ Use Chrome or Edge'); return; }

    _sr_obj = new SR();
    _sr_obj.lang            = _sr_lang;
    _sr_obj.continuous      = false;
    _sr_obj.interimResults  = true;
    _sr_obj.maxAlternatives = 1;

    _sr_obj.onstart = function() {
        _sr_active = true;
        var btn = getMicBtn();
        if (btn) { btn.classList.add('listening'); btn.innerText = '🔴'; btn.style.color='#e05c2a'; }
        showStatus('🎤 Listening...');
    };
    _sr_obj.onresult = function(e) {
        var interim='', final_t='';
        for (var i=e.resultIndex;i<e.results.length;i++) {
            if (e.results[i].isFinal) final_t += e.results[i][0].transcript;
            else interim += e.results[i][0].transcript;
        }
        showStatus(interim || final_t);
        if (final_t) injectAndSubmit(final_t.trim());
    };
    _sr_obj.onerror = function(e) {
        _sr_active=false; resetBtn();
        if (e.error!=='no-speech') showStatus('Error: '+e.error);
        else hideStatus();
    };
    _sr_obj.onend = function() {
        _sr_active=false; resetBtn();
        setTimeout(hideStatus, 1800);
    };
    _sr_obj.start();
}

window.setMicLang = function(l) { _sr_lang = l; };
window.parent.setMicLang = window.setMicLang;
</script>
""", height=0)


# ==============================
# SESSION STATE
# ==============================

if "feedback_cache" not in st.session_state:
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            st.session_state.feedback_cache = json.load(f)
    else:
        st.session_state.feedback_cache = {}

if "lang" not in st.session_state:
    st.session_state.lang = "en"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_feedback" not in st.session_state:
    st.session_state.pending_feedback = {}


# ==============================
# LANGUAGE TOGGLE
# ==============================

UI = {
    "en": {
        "title":       "Samridhi – M-CADWM",
        "subtitle":    "AI Assistant — M-CADWM & SMIS",
        "welcome":     "This assistant is designated for queries relating to the M-CADWM scheme, its official documents, website, and the SMIS portal. Please state your query.",
        "placeholder": "Enter your query or click 🎤 to speak...",
        "spinner":     "Searching M-CADWM documents...",
        "no_result":   "No relevant information was found in the M-CADWM documents. Please refine your query.",
        "cached_note": "*(Retrieved from feedback cache)*\n\n",
        "fb_up":       "Feedback recorded.",
        "fb_dn":       "Feedback recorded. Response will be reviewed.",
        "btn_en":      "🇬🇧 English",
        "btn_hi":      "🇮🇳 हिंदी",
    },
    "hi": {
        "title":       "समृद्धि – M-CADWM",
        "subtitle":    "AI सहायक — M-CADWM और SMIS",
        "welcome":     "यह सहायक M-CADWM योजना, इसके आधिकारिक दस्तावेज़ों, वेबसाइट और SMIS पोर्टल से संबंधित प्रश्नों के लिए निर्धारित है। कृपया अपना प्रश्न दर्ज करें।",
        "placeholder": "प्रश्न दर्ज करें या 🎤 दबाकर बोलें...",
        "spinner":     "M-CADWM दस्तावेज़ खोज रहे हैं...",
        "no_result":   "M-CADWM दस्तावेज़ों में कोई प्रासंगिक जानकारी नहीं मिली। कृपया अपना प्रश्न पुनः दर्ज करें।",
        "cached_note": "*(फीडबैक कैश से प्राप्त)*\n\n",
        "fb_up":       "फीडबैक दर्ज किया गया।",
        "fb_dn":       "फीडबैक दर्ज किया गया। उत्तर की समीक्षा की जाएगी।",
        "btn_en":      "🇬🇧 English",
        "btn_hi":      "🇮🇳 हिंदी",
    }
}

lang = st.session_state.lang
ui   = UI[lang]

lcol1, lcol2, lcol3 = st.columns([5, 1, 1])
with lcol2:
    en_type = "primary" if lang == "en" else "secondary"
    if st.button("🇬🇧 EN", type=en_type, use_container_width=True):
        st.session_state.lang = "en"
        st.session_state.messages = [{"role":"assistant","content": UI["en"]["welcome"]}]
        st.session_state.pending_feedback = {}
        st.rerun()
with lcol3:
    hi_type = "primary" if lang == "hi" else "secondary"
    if st.button("🇮🇳 HI", type=hi_type, use_container_width=True):
        st.session_state.lang = "hi"
        st.session_state.messages = [{"role":"assistant","content": UI["hi"]["welcome"]}]
        st.session_state.pending_feedback = {}
        st.rerun()

speech_lang = "hi-IN"             if lang == "hi" else "en-IN"
tts_voice   = "hi-IN-SwaraNeural" if lang == "hi" else "en-IN-NeerjaNeural"
tts_rate    = "+5%"               if lang == "hi" else "+10%"


# ==============================
# HEADER
# ==============================

header_logo = ""
if os.path.exists(LOGO_PATH):
    with open(LOGO_PATH, "rb") as f:
        header_logo = f'<img src="data:image/png;base64,{base64.b64encode(f.read()).decode()}" width="110"><br>'

st.markdown(f"""
<div style="text-align:center; margin-top:-10px;">
    {header_logo}
    <h1 style="margin-bottom:4px;">{ui["title"]}</h1>
    <p style="color:#aaa; margin-top:0;">{ui["subtitle"]}</p>
</div>
<hr style="margin:10px 0 18px 0;">
""", unsafe_allow_html=True)


# ==============================
# LOAD VECTOR DB
# ==============================

@st.cache_resource
def load_vector_db():
    emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder=os.path.join(BASE_DIR, "models")
    )
    return FAISS.load_local(FAISS_PATH, emb, allow_dangerous_deserialization=True)

vector_db = load_vector_db()


# ==============================
# LIVE WEB FETCH (fallback)
# ==============================

CADWM_BASE = "http://cadwm.gov.in/"
WEB_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; SamridhiBot/1.0)"}

def fetch_live_context(question, max_pages=3):
    """Search cadwm.gov.in live for the question keywords."""
    try:
        # Search Google for site-specific results
        query = f"site:cadwm.gov.in {question}"
        search_url = f"https://www.google.com/search?q={requests.utils.quote(query)}&num=5"
        resp = requests.get(search_url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }, timeout=8)
        soup = BeautifulSoup(resp.text, "html.parser")

        # Extract result URLs from Google
        urls = []
        for a in soup.select("a[href]"):
            href = a["href"]
            if "/url?q=http://cadwm.gov.in" in href:
                actual = href.split("/url?q=")[1].split("&")[0]
                if actual not in urls:
                    urls.append(actual)
            if len(urls) >= max_pages:
                break

        # Fallback: just fetch homepage if no results
        if not urls:
            urls = [CADWM_BASE]

        # Scrape each URL
        texts = []
        for url in urls[:max_pages]:
            try:
                r = requests.get(url, headers=WEB_HEADERS, timeout=8)
                s = BeautifulSoup(r.text, "html.parser")
                for tag in s(["script","style","nav","footer","header","aside"]):
                    tag.decompose()
                text = s.get_text(separator=" ", strip=True)
                text = re.sub(r'\s{2,}', ' ', text)
                if len(text) > 200:
                    texts.append(f"[Source: {url}]\n{text[:3000]}")
            except Exception:
                pass

        return "\n\n---\n\n".join(texts) if texts else None

    except Exception as e:
        return None


# ==============================
# HYBRID RETRIEVAL
# ==============================

def keyword_score(text, words):
    tl = text.lower()
    return sum(1 for w in words if w in tl) / max(len(words), 1)

def retrieve_docs(question, k=10, max_dist=1.4):
    candidates = vector_db.similarity_search_with_score(question, k=k * 2)
    stops = {"a","an","the","is","are","in","on","at","to","of","and","or",
             "for","what","tell","me","how","do","can","please","give",
             "kya","hai","mujhe","batao","ke","ka","ki","aur","se"}
    words = [w for w in re.findall(r'\w+', question.lower())
             if w not in stops and len(w) > 2]
    scored = []
    for doc, dist in candidates:
        if dist >= max_dist: continue
        sem = 1 / (1 + dist)
        kw  = keyword_score(doc.page_content, words)
        scored.append((doc, 0.6 * sem + 0.4 * kw))
    scored.sort(key=lambda x: x[1], reverse=True)
    best_score = scored[0][1] if scored else 0.0
    return [d for d, _ in scored[:k]], best_score


# ==============================
# LLM
# ==============================

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0.15
)


# ==============================
# FEEDBACK HELPERS
# ==============================

def save_cache():
    with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state.feedback_cache, f, indent=2, ensure_ascii=False)

def get_cached(q):
    h = hashlib.md5(q.lower().strip().encode()).hexdigest()
    e = st.session_state.feedback_cache.get(h)
    if e and e.get("thumbs_up", 0) > e.get("thumbs_down", 0):
        return e.get("answer")
    return None

def record_feedback(q, ans, vote):
    h = hashlib.md5(q.lower().strip().encode()).hexdigest()
    c = st.session_state.feedback_cache
    if h not in c:
        c[h] = {"question":q,"answer":ans,"thumbs_up":0,"thumbs_down":0,"last_updated":""}
    c[h]["answer"] = ans
    c[h]["thumbs_up" if vote=="up" else "thumbs_down"] += 1
    c[h]["last_updated"] = datetime.now().isoformat()
    save_cache()


# ==============================
# GREETING
# ==============================

GREETINGS = [
    "hi","hello","hey","good morning","good evening","good afternoon",
    "namaste","namaskar","नमस्ते","हेलो","हाय","howdy","greetings",
    "helo","hii","helo","hiya","sup","wassup"
]

def is_greeting(t):
    t = t.lower().strip()
    return any(t.startswith(g) for g in GREETINGS) and len(t.split()) <= 5


# ==============================
# TYPO CORRECTION
# ==============================

def correct_typos(question):
    """Use LLM to lightly correct typos before searching."""
    try:
        prompt = (
            "Fix any spelling mistakes or typos in the following query. "
            "Return ONLY the corrected query, nothing else. "
            "Do not change meaning or add words.\n\n"
            f"Query: {question}"
        )
        corrected = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        # Safety: if LLM returns something wildly different, use original
        if len(corrected) > len(question) * 2 or len(corrected) < 2:
            return question
        return corrected
    except Exception:
        return question


# ==============================
# GENERAL KNOWLEDGE FALLBACK
# ==============================

RELATED_TOPICS = [
    "water","irrigation","agriculture","farming","crop","dam","canal",
    "river","scheme","government","ministry","portal","smis","cadwm",
    "mcadwm","pmksy","kisan","farmer","soil","drainage","flood",
    "watershed","groundwater","silt","wus","imti","fund","subsidy",
    "registration","application","form","document","certificate",
    "sarkar","yojana","विभाग","सिंचाई","पानी","कृषि","किसान","योजना"
]

def is_related_topic(question):
    q = question.lower()
    return any(kw in q for kw in RELATED_TOPICS)

def general_fallback(question, corrected_q):
    """Answer from general knowledge, clearly labeled."""
    if lang == "hi":
        prompt = f"""आप समृद्धि हैं — M-CADWM योजना, इसके आधिकारिक दस्तावेज़ों, वेबसाइट और SMIS पोर्टल के लिए एक पेशेवर AI सहायक।

उपयोगकर्ता का प्रश्न: "{corrected_q}"

M-CADWM दस्तावेज़ों और वेबसाइट में इस विषय पर सीधी जानकारी उपलब्ध नहीं है।

निर्देश:
- स्पष्ट रूप से सूचित करें कि यह विषय M-CADWM दस्तावेज़ों या वेबसाइट में उपलब्ध नहीं है।
- यदि प्रश्न M-CADWM या SMIS के व्यापक दायरे से संबंधित है, तो सामान्य ज्ञान से संक्षिप्त जानकारी प्रदान करें।
- यदि प्रश्न M-CADWM या SMIS के दायरे से बाहर है, तो विनम्रता से उपयोगकर्ता को M-CADWM या SMIS से संबंधित प्रश्न पूछने के लिए निर्देशित करें।
- कोई emoji उपयोग न करें — उत्तर पूर्णतः पेशेवर एवं औपचारिक होना चाहिए।
- व्यक्तिगत या आत्मीय भाषा का उपयोग न करें।
- प्रश्न की श्रेणी का उल्लेख न करें।

उत्तर:"""
    else:
        scope_note = (
            "If the query falls within the broader domain of M-CADWM or SMIS, "
            "provide a brief and factual response from general knowledge. "
            "Clearly note that this information is not sourced from M-CADWM official documents."
            if is_related_topic(corrected_q) else
            "Inform the user that this query falls outside the scope of M-CADWM and SMIS, "
            "and direct them to submit queries relevant to M-CADWM or SMIS."
        )
        prompt = f"""You are Samridhi — a professional AI assistant designated for the M-CADWM scheme, its official documents, website, portal, and the SMIS portal.

Query received: "{corrected_q}"

This query is not covered in the M-CADWM official documents or website.

Instructions:
- Clearly state that this information is not available in the M-CADWM official documents or website.
- {scope_note}
- Do NOT use any emojis — maintain a formal, government-appropriate tone throughout.
- Do NOT use informal, warm, or personable language.
- Do NOT begin with conversational openers.
- Do NOT mention query categories to the user.
- Close by directing the user to submit queries specific to M-CADWM or SMIS.

Response:"""

    return llm.invoke([HumanMessage(content=prompt)]).content


# ==============================
# RAG
# ==============================

def ask_rag(question):

    # ── Greeting ──────────────────────────────────────────────────
    if is_greeting(question):
        if lang == "hi":
            return (
                "यह सहायक M-CADWM योजना, इसके आधिकारिक दस्तावेज़ों, वेबसाइट और SMIS पोर्टल से संबंधित प्रश्नों के लिए निर्धारित है। कृपया अपना प्रश्न दर्ज करें।"
            )
        return (
            "This assistant is designated for queries relating to the M-CADWM scheme, its official documents, website, and the SMIS portal. Please state your query."
        )

    # ── Feedback cache ────────────────────────────────────────────
    cached = get_cached(question)
    if cached:
        return ui["cached_note"] + cached

    # ── Typo correction ───────────────────────────────────────────
    corrected_q = correct_typos(question)
    if corrected_q.lower() != question.lower():
        typo_note = f"*Note: A typographical correction was applied. Response addresses: \"{corrected_q}\"*\n\n"
    else:
        typo_note = ""

    search_q = corrected_q

    # ── Layer 1: FAISS ────────────────────────────────────────────
    docs, confidence = retrieve_docs(search_q)
    source_note = ""

    if confidence >= 0.35 and docs:
        context = "\n\n---\n\n".join(d.page_content for d in docs)

        if lang == "hi":
            prompt = f"""आप समृद्धि हैं — M-CADWM योजना, इसके आधिकारिक दस्तावेज़ों, वेबसाइट और SMIS पोर्टल के लिए एक पेशेवर AI सहायक।

निर्देश:
- M-CADWM दस्तावेज़ों, वेबसाइट और SMIS पोर्टल से संबंधित जानकारी को सर्वोच्च प्राथमिकता दें।
- नीचे दिए गए संदर्भ के आधार पर पूर्ण, विस्तृत और सुव्यवस्थित उत्तर हिंदी में प्रदान करें।
- शीर्षक, बुलेट पॉइंट और क्रमांकित सूचियाँ उपयोग करें।
- कोई emoji उपयोग न करें — उत्तर पूर्णतः पेशेवर एवं औपचारिक होना चाहिए।
- व्यक्तिगत या आत्मीय भाषा का उपयोग न करें।
- यदि जानकारी संदर्भ में उपलब्ध नहीं है, तो विनम्रता से सूचित करें।

संदर्भ (M-CADWM आधिकारिक दस्तावेज़):
{context}

प्रश्न: {search_q}

उत्तर:"""
        else:
            prompt = f"""You are Samridhi — a professional AI assistant designated for the M-CADWM scheme, its official documents, website, portal, and the SMIS portal.

Instructions:
- Prioritise information from M-CADWM official documents, website, and SMIS portal above all else.
- Provide a COMPLETE, DETAILED, and WELL-STRUCTURED response based strictly on the context below.
- Use ## headings, bullet points, and numbered lists where appropriate.
- Do NOT use any emojis.
- Do NOT use informal, warm, or personable language — maintain a formal, government-appropriate tone throughout.
- Do NOT begin with conversational openers or acknowledgements.
- Do NOT invent information not present in the context.
- If the context does not contain sufficient information, state this clearly and concisely.

Context (M-CADWM official documents):
{context}

Query: {search_q}

Response:"""

        answer = llm.invoke([HumanMessage(content=prompt)]).content
        return typo_note + answer

    # ── Layer 2: Live cadwm.gov.in ────────────────────────────────
    with st.spinner("Checking cadwm.gov.in for the latest information..."):
        live_context = fetch_live_context(search_q)

    if live_context:
        source_note = "\n\n---\n*Source: cadwm.gov.in (live)*"
        if lang == "hi":
            prompt = f"""आप समृद्धि हैं — M-CADWM योजना, इसके आधिकारिक दस्तावेज़ों, वेबसाइट और SMIS पोर्टल के लिए एक पेशेवर AI सहायक।

निर्देश:
- M-CADWM वेबसाइट और SMIS पोर्टल की जानकारी को सर्वोच्च प्राथमिकता दें।
- नीचे दिए गए संदर्भ के आधार पर विस्तृत और पेशेवर उत्तर हिंदी में प्रदान करें।
- कोई emoji उपयोग न करें। व्यक्तिगत या आत्मीय भाषा का उपयोग न करें।

संदर्भ (cadwm.gov.in से):
{live_context}

प्रश्न: {search_q}

उत्तर:"""
        else:
            prompt = f"""You are Samridhi — a professional AI assistant designated for the M-CADWM scheme, its official documents, website, portal, and the SMIS portal.

Instructions:
- Prioritise M-CADWM website and SMIS portal information above all else.
- Provide a professional, detailed, and well-structured response based strictly on the context below.
- Do NOT use any emojis. Do NOT use informal or personable language.
- Maintain a formal, government-appropriate tone throughout.

Context (from cadwm.gov.in):
{live_context}

Query: {search_q}

Response:"""

        answer = llm.invoke([HumanMessage(content=prompt)]).content
        return typo_note + answer + source_note

    # ── Layer 3: General knowledge fallback ───────────────────────
    return typo_note + general_fallback(question, corrected_q)


# ==============================
# TTS
# ==============================

def clean_tts(text):
    text = re.sub(r"\*\(.*?\)\*\n\n", "", text)
    text = re.sub(r"#{1,6}\s*", "", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*",   r"\1", text)
    text = re.sub(r"`(.*?)`",     r"\1", text)
    text = re.sub(r"^\s*[-•]\s*", ", ", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s*",", ", text, flags=re.MULTILINE)
    if lang == "en":
        for a, b in {"MCADWM":"M-CADWM","SMIS":"S-MIS","O&M":"Operations and Maintenance",
                     "&":"and","%":"percent","i.e.":"that is","e.g.":"for example",
                     "etc.":"and so on","Govt":"Government","Rs.":"Rupees"}.items():
            text = text.replace(a, b)
    text = text.replace(":"," , ").replace(";"," , ")
    text = re.sub(r"\n{2,}", ". ", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\.{2,}", ".", text)
    return text.strip()

async def _gen_audio(text, voice, rate):
    await edge_tts.Communicate(text=text, voice=voice, rate=rate).save(
        os.path.join(BASE_DIR, "response.mp3")
    )

def speak(text):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(_gen_audio(clean_tts(text), tts_voice, tts_rate))
    loop.close()

def autoplay_audio():
    with open(os.path.join(BASE_DIR, "response.mp3"), "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    st.markdown(f"""
    <audio autoplay controls style="width:100%;margin-top:8px;border-radius:8px;">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>""", unsafe_allow_html=True)


# ==============================
# INIT WELCOME MESSAGE
# ==============================

if not st.session_state.messages:
    st.session_state.messages = [
        {"role": "assistant", "content": ui["welcome"]}
    ]


# ==============================
# RENDER CHAT HISTORY
# ==============================

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and i in st.session_state.pending_feedback:
            pf = st.session_state.pending_feedback[i]
            rk = f"rated_{i}"
            if rk not in st.session_state:
                c1, c2, _ = st.columns([1,1,8])
                with c1:
                    if st.button("👍", key=f"up_{i}"):
                        record_feedback(pf["q"], pf["a"], "up")
                        st.session_state[rk] = "up"; st.rerun()
                with c2:
                    if st.button("👎", key=f"dn_{i}"):
                        record_feedback(pf["q"], pf["a"], "down")
                        st.session_state[rk] = "down"; st.rerun()
            else:
                st.caption(ui["fb_up"] if st.session_state[rk]=="up" else ui["fb_dn"])


# ==============================
# USER INPUT
# ==============================

if question := st.chat_input(ui["placeholder"]):
    st.session_state.messages.append({"role":"user","content":question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner(ui["spinner"]):
            answer = ask_rag(question)
        st.markdown(answer)

        idx = len(st.session_state.messages)
        st.session_state.pending_feedback[idx] = {"q": question, "a": answer}

        c1, c2, _ = st.columns([1,1,8])
        rk = f"rated_{idx}"
        with c1:
            if st.button("👍", key=f"up_{idx}"):
                record_feedback(question, answer, "up")
                st.session_state[rk] = "up"; st.rerun()
        with c2:
            if st.button("👎", key=f"dn_{idx}"):
                record_feedback(question, answer, "down")
                st.session_state[rk] = "down"; st.rerun()

        st.session_state.messages.append({"role":"assistant","content":answer})
        speak(answer)
        autoplay_audio()
