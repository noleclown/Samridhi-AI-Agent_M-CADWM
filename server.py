"""
server.py — Samridhi AI  FastAPI Server
========================================
Production-ready API server. Replaces app.py (Streamlit).

Endpoints:
    POST /api/chat          → full JSON response
    POST /api/stream        → Server-Sent Events (streaming tokens)
    GET  /api/health        → system status
    GET  /api/session       → create new session
    DELETE /api/session/{id}→ clear session
    GET  /                  → full-page chat UI (static/index.html)
    GET  /widget.js         → embeddable floating widget script
    GET  /static/*          → static assets

Run locally:
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload

Run in production (Docker):
    uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1

Note: Use --workers 1 (single process) since SessionManager is in-memory.
For multi-process scale, swap SessionManager for Redis-backed sessions.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse, HTMLResponse, JSONResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv()

# ── NumPy-safe JSON encoder ───────────────────────────────────
# FAISS returns numpy float32/int64 values. Standard json.dumps
# can't handle them. This encoder converts them to native Python types.
import json as _json

class _SafeEncoder(_json.JSONEncoder):
    def default(self, obj):
        try:
            import numpy as np
            if isinstance(obj, (np.floating,)):  return float(obj)
            if isinstance(obj, (np.integer,)):   return int(obj)
            if isinstance(obj, np.ndarray):      return obj.tolist()
        except ImportError:
            pass
        return super().default(obj)

def _dumps(obj) -> str:
    return _json.dumps(obj, cls=_SafeEncoder)

# ── Samridhi package ──────────────────────────────────────────
from samridhi.api import ChatResponse, SessionManager, handle_chat
from samridhi.cache import AnalyticsLog, ExpansionStore, FeedbackCache, WebCache
from samridhi.config import BASE_DIR, cfg
from samridhi.logger import get_logger
from samridhi.pipeline import Pipeline
from samridhi.ui_strings import UI

log = get_logger()

BASE_PATH   = Path(__file__).parent
STATIC_PATH = BASE_PATH / "static"
STATIC_PATH.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════
# RESOURCE INITIALISATION  (once at startup)
# ══════════════════════════════════════════════════════════════

_resources: dict = {}

def _load_resources():
    """Load all heavy resources once at startup."""
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise RuntimeError("GROQ_API_KEY environment variable not set.")

    from langchain_community.vectorstores import FAISS
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings

    log.info("Loading embedding model...")
    emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder=str(BASE_DIR / "models"),
    )

    log.info("Loading FAISS index...")
    vector_db = FAISS.load_local(
        str(BASE_DIR / "faiss_index"), emb,
        allow_dangerous_deserialization=True,
    )

    log.info("Initialising LLM client...")
    llm = ChatGroq(
        groq_api_key=groq_key,
        model_name=cfg["llm"]["model"],
        temperature=cfg["llm"]["temperature"],
        max_tokens=cfg["llm"]["max_tokens"],
    )

    feedback_db = FeedbackCache()
    web_cache   = WebCache()
    analytics   = AnalyticsLog()
    expansions  = ExpansionStore()
    sessions    = SessionManager()

    pipeline = Pipeline(
        llm, vector_db, feedback_db, web_cache, analytics,
        expansion_store=expansions,
    )

    _resources.update({
        "pipeline":    pipeline,
        "feedback_db": feedback_db,
        "web_cache":   web_cache,
        "analytics":   analytics,
        "expansions":  expansions,
        "sessions":    sessions,
        "vector_db":   vector_db,
    })
    log.info("All resources loaded — server ready.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_resources()
    yield
    log.info("Server shutting down.")


# ══════════════════════════════════════════════════════════════
# FASTAPI APP
# ══════════════════════════════════════════════════════════════

app = FastAPI(
    title       = "Samridhi AI API",
    description = "AI Assistant for M-CADWM & SMIS — Ministry of Jal Shakti",
    version     = "1.0.0",
    lifespan    = lifespan,
)

# CORS — allow cadwm.gov.in and any origin during development
_allowed_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:8000,"
    "https://mcad.one1sewa.com,https://www.mcad.one1sewa.com,"
    "https://cadwm.gov.in,https://www.cadwm.gov.in"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins       = ["*"],   # open during development — restrict before production
    allow_credentials   = False,   # must be False when allow_origins=["*"]
    allow_methods       = ["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers       = ["*"],
)

# Serve static files (index.html, widget.js, assets)
if STATIC_PATH.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_PATH)), name="static")


# ══════════════════════════════════════════════════════════════
# REQUEST / RESPONSE SCHEMAS
# ══════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    question:   str          = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = Field(None, description="Omit to start a new session")
    lang:       str          = Field("en", pattern="^(en|hi)$")

class SessionResponse(BaseModel):
    session_id: str
    lang:       str
    created_at: float


# ══════════════════════════════════════════════════════════════
# API ENDPOINTS
# ══════════════════════════════════════════════════════════════

@app.post("/api/chat", response_class=JSONResponse)
async def chat(req: ChatRequest):
    """
    Send a message and receive the full response.
    Returns JSON with answer, source layer, follow-up suggestions.

    Example:
        POST /api/chat
        {"question": "What is M-CADWM?", "lang": "en"}

    Response:
        {"session_id": "...", "answer": "...", "layer": "faiss",
         "follow_ups": [...], "confidence": 0.72, ...}
    """
    sessions: SessionManager = _resources["sessions"]
    pipeline: Pipeline       = _resources["pipeline"]

    session = sessions.get_or_create(req.session_id, req.lang)
    session.lang = req.lang   # allow lang switch mid-session

    try:
        response: ChatResponse = handle_chat(req.question, session, pipeline)
        return JSONResponse(content=response.to_dict())
    except Exception as e:
        log.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stream")
async def stream(req: ChatRequest):
    """
    Send a message and receive a Server-Sent Events stream.
    Each event is a JSON chunk:

        data: {"type": "token", "content": "The "}
        data: {"type": "token", "content": "M-CADWM "}
        data: {"type": "done",  "session_id": "...", "layer": "faiss",
               "follow_ups": [...], "confidence": 0.72}

    The frontend appends tokens as they arrive for a streaming effect.
    Note: The pipeline runs synchronously; we stream the final answer
    word-by-word to simulate streaming until true LLM streaming is added.
    """
    sessions: SessionManager = _resources["sessions"]
    pipeline: Pipeline       = _resources["pipeline"]

    session = sessions.get_or_create(req.session_id, req.lang)
    session.lang = req.lang

    async def event_generator():
        try:
            response: ChatResponse = handle_chat(req.question, session, pipeline)
            # Stream answer word by word
            words = response.answer.split(" ")
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words) - 1 else "")
                data  = _dumps({"type": "token", "content": chunk})
                yield f"data: {data}\n\n"
                import asyncio
                await asyncio.sleep(0.012)
            # Final done event with metadata — cast numpy floats to Python float
            done_data = _dumps({
                "type":        "done",
                "session_id":  response.session_id,
                "layer":       response.layer,
                "follow_ups":  response.follow_ups,
                "confidence":  float(response.confidence),
                "intent":      response.intent,
                "response_ms": float(response.response_ms),
            })
            yield f"data: {done_data}\n\n"
        except Exception as e:
            log.error(f"Stream error: {e}")
            err = _dumps({"type": "error", "message": str(e)})
            yield f"data: {err}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.get("/api/session", response_model=SessionResponse)
async def create_session(lang: str = "en"):
    """Create a new chat session. Returns session_id to use in subsequent requests."""
    sessions: SessionManager = _resources["sessions"]
    session = sessions.create(lang)
    return SessionResponse(
        session_id = session.session_id,
        lang       = session.lang,
        created_at = session.created_at,
    )


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """Clear a session (start fresh conversation)."""
    sessions: SessionManager = _resources["sessions"]
    sessions.delete(session_id)
    return {"status": "deleted", "session_id": session_id}


@app.post("/api/tts")
async def tts(request: Request):
    """
    Generate neural TTS audio using Microsoft Edge TTS.
    Voices: en-IN-NeerjaNeural (EN) · hi-IN-SwaraNeural (HI)
    Returns MP3 audio bytes.
    """
    import tempfile
    import edge_tts
    from pathlib import Path as _Path

    body = await request.json()
    text = str(body.get("text", "")).strip()
    lang = str(body.get("lang", "en")).strip()

    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    if lang not in ("en", "hi"):
        lang = "en"

    from samridhi.tts import clean_tts
    voice = cfg["tts"]["hi_voice"] if lang == "hi" else cfg["tts"]["en_voice"]
    rate  = cfg["tts"]["hi_rate"]  if lang == "hi" else cfg["tts"]["en_rate"]

    try:
        cleaned = clean_tts(text, lang)
        tmp     = tempfile.mktemp(suffix=".mp3", dir=str(BASE_DIR))

        # Use await directly — we are already inside an async FastAPI handler.
        # Never create a new event loop inside an async function.
        communicate = edge_tts.Communicate(text=cleaned, voice=voice, rate=rate)
        await communicate.save(tmp)

        p = _Path(tmp)
        if not p.exists() or p.stat().st_size == 0:
            raise HTTPException(status_code=500, detail="TTS file empty or missing")

        audio_bytes = p.read_bytes()
        p.unlink(missing_ok=True)

        from fastapi.responses import Response
        return Response(
            content    = audio_bytes,
            media_type = "audio/mpeg",
            headers    = {"Cache-Control": "no-store"},
        )
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"TTS endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"TTS failed: {e}")


@app.get("/api/health")
async def health():
    """System health check — used by load balancers and monitoring."""
    sessions: SessionManager = _resources.get("sessions")
    web_cache: WebCache      = _resources.get("web_cache")
    feedback_db: FeedbackCache = _resources.get("feedback_db")
    expansions: ExpansionStore = _resources.get("expansions")

    faiss_ok = (BASE_DIR / "faiss_index").exists()

    return {
        "status":      "ok" if faiss_ok else "degraded",
        "version":     "1.0.0",
        "faiss":       faiss_ok,
        "sessions":    sessions.stats() if sessions else {},
        "web_cache":   web_cache.stats() if web_cache else {},
        "feedback":    feedback_db.stats() if feedback_db else {},
        "expansions":  expansions.stats() if expansions else {},
        "uptime_s":    round(time.time() - _start_time, 1),
    }


@app.get("/api/starters")
async def starters(lang: str = "en"):
    """Return the list of starter questions for the given language."""
    key = "starter_questions_hi" if lang == "hi" else "starter_questions_en"
    return {"lang": lang, "starters": cfg["ui"].get(key, [])}


# ══════════════════════════════════════════════════════════════
# STATIC PAGES
# ══════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def index():
    """Full-page chat UI."""
    html_file = STATIC_PATH / "index.html"
    if not html_file.exists():
        return HTMLResponse("<h2>static/index.html not found</h2>", status_code=404)
    return HTMLResponse(html_file.read_text(encoding="utf-8"))


@app.get("/embed", response_class=HTMLResponse)
async def embed():
    """
    Embeddable full agent — no sidebar, iframe-friendly.
    Use this URL in <iframe src="..."> or open as a dedicated page.
    """
    html_file = STATIC_PATH / "embed.html"
    if not html_file.exists():
        return HTMLResponse("<h2>static/embed.html not found</h2>", status_code=404)
    return HTMLResponse(
        html_file.read_text(encoding="utf-8"),
        headers={"X-Frame-Options": "ALLOWALL"},  # allow iframe embedding
    )


@app.get("/widget.js")
async def widget_js():
    """Embeddable floating chat widget script."""
    js_file = STATIC_PATH / "widget.js"
    if not js_file.exists():
        return HTMLResponse("// widget.js not found", status_code=404,
                            media_type="application/javascript")
    return FileResponse(str(js_file), media_type="application/javascript")


# ── startup timestamp ─────────────────────────────────────────
_start_time = time.time()
