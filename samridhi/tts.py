"""
samridhi/tts.py
===============
Text-to-speech generation via edge-tts.

Exports:
    clean_tts(text, lang)         → str   (markdown stripped, abbrs expanded)
    speak(text, lang, cfg)        → str   (path to temp .mp3 file, "" on failure)
    autoplay_audio(path, st_mod)         (renders audio player in Streamlit)

Per-session temp files:  avoids concurrent-user MP3 collision.
File is deleted after being served to the browser.
"""

from __future__ import annotations

import asyncio
import re
import tempfile
from pathlib import Path

from samridhi.config import BASE_DIR, cfg
from samridhi.logger import get_logger

log = get_logger()


# ══════════════════════════════════════════════════════════════
# TTS TEXT CLEANER
# ══════════════════════════════════════════════════════════════

_TTS_ABBR: dict[str, str] = {
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


def clean_tts(text: str, lang: str) -> str:
    """
    Strip markdown and expand abbreviations for natural TTS output.
    EN only for abbreviation expansion (Hindi TTS handles its own phonetics).
    """
    text = re.sub(r"\*\(.*?\)\*\n\n", "", text)
    text = re.sub(r"#{1,6}\s*",        "", text)
    text = re.sub(r"\*\*(.*?)\*\*",   r"\1", text)
    text = re.sub(r"\*(.*?)\*",        r"\1", text)
    text = re.sub(r"`(.*?)`",          r"\1", text)
    text = re.sub(r"^\s*[-•]\s*",     ", ",  text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s*",    ", ",  text, flags=re.MULTILINE)
    if lang == "en":
        for abbr, expansion in _TTS_ABBR.items():
            text = text.replace(abbr, expansion)
    text = text.replace(":", " , ").replace(";", " , ")
    text = re.sub(r"\n{2,}", ". ", text)
    text = re.sub(r"\n",     " ",  text)
    text = re.sub(r"\s{2,}", " ",  text)
    text = re.sub(r"\.{2,}", ".",  text)
    return text.strip()


# ══════════════════════════════════════════════════════════════
# TTS GENERATION
# ══════════════════════════════════════════════════════════════

async def _generate(text: str, voice: str, rate: str, path: str):
    import edge_tts
    await edge_tts.Communicate(text=text, voice=voice, rate=rate).save(path)


def speak(text: str, lang: str) -> str:
    """
    Generate TTS audio to a per-session temp file.
    Returns the file path on success, "" on failure.
    Uses asyncio.run() which correctly creates and closes an event loop,
    or falls back to a new loop on Python < 3.11.
    """
    voice = cfg["tts"]["hi_voice"] if lang == "hi" else cfg["tts"]["en_voice"]
    rate  = cfg["tts"]["hi_rate"]  if lang == "hi" else cfg["tts"]["en_rate"]
    try:
        tmp = tempfile.mktemp(suffix=".mp3", dir=str(BASE_DIR))
        try:
            asyncio.run(_generate(clean_tts(text, lang), voice, rate, tmp))
        except RuntimeError:
            # Already inside an event loop (e.g. Jupyter / some Streamlit contexts)
            loop = asyncio.new_event_loop()
            loop.run_until_complete(_generate(clean_tts(text, lang), voice, rate, tmp))
            loop.close()
        return tmp
    except Exception as e:
        log.warning(f"TTS generation failed: {e}")
        return ""


# ══════════════════════════════════════════════════════════════
# AUDIO PLAYER
# ══════════════════════════════════════════════════════════════

def autoplay_audio(path: str, st_module) -> None:
    """
    Render an HTML5 audio player with autoplay.
    Deletes the temp file after reading to prevent disk accumulation.

    Parameters
    ----------
    path      : file path returned by speak()
    st_module : the streamlit module (passed in to avoid import-time coupling)
    """
    if not path:
        return
    p = Path(path)
    if not p.exists():
        return
    try:
        import base64
        b64 = base64.b64encode(p.read_bytes()).decode()
        st_module.markdown(
            f'<audio autoplay controls '
            f'style="width:100%;margin-top:8px;border-radius:8px;">'
            f'<source src="data:audio/mp3;base64,{b64}" type="audio/mp3">'
            f'</audio>',
            unsafe_allow_html=True,
        )
    except Exception as e:
        log.warning(f"Audio playback failed: {e}")
    finally:
        try:
            p.unlink()
        except Exception:
            pass
