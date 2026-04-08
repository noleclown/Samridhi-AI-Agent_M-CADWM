"""
samridhi/prompts.py
===================
Single source of truth for all LLM prompt construction.

Exports:
    build_faiss_messages(context, query, intent, history, lang)
    build_live_messages(context, query, intent, history, lang)
    build_fallback_messages(query, is_related, lang)
    build_followup_prompt(query, answer, lang)
    build_typo_prompt(question)
    build_intent_prompt(query)

All functions are pure — no Streamlit or network I/O.
`lang` is always passed explicitly so prompts are never dependent on global state.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from samridhi.config import cfg


# ══════════════════════════════════════════════════════════════
# INTENT INSTRUCTION STRINGS  (injected into FAISS + live prompts)
# ══════════════════════════════════════════════════════════════

_INTENT_INSTR: dict[str, tuple[str, str]] = {
    "definition": (
        "Focus on a clear, precise definition. Cite the document or section where it appears.",
        "स्पष्ट और संक्षिप्त परिभाषा दें। दस्तावेज़ या अनुभाग का उल्लेख करें।",
    ),
    "procedure": (
        "Provide numbered step-by-step procedure. Include prerequisites and required documents.",
        "क्रमांकित चरण-दर-चरण प्रक्रिया दें। पूर्वापेक्षाएँ और आवश्यक दस्तावेज़ शामिल करें।",
    ),
    "data": (
        "Present data in tables or structured lists. Cite source document and year for all figures.",
        "डेटा को तालिका या सूची में प्रस्तुत करें। सभी आँकड़ों के लिए स्रोत और वर्ष का उल्लेख करें।",
    ),
    "smis": (
        "This is a SMIS portal query. Provide step-by-step portal workflow with screen/module names.",
        "यह SMIS पोर्टल प्रश्न है। स्क्रीन/मॉड्यूल नामों सहित चरण-दर-चरण पोर्टल वर्कफ्लो दें।",
    ),
    "general": (
        "Provide a comprehensive, well-structured response covering all relevant aspects.",
        "सभी प्रासंगिक पहलुओं को कवर करते हुए व्यापक और सुव्यवस्थित उत्तर दें।",
    ),
}


def _intent_instruction(intent: str, lang: str) -> str:
    pair = _INTENT_INSTR.get(intent, _INTENT_INSTR["general"])
    return pair[1] if lang == "hi" else pair[0]


# ══════════════════════════════════════════════════════════════
# CONVERSATION HISTORY BUILDER
# Converts stored message dicts into LangChain message objects.
# Capped at context_turns pairs for token economy.
# ══════════════════════════════════════════════════════════════

def _build_history(history: list[dict]) -> list:
    """
    history is st.session_state.messages (list of {role, content}).
    Returns list of HumanMessage / AIMessage for the last N pairs.
    Assistant content is truncated to 400 chars for token economy.
    """
    n     = cfg["retrieval"]["context_turns"]
    pairs = []
    i     = len(history) - 1
    while i >= 0 and len(pairs) < n:
        m = history[i]
        if m["role"] == "assistant" and i > 0 and history[i - 1]["role"] == "user":
            pairs.insert(0, (history[i - 1]["content"], m["content"]))
            i -= 2
        else:
            i -= 1
    msgs = []
    for user_t, asst_t in pairs:
        msgs.append(HumanMessage(content=user_t))
        msgs.append(AIMessage(content=asst_t[:400]))
    return msgs


# ══════════════════════════════════════════════════════════════
# SYSTEM PROMPT HEADER
# ══════════════════════════════════════════════════════════════

def _sys_header(source_label: str, lang: str) -> str:
    if lang == "hi":
        return (
            f"आप समृद्धि हैं — M-CADWM योजना और SMIS पोर्टल के लिए पेशेवर AI सहायक।\n"
            f"स्रोत: {source_label}\n"
            "निर्देश: कोई emoji न उपयोग करें। औपचारिक सरकारी भाषा-शैली बनाए रखें। "
            "व्यक्तिगत भाषा उपयोग न करें। संदर्भ में अनुपलब्ध जानकारी न बनाएं।"
        )
    return (
        f"You are Samridhi — the professional AI Assistant for the M-CADWM scheme and SMIS portal.\n"
        f"Source: {source_label}\n"
        "Instructions: No emojis. Formal government-appropriate tone. "
        "No informal language. Do not invent information absent from the provided context."
    )


# ══════════════════════════════════════════════════════════════
# PUBLIC PROMPT BUILDERS
# Each returns a list[HumanMessage | AIMessage] ready for llm.invoke()
# ══════════════════════════════════════════════════════════════

def build_faiss_messages(
    context: str,
    query: str,
    intent: str,
    history: list[dict],
    lang: str,
) -> list:
    ii  = _intent_instruction(intent, lang)
    sys = _sys_header("M-CADWM official documents", lang)

    if lang == "hi":
        body = (
            f"{sys}\n\nअतिरिक्त निर्देश: {ii}\n"
            "- ## शीर्षक, बुलेट और क्रमांकित सूचियाँ उचित स्थान पर उपयोग करें।\n"
            "- यदि संदर्भ में जानकारी अनुपलब्ध है, स्पष्ट रूप से सूचित करें।\n\n"
            f"संदर्भ (M-CADWM आधिकारिक दस्तावेज़):\n{context}\n\n"
            f"प्रश्न: {query}\n\nउत्तर:"
        )
    else:
        body = (
            f"{sys}\n\nAdditional: {ii}\n"
            "- Use ## headings, bullets, and numbered lists where appropriate.\n"
            "- If the context lacks sufficient information, state this clearly.\n\n"
            f"Context (M-CADWM official documents):\n{context}\n\n"
            f"Query: {query}\n\nResponse:"
        )

    return _build_history(history) + [HumanMessage(content=body)]


def build_live_messages(
    context: str,
    query: str,
    intent: str,
    history: list[dict],
    lang: str,
) -> list:
    ii  = _intent_instruction(intent, lang)
    sys = _sys_header("cadwm.gov.in (live)", lang)

    if lang == "hi":
        body = (
            f"{sys}\n\nअतिरिक्त निर्देश: {ii}\n"
            "- ## शीर्षक और बुलेट उपयोग करें जहाँ उचित हो।\n\n"
            f"संदर्भ (cadwm.gov.in से):\n{context}\n\n"
            f"प्रश्न: {query}\n\nउत्तर:"
        )
    else:
        body = (
            f"{sys}\n\nAdditional: {ii}\n"
            "- Use ## headings and bullets where appropriate.\n\n"
            f"Context (from cadwm.gov.in):\n{context}\n\n"
            f"Query: {query}\n\nResponse:"
        )

    return _build_history(history) + [HumanMessage(content=body)]


def build_fallback_messages(
    query: str,
    is_related: bool,
    lang: str,
) -> list:
    if lang == "hi":
        scope = (
            "यदि प्रश्न M-CADWM या SMIS के व्यापक दायरे से संबंधित है, सामान्य ज्ञान से "
            "संक्षिप्त और तथ्यात्मक जानकारी दें — स्पष्ट करें कि यह आधिकारिक स्रोत नहीं है।"
            if is_related else
            "सूचित करें कि यह प्रश्न M-CADWM/SMIS के दायरे से बाहर है और उपयुक्त प्रश्न पूछने का निर्देश दें।"
        )
        body = (
            "आप समृद्धि हैं — M-CADWM और SMIS पोर्टल के लिए पेशेवर AI सहायक।\n"
            f"प्रश्न: \"{query}\"\n"
            f"निर्देश: {scope} कोई emoji न उपयोग करें। औपचारिक भाषा बनाए रखें।\n\nउत्तर:"
        )
    else:
        scope = (
            "Provide a brief, factual response from general knowledge if the query is "
            "broadly related to M-CADWM or SMIS — clearly note it is not from official documents."
            if is_related else
            "Inform the user this query falls outside the scope of M-CADWM and SMIS, "
            "and direct them to submit M-CADWM/SMIS-specific queries."
        )
        body = (
            "You are Samridhi — professional AI Assistant for M-CADWM and SMIS.\n"
            f"Query: \"{query}\"\n"
            f"Instructions: {scope} No emojis. Formal government-appropriate tone.\n\nResponse:"
        )

    return [HumanMessage(content=body)]


def build_followup_prompt(query: str, answer: str, lang: str) -> list:
    if lang == "hi":
        body = (
            "नीचे दिए प्रश्न और उत्तर के आधार पर M-CADWM/SMIS से संबंधित "
            "2-3 उपयोगी अनुवर्ती प्रश्न सुझाइए। केवल प्रश्न, प्रत्येक अलग पंक्ति पर, "
            "कोई क्रमांक या बुलेट नहीं।\n\n"
            f"प्रश्न: {query}\nउत्तर: {answer[:400]}\n\nअनुवर्ती प्रश्न:"
        )
    else:
        body = (
            "Based on the question and answer below, suggest 2-3 useful M-CADWM/SMIS "
            "follow-up questions. One per line, no numbering or bullets.\n\n"
            f"Question: {query}\nAnswer: {answer[:400]}\n\nFollow-up questions:"
        )
    return [HumanMessage(content=body)]


def build_typo_prompt(question: str) -> list:
    body = (
        "Fix any spelling mistakes or typos in the following query. "
        "Return ONLY the corrected query, nothing else. "
        "Do not change meaning, language, or add words.\n\nQuery: " + question
    )
    return [HumanMessage(content=body)]


def build_intent_prompt(query: str) -> list:
    body = (
        "Classify the following query into exactly one category. "
        "Reply with ONE word only — no explanation:\n"
        "  definition  procedure  data  smis  general\n\n"
        f"Query: {query}\n\nCategory:"
    )
    return [HumanMessage(content=body)]
