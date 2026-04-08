"""
samridhi/ui_strings.py
======================
All user-facing strings (bilingual EN/HI) and static CSS/JS.

Exports:
    UI          — dict[lang → dict[key → str]]
    CSS         — str  (injected once via st.markdown)
    BRIDGE_JS   — str  (injected once via components.html)
"""

from __future__ import annotations

# ══════════════════════════════════════════════════════════════
# UI STRINGS  (100% bilingual — zero cross-language text)
# ══════════════════════════════════════════════════════════════

UI: dict[str, dict[str, str]] = {
    "en": {
        "title":        "Samridhi – M-CADWM",
        "subtitle":     "AI Assistant — M-CADWM & SMIS",
        "welcome": (
            "Hello. I am Samridhi, the AI Assistant for the M-CADWM scheme and the SMIS portal.\n\n"
            "Please state your query, or select a topic below."
        ),
        "placeholder":   "Enter your query or click 🎤 to speak...",
        "spinner":       "Searching M-CADWM documents...",
        "spinner_web":   "Checking cadwm.gov.in for the latest information...",
        "no_result":     "No relevant information was found. Please refine your query.",
        "cached_note":   "*(Retrieved from cache)*\n\n",
        "fb_up":         "Feedback recorded.",
        "fb_dn":         "Feedback recorded. Response will be reviewed.",
        "typo_note":     "*Note: Typographical correction applied — response addresses: \"{q}\"*\n\n",
        "src_faiss":     "\n\n---\n*Source: M-CADWM official documents*",
        "src_live":      "\n\n---\n*Source: cadwm.gov.in (live)*",
        "src_general":   "\n\n---\n*Note: Response based on general knowledge, not M-CADWM official documents.*",
        "follow_header": "**You may also ask:**",
        "starter_label": "Common queries:",
        "copy_label":    "Copy answer",
        "pdf_label":     "Download last answer as PDF",
        "health_label":  "System Status",
        "btn_en":        "🇬🇧 EN",
        "btn_hi":        "🇮🇳 HI",
        "rate_limited":  "The system is briefly busy. Please try again in a moment.",
        "clear_chat":    "New conversation",
        "input_truncated":   "*Note: Your query was truncated to {n} characters to ensure accurate results.*",
        "operator_panel":    "Operator Panel",
        "operator_password": "Enter operator password",
        "operator_unlock":   "Unlock",
        "operator_locked":   "🔒 Operator access",
        "operator_wrong":    "Incorrect password.",
        "tts_toggle":        "Voice responses",
        "about_label":       "About Samridhi",
        "about_text":        "Samridhi v4.0 — AI Assistant for M-CADWM & SMIS\ncadwm.gov.in",
        "history_label":     "Past conversations",
        "history_empty":     "No past conversations yet.",
        "history_restore":   "Restore",
        "history_delete":    "Delete",
        "history_saved":     "Conversation saved.",
        "badge_faiss":       "✦ M-CADWM Official Documents",
        "badge_live":        "◉ cadwm.gov.in  (live)",
        "badge_general":     "◈ General Knowledge",
        "badge_cache":       "◇ Retrieved from cache",
        "appearance_label":  "Appearance",
        "theme_label":       "Theme",
        "fontsize_label":    "Font size",
    },
    "hi": {
        "title":        "समृद्धि – M-CADWM",
        "subtitle":     "AI सहायक — M-CADWM और SMIS",
        "welcome": (
            "नमस्ते। मैं समृद्धि हूँ — M-CADWM योजना और SMIS पोर्टल के लिए AI सहायक।\n\n"
            "कृपया अपना प्रश्न दर्ज करें, या नीचे दिए गए विषयों में से कोई चुनें।"
        ),
        "placeholder":   "प्रश्न दर्ज करें या 🎤 दबाकर बोलें...",
        "spinner":       "M-CADWM दस्तावेज़ खोज रहे हैं...",
        "spinner_web":   "cadwm.gov.in से नवीनतम जानकारी प्राप्त की जा रही है...",
        "no_result":     "कोई प्रासंगिक जानकारी नहीं मिली। कृपया अपना प्रश्न पुनः दर्ज करें।",
        "cached_note":   "*(कैश से प्राप्त)*\n\n",
        "fb_up":         "फीडबैक दर्ज किया गया।",
        "fb_dn":         "फीडबैक दर्ज किया गया। उत्तर की समीक्षा की जाएगी।",
        "typo_note":     "*सूचना: वर्तनी सुधार लागू किया गया — उत्तर इस प्रश्न के लिए है: \"{q}\"*\n\n",
        "src_faiss":     "\n\n---\n*स्रोत: M-CADWM आधिकारिक दस्तावेज़*",
        "src_live":      "\n\n---\n*स्रोत: cadwm.gov.in (लाइव)*",
        "src_general":   "\n\n---\n*सूचना: यह उत्तर सामान्य ज्ञान पर आधारित है।*",
        "follow_header": "**आप यह भी पूछ सकते हैं:**",
        "starter_label": "सामान्य प्रश्न:",
        "copy_label":    "उत्तर कॉपी करें",
        "pdf_label":     "अंतिम उत्तर PDF में डाउनलोड करें",
        "health_label":  "सिस्टम स्थिति",
        "btn_en":        "🇬🇧 EN",
        "btn_hi":        "🇮🇳 HI",
        "rate_limited":  "सिस्टम अभी व्यस्त है। कृपया थोड़ी देर बाद पुनः प्रयास करें।",
        "clear_chat":    "नई बातचीत",
        "input_truncated":   "*सूचना: सटीक परिणामों के लिए आपका प्रश्न {n} अक्षरों तक सीमित किया गया है।*",
        "operator_panel":    "ऑपरेटर पैनल",
        "operator_password": "ऑपरेटर पासवर्ड दर्ज करें",
        "operator_unlock":   "अनलॉक करें",
        "operator_locked":   "🔒 ऑपरेटर एक्सेस",
        "operator_wrong":    "गलत पासवर्ड।",
        "tts_toggle":        "वॉइस प्रतिक्रिया",
        "about_label":       "समृद्धि के बारे में",
        "about_text":        "समृद्धि v4.0 — M-CADWM और SMIS के लिए AI सहायक\ncadwm.gov.in",
        "history_label":     "पिछली बातचीत",
        "history_empty":     "अभी तक कोई पिछली बातचीत नहीं।",
        "history_restore":   "पुनः लोड करें",
        "history_delete":    "हटाएं",
        "history_saved":     "बातचीत सहेजी गई।",
        "badge_faiss":       "✦ M-CADWM आधिकारिक दस्तावेज़",
        "badge_live":        "◉ cadwm.gov.in (लाइव)",
        "badge_general":     "◈ सामान्य ज्ञान",
        "badge_cache":       "◇ कैश से प्राप्त",
        "appearance_label":  "दिखावट",
        "theme_label":       "थीम",
        "fontsize_label":    "अक्षर आकार",
    },
}


# ══════════════════════════════════════════════════════════════
# CSS  (injected once via st.markdown)
# ══════════════════════════════════════════════════════════════

CSS = """
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
.samridhi-follow-btn {
    background:none;border:1px solid #555;border-radius:6px;
    padding:4px 10px;font-size:13px;color:#ccc;cursor:pointer;
    margin:3px 3px 3px 0;transition:background 0.15s,color 0.15s;
}
.samridhi-follow-btn:hover{background:#e05c2a;color:#fff;border-color:#e05c2a;}
.samridhi-copy-btn {
    background:none;border:1px solid #444;border-radius:5px;
    padding:2px 8px;font-size:11px;color:#888;cursor:pointer;
    margin-top:6px;transition:background 0.15s;
}
.samridhi-copy-btn:hover{background:#333;color:#fff;}

/* ── Source badges ─────────────────────────────────────────── */
.src-badge {
    display:inline-block;
    font-size:11px;font-weight:600;letter-spacing:0.3px;
    padding:3px 10px;border-radius:12px;margin-bottom:8px;
    border:1px solid transparent;
}
.src-badge-green {
    background:rgba(34,197,94,0.12);
    color:#22c55e;
    border-color:rgba(34,197,94,0.35);
}
.src-badge-amber {
    background:rgba(251,146,60,0.12);
    color:#fb923c;
    border-color:rgba(251,146,60,0.35);
}
.src-badge-grey {
    background:rgba(148,163,184,0.12);
    color:#94a3b8;
    border-color:rgba(148,163,184,0.3);
}
</style>
<div id="samridhi-mic-status"></div>
"""


# ══════════════════════════════════════════════════════════════
# JS BRIDGE  (voice input + follow-up submit + clipboard copy)
# ══════════════════════════════════════════════════════════════

BRIDGE_JS = """
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
    btn.id='samridhi-mic';btn.title='Click to speak';btn.innerHTML='🎤';
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
window.parent.samridhiSubmit=function(q){injectAndSubmit(q);};
window.parent.samridhiCopy=function(text){
    navigator.clipboard.writeText(text).catch(function(){
        var ta=P.createElement('textarea');ta.value=text;
        P.body.appendChild(ta);ta.select();P.execCommand('copy');P.body.removeChild(ta);
    });
};
</script>
"""
