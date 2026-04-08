/**
 * Samridhi AI — Floating Chat Widget  v1.0
 * ==========================================
 * Drop this single <script> tag into any website:
 *
 *   <script
 *     src="https://your-server.com/widget.js"
 *     data-server="https://your-server.com"
 *     data-lang="en"
 *     data-position="bottom-right"
 *     data-accent="#4A90D9"
 *   ></script>
 *
 * Configuration attributes:
 *   data-server    — URL of the Samridhi FastAPI server (required)
 *   data-lang      — default language: "en" or "hi" (default: "en")
 *   data-position  — "bottom-right" or "bottom-left" (default: "bottom-right")
 *   data-accent    — accent colour hex (default: "#4A90D9")
 *   data-title     — widget header title (default: "Samridhi AI")
 */
(function() {
  'use strict';

  // ── Read configuration from script tag ─────────────────────
  const scriptTag  = document.currentScript ||
    document.querySelector('script[src*="widget.js"]');
  const SERVER     = ((scriptTag && scriptTag.dataset.server) || window.location.origin).replace(/\/$/, '');
  const LANG_INIT  = (scriptTag && scriptTag.dataset.lang)     || 'en';
  const POSITION   = (scriptTag && scriptTag.dataset.position) || 'bottom-right';
  const ACCENT     = (scriptTag && scriptTag.dataset.accent)   || '#4A90D9';
  const TITLE      = (scriptTag && scriptTag.dataset.title)    || 'Samridhi AI';

  // Prevent double init
  if (window.__SamridhiWidget) return;
  window.__SamridhiWidget = true;

  // ── State ───────────────────────────────────────────────────
  let sessionId  = null;
  let lang       = LANG_INIT;
  let open       = false;
  let isTyping   = false;
  let msgCount   = 0;

  const STR = {
    en: {
      welcome: "Hello. I am Samridhi, the AI Assistant for the M-CADWM scheme and the SMIS portal.\n\nPlease state your query, or select a topic below.",
      placeholder: "Ask about M-CADWM or SMIS...",
      thinking: "Searching...",
      follow_label: "You may also ask:",
      badge_faiss:   "✦ Official Documents",
      badge_live:    "◉ Live",
      badge_general: "◈ General",
      badge_cache:   "◇ Cached",
    },
    hi: {
      welcome: "नमस्ते। मैं समृद्धि हूँ — M-CADWM के लिए AI सहायक। मैं आपकी कैसे सहायता कर सकती हूँ?",
      placeholder: "M-CADWM या SMIS के बारे में पूछें...",
      thinking: "खोज रहे हैं...",
      follow_label: "आप यह भी पूछ सकते हैं:",
      badge_faiss:   "✦ आधिकारिक दस्तावेज़",
      badge_live:    "◉ लाइव",
      badge_general: "◈ सामान्य",
      badge_cache:   "◇ कैश",
    }
  };
  const t = () => STR[lang] || STR.en;

  // ── CSS ─────────────────────────────────────────────────────
  const css = `
  #srw-launcher {
    position: fixed;
    ${POSITION === 'bottom-left' ? 'left: 24px;' : 'right: 24px;'}
    bottom: 24px;
    z-index: 999999;
    width: 56px; height: 56px;
    border-radius: 50%;
    background: ${ACCENT};
    border: none;
    cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.35);
    transition: transform 0.2s, box-shadow 0.2s;
    font-size: 22px;
  }
  #srw-launcher:hover {
    transform: scale(1.08);
    box-shadow: 0 6px 28px rgba(0,0,0,0.45);
  }
  #srw-container {
    position: fixed;
    ${POSITION === 'bottom-left' ? 'left: 24px;' : 'right: 24px;'}
    bottom: 92px;
    z-index: 999998;
    width: 380px;
    height: 580px;
    max-height: calc(100vh - 110px);
    border-radius: 16px;
    overflow: hidden;
    display: flex; flex-direction: column;
    background: #0E1117;
    border: 1px solid rgba(255,255,255,0.10);
    box-shadow: 0 12px 48px rgba(0,0,0,0.5);
    transform-origin: bottom right;
    transition: transform 0.22s cubic-bezier(.4,0,.2,1), opacity 0.22s;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  }
  #srw-container.hidden {
    transform: scale(0.85) translateY(20px);
    opacity: 0;
    pointer-events: none;
  }
  .srw-header {
    background: #12141F;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    padding: 12px 14px;
    display: flex; align-items: center; gap: 10px;
    flex-shrink: 0;
  }
  .srw-logo {
    width: 30px; height: 30px;
    border-radius: 6px; object-fit: contain;
    background: ${ACCENT};
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; flex-shrink: 0;
  }
  .srw-title { font-weight: 700; font-size: 14px; color: #E8ECF4; }
  .srw-sub   { font-size: 10px; color: #8892A4; }
  .srw-header-actions {
    margin-left: auto; display: flex; gap: 4px; align-items: center;
  }
  .srw-lang-btn {
    background: none; border: 1px solid rgba(255,255,255,0.12);
    border-radius: 5px; padding: 3px 7px;
    color: #8892A4; font-size: 11px; cursor: pointer;
    transition: all 0.15s;
  }
  .srw-lang-btn.active { background: ${ACCENT}; color: #fff; border-color: ${ACCENT}; }
  .srw-close {
    background: none; border: none;
    color: #8892A4; cursor: pointer; font-size: 18px;
    padding: 2px 6px; border-radius: 4px;
    transition: color 0.15s;
  }
  .srw-close:hover { color: #E8ECF4; }
  .srw-expand-btn {
    background: none; border: 1px solid rgba(255,255,255,0.15);
    border-radius: 5px; color: #8892A4;
    font-size: 14px; padding: 2px 6px;
    cursor: pointer; text-decoration: none;
    transition: all 0.15s; display: flex;
    align-items: center; justify-content: center;
    line-height: 1;
  }
  .srw-expand-btn:hover { color: #E8ECF4; border-color: rgba(255,255,255,0.4); }
  .srw-messages {
    flex: 1; overflow-y: auto;
    padding: 14px 12px;
    display: flex; flex-direction: column; gap: 10px;
    scroll-behavior: smooth;
  }
  .srw-messages::-webkit-scrollbar { width: 3px; }
  .srw-messages::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 2px; }
  .srw-msg { display: flex; gap: 8px; }
  .srw-msg.user { flex-direction: row-reverse; }
  .srw-av {
    width: 26px; height: 26px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 12px; flex-shrink: 0; margin-top: 2px;
  }
  .srw-av.bot  { background: ${ACCENT}; color: #fff; }
  .srw-av.usr  { background: #2d3a55; color: #E8ECF4; }
  .srw-bubble {
    padding: 9px 12px;
    border-radius: 12px;
    font-size: 13px;
    line-height: 1.55;
    max-width: calc(100% - 38px);
  }
  .srw-msg.user .srw-bubble {
    background: ${ACCENT}; color: #fff;
    border-bottom-right-radius: 4px;
  }
  .srw-msg.bot .srw-bubble {
    background: #1A1D2E;
    border: 1px solid rgba(255,255,255,0.08);
    color: #E8ECF4;
    border-bottom-left-radius: 4px;
  }
  .srw-bubble strong { font-weight: 700; }
  .srw-bubble em { font-style: italic; }
  .srw-bubble h1,.srw-bubble h2,.srw-bubble h3 { font-weight: 700; margin: 6px 0 3px; }
  .srw-bubble ul,.srw-bubble ol { padding-left: 16px; margin: 4px 0; }
  .srw-bubble li { margin: 2px 0; }
  .srw-badge {
    display: inline-block; font-size: 10px; font-weight: 600;
    padding: 1px 7px; border-radius: 20px; margin-bottom: 5px;
  }
  .srw-b-green { background: rgba(74,222,128,0.12); color: #4ade80; }
  .srw-b-amber { background: rgba(251,191,36,0.12);  color: #fbbf24; }
  .srw-b-grey  { background: rgba(148,163,184,0.10); color: #94a3b8; }
  .srw-follow {
    margin-top: 8px; display: flex; flex-direction: column; gap: 4px;
  }
  .srw-follow-label { font-size: 10px; color: #8892A4; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 2px; }
  .srw-fup {
    background: rgba(74,144,217,0.08);
    border: 1px solid rgba(74,144,217,0.2);
    border-radius: 6px; padding: 5px 8px;
    font-size: 12px; color: #93c5fd; cursor: pointer;
    text-align: left; transition: all 0.15s; line-height: 1.3;
  }
  .srw-fup:hover { background: ${ACCENT}; color: #fff; border-color: ${ACCENT}; }
  .srw-typing {
    display: flex; gap: 4px; align-items: center;
    padding: 10px 12px; background: #1A1D2E;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px; border-bottom-left-radius: 4px;
  }
  .srw-dot {
    width: 5px; height: 5px; border-radius: 50%;
    background: #8892A4; animation: srw-bounce 1.2s infinite;
  }
  .srw-dot:nth-child(2) { animation-delay: 0.2s; }
  .srw-dot:nth-child(3) { animation-delay: 0.4s; }
  @keyframes srw-bounce {
    0%,80%,100% { transform: translateY(0); }
    40% { transform: translateY(-5px); }
  }
  .srw-input-area {
    padding: 10px 12px;
    border-top: 1px solid rgba(255,255,255,0.08);
    background: #12141F;
    flex-shrink: 0;
  }
  .srw-input-row {
    display: flex; gap: 6px; align-items: flex-end;
    background: #1A1D2E;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px; padding: 6px 10px;
    transition: border-color 0.15s;
  }
  .srw-input-row:focus-within { border-color: rgba(74,144,217,0.4); }
  .srw-ta {
    flex: 1; background: none; border: none; outline: none;
    color: #E8ECF4; font-size: 13px;
    font-family: inherit; resize: none;
    min-height: 20px; max-height: 100px;
    line-height: 1.5; padding: 2px 0;
  }
  .srw-ta::placeholder { color: #8892A4; }
  .srw-send {
    background: ${ACCENT}; border: none; border-radius: 7px;
    width: 28px; height: 28px;
    display: flex; align-items: center; justify-content: center;
    cursor: pointer; flex-shrink: 0;
    transition: background 0.15s, opacity 0.15s;
  }
  .srw-send:disabled { opacity: 0.3; cursor: not-allowed; }
  .srw-send:hover:not(:disabled) { background: ${ACCENT}cc; }
  .srw-branding {
    text-align: center; font-size: 10px; color: #8892A4;
    margin-top: 6px; padding-bottom: 2px;
  }
  `;

  // Inject CSS
  const style = document.createElement('style');
  style.textContent = css;
  document.head.appendChild(style);

  // ── HTML ────────────────────────────────────────────────────
  const launcherEl = document.createElement('button');
  launcherEl.id = 'srw-launcher';
  launcherEl.title = 'Chat with Samridhi AI';
  launcherEl.innerHTML = '💬';
  launcherEl.onclick = toggleWidget;

  const containerEl = document.createElement('div');
  containerEl.id = 'srw-container';
  containerEl.className = 'hidden';
  containerEl.innerHTML = `
  <div class="srw-header">
    <div class="srw-logo">🤖</div>
    <div>
      <div class="srw-title">${TITLE}</div>
      <div class="srw-sub">M-CADWM &amp; SMIS</div>
    </div>
    <div class="srw-header-actions">
      <button class="srw-lang-btn active" id="srw-en" onclick="window.__SrwSetLang('en')">EN</button>
      <button class="srw-lang-btn"        id="srw-hi" onclick="window.__SrwSetLang('hi')">हि</button>
      <a class="srw-expand-btn" id="srw-expand" href="${SERVER}/" target="_blank" title="Open full chat">⛶</a>
      <button class="srw-close" onclick="window.__SrwClose()">✕</button>
    </div>
  </div>
  <div class="srw-messages" id="srw-msgs"></div>
  <div class="srw-input-area">
    <div class="srw-input-row">
      <textarea class="srw-ta" id="srw-input" rows="1"
        placeholder="${t().placeholder}"
        onkeydown="window.__SrwKey(event)"
        oninput="window.__SrwResize(this)"></textarea>
      <button class="srw-send" id="srw-send" disabled onclick="window.__SrwSend()">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="white">
          <path d="M2 21l21-9L2 3v7l15 2-15 2v7z"/>
        </svg>
      </button>
    </div>
    <div class="srw-branding">Powered by Samridhi AI · cadwm.gov.in</div>
  </div>`;

  document.body.appendChild(launcherEl);
  document.body.appendChild(containerEl);

  // ── API ─────────────────────────────────────────────────────
  async function initSession() {
    try {
      const r = await fetch(`${SERVER}/api/session?lang=${lang}`);
      const d = await r.json();
      sessionId = d.session_id;
      // Update expand link so full page continues this exact session
      const expandBtn = document.getElementById('srw-expand');
      if (expandBtn) {
        expandBtn.href = `${SERVER}/?session_id=${sessionId}&lang=${lang}`;
      }
    } catch(e) { console.error('[Samridhi] Session init failed', e); }
  }

  async function sendMsg(question) {
    if (isTyping || !question.trim()) return;
    isTyping = true;

    appendUser(question);
    const tid = appendTyping();

    try {
      // Use /api/chat (JSON) instead of /api/stream (SSE)
      // SSE is often blocked by proxies, Cloudflare tunnels, and CDNs.
      // JSON POST works everywhere reliably.
      const r = await fetch(`${SERVER}/api/chat`, {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({question, session_id: sessionId, lang}),
      });

      if (!r.ok) throw new Error(`Server error: ${r.status}`);

      const data = await r.json();
      sessionId  = data.session_id;

      removeEl(tid);
      finaliseBot(
        appendBot('', '', []),
        data.answer || '',
        {
          layer:       data.layer       || '',
          follow_ups:  data.follow_ups  || [],
          session_id:  data.session_id  || '',
          confidence:  data.confidence  || 0,
        }
      );

    } catch(e) {
      removeEl(tid);
      console.error('[Samridhi Widget]', e);
      appendBot('Sorry, something went wrong. Please check your connection and try again.', '', []);
    } finally {
      isTyping = false;
    }
  }

  // ── Rendering ─────────────────────────────────────────────
  function appendUser(text) {
    const id = ++msgCount;
    const el = document.createElement('div');
    el.className = 'srw-msg user';
    el.id = `srw-m-${id}`;
    el.innerHTML = `
      <div class="srw-av usr">U</div>
      <div class="srw-bubble">${esc(text)}</div>`;
    msgs().appendChild(el);
    srwScroll();
    return id;
  }

  function appendBot(text, layer, fups) {
    const id = ++msgCount;
    const el = document.createElement('div');
    el.className = 'srw-msg bot';
    el.id = `srw-m-${id}`;
    el.innerHTML = `
      <div class="srw-av bot">🤖</div>
      <div class="srw-bubble">
        ${badge(layer)}
        <div class="srw-body">${srwMd(stripSrc(text))}</div>
        ${fupsHtml(fups, id)}
      </div>`;
    msgs().appendChild(el);
    srwScroll();
    return id;
  }

  function finaliseBot(id, text, meta) {
    const el = document.getElementById(`srw-m-${id}`);
    if (!el) return;
    const bubble = el.querySelector('.srw-bubble');
    const body   = el.querySelector('.srw-body');
    body.innerHTML = srwMd(stripSrc(text));
    const bdg = document.createElement('div');
    bdg.innerHTML = badge(meta.layer);
    bubble.insertBefore(bdg, body);
    if (meta.follow_ups && meta.follow_ups.length) {
      const div = document.createElement('div');
      div.innerHTML = fupsHtml(meta.follow_ups, id);
      bubble.appendChild(div);
      div.querySelectorAll('.srw-fup').forEach(b => {
        b.onclick = () => { document.getElementById('srw-input').value = b.dataset.q; window.__SrwSend(); };
      });
    }
    srwScroll();
  }

  function appendTyping() {
    const id = ++msgCount;
    const el = document.createElement('div');
    el.className = 'srw-msg bot';
    el.id = `srw-m-${id}`;
    el.innerHTML = `
      <div class="srw-av bot">🤖</div>
      <div class="srw-typing">
        <span style="font-size:11px;color:#8892A4;">${t().thinking}</span>
        <div class="srw-dot"></div><div class="srw-dot"></div><div class="srw-dot"></div>
      </div>`;
    msgs().appendChild(el);
    srwScroll();
    return id;
  }

  function removeEl(id) {
    const el = document.getElementById(`srw-m-${id}`);
    if (el) el.remove();
  }

  function badge(layer) {
    const map = {
      faiss:    ['srw-b-green', t().badge_faiss],
      live:     ['srw-b-amber', t().badge_live],
      fallback: ['srw-b-grey',  t().badge_general],
      cache:    ['srw-b-grey',  t().badge_cache],
    };
    if (!map[layer]) return '';
    return `<span class="srw-badge ${map[layer][0]}">${map[layer][1]}</span><br>`;
  }

  function fupsHtml(fups, id) {
    if (!fups || !fups.length) return '';
    const btns = fups.map(q =>
      `<button class="srw-fup" data-q="${esc(q)}">${esc(q)}</button>`
    ).join('');
    return `<div class="srw-follow">
      <div class="srw-follow-label">${t().follow_label}</div>${btns}</div>`;
  }

  function srwMd(md) {
    if (!md) return '';
    return md
      .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
      .replace(/^### (.+)$/gm,'<h3>$1</h3>')
      .replace(/^## (.+)$/gm, '<h2>$1</h2>')
      .replace(/^# (.+)$/gm,  '<h1>$1</h1>')
      .replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>')
      .replace(/\*(.+?)\*/g,'<em>$1</em>')
      .replace(/^\s*[-•]\s+(.+)$/gm,'<li>$1</li>')
      .replace(/(<li>.*<\/li>\n?)+/g, m=>`<ul>${m}</ul>`)
      .replace(/\n\n/g,'<br><br>').replace(/\n/g,'<br>');
  }

  function stripSrc(text) {
    const suffixes = [
      '\n\n---\n*Source: M-CADWM official documents*',
      '\n\n---\n*Source: cadwm.gov.in (live)*',
      '\n\n---\n*Note: Response based on general knowledge, not M-CADWM official documents.*',
    ];
    for (const s of suffixes) {
      if (text.trimEnd().endsWith(s.trim()))
        return text.slice(0, text.trimEnd().length - s.trim().length).trimEnd();
    }
    return text;
  }

  function esc(s) { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }
  function msgs()  { return document.getElementById('srw-msgs'); }
  function srwScroll() { const m = msgs(); if(m) requestAnimationFrame(()=>m.scrollTop=m.scrollHeight); }

  // ── Controls ───────────────────────────────────────────────
  function toggleWidget() {
    open = !open;
    containerEl.classList.toggle('hidden', !open);
    launcherEl.innerHTML = open ? '✕' : '💬';
    if (open && !sessionId) {
      initSession().then(() => {
        appendBot(t().welcome, '', []);
        document.getElementById('srw-input').focus();
      });
    }
  }

  window.__SrwClose = () => {
    open = false;
    containerEl.classList.add('hidden');
    launcherEl.innerHTML = '💬';
  };

  window.__SrwSetLang = async (l) => {
    lang = l;
    document.getElementById('srw-en').classList.toggle('active', l==='en');
    document.getElementById('srw-hi').classList.toggle('active', l==='hi');
    document.getElementById('srw-input').placeholder = t().placeholder;
    await initSession();
    msgs().innerHTML = '';
    msgCount = 0;
    appendBot(t().welcome, '', []);
  };

  window.__SrwKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); window.__SrwSend(); }
    document.getElementById('srw-send').disabled = !document.getElementById('srw-input').value.trim();
  };

  window.__SrwResize = (el) => {
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 100) + 'px';
    document.getElementById('srw-send').disabled = !el.value.trim();
  };

  window.__SrwSend = () => {
    const ta = document.getElementById('srw-input');
    const q  = ta.value.trim();
    if (!q || isTyping) return;
    ta.value = '';
    ta.style.height = 'auto';
    document.getElementById('srw-send').disabled = true;
    sendMsg(q);
  };

})();
