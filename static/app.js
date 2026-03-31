/* ═══════════════════════════════════════════════════════════════
   MalariaGEN NLQ  ·  app.js
   ═══════════════════════════════════════════════════════════════ */
"use strict";

// ── state ─────────────────────────────────────────────────────────────────────
let pins = [];
try { pins = JSON.parse(localStorage.getItem("mgen-pins") || "[]"); } catch {}

// ── DOM ───────────────────────────────────────────────────────────────────────
const messagesEl        = document.getElementById("messages");
const queryInput        = document.getElementById("queryInput");
const sendBtn           = document.getElementById("sendBtn");
const clearBtn          = document.getElementById("clearBtn");
const darkToggle        = document.getElementById("darkToggle");
const suggestions       = document.getElementById("suggestions");
const tpl               = document.getElementById("responseTpl");

// pins – desktop
const pinCount          = document.getElementById("pinCount");
const pinsEmpty         = document.getElementById("pinsEmpty");
const pinsList          = document.getElementById("pinsList");

// pins – mobile drawer
const mobilePinsToggle  = document.getElementById("mobilePinsToggle");
const mobilePinLabel    = document.getElementById("mobilePinLabel");
const pinsDrawer        = document.getElementById("pinsDrawer");
const drawerBackdrop    = document.getElementById("drawerBackdrop");
const drawerClose       = document.getElementById("drawerClose");
const pinCountDrawer    = document.getElementById("pinCountDrawer");
const pinsEmptyDrawer   = document.getElementById("pinsEmptyDrawer");
const pinsListDrawer    = document.getElementById("pinsListDrawer");

// ── dark mode ─────────────────────────────────────────────────────────────────
const DARK_KEY = "mgen-dark";
function applyDark(on) {
  document.body.classList.toggle("dark", on);
  darkToggle.textContent = on ? "☀" : "☾";
  try { localStorage.setItem(DARK_KEY, on ? "1" : "0"); } catch {}
}
applyDark(
  localStorage.getItem(DARK_KEY) === "1" ||
  (localStorage.getItem(DARK_KEY) === null &&
   window.matchMedia("(prefers-color-scheme: dark)").matches)
);
darkToggle.addEventListener("click", () => applyDark(!document.body.classList.contains("dark")));

// ── init ──────────────────────────────────────────────────────────────────────
renderAllPins();

// ── events ────────────────────────────────────────────────────────────────────
sendBtn.addEventListener("click", handleSend);
queryInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); }
});
clearBtn.addEventListener("click", () => {
  if (!confirm("Clear all messages?")) return;
  messagesEl.innerHTML = "";
  appendWelcome();
});
suggestions.addEventListener("click", (e) => {
  const chip = e.target.closest(".chip");
  if (!chip) return;
  queryInput.value = chip.dataset.query;
  queryInput.focus();
});

// mobile drawer
mobilePinsToggle.addEventListener("click", () => pinsDrawer.classList.add("open"));
drawerBackdrop.addEventListener("click",   () => pinsDrawer.classList.remove("open"));
drawerClose.addEventListener("click",      () => pinsDrawer.classList.remove("open"));


// ── core flow ─────────────────────────────────────────────────────────────────
async function handleSend() {
  const query = queryInput.value.trim();
  if (!query) return;
  queryInput.value = "";
  queryInput.style.height = "";
  sendBtn.disabled = true;
  hideWelcome();
  appendUserMessage(query);
  const loadingEl = appendLoading();

  try {
    const data = await fetchQuery(query);
    loadingEl.remove();
    if (data.off_topic) appendOffTopic(data.answer);
    else                appendAssistantMessage(query, data);
  } catch (err) {
    loadingEl.remove();
    appendOffTopic(`⚠️ Request failed: ${err.message}`);
    console.error(err);
  } finally {
    sendBtn.disabled = false;
    queryInput.focus();
  }
}

async function fetchQuery(query) {
  const resp = await fetch("/api/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    throw new Error(err.error || `HTTP ${resp.status}`);
  }
  return resp.json();
}


// ── message builders ──────────────────────────────────────────────────────────
function appendUserMessage(text) {
  const d = document.createElement("div");
  d.className = "message-block user-block";
  d.textContent = text;
  messagesEl.appendChild(d);
  scrollBottom();
}

function appendLoading() {
  const d = document.createElement("div");
  d.className = "loading-block message-block";
  d.innerHTML = `
    <div class="loading-dots">
      <div class="dots"><i></i><i></i><i></i></div>
      <span>Querying ChromaDB + Gemini…</span>
    </div>
    <div class="skeleton-line"></div>
    <div class="skeleton-line"></div>
    <div class="skeleton-line"></div>`;
  messagesEl.appendChild(d);
  scrollBottom();
  return d;
}

function appendAssistantMessage(query, data) {
  const node  = tpl.content.cloneNode(true);
  const block = node.querySelector(".assistant-block");

  // confidence
  const badge = block.querySelector(".confidence-badge");
  const conf  = data.confidence ?? 0;
  badge.textContent = `${conf}%`;
  badge.dataset.level = conf >= 70 ? "high" : conf >= 40 ? "medium" : "low";

  // answer
  block.querySelector(".msg-answer").textContent = data.answer || "";

  // reasoning
  if (data.reasoning) {
    block.querySelector(".reasoning-body").textContent = data.reasoning;
  } else {
    block.querySelector(".reasoning-details").style.display = "none";
  }

  // code block with syntax highlighting + line numbers
  const code = (data.code || "").trim();
  if (code) {
    const codeEl    = block.querySelector(".code-block code");
    const lnEl      = block.querySelector(".line-numbers");
    const copyBtn   = block.querySelector(".copy-btn");

    const lines = code.split("\n");
    lnEl.innerHTML = lines.map((_, i) => `<span>${i + 1}</span>`).join("");
    codeEl.innerHTML = highlightPython(code);
    copyBtn.addEventListener("click", () => copyCode(code, copyBtn));
  } else {
    block.querySelector(".code-block-wrapper").style.display = "none";
  }

  // function tags
  const tagsEl = block.querySelector(".func-tags");
  (data.relevant_functions || []).forEach((fn) => {
    const s = document.createElement("span");
    s.className = "func-tag";
    s.textContent = `ag3.${fn}()`;
    tagsEl.appendChild(s);
  });
  if (!data.relevant_functions?.length) tagsEl.style.display = "none";

  // exec output
  const execDetails = block.querySelector(".exec-details");
  if (data.execution) {
    const out = [
      data.execution.stdout && `STDOUT:\n${data.execution.stdout}`,
      data.execution.stderr && `STDERR:\n${data.execution.stderr}`,
    ].filter(Boolean).join("\n\n") || "(no output)";
    block.querySelector(".exec-output").textContent = out;
  } else {
    execDetails.style.display = "none";
  }

  // verifier
  const vbarScores = block.querySelector(".vbar-scores");
  if (data.verification) {
    const v = data.verification;
    [["Intent", v.intent_match], ["Logic", v.logic_consistent]].forEach(([lbl, sc]) => {
      if (sc == null) return;
      const s = document.createElement("span");
      s.className = "vbar-score";
      s.textContent = `${lbl}: ${sc}%`;
      vbarScores.appendChild(s);
    });
    if (v.issues?.length) {
      const s = document.createElement("span");
      s.className = "vbar-score";
      s.style.cssText = "color:var(--red);border-color:var(--red-bg)";
      s.textContent = `⚠ ${v.issues[0]}`;
      vbarScores.appendChild(s);
    }
  } else {
    block.querySelector(".verification-bar").style.display = "none";
  }

  // pin button
  const pinBtn = block.querySelector(".pin-btn");
  pinBtn.addEventListener("click", () => {
    // prevent double-pin
    if (pinBtn.classList.contains("pinned")) return;
    addPin(query, data);
    pinBtn.classList.add("pinned");
    pinBtn.title = "Pinned!";
  });

  messagesEl.appendChild(block);
  scrollBottom();
}

function appendOffTopic(text) {
  const d = document.createElement("div");
  d.className = "off-topic-block message-block";
  d.textContent = text;
  messagesEl.appendChild(d);
  scrollBottom();
}

function appendWelcome() {
  const d = document.createElement("div");
  d.className = "welcome-card";
  d.innerHTML = `
    <div class="wc-icon">🦟</div>
    <div class="wc-text">
      <strong>Welcome to MalariaGEN NLQ</strong>
      <p>Ask about <em>Anopheles</em> genomics — sample metadata, SNPs, CNVs, haplotypes, selection, Fst. I retrieve API context from ChromaDB and generate verified Python code.</p>
    </div>`;
  messagesEl.appendChild(d);
}

function hideWelcome() {
  const w = messagesEl.querySelector(".welcome-card");
  if (w) w.remove();
}


// ── pins ──────────────────────────────────────────────────────────────────────
function addPin(query, data) {
  // don't add duplicates
  if (pins.some(p => p.query === query)) return;
  pins.unshift({
    id:         Date.now(),
    query,
    answer:     data.answer,
    confidence: data.confidence,
    code:       data.code || "",
  });
  if (pins.length > 25) pins.length = 25;
  savePins();
  renderAllPins();
}

function removePin(id) {
  pins = pins.filter(p => p.id !== id);
  savePins();
  renderAllPins();
}

function savePins() {
  try { localStorage.setItem("mgen-pins", JSON.stringify(pins)); } catch {}
}

function buildPinCard(pin) {
  const card = document.createElement("div");
  card.className = "pin-card";
  card.innerHTML = `
    <div class="pin-card-query">${esc(pin.query)}</div>
    <div class="pin-card-answer">${esc(pin.answer || "")}</div>
    <div class="pin-card-footer">
      <span class="pin-card-conf">${pin.confidence ?? "—"}%</span>
      <button class="pin-card-remove" data-id="${pin.id}">✕ Remove</button>
    </div>`;
  card.querySelector(".pin-card-remove")
      .addEventListener("click", () => removePin(pin.id));
  return card;
}

function renderAllPins() {
  // count badges
  const n = pins.length;
  pinCount.textContent       = n;
  pinCountDrawer.textContent = n;
  mobilePinLabel.textContent = `Pins (${n})`;

  // desktop
  renderPinList(pinsList, pinsEmpty);
  // drawer
  renderPinList(pinsListDrawer, pinsEmptyDrawer);
}

function renderPinList(listEl, emptyEl) {
  if (!pins.length) {
    emptyEl.style.display = "block";
    listEl.innerHTML = "";
    return;
  }
  emptyEl.style.display = "none";
  listEl.innerHTML = "";
  pins.forEach(pin => listEl.appendChild(buildPinCard(pin)));
}


// ── syntax highlighting ───────────────────────────────────────────────────────
function highlightPython(code) {
  // escape HTML first
  let s = code
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  // order matters — longer/more specific patterns first
  s = s
    // comments
    .replace(/(#[^\n]*)/g, '<span class="tok-cmt">$1</span>')
    // triple-quoted strings
    .replace(/("""[\s\S]*?"""|'''[\s\S]*?''')/g, '<span class="tok-str">$1</span>')
    // single/double quoted strings
    .replace(/("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')/g, '<span class="tok-str">$1</span>')
    // malariagen_data references (highlight brand)
    .replace(/\b(malariagen_data|ag3)\b/g, '<span class="tok-mg">$1</span>')
    // keywords
    .replace(/\b(import|from|as|def|class|return|if|elif|else|for|in|while|try|except|with|pass|None|True|False|and|or|not|lambda|yield|raise|assert|del|global|nonlocal|is)\b/g, '<span class="tok-kw">$1</span>')
    // builtins / common functions
    .replace(/\b(print|len|range|list|dict|set|tuple|str|int|float|bool|type|isinstance|zip|map|filter|sorted|enumerate|sum|max|min|open|hasattr|getattr|setattr)\b/g, '<span class="tok-fn">$1</span>')
    // numbers
    .replace(/\b(\d[\d_]*(?:\.\d+)?(?:[eE][+-]?\d+)?)\b/g, '<span class="tok-num">$1</span>')
    // operators
    .replace(/([=+\-*\/&|^~<>!%@]+)/g, '<span class="tok-op">$1</span>');

  return s;
}


// ── helpers ───────────────────────────────────────────────────────────────────
function scrollBottom() {
  requestAnimationFrame(() => { messagesEl.scrollTop = messagesEl.scrollHeight; });
}

function copyCode(code, btn) {
  navigator.clipboard.writeText(code).then(() => {
    btn.textContent = "Copied!";
    btn.classList.add("copied");
    setTimeout(() => { btn.textContent = "Copy"; btn.classList.remove("copied"); }, 2000);
  }).catch(() => {
    btn.textContent = "Failed";
    setTimeout(() => { btn.textContent = "Copy"; }, 1500);
  });
}

function esc(str) {
  const d = document.createElement("div");
  d.appendChild(document.createTextNode(str));
  return d.innerHTML;
}

// auto-grow textarea
queryInput.addEventListener("input", function () {
  this.style.height = "auto";
  this.style.height = Math.min(this.scrollHeight, 120) + "px";
});