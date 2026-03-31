"""
app.py  –  Flask backend for MalariaGEN NLQ POC (Gemini edition).

LLM stack:
  Programmer : gemini-2.0-flash  (fast, strong at code)
  Verifier   : gemini-2.0-flash  (separate call, lighter prompt)
  Embeddings : text-embedding-004 via ChromaDB integration  (in rag.py)

Environment variables:
  GEMINI_API_KEY   (required)
  FLASK_PORT       (default 5000)
  FLASK_DEBUG      (default 0)
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from google import genai
from google.genai import types

from rag import get_rag

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

_gemini_client: genai.Client | None = None

PROGRAMMER_MODEL = "gemini-2.5-flash"
VERIFIER_MODEL   = "gemini-2.5-flash"


def _client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not set.  Export it before running:\n"
                "  export GEMINI_API_KEY=AIza..."
            )
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


# ── prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are MalariaGEN Assistant, an AI interface for the MalariaGEN \
malaria vector genomic data API (malariagen_data Python package, Ag3 dataset).

Your ONLY job is to help researchers query Anopheles gambiae complex genomic data \
using the malariagen_data Python API.

RULES:
1. Only answer questions about malaria vector genomics, the malariagen_data API, \
   mosquito biology, or related public health topics. Politely decline anything else.
2. Always generate executable Python code using ag3 = malariagen_data.Ag3() as entry point.
3. Provide a confidence score 0–100 in your JSON response.
4. Show your reasoning before the code.

RESPONSE FORMAT — always return ONLY valid JSON, no markdown fences:
{
  "answer": "<plain English explanation, 2-4 sentences>",
  "reasoning": "<step-by-step thought process, max 5 steps>",
  "code": "<executable Python code, or empty string>",
  "confidence": <integer 0-100>,
  "relevant_functions": ["list", "of", "api", "function", "names"],
  "off_topic": false
}

If the query is off-topic, set off_topic to true and leave code empty."""




# ── guards ────────────────────────────────────────────────────────────────────

_OFF_TOPIC_RE = re.compile(
    r"\b(bitcoin|crypto|weather|recipe|poem|joke|write.?a.?story|stock.?price|"
    r"president|election|sports.?score|movie|music|celebrity)\b",
    re.IGNORECASE,
)

def _is_off_topic(text: str) -> bool:
    return bool(_OFF_TOPIC_RE.search(text))


# ── sandbox ───────────────────────────────────────────────────────────────────

_BLOCKED = {"os", "subprocess", "sys", "socket", "urllib", "requests",
            "http", "ftplib", "smtplib", "shutil"}
_IMPORT_RE = re.compile(r"^\s*(?:import|from)\s+(\w+)", re.MULTILINE)

def _safe(code: str) -> tuple[bool, str]:
    for m in _IMPORT_RE.finditer(code):
        if m.group(1) in _BLOCKED:
            return False, f"Blocked import: {m.group(1)}"
    if any(tok in code for tok in ("__import__", "eval(", "exec(")):
        return False, "Blocked: eval/exec"
    return True, ""

def _run(code: str, timeout: int = 20) -> dict:
    ok, reason = _safe(code)
    if not ok:
        return {"stdout": "", "stderr": f"[BLOCKED] {reason}", "returncode": -1}
    wrapper = textwrap.dedent(f"""\
        import warnings; warnings.filterwarnings('ignore')
        try:
            import malariagen_data
        except ImportError:
            print("[INFO] malariagen_data not installed — code generation only mode")
        {textwrap.indent(code, '    ')}
    """)
    try:
        r = subprocess.run([sys.executable, "-c", wrapper],
                           capture_output=True, text=True, timeout=timeout)
        return {"stdout": r.stdout[:2000], "stderr": r.stderr[:1000], "returncode": r.returncode}
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "[TIMEOUT]", "returncode": -1}
    except Exception as exc:
        return {"stdout": "", "stderr": str(exc), "returncode": -1}


# ── Gemini calls ──────────────────────────────────────────────────────────────

def _clean_json(raw: str) -> str:
    """Strip markdown fences and extract the first complete JSON object."""
    # remove ```json ... ``` fences
    raw = re.sub(r"```json\s*", "", raw)
    raw = re.sub(r"```\s*", "", raw)
    raw = raw.strip()

    # find the first '{' ... last '}' to isolate the JSON object
    start = raw.find("{")
    end   = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        raw = raw[start : end + 1]

    return raw


def _safe_json_loads(raw: str) -> dict | None:
    """Try json.loads; on failure attempt field-by-field extraction."""
    cleaned = _clean_json(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # last-resort: extract individual string/number fields with regex
    obj: dict = {}
    for key in ("answer", "reasoning", "code"):
        m = re.search(rf'"{key}"\s*:\s*"(.*?)(?<!\\)"', cleaned, re.DOTALL)
        if m:
            obj[key] = m.group(1).replace('\\"', '"')
    for key in ("confidence", "intent_match", "logic_consistent", "overall_confidence"):
        m = re.search(rf'"{key}"\s*:\s*(\d+)', cleaned)
        if m:
            obj[key] = int(m.group(1))
    for key in ("off_topic",):
        m = re.search(rf'"{key}"\s*:\s*(true|false)', cleaned, re.IGNORECASE)
        if m:
            obj[key] = m.group(1).lower() == "true"
    # relevant_functions array
    m = re.search(r'"relevant_functions"\s*:\s*\[(.*?)\]', cleaned, re.DOTALL)
    if m:
        obj["relevant_functions"] = re.findall(r'"([^"]+)"', m.group(1))
    return obj if obj else None

def _call_programmer(query: str, context: str) -> dict:
    prompt = (
        f"API CONTEXT (from malariagen_data documentation):\n{context}\n\n"
        f"USER QUERY: {query}"
    )
    resp = _client().models.generate_content(
        model=PROGRAMMER_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,
            max_output_tokens=2048,
        ),
    )
    result = _safe_json_loads(resp.text)
    if not result:
        raise ValueError(f"Could not parse programmer response: {resp.text[:200]}")
    return result

VERIFIER_PROMPT = """You are a strict quality-control verifier for genomic data query results.
Return ONLY a raw JSON object with no markdown, no code fences, no extra text.
Required format: {"intent_match": <0-100>, "logic_consistent": <0-100>, "overall_confidence": <0-100>, "issues": []}"""


def _call_verifier(query: str, code: str, exec_result: dict, orig_conf: int) -> dict:
    _default = {"intent_match": orig_conf, "logic_consistent": orig_conf,
                "overall_confidence": orig_conf, "issues": []}
    if not code.strip():
        return _default
    payload = (
        f"query: {query[:200]}\n"
        f"code: {code[:400]}\n"
        f"stderr: {exec_result.get('stderr','')[:100]}"
    )
    try:
        resp = _client().models.generate_content(
            model=VERIFIER_MODEL,
            contents=payload,
            config=types.GenerateContentConfig(
                system_instruction=VERIFIER_PROMPT,
                temperature=0.0,
                max_output_tokens=120,
            ),
        )
        verdict = _safe_json_loads(resp.text)
        if not verdict or "overall_confidence" not in verdict:
            return _default
        verdict["overall_confidence"] = int(
            0.5 * verdict.get("overall_confidence", orig_conf) + 0.5 * orig_conf
        )
        verdict.setdefault("issues", [])
        return verdict
    except Exception as exc:
        print(f"[Verifier] failed ({exc}), using original confidence")
        return _default


# ── query endpoint ─────────────────────────────────────────────────────────────

MAX_RETRIES = 3

@app.post("/api/query")
def handle_query():
    body  = request.get_json(silent=True) or {}
    query = (body.get("query") or "").strip()
    if not query:
        return jsonify({"error": "No query provided"}), 400

    if _is_off_topic(query):
        return jsonify({
            "answer": "I can only help with MalariaGEN genomic data queries. Please ask about Anopheles mosquito genomics, sample metadata, SNPs, CNVs, or related analyses.",
            "reasoning": "Off-topic guard triggered.",
            "code": "", "confidence": 0, "off_topic": True,
            "relevant_functions": [], "execution": None, "verification": None,
        })

    rag     = get_rag()
    chunks  = rag.query(query, top_k=4)
    context = rag.format_context(chunks)

    result: dict = {}
    last_error = ""
    for attempt in range(MAX_RETRIES):
        try:
            q = query if attempt == 0 else f"{query}\n\n[RETRY {attempt}: previous error: {last_error}]"
            result = _call_programmer(q, context)
            if result.get("confidence", 0) >= 30:
                break
            last_error = f"low confidence ({result.get('confidence')})"
        except (json.JSONDecodeError, Exception) as exc:
            last_error = str(exc)
            result = {}

    if not result:
        return jsonify({"error": f"All attempts failed: {last_error}"}), 500

    exec_result = _run(result.get("code", "")) if result.get("code", "").strip() else None
    verification = _call_verifier(query, result.get("code", ""), exec_result or {}, result.get("confidence", 50))

    return jsonify({
        "answer":             result.get("answer", ""),
        "reasoning":          result.get("reasoning", ""),
        "code":               result.get("code", ""),
        "confidence":         verification["overall_confidence"],
        "off_topic":          result.get("off_topic", False),
        "relevant_functions": result.get("relevant_functions", []),
        "execution":          exec_result,
        "verification":       verification,
        "model_used":         PROGRAMMER_MODEL,
    })


@app.get("/api/health")
def health():
    rag = get_rag()
    return jsonify({
        "status": "ok",
        "vector_db": "ChromaDB",
        "embedding_model": "text-embedding-004 (Gemini) or MiniLM fallback",
        "llm": PROGRAMMER_MODEL,
        "chunks_loaded": rag.chunk_count,
        "functions": rag.all_function_names[:20],
    })

@app.get("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    port  = int(os.environ.get("FLASK_PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    print(f"\n🦟  MalariaGEN NLQ (Gemini + ChromaDB)  →  http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=debug)
