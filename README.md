# MalariaGEN NLQ — Local POC (Gemini + ChromaDB edition)

> Natural-language interface for the MalariaGEN `malariagen_data` Python API.

---

## Architecture

```
User query
   │
   ▼
Off-topic guard (regex)
   │
   ▼
ChromaDB semantic search  ←── Gemini text-embedding-004  (768-dim vectors)
   │                           OR local MiniLM-L6 fallback (no API needed)
   ▼
Context (top-4 API doc chunks)
   │
   ▼
Gemini 2.0 Flash — Programmer
   │
   ▼
Sandboxed subprocess execution (optional)
   │
   ▼
Gemini 2.0 Flash — Verifier  (separate call)
   │
   ▼
Blended confidence score → Frontend
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your Gemini API key
Get a free key at https://aistudio.google.com/
```bash
export GEMINI_API_KEY=AIza...
```

### 3. Scrape the Ag3 API docs (run once)
```bash
python scrape_docs.py
```
Saves structured chunks to `docs/ag3_chunks.json`.  
> If the page is unreachable the app falls back to a built-in 10-function knowledge base.

### 4. Start the server
```bash
python app.py
```
Open **http://localhost:5000**

---

## ChromaDB Details

- Persisted to `./chroma_db/` (auto-created on first run)
- Collection is rebuilt automatically if docs change (SHA-256 hash check)
- Embeddings: **Gemini `text-embedding-004`** (768-dim, cosine space) if `GEMINI_API_KEY` is set; otherwise ChromaDB's bundled `all-MiniLM-L6-v2` (runs fully locally)
- HNSW index — approximate nearest-neighbour search

---

## Gemini Models Used

| Layer | Model | Purpose |
|-------|-------|---------|
| Embeddings | `text-embedding-004` | Vectorise docs + queries |
| Programmer | `gemini-2.0-flash` | NLQ → Python code |
| Verifier | `gemini-2.0-flash` | Cross-check intent & logic |

---

## Project Layout

```
mgen-nlq-poc/
├── scrape_docs.py        # one-shot Ag3 API docs scraper
├── rag.py                # ChromaDB RAG engine
├── app.py                # Flask backend (Gemini)
├── requirements.txt
├── README.md
├── docs/
│   └── ag3_chunks.json   # created by scrape_docs.py
├── chroma_db/            # auto-created — persisted vectors
├── static/
│   ├── style.css
│   └── app.js
└── templates/
    └── index.html
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Frontend |
| `POST` | `/api/query` | `{"query": "..."}` → full NLQ pipeline |
| `GET` | `/api/health` | Liveness + chunk count + model info |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | — | **Required** (or `GOOGLE_API_KEY`) |
| `FLASK_PORT` | `5000` | Server port |
| `FLASK_DEBUG` | `0` | Flask debug mode |
