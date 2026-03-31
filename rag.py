"""
rag.py  –  ChromaDB-backed RAG system for the MalariaGEN NLQ POC.

Embedding strategy (in priority order):
  1. Gemini text-embedding-004  — if GEMINI_API_KEY is set
  2. Pure-Python TF-IDF fallback — always available, zero downloads

The ChromaDB collection is persisted to ./chroma_db/ and rebuilt
automatically when docs change (SHA-256 hash check).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings

DOCS_PATH       = Path("docs/ag3_chunks.json")
CHROMA_DIR      = Path("chroma_db")
COLLECTION_NAME = "mgen_ag3_docs"
HASH_FILE       = CHROMA_DIR / ".docs_hash"

# ── pure-python TF-IDF embedding (no downloads, implements ChromaDB protocol) ─

class TfidfEmbeddingFunction(EmbeddingFunction):
    """
    Lightweight TF-IDF vectoriser that produces fixed 512-dim embeddings.
    Completely self-contained — no model downloads, no external calls.
    Used as the local fallback when Gemini API is unavailable.
    """

    VOCAB_SIZE = 512

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self._idf: dict[int, float] = {}
        self._fitted = False

    def fit(self, corpus: list[str]) -> None:
        N = len(corpus)
        df: dict[str, int] = defaultdict(int)
        for doc in corpus:
            for t in set(self._tokenize(doc)):
                df[t] += 1

        # pick top VOCAB_SIZE tokens by df (most common = most stable)
        top = sorted(df.items(), key=lambda x: -x[1])[: self.VOCAB_SIZE]
        self._vocab = {tok: i for i, (tok, _) in enumerate(top)}
        self._idf = {
            i: math.log((N + 1) / (df[tok] + 1)) + 1
            for tok, i in self._vocab.items()
        }
        self._fitted = True

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-z0-9_]+", text.lower())

    def _vectorize(self, text: str) -> list[float]:
        tokens = self._tokenize(text)
        freq: dict[str, int] = defaultdict(int)
        for t in tokens:
            freq[t] += 1
        total = len(tokens) or 1
        vec = [0.0] * self.VOCAB_SIZE
        for tok, cnt in freq.items():
            if tok in self._vocab:
                idx = self._vocab[tok]
                vec[idx] = (cnt / total) * self._idf.get(idx, 1.0)
        # L2-normalise
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def __call__(self, input: Documents) -> Embeddings:
        if not self._fitted:
            self.fit(list(input))
        return [self._vectorize(doc) for doc in input]


# ── Gemini embedding function wrapper ─────────────────────────────────────────

class GeminiEmbeddingFunction(EmbeddingFunction):
    """
    Custom embedding function using the new google-genai SDK.
    Avoids the ChromaDB built-in wrapper which requires the deprecated
    google-generativeai package.
    """
    MODEL = "models/text-embedding-004"

    def __init__(self, api_key: str) -> None:
        from google import genai
        self._client = genai.Client(api_key=api_key)

    def __call__(self, input: Documents) -> Embeddings:
        from google.genai import types as gtypes
        result = self._client.models.embed_content(
            model=self.MODEL,
            contents=list(input),
            config=gtypes.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        return [e.values for e in result.embeddings]


def _make_embedding_fn(api_key: str | None):
    if api_key:
        try:
            fn = GeminiEmbeddingFunction(api_key=api_key)
            # smoke-test with one short string
            fn(["test"])
            print("[RAG] embedding: Gemini text-embedding-004 (768-dim)")
            return fn, "gemini"
        except Exception as exc:
            print(f"[RAG] Gemini embeddings unavailable ({exc})")
    print("[RAG] embedding: local TF-IDF fallback (512-dim, no downloads)")
    return TfidfEmbeddingFunction(), "tfidf"


# ── knowledge base ────────────────────────────────────────────────────────────

FALLBACK_CHUNKS: list[dict] = [
    {
        "name": "__overview__",
        "signature": "import malariagen_data; ag3 = malariagen_data.Ag3()",
        "description": (
            "The malariagen_data Python package provides access to Anopheles gambiae "
            "complex genomic data (Ag3 dataset). Initialise with ag3 = malariagen_data.Ag3(). "
            "Requires google-cloud authentication unless running in Google Colab."
        ),
        "parameters": [],
        "example": "import malariagen_data\nag3 = malariagen_data.Ag3()\nag3",
        "intent_keywords": ["setup", "install", "import", "initialise", "ag3", "overview", "getting started"],
        "source_url": "https://malariagen.github.io/malariagen-data-python/latest/Ag3.html",
    },
    {
        "name": "sample_metadata",
        "signature": "ag3.sample_metadata(sample_sets=None, sample_query=None)",
        "description": (
            "Access sample metadata as a pandas DataFrame. Columns include sample_id, "
            "country, location, year, month, latitude, longitude, sex_call, "
            "species_gambcolu_arabiensis, species_gambiae_coluzzii."
        ),
        "parameters": [
            "sample_sets (str or list): e.g. '3.0' or ['AG1000G-BF-A']",
            "sample_query (str): pandas query string e.g. 'country == \"Ghana\"'",
        ],
        "example": "df = ag3.sample_metadata(sample_sets='3.0')\ndf.groupby('country').size()",
        "intent_keywords": ["samples", "metadata", "country", "location", "year", "species",
                            "collection", "specimens", "coordinates", "latitude", "longitude"],
        "source_url": "https://malariagen.github.io/malariagen-data-python/latest/Ag3.html",
    },
    {
        "name": "snp_calls",
        "signature": "ag3.snp_calls(region, sample_sets=None, sample_query=None, site_mask=None)",
        "description": (
            "Access SNP genotype calls as an xarray Dataset (variants × samples × ploidy=2). "
            "Use region like '3L', '2R:1000000-2000000', or dict {contig, start, end}."
        ),
        "parameters": [
            "region (str): e.g. '3L' or '2R:1000000-2000000'",
            "site_mask (str): 'gamb_colu', 'arab', or 'gamb_colu_arab'",
        ],
        "example": "ds = ag3.snp_calls(region='3L', sample_sets='3.0', site_mask='gamb_colu')",
        "intent_keywords": ["snp", "variant", "genotype", "allele", "calls", "mutation",
                            "site filter", "chromosome", "contig", "vcf"],
        "source_url": "https://malariagen.github.io/malariagen-data-python/latest/Ag3.html",
    },
    {
        "name": "snp_allele_frequencies",
        "signature": "ag3.snp_allele_frequencies(transcript, cohorts, sample_query=None, min_cohort_size=10, site_mask='gamb_colu', sample_sets=None)",
        "description": (
            "Compute allele frequencies for SNPs in a gene/transcript across cohorts. "
            "Returns a DataFrame (variants × cohorts). Useful for kdr/Vgsc resistance mutations."
        ),
        "parameters": [
            "transcript (str): Ensembl ID, e.g. 'AGAP004707-RA' (Vgsc/kdr gene)",
            "cohorts (str or dict): e.g. 'admin1_year' or custom cohort dict",
            "min_cohort_size (int): minimum samples per cohort (default 10)",
        ],
        "example": "df_freq = ag3.snp_allele_frequencies(\n    transcript='AGAP004707-RA',\n    cohorts='admin1_year',\n    sample_sets='3.0'\n)",
        "intent_keywords": ["frequency", "allele frequency", "snp frequency", "cohort",
                            "prevalence", "kdr", "vgsc", "resistance mutation"],
        "source_url": "https://malariagen.github.io/malariagen-data-python/latest/Ag3.html",
    },
    {
        "name": "plot_frequencies_map",
        "signature": "ag3.plot_frequencies_map(df, map_type='mapbox', title=None, zoom=3)",
        "description": (
            "Plot allele or haplotype frequencies as geographic bubble map using Plotly. "
            "Takes the DataFrame from snp_allele_frequencies or haplotype_frequencies."
        ),
        "parameters": [
            "df (DataFrame): output of *_frequencies",
            "map_type (str): 'mapbox' or 'geo'",
        ],
        "example": "fig = ag3.plot_frequencies_map(df_freq)\nfig.show()",
        "intent_keywords": ["map", "geographic", "plot", "visualise", "frequency map", "africa", "plotly"],
        "source_url": "https://malariagen.github.io/malariagen-data-python/latest/Ag3.html",
    },
    {
        "name": "pca",
        "signature": "ag3.pca(region, n_components=20, sample_sets=None, sample_query=None, site_mask='gamb_colu')",
        "description": (
            "PCA on SNP data to explore population structure. "
            "Returns (coords DataFrame, sklearn PCA model). Follow with plot_pca_coords."
        ),
        "parameters": [
            "region (str): e.g. '3L'",
            "n_components (int): default 20",
        ],
        "example": "coords, model = ag3.pca(region='3L', sample_sets='3.0', n_components=10)\nag3.plot_pca_coords(coords, color='country')",
        "intent_keywords": ["pca", "principal component", "population structure",
                            "genetic structure", "cluster", "admixture"],
        "source_url": "https://malariagen.github.io/malariagen-data-python/latest/Ag3.html",
    },
    {
        "name": "haplotype_frequencies",
        "signature": "ag3.haplotype_frequencies(region, cohorts, sample_query=None, min_cohort_size=10, sample_sets=None)",
        "description": (
            "Compute haplotype frequencies across cohorts. "
            "Useful for tracking insecticide-resistance haplotypes (Ace1, Cyp6)."
        ),
        "parameters": [
            "region (str): e.g. '2L:2358158-2431617'",
            "cohorts (str): e.g. 'admin1_year'",
        ],
        "example": "df_hap = ag3.haplotype_frequencies(\n    region='2L:2358158-2431617',\n    cohorts='admin1_year',\n    sample_sets='3.0'\n)",
        "intent_keywords": ["haplotype", "haplotype frequency", "resistance haplotype",
                            "phasing", "ace1", "cyp6", "track spread"],
        "source_url": "https://malariagen.github.io/malariagen-data-python/latest/Ag3.html",
    },
    {
        "name": "ihs",
        "signature": "ag3.ihs(region, sample_sets=None, sample_query=None, site_mask='gamb_colu', window_size=200)",
        "description": "Compute integrated haplotype score (iHS) for detecting positive selection / selective sweeps.",
        "parameters": ["region (str): e.g. '3L'", "window_size (int): default 200"],
        "example": "ds_ihs = ag3.ihs(region='3L', sample_sets='3.0')",
        "intent_keywords": ["ihs", "selection", "positive selection", "sweep",
                            "haplotype score", "selective sweep"],
        "source_url": "https://malariagen.github.io/malariagen-data-python/latest/Ag3.html",
    },
    {
        "name": "cnv_hmm",
        "signature": "ag3.cnv_hmm(region, sample_sets=None, sample_query=None)",
        "description": (
            "Access copy number variant (CNV) HMM calls as xarray Dataset. "
            "Contains normalised coverage and copy number states. "
            "Useful for insecticide resistance gene amplifications (Cyp6m2, Cyp6p3, Ace1)."
        ),
        "parameters": ["region (str): e.g. '2RL'"],
        "example": "ds_cnv = ag3.cnv_hmm(region='2RL', sample_sets='3.0')",
        "intent_keywords": ["cnv", "copy number", "amplification", "duplication",
                            "coverage", "resistance", "cyp6m2", "cyp6p3", "ace1"],
        "source_url": "https://malariagen.github.io/malariagen-data-python/latest/Ag3.html",
    },
    {
        "name": "fst",
        "signature": "ag3.fst(region, cohorts, cohort_size=30, sample_sets=None, site_mask='gamb_colu', window_size=2000)",
        "description": "Windowed Fst (fixation index) between cohort pairs. Measures genetic differentiation.",
        "parameters": [
            "region (str): e.g. '3L'",
            "cohorts (dict): {label: sample_query_string}",
            "window_size (int): SNPs per window (default 2000)",
        ],
        "example": "cohorts = {'BF': 'country==\"Burkina Faso\"', 'GH': 'country==\"Ghana\"'}\ndf_fst = ag3.fst(region='3L', cohorts=cohorts, sample_sets='3.0')",
        "intent_keywords": ["fst", "fixation index", "differentiation", "divergence",
                            "genetic distance", "population comparison"],
        "source_url": "https://malariagen.github.io/malariagen-data-python/latest/Ag3.html",
    },
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _chunk_to_document(c: dict) -> str:
    return "\n".join(filter(None, [
        f"Function: {c.get('name','')}",
        f"Signature: {c.get('signature','')}",
        f"Description: {c.get('description','')}",
        "Parameters: " + "; ".join(c.get("parameters", [])),
        "Keywords: " + ", ".join(c.get("intent_keywords", [])),
        f"Example:\n{c.get('example','')}",
    ]))

def _docs_hash(chunks: list[dict]) -> str:
    return hashlib.sha256(json.dumps(chunks, sort_keys=True).encode()).hexdigest()[:16]


# ── RAG class ─────────────────────────────────────────────────────────────────

class RAG:
    def __init__(self) -> None:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self._embed_fn, self._embed_backend = _make_embedding_fn(api_key)

        CHROMA_DIR.mkdir(exist_ok=True)
        self._chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))

        raw_chunks = self._load_chunks()
        self._chunks_by_id: dict[str, dict] = {c["name"]: c for c in raw_chunks}
        self._collection = self._get_or_build_collection(raw_chunks)
        print(f"[RAG] ready — {self._collection.count()} vectors in ChromaDB", flush=True)

    def _load_chunks(self) -> list[dict]:
        if DOCS_PATH.exists():
            try:
                data = json.loads(DOCS_PATH.read_text())
                if data:
                    print(f"[RAG] loaded scraped docs ({len(data)} chunks)")
                    return data
            except Exception:
                pass
        print("[RAG] using built-in fallback knowledge base")
        return FALLBACK_CHUNKS

    def _get_or_build_collection(self, chunks: list[dict]):
        current_hash = _docs_hash(chunks)
        needs_rebuild = True
        if HASH_FILE.exists():
            needs_rebuild = HASH_FILE.read_text().strip() != current_hash

        try:
            col = self._chroma.get_collection(COLLECTION_NAME, embedding_function=self._embed_fn)
            if not needs_rebuild and col.count() == len(chunks):
                print("[RAG] reusing persisted ChromaDB collection")
                return col
            print("[RAG] rebuilding ChromaDB collection …")
            self._chroma.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

        # For TF-IDF we need to fit on the full corpus first
        if isinstance(self._embed_fn, TfidfEmbeddingFunction):
            corpus = [_chunk_to_document(c) for c in chunks]
            self._embed_fn.fit(corpus)

        col = self._chroma.create_collection(
            COLLECTION_NAME,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
        ids   = [c["name"] for c in chunks]
        docs  = [_chunk_to_document(c) for c in chunks]
        metas = [{"name": c["name"], "source_url": c.get("source_url", "")} for c in chunks]
        for i in range(0, len(ids), 50):
            col.upsert(ids=ids[i:i+50], documents=docs[i:i+50], metadatas=metas[i:i+50])
        print(f"[RAG] indexed {len(ids)} chunks ({self._embed_backend} embeddings)")
        HASH_FILE.write_text(current_hash)
        return col

    def query(self, user_query: str, top_k: int = 4) -> list[dict]:
        # For Gemini embeddings, query vectors should use RETRIEVAL_QUERY task type.
        # ChromaDB's query_texts path will call __call__ again (RETRIEVAL_DOCUMENT),
        # so we embed the query ourselves and use query_embeddings instead.
        if self._embed_backend == "gemini":
            from google.genai import types as gtypes
            resp = self._embed_fn._client.models.embed_content(
                model=GeminiEmbeddingFunction.MODEL,
                contents=[user_query],
                config=gtypes.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
            )
            q_vec = resp.embeddings[0].values
            results = self._collection.query(
                query_embeddings=[q_vec],
                n_results=min(top_k, self._collection.count()),
                include=["metadatas", "distances"],
            )
        else:
            results = self._collection.query(
                query_texts=[user_query],
                n_results=min(top_k, self._collection.count()),
                include=["metadatas", "distances"],
            )
        out = []
        for meta in (results.get("metadatas") or [[]])[0]:
            chunk = self._chunks_by_id.get(meta.get("name", ""))
            if chunk:
                out.append(chunk)
        return out

    def format_context(self, chunks: list[dict]) -> str:
        parts = []
        for c in chunks:
            block = [f"## {c['name']}"]
            if c.get("signature"):   block.append(f"Signature: {c['signature']}")
            if c.get("description"): block.append(f"Description: {c['description']}")
            if c.get("parameters"):  block.append("Parameters:\n" + "\n".join(f"  - {p}" for p in c["parameters"]))
            if c.get("example"):     block.append(f"Example:\n```python\n{c['example']}\n```")
            parts.append("\n".join(block))
        return "\n\n---\n\n".join(parts)

    @property
    def all_function_names(self) -> list[str]:
        return [n for n in self._chunks_by_id if n != "__overview__"]

    @property
    def chunk_count(self) -> int:
        return self._collection.count()

    @property
    def embedding_backend(self) -> str:
        return self._embed_backend


_rag_instance: RAG | None = None

def get_rag() -> RAG:
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAG()
    return _rag_instance
