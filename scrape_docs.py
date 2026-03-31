"""
scrape_docs.py  –  scrapes the MalariaGEN Ag3 API reference and saves
structured chunks to docs/ag3_chunks.json for the RAG system.

Tries multiple URL variants to handle readthedocs / GitHub Pages differences.
Run once before starting the app:
    python scrape_docs.py
"""

import json
import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup, Tag

OUTPUT_PATH = Path("docs/ag3_chunks.json")

# Try these URLs in order — first one that returns a parseable page wins
CANDIDATE_URLS = [
    "https://malariagen.github.io/malariagen-data-python/latest/Ag3.html",
    "https://malariagen-data-python.readthedocs.io/en/latest/Ag3.html",
    "https://malariagen.github.io/malariagen-data-python/stable/Ag3.html",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120 Safari/537.36"
    )
}


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def fetch(url: str) -> BeautifulSoup | None:
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            r.raise_for_status()
            print(f"  ✓ fetched {url}")
            return BeautifulSoup(r.text, "html.parser")
        except requests.RequestException as exc:
            print(f"  attempt {attempt+1} failed: {exc}")
            time.sleep(2 ** attempt)
    return None


def _intent_keywords(name: str, description: str) -> list[str]:
    word_map = {
        "sample":     ["samples", "metadata", "collection", "specimens"],
        "snp":        ["snp", "variant", "mutation", "allele", "genotype"],
        "hap":        ["haplotype", "haplotypes", "phasing"],
        "cnv":        ["copy number", "cnv", "amplification", "duplication"],
        "pca":        ["pca", "principal component", "population structure"],
        "fst":        ["fst", "differentiation", "divergence", "fixation"],
        "freq":       ["frequency", "allele frequency", "prevalence"],
        "divers":     ["diversity", "pi", "tajima"],
        "tree":       ["tree", "phylogeny", "phylogenetic"],
        "karyotype":  ["karyotype", "inversion"],
        "ihs":        ["ihs", "selection", "sweep"],
        "xpehh":      ["xpehh", "cross-population", "selection"],
        "g123":       ["g123", "selection", "sweep"],
        "plot":       ["plot", "visualise", "chart", "figure"],
        "map":        ["map", "geographic", "africa"],
        "h12":        ["h12", "selection", "sweep"],
        "cohort":     ["cohort", "group", "population"],
    }
    kw: set[str] = set()
    lower = (name + " " + description).lower()
    for key, synonyms in word_map.items():
        if key in lower:
            kw.update(synonyms)
    kw.update(re.findall(r"[a-z][a-z0-9_]+", name.lower()))
    return sorted(kw)[:20]


def parse_overview(soup: BeautifulSoup) -> dict:
    paras = []
    for el in soup.select("section p, article p, .rst-content p"):
        t = clean(el.get_text())
        if len(t) > 40:
            paras.append(t)
        if len(paras) >= 4:
            break

    code_ex = ""
    for pre in soup.select("pre, .highlight pre")[:2]:
        t = clean(pre.get_text())
        if "malariagen" in t.lower() or "ag3" in t.lower():
            code_ex += t + "\n"

    return {
        "name": "__overview__",
        "signature": "import malariagen_data; ag3 = malariagen_data.Ag3()",
        "description": " ".join(paras)[:600] or "MalariaGEN Ag3 API for Anopheles gambiae complex genomic data.",
        "parameters": [],
        "example": code_ex.strip()[:400],
        "intent_keywords": ["setup", "import", "initialise", "ag3", "overview", "getting started"],
        "source_url": "",
    }


def parse_dl_method(dl: Tag) -> dict | None:
    """
    Sphinx wraps each method in a <dl class="py method"> (or function/attribute).
    Structure:
      <dt id="..."> signature </dt>
      <dd> description, field-list (params), examples </dd>
    """
    dt = dl.find("dt")
    if not dt:
        return None

    name_el = dt.find("span", class_=re.compile(r"sig-name|descname"))
    if not name_el:
        return None
    func_name = clean(name_el.get_text())
    if not func_name or func_name.startswith("_"):
        return None

    signature = clean(dt.get_text())

    dd = dl.find("dd")
    if not dd:
        return None

    # description = text before the field-list
    desc_parts = []
    for child in dd.children:
        if not hasattr(child, "get_text"):
            continue
        cls = getattr(child, "get", lambda k, d=None: None)("class") or []
        cls_str = " ".join(cls) if isinstance(cls, list) else str(cls)
        if any(x in cls_str for x in ("field-list", "rubric", "docutils")):
            break
        tag = getattr(child, "name", None)
        if tag in ("dl", "div") and "highlight" in cls_str:
            break
        t = clean(child.get_text())
        if t:
            desc_parts.append(t)
    description = " ".join(desc_parts)[:600]

    # parameters from field-list
    params: list[str] = []
    field_list = dd.find("dl", class_=re.compile(r"field-list|simple"))
    if field_list:
        for li in field_list.find_all("li"):
            t = clean(li.get_text())
            if t:
                params.append(t)
        if not params:
            for p in field_list.find_all("dd"):
                t = clean(p.get_text())
                if t:
                    params.append(t)

    # examples from highlight blocks
    examples: list[str] = []
    for pre in dd.select(".highlight pre, pre")[:2]:
        t = clean(pre.get_text())
        if t:
            examples.append(t)
    example = "\n".join(examples)[:400]

    if not description and not params:
        return None

    return {
        "name": func_name,
        "signature": signature[:300],
        "description": description,
        "parameters": params[:12],
        "example": example,
        "intent_keywords": _intent_keywords(func_name, description),
        "source_url": "",
    }


def parse_section_headers(soup: BeautifulSoup) -> list[dict]:
    """
    Fallback: some Sphinx themes group methods under h2/h3 headings
    with code blocks nearby. Extract those.
    """
    chunks = []
    for heading in soup.find_all(re.compile(r"^h[23]$")):
        name = clean(heading.get_text())
        if not re.match(r"^[a-z][a-z0-9_]+$", name):
            continue
        # grab next sibling paragraphs
        desc_parts = []
        sib = heading.find_next_sibling()
        while sib and sib.name in ("p", "div"):
            desc_parts.append(clean(sib.get_text()))
            sib = sib.find_next_sibling()
        if desc_parts:
            chunks.append({
                "name": name,
                "signature": f"ag3.{name}(...)",
                "description": " ".join(desc_parts)[:400],
                "parameters": [],
                "example": "",
                "intent_keywords": _intent_keywords(name, " ".join(desc_parts)),
                "source_url": "",
            })
    return chunks


def scrape(url: str) -> list[dict] | None:
    soup = fetch(url)
    if not soup:
        return None

    chunks = [parse_overview(soup)]
    chunks[0]["source_url"] = url

    # primary: Sphinx <dl class="py method|function|...">
    seen: set[str] = {"__overview__"}
    for dl in soup.find_all("dl", class_=re.compile(r"\bpy\b")):
        c = parse_dl_method(dl)
        if c and c["name"] not in seen:
            c["source_url"] = url
            seen.add(c["name"])
            chunks.append(c)

    # secondary fallback
    if len(chunks) < 5:
        for c in parse_section_headers(soup):
            if c["name"] not in seen:
                c["source_url"] = url
                seen.add(c["name"])
                chunks.append(c)

    return chunks if len(chunks) >= 2 else None


def main():
    result = None
    for url in CANDIDATE_URLS:
        print(f"\nTrying {url} …")
        result = scrape(url)
        if result:
            print(f"  → got {len(result)} chunks")
            break
        else:
            print(f"  → too few chunks, trying next URL")

    if not result:
        print(
            "\n⚠  Could not scrape any meaningful content from the docs pages.\n"
            "   This is usually a network/proxy issue in restricted environments.\n"
            "   The app will use the built-in fallback knowledge base (10 functions).\n"
            "   To force-populate, download Ag3.html manually and run:\n"
            "     python scrape_docs.py --local Ag3.html"
        )
        # if --local flag given, parse local file
        if "--local" in sys.argv:
            idx = sys.argv.index("--local") + 1
            local_path = Path(sys.argv[idx])
            if local_path.exists():
                print(f"\nParsing local file {local_path} …")
                soup = BeautifulSoup(local_path.read_text(encoding="utf-8"), "html.parser")
                chunks = [parse_overview(soup)]
                seen: set[str] = {"__overview__"}
                for dl in soup.find_all("dl", class_=re.compile(r"\bpy\b")):
                    c = parse_dl_method(dl)
                    if c and c["name"] not in seen:
                        c["source_url"] = str(local_path)
                        seen.add(c["name"])
                        chunks.append(c)
                result = chunks
                print(f"  → got {len(result)} chunks from local file")
        if not result:
            sys.exit(1)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2))
    print(f"\n✓ Saved {len(result)} chunks → {OUTPUT_PATH}")
    if len(result) < 5:
        print("⚠  Still few chunks — the app fallback knowledge base covers 10 core functions.")


if __name__ == "__main__":
    main()
