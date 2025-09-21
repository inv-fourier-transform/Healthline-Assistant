"""
Healthcare Assistant - Run directly from CLI (main.py)

Features:
- Enter 1–10 Healthline URLs, validate & build a fresh vectorstore (old embeddings cleared).
- Ask a query:
  - If it asks to "summarize ... separately", it summarizes each supplied Healthline URL independently.
  - Otherwise, it answers strictly from the retrieved Healthline chunks and cites Healthline sources only.
- Strict fallback: if no relevant chunks are found, prints:
  "No relevant information could be found in the provided sources."
"""

import sys
import re
from urllib.parse import urlparse, urlunparse
from pathlib import Path

# Absolute imports based on project structure:
#   core/      (backend code: indexer, qa, retrieval....)
#   frontend/  (streamlit UI)
#   config/    (config.yaml, .env)
#
# Run from the project root:
#   python main.py
#
# Ensure 'core' is a package (core/__init__.py present).

try:
    from core.indexer import build_index
    from core.qa import answer_query, summarize_per_source
except ModuleNotFoundError as e:
    print("Import error:", e)
    print(
        "Ensure that you are running from the project root (the folder that contains core/, frontend/, config/, main.py)."
    )
    print("Also ensure core/__init__.py exists so Python treats 'core' as a package.")
    sys.exit(1)


# ---------------- Validation ----------------

def is_valid_healthline_prefix(u: str) -> bool:
    """
    Accept only:
    - https://www.healthline.com...
    - www.healthline.com...
    - healthline.com...
    """
    if not isinstance(u, str):
        return False
    u = u.strip()
    allowed = (
        "https://www.healthline.com",
        "www.healthline.com",
        "healthline.com",
    )
    return any(u.startswith(p) for p in allowed)


def canonicalize_healthline(u: str) -> str | None:
    """
    Normalize to:
    - scheme=https
    - netloc=www.healthline.com
    - lowercase path
    - remove query/fragment
    - collapse slashes & trim trailing slash (except root)
    """
    if not isinstance(u, str):
        return None
    u = u.strip()
    if not is_valid_healthline_prefix(u):
        return None

    if u.startswith("https://www.healthline.com"):
        fixed = u
    elif u.startswith("www.healthline.com"):
        fixed = "https://" + u
    elif u.startswith("healthline.com"):
        fixed = "https://www." + u
    else:
        return None

    parsed = urlparse(fixed)
    scheme = "https"
    netloc = "www.healthline.com"
    path = parsed.path or "/"
    path = re.sub(r"/{2,}", "/", path).strip()
    if not path.startswith("/"):
        path = "/" + path
    path = path.lower()
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    # Ignore params/query/fragment for canonical equivalence
    return urlunparse((scheme, netloc, path, "", "", ""))


def read_urls_from_cli() -> list[str]:
    print("Enter 1–10 Healthline URLs (press Enter on a blank line to finish):")
    urls = []
    while len(urls) < 10:
        line = input(f"URL {len(urls)+1}: ").strip()
        if not line:
            break
        urls.append(line)

    # Validate + canonicalize + deduplicate
    invalids = [u for u in urls if not is_valid_healthline_prefix(u)]
    if invalids:
        print("\nInvalid URL(s) detected (must start with one of: https://www.healthline.com, www.healthline.com, healthline.com):")
        for u in invalids:
            print(" -", u)
        sys.exit(1)

    seen = set()
    canon_urls = []
    for u in urls:
        c = canonicalize_healthline(u)
        if c is None:
            print("\nInvalid (failed canonicalization):", u)
            sys.exit(1)
        if c in seen:
            print("\nDuplicate URL detected across formats:", u)
            sys.exit(1)
        seen.add(c)
        canon_urls.append(c)

    if not canon_urls:
        print("No URLs provided.")
        sys.exit(1)

    return canon_urls


# ---------------- Query Routing ----------------

def wants_separate_summaries(q: str) -> bool:
    ql = q.lower()
    return ("summarize" in ql) and ("separate" in ql or "separately" in ql)


def run_from_cli():
    # 1) URLs: read, validate, canonicalize, deduplicate
    urls = read_urls_from_cli()

    # 2) Build a fresh index (clears old embeddings, then indexes new ones)
    print("\nBuilding index (this clears previous embeddings and recreates from the supplied URLs)...")
    result = build_index(urls)
    if result.get("status") != "ok":
        print("Indexing failed.")
        for er in result.get("errors") or []:
            print(" -", er)
        sys.exit(1)

    print(f"Indexed {result.get('chunks_indexed', 0)} chunks.\n")

    # 3) Query input
    query = input("Enter your query: ").strip()
    if not query:
        print("No query entered; exiting.")
        sys.exit(0)

    # 4) Route
    if wants_separate_summaries(query):
        print("\nSummarizing each article strictly from its own content...\n")
        res = summarize_per_source()
        summaries = res.get("summaries", [])
        if not summaries:
            print("No relevant information could be found in the provided sources.")
            return
        for item in summaries:
            print(f"Source: {item.get('source', '')}")
            summary = item.get("summary") or "No relevant information could be found in the provided sources."
            print(summary)
            print("-" * 60)
    else:
        print("\nAnswering...")
        resp = answer_query(query)
        answer = resp.get("answer", "")
        sources = resp.get("sources", [])

        print("\n=== Response ===\n")
        print(answer or "No relevant information could be found in the provided sources.")

        print("\n=== Sources ===")
        if sources:
            for s in sources:
                print("-", s)
        else:
            print("(No sources)")


if __name__ == "__main__":
    run_from_cli()
