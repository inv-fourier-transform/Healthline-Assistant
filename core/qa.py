# /core/qa.py
import json
from pathlib import Path
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from core.config_loader import load_config
from core.retrieval import get_retriever, get_retriever_for_source

FINGERPRINT_FILE = "collection_fingerprint.txt"
CACHE_FILE = "qa_cache.json"
SOURCES_FILE = "sources.json"

SYSTEM_PROMPT = """You are an assistant that must answer strictly and only from the provided context.
- Do not use any knowledge outside the context.
- Do not include references, citations, links, or article titles that are not present in the context.
- If the context does not contain the answer, reply exactly: "No relevant information could be found in the provided sources."
- If multiple questions are asked, answer each separately using only the context.
- Keep responses concise and factual.
Context:
{context}"""

SYSTEM_PROMPT_SUMMARIZE = """You are an assistant that must summarize strictly and only from the provided context.
- Summarize the key points clearly and concisely.
- Do not include references, citations, links, or any article titles not present verbatim in the context.
- Do not invent or mention any content or article not present in the context.
- If the context does not contain enough information to summarize, reply exactly: "No relevant information could be found in the provided sources."
Context:
{context}"""

def _cache_path(persist_directory: str) -> Path:
    return Path(persist_directory) / CACHE_FILE

def _fingerprint(persist_directory: str) -> str:
    fp = Path(persist_directory) / FINGERPRINT_FILE
    return fp.read_text(encoding="utf-8").strip() if fp.exists() else "no-fingerprint"

def _sources_manifest(persist_directory: str) -> list[str]:
    sp = Path(persist_directory) / SOURCES_FILE
    if sp.exists():
        try:
            return json.loads(sp.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []

def _load_cache(persist_directory: str) -> dict:
    p = _cache_path(persist_directory)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _save_cache(persist_directory: str, cache: dict):
    p = _cache_path(persist_directory)
    p.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

def _build_doc_chain(llm, system_prompt: str):
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{input}")]
    )
    return create_stuff_documents_chain(llm, prompt)

def _fallback():
    return "No relevant information could be found in the provided sources."

def answer_query(query: str) -> dict:
    """
    Strictly grounded QA:
    - Retrieve first (robust k, no threshold).
    - If no docs, return exact fallback with no sources.
    - Otherwise, answer only from context; sources appended from metadata.source.
    """
    cfg = load_config()
    persist_directory = cfg["vectorstore"]["persist_directory"]
    fp = _fingerprint(persist_directory)
    key = f"{fp}::{query.strip()}"

    cache = _load_cache(persist_directory)
    if key in cache:
        return {"answer": cache[key]["answer"], "sources": cache[key]["sources"], "cached": True}

    retriever = get_retriever()
    retrieved_docs = retriever.get_relevant_documents(query)

    if not retrieved_docs:
        fb = _fallback()
        cache[key] = {"answer": fb, "sources": []}
        _save_cache(persist_directory, cache)
        return {"answer": fb, "sources": [], "cached": False}

    from core.llm import get_llm
    llm = get_llm()
    doc_chain = _build_doc_chain(llm, SYSTEM_PROMPT)
    resp = doc_chain.invoke({"input": query, "context": retrieved_docs})

    # Collect Healthline sources from retrieved docs only
    sources = []
    for d in retrieved_docs:
        src = getattr(d, "metadata", {}).get("source", "")
        if src:
            sources.append(src)
    seen = set()
    unique_sources = []
    for s in sources:
        if s not in seen:
            seen.add(s)
            unique_sources.append(s)

    answer = (resp or "") if isinstance(resp, str) else (resp.get("answer") or resp.get("output") or "")
    normalized = (answer or "").strip().lower()
    if not normalized or "no relevant information could be found in the provided sources." in normalized:
        answer = _fallback()
        unique_sources = []

    cache[key] = {"answer": answer, "sources": unique_sources}
    _save_cache(persist_directory, cache)
    return {"answer": answer, "sources": unique_sources, "cached": False}

def summarize_per_source() -> dict:
    """
    Summarize each indexed Healthline URL separately with strict grounding.
    Returns: { "summaries": [ { "source": url, "summary": str } ] }
    """
    cfg = load_config()
    persist_directory = cfg["vectorstore"]["persist_directory"]
    urls = _sources_manifest(persist_directory)

    from core.llm import get_llm
    llm = get_llm()
    doc_chain = _build_doc_chain(llm, SYSTEM_PROMPT_SUMMARIZE)

    results = []
    for url in urls:
        retriever = get_retriever_for_source(url)
        docs = retriever.get_relevant_documents("summarize")
        if not docs:
            results.append({"source": url, "summary": _fallback()})
            continue
        resp = doc_chain.invoke({"input": "Summarize the article.", "context": docs})
        summary = (resp or "") if isinstance(resp, str) else (resp.get("answer") or resp.get("output") or "")
        normalized = (summary or "").strip().lower()
        if not normalized or "no relevant information could be found in the provided sources." in normalized:
            summary = _fallback()
        results.append({"source": url, "summary": summary})
    return {"summaries": results}
