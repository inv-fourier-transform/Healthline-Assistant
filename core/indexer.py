# /core/indexer.py
import uuid
import json
from pathlib import Path
from core.config_loader import load_config
from core.loader import load_healthline_urls
from core.chunker import split_docs
from core.embeddings import get_embeddings
from core.vector_store import reset_persist_dir
from langchain_chroma import Chroma  # current integration

FINGERPRINT_FILE = "collection_fingerprint.txt"
SOURCES_FILE = "sources.json"

def build_index(urls: list[str]):
    """
    Full rebuild on each new set of URLs:
    - Load & filter Healthline URLs
    - Chunk
    - Reset Chroma directory
    - Embed & persist
    - Save a fingerprint & an input URL manifest (sources.json)
    Returns: dict(status, chunks_indexed, errors)
    """
    cfg = load_config()
    allowed_domain = cfg["ingestion"]["allowed_domain"]
    mode = cfg["ingestion"].get("unstructured_mode", "single")
    chunk_size = cfg["chunking"]["chunk_size"]
    chunk_overlap = cfg["chunking"]["chunk_overlap"]
    persist_directory = cfg["vectorstore"]["persist_directory"]
    collection_name = cfg["vectorstore"]["collection_name"]
    embed_model = cfg["embeddings"]["model"]

    errors = []

    # 1) Load docs
    docs = load_healthline_urls(urls, allowed_domain=allowed_domain, unstructured_mode=mode)
    if not docs:
        return {"status": "no_content", "chunks_indexed": 0, "errors": ["No valid Healthline content loaded."]}

    # 2) Perform chunking
    chunks = split_docs(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        return {"status": "no_chunks", "chunks_indexed": 0, "errors": ["Content could not be chunked."]}

    # 3) Reset vectorstore directory
    reset_persist_dir(persist_directory)

    # 4) Embeddings + Chroma (auto-persist via persist_directory)

    embeddings = get_embeddings(embed_model, normalize=True)
    _ = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )

    # 5) Save a new fingerprint & source manifest
    fp = Path(persist_directory) / FINGERPRINT_FILE
    fp.write_text(str(uuid.uuid4()), encoding="utf-8")

    # Collect exact input URLs for per-source operations
    srcs = sorted({d.metadata.get("source", "") for d in docs if d.metadata.get("source")})
    (Path(persist_directory) / SOURCES_FILE).write_text(json.dumps(srcs, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"status": "ok", "chunks_indexed": len(chunks), "errors": errors}
