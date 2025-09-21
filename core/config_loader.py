# /core/config_loader.py
from pathlib import Path
import os
import yaml
from dotenv import load_dotenv

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
CONFIG_PATH = CONFIG_DIR / "config.yaml"
ENV_PATH = CONFIG_DIR / ".env"

def load_config():
    # Load .env first so YAML can reference environment variables if needed
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Resolve persist_directory relative to the config file location
    # Final path should be: Path(__file__).parent / "vector_resources/vectorstore"
    # Get it from the config directory location for portability.
    persist_dir = (CONFIG_DIR / "vector_resources" / "vectorstore").resolve()
    cfg.setdefault("vectorstore", {})
    cfg["vectorstore"]["persist_directory"] = str(persist_dir)

    # Allow CHROMA_DIR from env to override, if provided
    chroma_env = os.getenv("CHROMA_DIR", "").strip()
    if chroma_env:
        # If absolute, trust it; if relative, resolve against CONFIG_DIR
        p = Path(chroma_env)
        cfg["vectorstore"]["persist_directory"] = str(p if p.is_absolute() else (CONFIG_DIR / p).resolve())

    # Embedding model from env
    embed_model = os.getenv("EMBEDDING_MODEL", "").strip()
    if embed_model:
        cfg.setdefault("embeddings", {})
        cfg["embeddings"]["model"] = embed_model

    # Ensure required keys exist with default values
    cfg.setdefault("ingestion", {}).setdefault("allowed_domain", "healthline.com")
    cfg.setdefault("ingestion", {}).setdefault("max_urls", 10)
    cfg.setdefault("chunking", {}).setdefault("chunk_size", 1000)
    cfg.setdefault("chunking", {}).setdefault("chunk_overlap", 20)
    cfg.setdefault("vectorstore", {}).setdefault("collection_name", "healthline_rag")
    cfg.setdefault("llm", {}).setdefault("groq_model", "llama-3.1-8b-instant")
    cfg.setdefault("llm", {}).setdefault("temperature", 0.3)
    cfg.setdefault("llm", {}).setdefault("max_tokens", 512)
    cfg.setdefault("retrieval", {}).setdefault("k", 4)

    return cfg
