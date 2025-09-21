# /core/retrieval.py
from langchain_chroma import Chroma
from core.config_loader import load_config
from core.embeddings import get_embeddings

def get_vectorstore():
    cfg = load_config()
    persist_directory = cfg["vectorstore"]["persist_directory"]
    collection_name = cfg["vectorstore"]["collection_name"]
    embed_model = cfg["embeddings"]["model"]

    embeddings = get_embeddings(embed_model, normalize=True)
    vectordb = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
    return vectordb

def get_retriever(k_override: int | None = None):
    """
    Create a retriever tuned for robust recall:
    - Use std similarity with k>=8 to reduce false negatives for multi-part queries.
    """
    cfg = load_config()
    k_cfg = cfg["retrieval"]["k"]
    k = max(8, int(k_override if k_override is not None else k_cfg))
    vectordb = get_vectorstore()
    return vectordb.as_retriever(search_kwargs={"k": k})

def get_retriever_for_source(source_url: str, k_override: int | None = None):
    """
    Create a retriever that only returns chunks from a specific Healthline URL via metadata filter.
    """
    cfg = load_config()
    k_cfg = cfg["retrieval"]["k"]
    k = max(8, int(k_override if k_override is not None else k_cfg))
    vectordb = get_vectorstore()

    return vectordb.as_retriever(search_kwargs={"k": k, "filter": {"source": source_url}})
