# /core/loader.py
from urllib.parse import urlparse
from langchain_community.document_loaders import UnstructuredURLLoader

def is_healthline_url(u: str, allowed_domain: str) -> bool:
    try:
        netloc = urlparse(u).netloc.lower()
        return netloc.endswith(allowed_domain)
    except Exception:
        return False

def load_healthline_urls(urls, allowed_domain: str, unstructured_mode: str = "single"):
    # Filter invalid domain early
    clean_urls = [u.strip() for u in urls if u.strip() and is_healthline_url(u.strip(), allowed_domain)]
    if not clean_urls:
        return []

    loader = UnstructuredURLLoader(
        urls=clean_urls,
        continue_on_failure=True,
        mode=unstructured_mode,
        show_progress_bar=False,
    )
    docs = loader.load()
    # Normalize metadata & drop empty ones
    filtered = []
    for d in docs:
        if getattr(d, "page_content", "").strip():
            if "source" not in d.metadata:
                # ensure "source" exists
                d.metadata["source"] = d.metadata.get("url", "") or d.metadata.get("source", "")
            filtered.append(d)
    return filtered
