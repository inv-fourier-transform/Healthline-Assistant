# ui_interface.py
import os
import re
from urllib.parse import urlparse, urlunparse
from pathlib import Path
import streamlit as st

# ================= Page & runtime config =================
st.set_page_config(page_title="Healthline Assistant", layout="wide")

# Streamlit Cloud writable vectorstore directory
APP_ROOT = Path(__file__).resolve().parent
CHROMA_DIR = st.secrets.get("paths", {}).get("CHROMA_DIR", "vector_resources/vectorstore")
VECTORSTORE_PATH = (APP_ROOT / CHROMA_DIR).resolve()
VECTORSTORE_PATH.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("CHROMA_DIR", str(VECTORSTORE_PATH))

# ================= Robust embedding loader =================
def build_embeddings(model_name: str, normalize: bool = True):
    """
    Priority:
      1) langchain_huggingface.HuggingFaceEmbeddings
      2) langchain_community.embeddings.HuggingFaceEmbeddings
      3) sentence-transformers direct (if installed)
    """
    try:
        from langchain_huggingface.embeddings import HuggingFaceEmbeddings as LC_HFEmb
        return LC_HFEmb(
            model_name=model_name,
            model_kwargs={"trust_remote_code": True},
            encode_kwargs={"normalize_embeddings": normalize},
        )
    except Exception:
        pass
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings as LC_CommEmb
        return LC_CommEmb(
            model_name=model_name,
            model_kwargs={"trust_remote_code": True},
            encode_kwargs={"normalize_embeddings": normalize},
        )
    except Exception:
        pass
    try:
        from sentence_transformers import SentenceTransformer
        from langchain_core.embeddings import Embeddings

        class _STWrap(Embeddings):
            def __init__(self, m, norm):
                self.m = m
                self.norm = norm
            def embed_documents(self, texts):
                return self.m.encode(texts, normalize_embeddings=self.norm, convert_to_numpy=False, show_progress_bar=False)
            def embed_query(self, text):
                return self.m.encode([text], normalize_embeddings=self.norm, convert_to_numpy=False, show_progress_bar=False)[0]
        st_model = SentenceTransformer(model_name, trust_remote_code=True)
        return _STWrap(st_model, normalize)
    except Exception:
        st.error("Embeddings backend unavailable. Ensure transformers and sentence-transformers are installed.")
        st.stop()

# ================= Strict QA primitives (always grounded) =================
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq

FALLBACK = 'No relevant information could be found in the provided sources.'
COLLECTION = "healthline_rag"

def _llm():
    groq_key = st.secrets.get("api", {}).get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", "")).strip()
    if not groq_key:
        st.error("GROQ_API_KEY is not set. Add it in Streamlit Secrets under [api].")
        st.stop()
    os.environ["GROQ_API_KEY"] = groq_key
    model = st.secrets.get("llm", {}).get("MODEL", os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"))
    temperature = float(st.secrets.get("llm", {}).get("TEMPERATURE", os.getenv("GROQ_TEMPERATURE", "0.0")))
    max_tokens = int(st.secrets.get("llm", {}).get("MAX_TOKENS", os.getenv("GROQ_MAX_TOKENS", "512")))
    return ChatGroq(model=model, temperature=temperature, max_tokens=max_tokens)

STRICT_SYSTEM = """You are a healthcare assistant that must answer strictly and only from the provided context.
- Do not use any external knowledge.
- Do not add links, references, or sources in your text.
- If the answer is not present, reply exactly: "No relevant information could be found in the provided sources."
Context:
{context}"""

def _load_urls(urls: list[str]):
    cleaned = []
    for u in urls:
        u = (u or "").strip()
        if u and u.startswith("http") and "healthline.com" in u:
            cleaned.append(u)
    if not cleaned:
        return []
    loader = UnstructuredURLLoader(urls=cleaned, continue_on_failure=True, show_progress_bar=False)
    docs = loader.load()
    out = []
    for d in docs:
        if getattr(d, "page_content", "").strip():
            if "source" not in d.metadata:
                d.metadata["source"] = d.metadata.get("url", "")
            out.append(d)
    return out

def _chunk(docs, size=1000, overlap=200):
    return RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap).split_documents(docs)

def reset_vector_dir(path: Path):
    import shutil
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)

def build_index(urls: list[str]) -> dict:
    docs = _load_urls(urls)
    if not docs:
        return {"status": "no_content", "chunks_indexed": 0, "errors": ["No valid Healthline content loaded."]}
    chunks = _chunk(docs)
    if not chunks:
        return {"status": "no_chunks", "chunks_indexed": 0, "errors": ["Chunking produced no segments."]}

    reset_vector_dir(VECTORSTORE_PATH)
    emb_model = st.secrets.get("embeddings", {}).get("MODEL", os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5"))
    emb = build_embeddings(emb_model, normalize=True)
    _ = Chroma.from_documents(
        documents=chunks,
        embedding=emb,
        collection_name=COLLECTION,
        persist_directory=str(VECTORSTORE_PATH),
    )
    return {"status": "ok", "chunks_indexed": len(chunks), "errors": []}

def get_retriever(k: int = 8):
    emb_model = st.secrets.get("embeddings", {}).get("MODEL", os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5"))
    emb = build_embeddings(emb_model, normalize=True)
    vs = Chroma(
        collection_name=COLLECTION,
        embedding_function=emb,
        persist_directory=str(VECTORSTORE_PATH),
    )
    return vs.as_retriever(search_kwargs={"k": max(8, int(k))})

def answer_query_strict(query: str) -> dict:
    """
    Strictly grounded QA:
    - Retrieve first; if no docs => exact fallback and no sources.
    - Otherwise, call LLM with strict prompt and pass only retrieved docs.
    - Sources are computed from doc metadata (not model output).
    """
    retriever = get_retriever()
    docs = retriever.get_relevant_documents(query)
    if not docs:
        return {"answer": FALLBACK, "sources": []}

    llm = _llm()
    prompt = ChatPromptTemplate.from_messages([("system", STRICT_SYSTEM), ("human", "{input}")])
    chain = create_stuff_documents_chain(llm, prompt)
    resp = chain.invoke({"input": query, "context": docs})
    answer = resp if isinstance(resp, str) else (resp.get("answer") or resp.get("output") or "")

    # Build sources strictly from retrieved docs
    seen, sources = set(), []
    for d in docs:
        src = d.metadata.get("source", "")
        if src and src not in seen:
            seen.add(src)
            sources.append(src)

    normalized = (answer or "").strip().lower()
    if not normalized or FALLBACK.lower() in normalized:
        return {"answer": FALLBACK, "sources": []}
    return {"answer": answer, "sources": sources}

# ================= UI styling =================
st.markdown("""
<style>
:root { --bg-primary:#0e1117; --bg-panel-1:#111827; --bg-panel-2:#0f172a; --bg-response:#1f2937;
--border-1:#1f2937; --border-2:#243041; --text-primary:#e5e7eb; --text-secondary:#cbd5e1;
--accent-blue:#3b82f6; --accent-blue-hover:#2563eb; --accent-green:#10b981; --accent-green-hover:#059669; }
html, body, [class^="css"] { font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; color: var(--text-primary); }
[data-testid="stAppViewContainer"] { background: radial-gradient(1200px 600px at 60% -20%, #121621 10%, var(--bg-primary) 60%); }
#app-title { font-size: 2.2rem; font-weight: 800; color: #f8fafc; text-align: center; margin: 0 0 1rem 0; letter-spacing: 0.3px; }
#app-title .icon { margin-right: 0.6rem; }
[data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-of-type(1) { background: var(--bg-panel-1); border: 1px solid var(--border-1); border-radius: 14px; padding: 18px 16px; }
[data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-of-type(2) { background: var(--bg-panel-2); border: 1px solid var(--border-2); border-radius: 14px; padding: 18px 16px; }
.section-label { font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 0.5rem; letter-spacing: 0.2px; font-weight: 600; }
input[type="text"], textarea { background:#0b1220!important; color:var(--text-primary)!important; border:1px solid #223049!important; border-radius:10px!important; }
input[type="text"]::placeholder, textarea::placeholder { color:#94a3b8!important; }
[data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-of-type(1) div.stButton > button { background:var(--accent-blue); color:#fff; border-radius:10px; border:1px solid var(--accent-blue-hover); padding:0.6rem 1rem; font-weight:700; }
[data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-of-type(1) div.stButton > button:hover { background:var(--accent-blue-hover); }
[data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-of-type(2) div.stButton > button { background:var(--accent-green); color:#fff; border-radius:10px; border:1px solid var(--accent-green-hover); padding:0.6rem 1rem; font-weight:700; }
[data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-of-type(2) div.stButton > button:hover { background:var(--accent-green-hover); }
button[disabled] { background:#2f3542!important; color:#9aa3af!important; border-color:#3a4150!important; }
.response-area { background:var(--bg-response); border:1px solid #334155; border-radius:14px; padding:16px; margin-top:14px; color:var(--text-primary); }
.response-title { font-weight:800; margin-bottom:8px; color:#dbeafe; }
.disclaimer { font-size:0.8rem; color:#9ca3af; text-align:center; margin-top:26px; }
</style>
""", unsafe_allow_html=True)

# ================= Title =================
st.markdown('<div id="app-title"><span class="icon">🩺</span>Healthline Assistant</div>', unsafe_allow_html=True)

# ================= Session state =================
if "urls_ok" not in st.session_state: st.session_state.urls_ok = False
if "validated_urls" not in st.session_state: st.session_state.validated_urls = []
if "index_ready" not in st.session_state: st.session_state.index_ready = False
if "last_query" not in st.session_state: st.session_state.last_query = ""
if "show_response" not in st.session_state: st.session_state.show_response = False
if "last_answer" not in st.session_state: st.session_state.last_answer = ""
if "last_sources" not in st.session_state: st.session_state.last_sources = []
for i in range(10):
    st.session_state.setdefault(f"url_{i}", "")

# ================= URL helpers =================
def is_valid_healthline_prefix(u: str) -> bool:
    if not isinstance(u, str):
        return False
    u = u.strip()
    allowed = ("https://www.healthline.com", "www.healthline.com", "healthline.com")
    return any(u.startswith(p) for p in allowed)

def canonicalize_healthline(u: str) -> str | None:
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
    return urlunparse((scheme, netloc, path, "", "", ""))

# ================= Layout =================
left, right = st.columns([1.2, 2.0])

# ----- Left: URLs + Index -----
with left:
    st.markdown('<div class="section-label">Enter up to 10 Healthline URLs (one per line)</div>', unsafe_allow_html=True)
    for i in range(10):
        st.text_input(
            label="URL",
            key=f"url_{i}",
            placeholder="Paste a Healthline URL (e.g., https://www.healthline.com/health/...)",
            label_visibility="collapsed",
        )

    if st.button("Validate & Submit URLs"):
        raw_urls = [(st.session_state.get(f"url_{i}", "") or "").strip() for i in range(10)]
        raw_urls = [u for u in raw_urls if u]
        if not raw_urls:
            st.session_state.urls_ok = False
            st.session_state.index_ready = False
            st.session_state.show_response = False
            st.error("No URLs uploaded.")
        else:
            invalids = [u for u in raw_urls if not is_valid_healthline_prefix(u)]
            canonical_list, seen, duplicates = [], set(), []
            for u in raw_urls:
                c = canonicalize_healthline(u)
                if c is None:
                    if u not in invalids:
                        invalids.append(u)
                else:
                    if c in seen:
                        duplicates.append(u)
                    else:
                        seen.add(c)
                        canonical_list.append(c)
            if invalids:
                st.session_state.urls_ok = False
                st.session_state.index_ready = False
                st.session_state.show_response = False
                st.error("Invalid URL format. Allowed starts: https://www.healthline.com, www.healthline.com, healthline.com.")
                for bad in invalids:
                    st.warning(f"Rejected: {bad}")
            elif duplicates:
                st.session_state.urls_ok = False
                st.session_state.index_ready = False
                st.session_state.show_response = False
                st.error("Duplicate URLs detected. Please remove duplicates.")
                for dup in duplicates:
                    st.warning(f"Duplicate: {dup}")
            else:
                st.session_state.validated_urls = canonical_list[:10]
                with st.spinner("Indexing vectorstore (clearing old embeddings and creating new ones)..."):
                    result = build_index(st.session_state.validated_urls)
                if result.get("status") == "ok":
                    st.session_state.urls_ok = True
                    st.session_state.index_ready = True
                    st.success(f"Indexed {result.get('chunks_indexed', 0)} chunks.")
                else:
                    st.session_state.urls_ok = False
                    st.session_state.index_ready = False
                    errs = result.get("errors") or []
                    st.error("Failed to index content.")
                    for e in errs:
                        st.warning(e)

# ----- Right: Query + Response -----
with right:
    query = st.text_area("Query", placeholder="Enter your query", height=220)
    submit_query_disabled = (not st.session_state.get("urls_ok")) or (not st.session_state.get("index_ready")) or (not query.strip())
    clicked = st.button("Submit query", disabled=submit_query_disabled)

    if clicked and not submit_query_disabled:
        st.session_state.last_query = query.strip()
        with st.spinner("Retrieving and generating response..."):
            resp = answer_query_strict(st.session_state.last_query)
        st.session_state.last_answer = resp.get("answer", "")
        st.session_state.last_sources = resp.get("sources", [])
        st.session_state.show_response = True

    if st.session_state.get("show_response"):
        st.markdown('<div class="response-area"><div class="response-title">Response</div>', unsafe_allow_html=True)
        if st.session_state.last_answer:
            st.markdown(st.session_state.last_answer)
        else:
            st.markdown(f"_{FALLBACK}_")
        if st.session_state.last_sources:
            st.markdown('<div class="response-title">Sources</div>', unsafe_allow_html=True)
            for src in st.session_state.last_sources:
                st.markdown(f"- {src}")
        st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown('<div class="disclaimer">“All rights to the content in the provided URLs belong solely to Healthline Media LLC.”</div>', unsafe_allow_html=True)
