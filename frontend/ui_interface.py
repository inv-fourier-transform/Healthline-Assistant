import re
import sys
from urllib.parse import urlparse, urlunparse
from pathlib import Path
import streamlit as st

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Backend imports
from core.indexer import build_index
from core.qa import answer_query

# ---------------- Page setup ----------------
st.set_page_config(page_title="Healthline Assistant", layout="wide")

# Dark theme - code snippet taken from ChatGPT
st.markdown("""
<style>
:root {
  --bg-primary: #0e1117;
  --bg-panel-1: #111827;
  --bg-panel-2: #0f172a;
  --bg-response: #1f2937;
  --border-1: #1f2937;
  --border-2: #243041;
  --text-primary: #e5e7eb;
  --text-secondary: #cbd5e1;
  --accent-blue: #3b82f6;
  --accent-blue-hover: #2563eb;
  --accent-green: #10b981;
  --accent-green-hover: #059669;
}
html, body, [class^="css"] {
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
  color: var(--text-primary);
}
[data-testid="stAppViewContainer"] {
  background: radial-gradient(1200px 600px at 60% -20%, #121621 10%, var(--bg-primary) 60%);
}
#app-title {
  font-size: 2.2rem;
  font-weight: 800;
  color: #f8fafc;
  text-align: center;
  margin: 0 0 1rem 0;
  letter-spacing: 0.3px;
}
#app-title .icon { margin-right: 0.6rem; }
[data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-of-type(1) {
  background: var(--bg-panel-1);
  border: 1px solid var(--border-1);
  border-radius: 14px;
  padding: 18px 16px;
}
[data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-of-type(2) {
  background: var(--bg-panel-2);
  border: 1px solid var(--border-2);
  border-radius: 14px;
  padding: 18px 16px;
}
.section-label {
  font-size: 0.9rem;
  color: var(--text-secondary);
  margin-bottom: 0.5rem;
  letter-spacing: 0.2px;
  font-weight: 600;
}
input[type="text"], textarea {
  background: #0b1220 !important;
  color: var(--text-primary) !important;
  border: 1px solid #223049 !important;
  border-radius: 10px !important;
}
input[type="text"]::placeholder, textarea::placeholder { color: #94a3b8 !important; }
[data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-of-type(1) div.stButton > button {
  background: var(--accent-blue); color: #ffffff; border-radius: 10px;
  border: 1px solid var(--accent-blue-hover); padding: 0.6rem 1rem; font-weight: 700;
}
[data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-of-type(1) div.stButton > button:hover { background: var(--accent-blue-hover); }
[data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-of-type(2) div.stButton > button {
  background: var(--accent-green); color: #ffffff; border-radius: 10px;
  border: 1px solid var(--accent-green-hover); padding: 0.6rem 1rem; font-weight: 700;
}
[data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-of-type(2) div.stButton > button:hover { background: var(--accent-green-hover); }
button[disabled] { background: #2f3542 !important; color: #9aa3af !important; border-color: #3a4150 !important; }
.response-area {
  background: var(--bg-response); border: 1px solid #334155; border-radius: 14px;
  padding: 16px; margin-top: 14px; color: var(--text-primary);
}
.response-title { font-weight: 800; margin-bottom: 8px; color: #dbeafe; }
.disclaimer { font-size: 0.8rem; color: #9ca3af; text-align: center; margin-top: 26px; }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div id="app-title"><span class="icon">🩺</span>Healthline Assistant</div>', unsafe_allow_html=True)

# ---------------- Session state: initialize first ----------------
if "urls_ok" not in st.session_state:
    st.session_state.urls_ok = False
if "validated_urls" not in st.session_state:
    st.session_state.validated_urls = []
if "index_ready" not in st.session_state:
    st.session_state.index_ready = False
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "show_response" not in st.session_state:
    st.session_state.show_response = False
if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

# Initialize fixed 10 URL slots
for i in range(10):
    k = f"url_{i}"
    if k not in st.session_state:
        st.session_state[k] = ""

# ---------------- Helpers (URL validation) ----------------
def is_valid_healthline_prefix(u: str) -> bool:
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

# ---------------- Layout ----------------
left, right = st.columns([1.2, 2.0])

# ================= LHS: URLs + Index =================
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
        # Gather non-empty inputs
        raw_urls = []
        for i in range(10):
            val = (st.session_state.get(f"url_{i}", "") or "").strip()
            if val:
                raw_urls.append(val)

        if not raw_urls:
            st.session_state.urls_ok = False
            st.session_state.index_ready = False
            st.session_state.show_response = False
            st.error("No URLs uploaded by the user.")
        else:
            # Validate prefixes & detect duplicates across formats
            invalids = [u for u in raw_urls if not is_valid_healthline_prefix(u)]
            canonical_list = []
            seen = set()
            duplicates = []
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
                st.error(
                    "Invalid URL format detected. A valid URL must start with one of: "
                    "https://www.healthline.com, www.healthline.com, or healthline.com."
                )
                for bad in invalids:
                    st.warning(f"Rejected: {bad}")
            elif duplicates:
                st.session_state.urls_ok = False
                st.session_state.index_ready = False
                st.session_state.show_response = False
                st.error("Duplicate URLs detected across different formats. Please remove duplicates.")
                for dup in duplicates:
                    st.warning(f"Duplicate: {dup}")
            else:
                # Validated set. now build index (reset & embed from scratch)
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

# ================= RHS: Query + Response =================
with right:
    query = st.text_area(
        "Query",
        placeholder="Enter your query",
        height=220,
    )
    submit_query_disabled = (not st.session_state.get("urls_ok")) or (not st.session_state.get("index_ready")) or (not query.strip())
    clicked = st.button("Submit query", disabled=submit_query_disabled)

    if clicked and not submit_query_disabled:
        st.session_state.last_query = query.strip()
        with st.spinner("Retrieving and generating response..."):
            resp = answer_query(st.session_state.last_query)
        st.session_state.last_answer = resp.get("answer", "")
        st.session_state.last_sources = resp.get("sources", [])
        st.session_state.show_response = True

    if st.session_state.get("show_response"):
        st.markdown('<div class="response-area"><div class="response-title">Response</div>', unsafe_allow_html=True)
        # Answer text
        if st.session_state.last_answer:
            st.markdown(st.session_state.last_answer)
        else:
            st.markdown("_No answer generated from the provided sources._")

        # Sources list
        if st.session_state.last_sources:
            st.markdown('<div class="response-title">Sources</div>', unsafe_allow_html=True)
            for src in st.session_state.last_sources:
                st.markdown(f"- {src}")
        st.markdown("</div>", unsafe_allow_html=True)

# Copyright disclaimer
st.markdown('<div class="disclaimer">“All rights to the content in the provided URLs belong solely to Healthline Media LLC.”</div>', unsafe_allow_html=True)
