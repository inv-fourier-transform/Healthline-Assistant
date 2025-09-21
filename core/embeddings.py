# /core/embeddings.py
from typing import List
from langchain_core.embeddings import Embeddings

# Preferred route
try:
    from langchain_huggingface.embeddings import HuggingFaceEmbeddings as LC_HFEmbeddings
    _HAS_LC_HF = True
except Exception:
    _HAS_LC_HF = False

# Community fallback
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings as LC_COMM_HFEmbeddings
    _HAS_COMM = True
except Exception:
    _HAS_COMM = False

# Direct ST import for robust fallback with trust_remote_code control
from sentence_transformers import SentenceTransformer

class _STWrapper(Embeddings):
    """Wrapper to use SentenceTransformer directly with LangChain."""
    def __init__(self, model: SentenceTransformer, normalize: bool = False):
        self.model = model
        self.normalize = normalize

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(
            texts,
            normalize_embeddings=self.normalize,
            convert_to_numpy=False,
            show_progress_bar=False,
        )

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(
            [text],
            normalize_embeddings=self.normalize,
            convert_to_numpy=False,
            show_progress_bar=False,
        )[0]

def get_embeddings(model_name: str, normalize: bool = False):
    """
    Create an Embeddings instance with trust_remote_code=True support for repos that need it.

    """
    # 1) Try the provider package
    if _HAS_LC_HF:
        try:
            return LC_HFEmbeddings(
                model_name=model_name,
                model_kwargs={"trust_remote_code": True},   # critical for GTE and similar repos
                encode_kwargs={"normalize_embeddings": normalize},
            )
        except Exception:
            pass

    # 2) Try the community provider (older import path)
    if _HAS_COMM:
        try:
            return LC_COMM_HFEmbeddings(
                model_name=model_name,
                model_kwargs={"trust_remote_code": True},
                encode_kwargs={"normalize_embeddings": normalize},
            )
        except Exception:
            pass

    # 3) Fallback option: load SentenceTransformer directly with trust_remote_code
    st_model = SentenceTransformer(model_name, trust_remote_code=True)
    return _STWrapper(st_model, normalize=normalize)
