# /core/vectorstore.py
import shutil
from pathlib import Path
from langchain_chroma import Chroma

def reset_persist_dir(persist_directory: str):
    p = Path(persist_directory)
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)
    p.mkdir(parents=True, exist_ok=True)

def create_chroma(embedding_function, collection_name: str, persist_directory: str):
    # Create a fresh Chroma instance against the given directory
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=persist_directory,
    )
