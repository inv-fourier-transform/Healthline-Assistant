# /core/chunker.py
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_docs(docs, chunk_size: int, chunk_overlap: int):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        separators=["\n\n", "\n", ".", " "],
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)
