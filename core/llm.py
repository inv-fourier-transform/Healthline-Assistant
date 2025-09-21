# /core/llm.py
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from core.config_loader import load_config

load_dotenv()

def get_llm():
    cfg = load_config()
    model = cfg["llm"]["groq_model"]
    temperature = cfg["llm"]["temperature"]
    max_tokens = cfg["llm"]["max_tokens"]

    # GROQ_API_KEY must be available in .env
    return ChatGroq(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
