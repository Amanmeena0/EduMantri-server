# src/core/config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY", "")
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")
    PERSIST_DIR: str = os.getenv("PERSIST_DIR", "./faiss_index")
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "dgft_collection")
    EMBEDDINGS_MODEL: str = os.getenv("EMBEDDINGS_MODEL", "BAAI/bge-small-en")

settings = Settings()
