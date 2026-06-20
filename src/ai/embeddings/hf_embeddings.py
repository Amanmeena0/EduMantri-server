# src/ai/embeddings/hf_embeddings.py
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.core.config import settings

def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Creates and returns a GoogleGenerativeAIEmbeddings instance.
    Utilizes models/embedding-001 from Gemini.
    """
    if not settings.GOOGLE_API_KEY:
        raise ValueError("Missing GOOGLE_API_KEY inside environment configuration.")
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=settings.GOOGLE_API_KEY
    )
