# src/ai/llms/gemini.py
from langchain_google_genai import ChatGoogleGenerativeAI
from src.core.config import settings

def get_llm(temperature: float = 0.2, model: str = "gemini-2.0-flash") -> ChatGoogleGenerativeAI:
    """
    Creates and returns a ChatGoogleGenerativeAI client instance.
    """
    if not settings.GOOGLE_API_KEY:
        raise ValueError("Missing GOOGLE_API_KEY inside environment configuration.")
    
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        top_p=0.9,
        google_api_key=settings.GOOGLE_API_KEY
    )
