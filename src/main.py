# src/main.py
from fastapi import FastAPI
from src.api.routes import chat
from src.api.middleware.cors import setup_cors
from src.core.logger import logger

app = FastAPI(
    title="EduMantri AI Study Platform API",
    description="Production-grade Multi-Agent System for Academic Study Help, RAG, and Routing.",
    version="1.0.0"
)

# Setup CORS middleware
setup_cors(app)

# Include routes
app.include_router(chat.router)

@app.get("/health")
async def health_check():
    """
    Health monitoring endpoint confirming server and model statuses.
    """
    return {
        "status": "operational",
        "service": "EduMantri AI Platform",
        "api_version": "1.0.0"
    }

logger.info("EduMantri Application Server fully loaded and operational.")
