# src/api/middleware/cors.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

def setup_cors(app: FastAPI):
    """
    Mounts CORS middleware on the FastAPI application instance.
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Adjust for production grade safety
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
