"""
TruthLens Backend Service (FastAPI)
Main application entrypoint.
Orchestrates Computer Vision and NLP pipelines for deepfake and misinformation detection.
"""

import os
import sys
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Add repository root to Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.database import db_manager
from backend.routes import media, text, results, history

# Setup structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("TruthLens.Backend")

# Ensure required directories exist
STATIC_DIR = os.getenv("STATIC_DIR", os.path.join(CURRENT_DIR, "static"))
HEATMAP_DIR = os.getenv("HEATMAP_DIR", os.path.join(STATIC_DIR, "heatmaps"))
UPLOAD_DIR = os.getenv("UPLOAD_DIR", os.path.join(CURRENT_DIR, "uploads"))
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(HEATMAP_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown events for the application."""
    logger.info("TruthLens Backend initializing...")
    await db_manager.connect()
    yield
    logger.info("TruthLens Backend shutting down...")
    await db_manager.disconnect()


app = FastAPI(
    title="TruthLens API",
    description="AI-Powered Platform for Deepfake and Misinformation Detection (Capstone Project)",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for Streamlit frontend and local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory for Grad-CAM heatmaps and uploaded files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Include API route modules
app.include_router(media.router)
app.include_router(text.router)
app.include_router(results.router)
app.include_router(history.router)


@app.get("/", tags=["System"])
async def root():
    return {
        "service": "TruthLens Backend API",
        "version": "1.0.0",
        "status": "online",
        "database": "connected" if db_manager.is_connected else "in-memory-fallback",
        "endpoints": {
            "upload_media": "POST /upload-media",
            "analyze_text": "POST /analyze-text",
            "results": "GET /results/{scan_id}",
            "history": "GET /history/{user_id}",
            "docs": "/docs"
        }
    }


@app.get("/health", tags=["System"])
async def health():
    return {
        "status": "healthy",
        "database_connected": db_manager.is_connected
    }


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("backend.main:app", host=host, port=port, reload=True)
