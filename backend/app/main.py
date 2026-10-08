"""
CardioCode – FastAPI Application Entry Point
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.v1.api import api_router
from app.core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load ML model on startup so first request isn't slow."""
    from app.services.ml_engine import _load
    _load()
    yield


app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "CardioCode – AI-powered cardiac diagnostic triage platform. "
        "Backed by Firebase Firestore + scikit-learn ML with SHAP explainability."
    ),
    version=settings.VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")


@app.get("/", tags=["Health"])
def health_check():
    return {
        "app":     settings.APP_NAME,
        "version": settings.VERSION,
        "status":  "running",
        "docs":    "/docs",
    }
