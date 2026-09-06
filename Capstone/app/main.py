import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.config import settings
from backend.app.api.routes_summarize import router as summarize_router
from backend.app.api.routes_evaluate import router as evaluate_router
from backend.app.api.routes_model import router as model_router
from backend.app.api.routes_health import router as health_router
from backend.app.utils.db import init_db
from backend.app.services.model_service import model_service

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Automatic Text Summarization Using a Transformer-Based T5 Model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Set CORS origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for development & Docker
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
init_db()

# Include API endpoints
app.include_router(health_router, prefix=f"{settings.API_V1_STR}/health", tags=["Health"])
app.include_router(summarize_router, prefix=f"{settings.API_V1_STR}/summarize", tags=["Summarize"])
app.include_router(evaluate_router, prefix=f"{settings.API_V1_STR}/evaluate", tags=["Evaluation"])
app.include_router(model_router, prefix=f"{settings.API_V1_STR}/model", tags=["Model Config & Training"])

@app.on_event("startup")
def startup_event():
    import threading
    print("FastAPI Backend started. Spawning background thread to pre-load T5 model weights...")
    # Load T5 weights in the background so status goes to "Ready" quickly without blocking uvicorn port binding
    threading.Thread(target=model_service.load_model, daemon=True).start()


@app.get("/")
def index():
    return {
        "app": "T5Summarizer",
        "description": "Automatic Text Summarization using Transformer-Based T5",
        "docs_url": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
