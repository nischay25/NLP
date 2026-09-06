from fastapi import APIRouter
from backend.app.services.model_service import model_service

router = APIRouter()

@router.get("")
@router.get("/health")
def health_check():
    status_info = model_service.get_status()
    return {
        "status": "healthy",
        "model": status_info["model_name"],
        "model_status": status_info["status"],
        "device": status_info["device"]
    }
