import os
import shutil
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query, Depends
from typing import Optional, List
from backend.app.services.model_service import model_service
from backend.app.services.training import start_training, get_training_status
from backend.app.utils.db import get_history, delete_history_item
from backend.app.utils.security import verify_api_key

router = APIRouter()

@router.get("/status")
def get_model_status():
    return model_service.get_status()

@router.post("/load")
def load_specified_model(
    checkpoint: str = Query(..., description="Checkpoint path or HF model id"),
    _ = Depends(verify_api_key)
):
    success = model_service.load_model(checkpoint)
    if not success:
        raise HTTPException(status_code=500, detail=f"Failed to load checkpoint '{checkpoint}'. Check logs.")
    return {"message": f"Successfully loaded checkpoint: {checkpoint}", "status": model_service.status}

@router.get("/history")
def get_history_list(query: Optional[str] = None):
    try:
        return get_history(search_query=query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.delete("/history/{item_id}")
def delete_history(item_id: int, _ = Depends(verify_api_key)):
    try:
        delete_history_item(item_id)
        return {"message": f"History item {item_id} deleted."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.post("/train")
def train_model(
    file: UploadFile = File(...),
    epochs: int = Form(3),
    batch_size: int = Form(4),
    learning_rate: float = Form(5e-5),
    max_input_length: int = Form(512),
    max_target_length: int = Form(150),
    _ = Depends(verify_api_key)
):
    # Save file temporarily
    temp_dir = "temp_datasets"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, file.filename)
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        success = start_training(
            dataset_path=temp_path,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_input_length=max_input_length,
            max_target_length=max_target_length
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="Training is already in progress.")
            
        return {"message": "Fine-tuning routine started in background", "dataset": file.filename}
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/train/status")
def train_status():
    return get_training_status()
