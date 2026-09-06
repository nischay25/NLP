import os
from fastapi import APIRouter, HTTPException, Request
from backend.app.schemas.evaluation import EvaluateRequest, EvaluateResponse, CompareEvaluateRequest, CompareEvaluateResponse
from backend.app.services.evaluation import calculate_rouge
from backend.app.services.summarization import summarize_text
from backend.app.services.model_service import model_service
from backend.app.utils.security import rate_limiter

router = APIRouter()

@router.post("", response_model=EvaluateResponse)
def evaluate_summary(payload: EvaluateRequest, request: Request):
    try:
        rate_limiter.check_rate_limit(request)
        if not payload.generated_summary.strip():
            raise HTTPException(status_code=400, detail="Generated summary cannot be empty")
        if not payload.reference_summary.strip():
            raise HTTPException(status_code=400, detail="Reference summary cannot be empty")
            
        result = calculate_rouge(payload.generated_summary, payload.reference_summary)
        return result
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/compare", response_model=CompareEvaluateResponse)
def compare_models(payload: CompareEvaluateRequest, request: Request):
    try:
        rate_limiter.check_rate_limit(request)
        if not payload.text.strip():
            raise HTTPException(status_code=400, detail="Input text cannot be empty")
        if not payload.reference_summary.strip():
            raise HTTPException(status_code=400, detail="Reference summary cannot be empty")
            
        # Back up active checkpoint
        original_checkpoint = model_service.checkpoint
        
        # 1. Base T5 summarization
        model_service.load_model("t5-small")
        base_res = summarize_text(
            text=payload.text,
            max_length=payload.max_length,
            min_length=payload.min_length
        )
        base_summary = base_res["summary"]
        base_eval = calculate_rouge(base_summary, payload.reference_summary)
        
        # 2. Fine-Tuned T5 summarization (if trained check exists)
        fine_tuned_checkpoint = "models/fine_tuned/t5_finetuned"
        fine_tuned_available = os.path.exists(os.path.join(fine_tuned_checkpoint, "config.json"))
        
        fine_tuned_summary = ""
        fine_tuned_eval = None
        
        if fine_tuned_available:
            try:
                model_service.load_model(fine_tuned_checkpoint)
                ft_res = summarize_text(
                    text=payload.text,
                    max_length=payload.max_length,
                    min_length=payload.min_length
                )
                fine_tuned_summary = ft_res["summary"]
                fine_tuned_eval = calculate_rouge(fine_tuned_summary, payload.reference_summary)
            except Exception as ft_err:
                print(f"Error executing fine-tuned model evaluation: {ft_err}")
                fine_tuned_available = False
                
        # Restore original checkpoint
        model_service.load_model(original_checkpoint)
        
        return CompareEvaluateResponse(
            base_t5_eval=base_eval,
            fine_tuned_t5_eval=fine_tuned_eval,
            base_summary=base_summary,
            fine_tuned_summary=fine_tuned_summary,
            fine_tuned_available=fine_tuned_available
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
