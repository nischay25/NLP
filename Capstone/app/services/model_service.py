import time
import torch
import os
from typing import Dict, Any, Optional
from transformers import T5ForConditionalGeneration, T5Tokenizer
from backend.app.config import settings

class ModelService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.status = "Not Loaded"
        self.checkpoint = settings.MODEL_CHECKPOINT
        self.device = settings.DEVICE
        
        # Determine device
        if not self.device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        print(f"ModelService initialized with device: {self.device}")

    def load_model(self, checkpoint: Optional[str] = None) -> bool:
        """
        Loads the T5 model and tokenizer.
        """
        if checkpoint:
            self.checkpoint = checkpoint
        else:
            self.checkpoint = settings.MODEL_CHECKPOINT
            
        self.status = "Loading"
        try:
            print(f"Loading tokenizer & T5 model from checkpoint: {self.checkpoint}...")
            # Load tokenizer
            # For T5, we use T5Tokenizer
            self.tokenizer = T5Tokenizer.from_pretrained(self.checkpoint)
            
            # Load model
            self.model = T5ForConditionalGeneration.from_pretrained(self.checkpoint)
            self.model.to(self.device)
            
            self.status = "Ready"
            print(f"T5 Model loaded successfully on device: {self.device}")
            return True
        except Exception as e:
            self.status = f"Error: {str(e)}"
            print(f"Failed to load T5 model: {str(e)}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """
        Returns info about model status.
        """
        # If not loaded yet, let's load it or return status
        return {
            "model_name": self.checkpoint,
            "status": self.status,
            "device": self.device,
            "framework": "Hugging Face Transformers / PyTorch",
            "is_gpu": self.device == "cuda"
        }

    def summarize_chunk(
        self, 
        text: str,
        max_length: int = 150,
        min_length: int = 40,
        temperature: float = 1.0,
        beam_size: int = 4,
        length_penalty: float = 1.0
    ) -> str:
        """
        Summarizes a single chunk of text.
        """
        if self.status != "Ready":
            # Direct lazy load
            success = self.load_model()
            if not success:
                raise ValueError(f"Model is not loaded. Current status: {self.status}")

        # T5 prompt prefix is required: "summarize: "
        input_text = "summarize: " + text
        
        # Tokenize
        inputs = self.tokenizer(
            input_text, 
            return_tensors="pt", 
            max_length=512,  # T5 standard max length limit
            truncation=True
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generation config
        gen_args = {
            "max_length": max_length,
            "min_length": min_length,
            "num_beams": beam_size,
            "length_penalty": length_penalty,
            "no_repeat_ngram_size": 2,
            "early_stopping": True
        }
        
        # Adjust temperature (only uses sampling if temperature != 1.0 or do_sample is True)
        if temperature != 1.0:
            gen_args["do_sample"] = True
            gen_args["temperature"] = temperature
            # num_beams should be 1 if we're sampling or we must handle it differently
            # Typically, beam search doesn't pair well with pure temperature sample.
            # We'll set num_beams = 1 if temperature is custom
            if temperature > 0.0 and temperature < 1.0 or temperature > 1.0:
                gen_args["num_beams"] = 1
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_args)
            
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary

# Singleton instance
model_service = ModelService()
