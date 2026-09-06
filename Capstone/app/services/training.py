import os
import time
import threading
import pandas as pd
import torch
from typing import Dict, Any, Optional, List
from backend.app.config import settings
from backend.app.services.model_service import model_service

# Global training state tracker
training_state = {
    "is_training": False,
    "current_epoch": 0,
    "total_epochs": 0,
    "progress_percent": 0.0,
    "logs": [],  # List of dicts: {"epoch": int, "step": int, "train_loss": float, "val_loss": float}
    "error": None,
    "completed": False,
    "saved_checkpoint": None
}

def get_training_status() -> Dict[str, Any]:
    return training_state

def run_training_loop(
    dataset_path: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_input_length: int,
    max_target_length: int
):
    global training_state
    try:
        # Load dataset
        ext = dataset_path.split(".")[-1].lower()
        if ext == "csv":
            df = pd.read_csv(dataset_path)
        elif ext == "json" or ext == "jsonl":
            df = pd.read_json(dataset_path, lines=(ext == "jsonl"))
        else:
            raise ValueError(f"Unsupported dataset format: .{ext}")

        # Find columns
        cols = df.columns.tolist()
        input_col = None
        target_col = None
        for col in cols:
            col_l = col.lower()
            if "text" in col_l or "input" in col_l or "document" in col_l or "article" in col_l:
                input_col = col
            if "summary" in col_l or "target" in col_l or "abstract" in col_l:
                target_col = col

        if not input_col:
            input_col = cols[0]
        if not target_col:
            target_col = cols[1] if len(cols) > 1 else cols[0]

        # Filter empty rows and convert to string
        df = df[[input_col, target_col]].dropna()
        df[input_col] = df[input_col].astype(str)
        df[target_col] = df[target_col].astype(str)
        
        dataset_size = len(df)
        if dataset_size == 0:
            raise ValueError("Dataset is empty.")
        
        training_state["logs"].append({
            "epoch": 0,
            "message": f"Successfully parsed dataset with {dataset_size} examples. Input: '{input_col}', Target: '{target_col}'"
        })

        # Set training device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model if not loaded
        if not model_service.model:
            model_service.load_model()
            
        model = model_service.model
        tokenizer = model_service.tokenizer

        if model is None or tokenizer is None:
            raise ValueError("T5 Model or Tokenizer could not be loaded.")

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        
        # Split into training and validation
        if dataset_size >= 2:
            train_df = df.sample(frac=0.8, random_state=42)
            if train_df.empty:
                train_df = df.iloc[:1]
            val_df = df.drop(train_df.index)
            if val_df.empty:
                val_df = df.iloc[-1:]
        else:
            train_df = df
            val_df = df
        
        train_examples = train_df.to_dict('records')
        val_examples = val_df.to_dict('records')
        
        training_state["logs"].append({
            "epoch": 0,
            "message": f"Split: {len(train_examples)} training assets, {len(val_examples)} validation assets"
        })

        # Real training epochs
        for epoch in range(1, epochs + 1):
            training_state["current_epoch"] = epoch
            
            # Training Phase
            model.train()
            total_train_loss = 0.0
            num_train_batches = 0
            
            # Shuffle training items
            import random
            random.seed(42 + epoch)
            random.shuffle(train_examples)
            
            for i in range(0, len(train_examples), batch_size):
                batch = train_examples[i:i+batch_size]
                inputs_text = ["summarize: " + str(b[input_col]) for b in batch]
                targets_text = [str(b[target_col]) for b in batch]
                
                inputs = tokenizer(inputs_text, max_length=max_input_length, truncation=True, padding=True, return_tensors="pt").to(device)
                targets = tokenizer(targets_text, max_length=max_target_length, truncation=True, padding=True, return_tensors="pt").to(device)
                
                labels = targets["input_ids"]
                labels[labels == tokenizer.pad_token_id] = -100
                
                optimizer.zero_grad()
                outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=labels)
                loss = outputs.loss
                
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
                num_train_batches += 1

            avg_train_loss = total_train_loss / num_train_batches if num_train_batches > 0 else 0.0
            
            # Validation Phase
            model.eval()
            total_val_loss = 0.0
            num_val_batches = 0
            
            with torch.no_grad():
                for i in range(0, len(val_examples), batch_size):
                    batch = val_examples[i:i+batch_size]
                    inputs_text = ["summarize: " + str(b[input_col]) for b in batch]
                    targets_text = [str(b[target_col]) for b in batch]
                    
                    inputs = tokenizer(inputs_text, max_length=max_input_length, truncation=True, padding=True, return_tensors="pt").to(device)
                    targets = tokenizer(targets_text, max_length=max_target_length, truncation=True, padding=True, return_tensors="pt").to(device)
                    
                    labels = targets["input_ids"]
                    labels[labels == tokenizer.pad_token_id] = -100
                    
                    outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=labels)
                    val_loss = outputs.loss.item()
                    
                    total_val_loss += val_loss
                    num_val_batches += 1
            
            avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0.0
            
            # Update state with real calculate results
            progress = (epoch / epochs) * 100
            training_state["progress_percent"] = round(progress, 1)
            
            epoch_log = {
                "epoch": epoch,
                "train_loss": round(avg_train_loss, 4),
                "val_loss": round(avg_val_loss, 4),
                "message": f"Epoch {epoch}/{epochs} completed. Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
            }
            training_state["logs"].append(epoch_log)
            print(f"Training log: {epoch_log['message']}")
            
        # Complete training, save checkpoint
        checkpoint_name = "models/fine_tuned/t5_finetuned"
        os.makedirs(checkpoint_name, exist_ok=True)
        if model is not None and tokenizer is not None:
            model.save_pretrained(checkpoint_name)
            tokenizer.save_pretrained(checkpoint_name)
            training_state["saved_checkpoint"] = checkpoint_name
            
        training_state["completed"] = True
        training_state["is_training"] = False
        training_state["logs"].append({
            "epoch": epochs,
            "message": f"Training completed successfully. Checkpoint saved under '{checkpoint_name}'"
        })
        print(f"Training completed successfully. Model saved to {checkpoint_name}.")
        
    except Exception as e:
        training_state["error"] = str(e)
        training_state["is_training"] = False
        training_state["logs"].append({
            "epoch": training_state["current_epoch"],
            "message": f"Training failed with error: {str(e)}"
        })
        print(f"Training failed: {str(e)}")

def start_training(
    dataset_path: str,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    max_input_length: int = 512,
    max_target_length: int = 150
) -> bool:
    global training_state
    if training_state["is_training"]:
        return False
        
    # Reset state
    training_state = {
        "is_training": True,
        "current_epoch": 0,
        "total_epochs": epochs,
        "progress_percent": 0.0,
        "logs": [{"epoch": 0, "message": "Starting T5 fine-tuning routine..."}],
        "error": None,
        "completed": False,
        "saved_checkpoint": None
    }
    
    # Run in background thread
    t = threading.Thread(
        target=run_training_loop,
        kwargs={
            "dataset_path": dataset_path,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_input_length": max_input_length,
            "max_target_length": max_target_length
        }
    )
    t.daemon = True
    t.start()
    return True
