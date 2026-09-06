import os
import time
import shutil
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query, Request
from typing import List, Optional
from backend.app.schemas.summarization import SummarizeRequest, SummarizeResponse, PreprocessRequest, PreprocessResponse
from backend.app.services.document_processor import validate_file, extract_text
from backend.app.services.preprocessing import clean_text, detect_sentences
from backend.app.services.chunking import create_chunks, estimate_tokens
from backend.app.services.summarization import summarize_text, summarize_multi_documents
from backend.app.utils.security import rate_limiter

router = APIRouter()

@router.post("", response_model=SummarizeResponse)
def summarize(payload: SummarizeRequest, request: Request):
    try:
        rate_limiter.check_rate_limit(request)
        if not payload.text.strip():
            raise HTTPException(status_code=400, detail="Input text cannot be empty")
            
        res = summarize_text(
            text=payload.text,
            max_length=payload.max_length,
            min_length=payload.min_length,
            chunk_size=payload.chunk_size,
            chunk_overlap=payload.chunk_overlap,
            temperature=payload.temperature,
            beam_size=payload.beam_size,
            length_penalty=payload.length_penalty
        )
        return res
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/preprocess", response_model=PreprocessResponse)
def preprocess_only(payload: PreprocessRequest, request: Request):
    try:
        rate_limiter.check_rate_limit(request)
        cleaned = clean_text(payload.text)
        sentences = detect_sentences(cleaned)
        chunks = create_chunks(sentences, chunk_size=payload.chunk_size, chunk_overlap=payload.chunk_overlap)
        
        chunk_word_counts = [len(c.split()) for c in chunks]
        total_words = len(cleaned.split())
        est_tokens = estimate_tokens(cleaned)
        
        return PreprocessResponse(
            cleaned_text=cleaned,
            detected_sentences=sentences,
            chunks=chunks,
            chunk_word_counts=chunk_word_counts,
            total_words=total_words,
            estimated_tokens=est_tokens
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/files")
def summarize_files(
    request: Request,
    files: List[UploadFile] = File(...),
    max_length: int = Form(150),
    min_length: int = Form(40),
    chunk_size: int = Form(500),
    chunk_overlap: int = Form(50),
    temperature: float = Form(1.0),
    beam_size: int = Form(4),
    length_penalty: float = Form(1.0)
):
    try:
        rate_limiter.check_rate_limit(request)
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded")
            
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        
        extracted_docs = []
        
        # Validate and process all files
        for uploaded_file in files:
            validate_file(uploaded_file)
            
            # Save file temporarily
            temp_file_path = os.path.join(temp_dir, uploaded_file.filename)
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(uploaded_file.file, buffer)
                
            try:
                # Extract text
                doc_text = extract_text(temp_file_path)
                if not doc_text.strip():
                    raise ValueError(f"File '{uploaded_file.filename}' is empty or has no readable text")
                extracted_docs.append((uploaded_file.filename, doc_text))
            finally:
                # Remove file after extraction
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                    
        # Apply correct pipeline: Single vs Multi document
        if len(extracted_docs) == 1:
            filename, text = extracted_docs[0]
            res = summarize_text(
                text=text,
                max_length=max_length,
                min_length=min_length,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                temperature=temperature,
                beam_size=beam_size,
                length_penalty=length_penalty,
                filename=filename
            )
            return res
        else:
            res = summarize_multi_documents(
                documents=extracted_docs,
                max_length=max_length,
                min_length=min_length,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                temperature=temperature,
                beam_size=beam_size,
                length_penalty=length_penalty
            )
            return res
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/long-text", response_model=SummarizeResponse)
def summarize_long_text(payload: SummarizeRequest, request: Request):
    """
    Explicit endpoint for long-text chunking & hierarchical T5 summarization.
    """
    return summarize(payload, request)

@router.post("/multi-document")
def summarize_multi_doc_alias(
    request: Request,
    files: List[UploadFile] = File(...),
    max_length: int = Form(150),
    min_length: int = Form(40),
    chunk_size: int = Form(500),
    chunk_overlap: int = Form(50),
    temperature: float = Form(1.0),
    beam_size: int = Form(4),
    length_penalty: float = Form(1.0)
):
    """
    Explicit endpoint for multi-document extraction, sentence redundancy removal, and unified T5 merging.
    """
    return summarize_files(
        request=request,
        files=files,
        max_length=max_length,
        min_length=min_length,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        temperature=temperature,
        beam_size=beam_size,
        length_penalty=length_penalty
    )

