from pydantic import BaseModel, Field
from typing import List, Optional

class SummarizeRequest(BaseModel):
    text: str = Field(..., description="The original document text to summarize")
    max_length: Optional[int] = Field(150, description="Max length of generated summary")
    min_length: Optional[int] = Field(40, description="Min length of generated summary")
    chunk_size: Optional[int] = Field(500, description="Pre-processing chunk size in words")
    chunk_overlap: Optional[int] = Field(50, description="Pre-processing overlap in words")
    temperature: Optional[float] = Field(1.0, description="Inference temperature")
    beam_size: Optional[int] = Field(4, description="Inference beam size")
    length_penalty: Optional[float] = Field(1.0, description="Length penalty")

class SummarizeResponse(BaseModel):
    summary: str
    original_word_count: int
    summary_word_count: int
    compression_ratio: float  # e.g., 0.12 (12%)
    processing_time: float
    model_used: str
    chunks: Optional[List[str]] = None
    cleaned_text: Optional[str] = None
    detected_sentences: Optional[List[str]] = None
    pipeline_details: Optional[dict] = None


class PreprocessRequest(BaseModel):
    text: str
    chunk_size: int = 500
    chunk_overlap: int = 50

class PreprocessResponse(BaseModel):
    cleaned_text: str
    detected_sentences: List[str]
    chunks: List[str]
    chunk_word_counts: List[int]
    total_words: int
    estimated_tokens: int
