from typing import List

def create_chunks(sentences: List[str], chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """
    Groups sentences into overlapping chunks of approx chunk_size words, respecting sentence boundaries.
    
    Args:
        sentences: List of detected sentences.
        chunk_size: Maximum word limit per chunk.
        chunk_overlap: Reusable word overlap between consecutive chunks.
    """
    if not sentences:
        return []
    
    chunks = []
    current_chunk_sentences = []
    current_word_count = 0
    
    # Store word count of each sentence
    sentence_word_counts = []
    for s in sentences:
        words = s.split()
        sentence_word_counts.append(len(words))
        
    i = 0
    n = len(sentences)
    
    while i < n:
        current_chunk_sentences = []
        current_word_count = 0
        
        # Build first chunk starting at i
        while i < n and current_word_count + sentence_word_counts[i] <= chunk_size:
            current_chunk_sentences.append(sentences[i])
            current_word_count += sentence_word_counts[i]
            i += 1
            
        # If the sentence itself is larger than chunk_size and current_chunk is empty, we must include it
        if not current_chunk_sentences and i < n:
            current_chunk_sentences.append(sentences[i])
            current_word_count += sentence_word_counts[i]
            i += 1
            
        # Put the chunk text together
        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))
            
        if i >= n:
            break
            
        # For the next starting index, we need to backtrack to implement overlap
        # Backtrack index so we have approximately chunk_overlap words of overlap
        overlap_words = 0
        backtrack_steps = 0
        k = i - 1
        while k >= 0 and overlap_words + sentence_word_counts[k] <= chunk_overlap:
            overlap_words += sentence_word_counts[k]
            backtrack_steps += 1
            k -= 1
            
        # Avoid infinite loops by ensuring we always advance at least 1 sentence
        # if backtrack_steps is equal to the amount we just processed
        if backtrack_steps > 0 and i - backtrack_steps < i:
            # We will start next chunk from i - backtrack_steps
            i = i - backtrack_steps
            
    # Edge case: if no chunks were created but there are sentence inputs
    if not chunks and sentences:
        chunks.append(" ".join(sentences))
        
    return chunks

def estimate_tokens(text: str) -> int:
    """
    Simple rule-of-thumb token estimator (approx 1.3 tokens per word for English)
    """
    if not text:
        return 0
    words = len(text.split())
    return int(words * 1.3)
