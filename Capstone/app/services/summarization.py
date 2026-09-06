import time
from typing import List, Dict, Any, Tuple
from backend.app.services.preprocessing import clean_text, detect_sentences, remove_redundant_sentences
from backend.app.services.chunking import create_chunks, estimate_tokens
from backend.app.services.model_service import model_service
from backend.app.utils.db import add_history

def count_tokens(text: str) -> int:
    """
    Counts tokens referencing the active model tokenizer.
    """
    if model_service.tokenizer:
        try:
            return len(model_service.tokenizer.encode(text))
        except Exception:
            pass
    return estimate_tokens(text)

def summarize_hierarchical(
    chunk_summaries: List[str],
    max_length: int = 155,
    min_length: int = 40,
    temperature: float = 1.0,
    beam_size: int = 4,
    length_penalty: float = 1.0,
    max_tokens: int = 400
) -> Tuple[str, List[str]]:
    """
    Summarize a collection of chunk summaries in batches if they exceed the model context window.
    """
    current_summaries = list(chunk_summaries)
    layer = 1
    logs = []
    
    while True:
        combined_text = " ".join(current_summaries)
        combined_tokens = count_tokens(combined_text)
        
        logs.append(f"Level {layer}: {len(current_summaries)} summaries, total token count = {combined_tokens}")
        
        # If combined chunks fit inside one single pass (approx 400 tokens / 300 words), we can finalize
        if combined_tokens <= max_tokens or len(current_summaries) == 1:
            break
            
        # Standard batch grouping
        batches = []
        current_batch = []
        current_batch_tokens = 0
        
        for summary in current_summaries:
            summary_tokens = count_tokens(summary)
            if current_batch_tokens + summary_tokens > max_tokens and current_batch:
                batches.append(" ".join(current_batch))
                current_batch = [summary]
                current_batch_tokens = summary_tokens
            else:
                current_batch.append(summary)
                current_batch_tokens += summary_tokens
                
        if current_batch:
            batches.append(" ".join(current_batch))
            
        logs.append(f"Partitioned into {len(batches)} batches for Level {layer} reduction.")
        
        next_summaries = []
        for idx, batch_content in enumerate(batches):
            b_tokens = count_tokens(batch_content)
            batch_words = len(batch_content.split())
            adjusted_min = min(min_length, max(5, batch_words // 3))
            adjusted_max = min(max_length, max(adjusted_min + 5, batch_words))
            
            batch_summary = model_service.summarize_chunk(
                batch_content,
                max_length=adjusted_max,
                min_length=adjusted_min,
                temperature=temperature,
                beam_size=beam_size,
                length_penalty=length_penalty
            )
            next_summaries.append(batch_summary)
            logs.append(f"Level {layer} - Batch {idx+1} (tokens={b_tokens}) summary: {batch_summary}")
            
        current_summaries = next_summaries
        layer += 1
        
    # Final pass to merge all sub-summaries into a coherent result
    final_context = " ".join(current_summaries)
    final_tokens = count_tokens(final_context)
    final_context_words = len(final_context.split())
    
    adjusted_min = min(min_length, max(5, final_context_words // 3))
    adjusted_max = min(max_length, max(adjusted_min + 5, final_context_words))
    
    logs.append(f"Final pipeline summarization pass on combined context of {final_tokens} tokens.")
    
    final_summary = model_service.summarize_chunk(
        final_context,
        max_length=adjusted_max,
        min_length=adjusted_min,
        temperature=temperature,
        beam_size=beam_size,
        length_penalty=length_penalty
    )
    
    return final_summary, logs

def summarize_text(
    text: str,
    max_length: int = 150,
    min_length: int = 40,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    temperature: float = 1.0,
    beam_size: int = 4,
    length_penalty: float = 1.0,
    filename: str = "Pasted Text"
) -> Dict[str, Any]:
    """
    Detailed flow mapping to academic modules:
    Module 1: Preprocessing & Tokenization
    Module 3: Long-Text Chunking & Merging Pipeline
    """
    start_time = time.time()
    
    # 1. Clean Text
    cleaned = clean_text(text)
    if not cleaned:
        return {
            "summary": "Empty input text.",
            "original_word_count": 0,
            "summary_word_count": 0,
            "compression_ratio": 0.0,
            "processing_time": 0.0,
            "model_used": model_service.checkpoint,
            "chunks": [],
            "cleaned_text": "",
            "detected_sentences": [],
            "pipeline_details": None
        }
        
    # 2. Sentence Boundaries
    sentences = detect_sentences(cleaned)
    
    # 3. Create Chunks
    chunks = create_chunks(sentences, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # Original token count
    original_token_count = count_tokens(text)
    number_of_chunks = len(chunks)
    
    # 4. Summarize each chunk
    chunk_summaries = []
    chunk_token_counts = []
    for chunk in chunks:
        chunk_token_counts.append(count_tokens(chunk))
        
        # Scale min/max parameters dynamically for chunk boundary lengths
        chunk_words = len(chunk.split())
        adjusted_min = min(min_length, max(5, chunk_words // 3))
        adjusted_max = min(max_length, max(adjusted_min + 5, chunk_words))
        
        c_summary = model_service.summarize_chunk(
            chunk,
            max_length=adjusted_max,
            min_length=adjusted_min,
            temperature=temperature,
            beam_size=beam_size,
            length_penalty=length_penalty
        )
        chunk_summaries.append(c_summary)
        
    # 5. Merge summaries
    merged_summary = " ".join(chunk_summaries)
    
    # 6. Final T5 pass to make summary coherent
    intermediate_logs = []
    if len(chunks) > 1:
        final_summary, intermediate_logs = summarize_hierarchical(
            chunk_summaries,
            max_length=max_length,
            min_length=min_length,
            temperature=temperature,
            beam_size=beam_size,
            length_penalty=length_penalty
        )
    else:
        intermediate_logs = ["Single chunk input; skipping hierarchical batch summarization."]
        final_summary = chunk_summaries[0]
        
    processing_time = round(time.time() - start_time, 2)
    original_word_count = len(text.split())
    summary_word_count = len(final_summary.split())
    
    compression_ratio = 0.0
    if original_word_count > 0:
        compression_ratio = round(float(summary_word_count) / float(original_word_count), 4)
        
    # Add to SQLite database history
    add_history(
        filename=filename,
        input_word_count=original_word_count,
        summary=final_summary,
        summary_word_count=summary_word_count,
        processing_time=processing_time,
        model=model_service.checkpoint
    )
    
    # Generate system debugging prints (Backend Console Logging)
    print("\n================== T5 PIPELINE LOGS ==================")
    print(f"Original Token Count: {original_token_count}")
    print(f"Number of Chunks: {number_of_chunks}")
    for idx, (t_count, chunk_sum) in enumerate(zip(chunk_token_counts, chunk_summaries)):
        print(f"  -> Chunk {idx + 1} | Tokens: {t_count} | Summary: {chunk_sum}")
    print(f"Merged Summary: {merged_summary}")
    print("Hierarchical Batch Iteration Logs:")
    for line in intermediate_logs:
         print(f"  [Pipeline Log] {line}")
    print(f"Coherent Final Summary: {final_summary}")
    print("======================================================\n")
    
    # Format viva logs exactly as requested
    viva_logs = []
    viva_logs.append(f"Original tokens: {original_token_count}")
    viva_logs.append(f"Chunks created: {number_of_chunks}")
    viva_logs.append("")
    for idx, (t_count, chunk_sum) in enumerate(zip(chunk_token_counts, chunk_summaries)):
        w_size = len(chunk_sum.split())
        viva_logs.append(f"Chunk {idx + 1}: {t_count} tokens → summary {w_size} words")
    viva_logs.append("")
    merged_tokens = count_tokens(merged_summary)
    viva_logs.append(f"Merged summaries: {merged_tokens} tokens")
    viva_logs.append("")
    viva_logs.append(f"Final summary: {summary_word_count} words")
    
    pipeline_details = {
        "original_token_count": original_token_count,
        "number_of_chunks": number_of_chunks,
        "chunk_token_counts": chunk_token_counts,
        "chunk_summaries": chunk_summaries,
        "merged_summary": merged_summary,
        "intermediate_logs": intermediate_logs,
        "viva_logs": viva_logs,
        "final_summary": final_summary
    }
    
    return {
        "summary": final_summary,
        "original_word_count": original_word_count,
        "summary_word_count": summary_word_count,
        "compression_ratio": compression_ratio,
        "processing_time": processing_time,
        "model_used": model_service.checkpoint,
        "chunks": chunks,
        "cleaned_text": cleaned,
        "detected_sentences": sentences,
        "pipeline_details": pipeline_details
    }


def summarize_multi_documents(
    documents: List[Tuple[str, str]],  # List of (filename, text)
    max_length: int = 150,
    min_length: int = 40,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    temperature: float = 1.0,
    beam_size: int = 4,
    length_penalty: float = 1.0
) -> Dict[str, Any]:
    """
    Multi-Document Summarization Pipeline:
    1. Text extraction & document preprocessing per file
    2. Document-level & chunk-level T5 summarization
    3. Cross-document sentence boundary detection & redundancy filtering
    4. Hierarchical summary merging into a unified coherent summary
    """
    start_time = time.time()
    
    doc_details = []
    all_summary_sentences = []
    total_original_words = 0
    all_chunks = []
    all_cleaned_texts = []
    
    for filename, text in documents:
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
        
        doc_summary = res['summary']
        doc_details.append({
            "filename": filename,
            "original_word_count": res['original_word_count'],
            "summary_word_count": res['summary_word_count'],
            "summary": doc_summary,
            "chunks": res.get('chunks', [])
        })
        
        total_original_words += res['original_word_count']
        all_chunks.extend(res.get('chunks', []))
        all_cleaned_texts.append(f"--- Document: {filename} ---\n" + res.get('cleaned_text', ''))
        
        # Extract sentences from each document summary
        doc_sentences = detect_sentences(doc_summary)
        all_summary_sentences.extend(doc_sentences)
        
    # Apply sentence-level redundancy elimination across combined summaries
    unique_sentences, redundancy_logs = remove_redundant_sentences(all_summary_sentences, similarity_threshold=0.65)
    
    deduplicated_combined_text = " ".join(unique_sentences)
    if not deduplicated_combined_text.strip():
        deduplicated_combined_text = " ".join(all_summary_sentences)
        
    # Perform final merging pass over deduplicated summaries
    final_summary, intermediate_logs = summarize_hierarchical(
        unique_sentences if unique_sentences else all_summary_sentences,
        max_length=max_length,
        min_length=min_length,
        temperature=temperature,
        beam_size=beam_size,
        length_penalty=length_penalty
    ) if len(unique_sentences) > 1 else (
        deduplicated_combined_text,
        ["Single deduplicated segment pass."]
    )

    processing_time = round(time.time() - start_time, 2)
    summary_word_count = len(final_summary.split())
    
    compression_ratio = 0.0
    if total_original_words > 0:
        compression_ratio = round(float(summary_word_count) / float(total_original_words), 4)
        
    # Build viva logs for multi-document workflow
    viva_logs = [
        f"Multi-document processing: {len(documents)} files combined",
        f"Total original words: {total_original_words}",
        f"Extracted sentences across summaries: {len(all_summary_sentences)}",
        f"Unique sentences after redundancy removal: {len(unique_sentences)}",
        f"Sentences removed as redundant: {len(redundancy_logs)}",
        f"Final summary length: {summary_word_count} words"
    ]
    
    filenames_str = ", ".join([f[0] for f in documents])
    add_history(
        filename=f"Multi-Document ({filenames_str})",
        input_word_count=total_original_words,
        summary=final_summary,
        summary_word_count=summary_word_count,
        processing_time=processing_time,
        model=model_service.checkpoint
    )
    
    pipeline_details = {
        "documents_count": len(documents),
        "total_original_words": total_original_words,
        "doc_details": doc_details,
        "redundancy_logs": redundancy_logs,
        "intermediate_logs": intermediate_logs,
        "viva_logs": viva_logs,
        "final_summary": final_summary
    }
    
    return {
        "summary": final_summary,
        "original_word_count": total_original_words,
        "summary_word_count": summary_word_count,
        "compression_ratio": compression_ratio,
        "processing_time": processing_time,
        "model_used": model_service.checkpoint,
        "documents_count": len(documents),
        "chunks": all_chunks,
        "cleaned_text": "\n\n".join(all_cleaned_texts),
        "detected_sentences": all_summary_sentences,
        "pipeline_details": pipeline_details
    }

