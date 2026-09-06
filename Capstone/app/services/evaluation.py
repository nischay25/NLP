from rouge_score import rouge_scorer
from backend.app.schemas.evaluation import EvaluateResponse, RougeMetric

def calculate_rouge(generated_summary: str, reference_summary: str) -> EvaluateResponse:
    """
    Calculate ROUGE-1, ROUGE-2, and ROUGE-L between generated and reference summaries.
    """
    # Clean whitespace
    gen = " ".join(generated_summary.split())
    ref = " ".join(reference_summary.split())
    
    # Initialize rouge scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(ref, gen)
    
    r1 = scores['rouge1']
    r2 = scores['rouge2']
    rL = scores['rougeL']
    
    original_words = len(ref.split())
    generated_words = len(gen.split())
    
    compression_ratio = 0.0
    if original_words > 0:
        compression_ratio = float(generated_words) / float(original_words)
        
    return EvaluateResponse(
        rouge1=RougeMetric(
            precision=round(r1.precision, 4),
            recall=round(r1.recall, 4),
            fmeasure=round(r1.fmeasure, 4)
        ),
        rouge2=RougeMetric(
            precision=round(r2.precision, 4),
            recall=round(r2.recall, 4),
            fmeasure=round(r2.fmeasure, 4)
        ),
        rougeL=RougeMetric(
            precision=round(rL.precision, 4),
            recall=round(rL.recall, 4),
            fmeasure=round(rL.fmeasure, 4)
        ),
        original_length=original_words,
        generated_length=generated_words,
        compression_ratio=round(compression_ratio, 4)
    )
