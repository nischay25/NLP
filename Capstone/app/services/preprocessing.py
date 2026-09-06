import re
from typing import List

def clean_text(text: str) -> str:
    """
    Remove unnecessary whitespace/noise.
    """
    if not text:
        return ""
    # Replace multiple spaces with a single space
    cleaned = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing space
    cleaned = cleaned.strip()
    return cleaned

def detect_sentences(text: str) -> List[str]:
    """
    Sentence Boundary Detection.
    Splits text on periods, exclamation marks, question marks, ensuring they are followed by spaces,
    while avoiding splits on common abbreviations or single-letter initials.
    """
    if not text:
        return []
    
    # Split text keeping the punctuation + space segments
    elements = re.split(r'([.!?]\s+)', text)
    
    sentences = []
    current_sentence = []
    
    # Normalize abbreviation set for checking
    abbrev_dict = {
        "dr", "mr", "mrs", "ms", "corp", "inc", "ltd", "co", "e.g", "i.e", "vs", "prof", 
        "sr", "jr", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec"
    }
    
    for i, el in enumerate(elements):
        if i % 2 == 0:
            current_sentence.append(el)
        else:
            current_sentence.append(el)
            # Get the previous text segment and find its last word to verify boundaries
            prev_text = elements[i - 1].strip()
            if prev_text:
                last_word = prev_text.split()[-1].strip(".,()[]{}")
                last_word_lower = last_word.lower()
                
                # Check criteria to skip sentence segmentation:
                # 1. Word is in the known abbreviations list (case-insensitive)
                # 2. Word is a single uppercase initial (e.g. John A. Smith)
                # 3. Word contains only letters and dot components (e.g., U.S. or Ph.D.)
                is_abbrev = last_word_lower in abbrev_dict
                is_initial = len(last_word) == 1 and last_word.isupper()
                is_acronym = bool(re.match(r'^[A-Za-z]\.[A-Za-z]\.?$', last_word))
                
                if is_abbrev or is_initial or is_acronym:
                    continue
            
            # If not an abbreviation, finalize the sentence
            sentences.append("".join(current_sentence).strip())
            current_sentence = []
            
    # Append any trailing parts
    if current_sentence:
        remaining = "".join(current_sentence).strip()
        if remaining:
            sentences.append(remaining)
            
    return [s for s in sentences if s]


def remove_redundant_sentences(sentences: List[str], similarity_threshold: float = 0.65):
    """
    Detects and filters out redundant or duplicate sentences based on word-level Jaccard similarity.
    
    Returns:
        (unique_sentences, removed_logs)
    """
    if not sentences:
        return [], []

    unique_sentences = []
    seen_word_sets = []
    removed_logs = []

    for s in sentences:
        s_clean = clean_text(s)
        if not s_clean:
            continue
            
        # Extract word tokens (alphanumeric, lowercase)
        words = set(re.findall(r'\w+', s_clean.lower()))
        if not words:
            continue

        is_redundant = False
        matching_reason = ""

        for existing_s, existing_words in zip(unique_sentences, seen_word_sets):
            # Calculate Jaccard similarity
            intersection = len(words.intersection(existing_words))
            union = len(words.union(existing_words))
            similarity = float(intersection) / float(union) if union > 0 else 0.0

            if similarity >= similarity_threshold or s_clean.lower() == existing_s.lower():
                is_redundant = True
                matching_reason = f"Overlap ({similarity * 100:.1f}%) with: '{existing_s[:50]}...'"
                break

        if is_redundant:
            removed_logs.append(f"Removed sentence: '{s_clean}' -> Reason: {matching_reason}")
        else:
            unique_sentences.append(s_clean)
            seen_word_sets.append(words)

    return unique_sentences, removed_logs


