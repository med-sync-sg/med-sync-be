import re
import json
import unicodedata
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union


# =============================================================================
# Text Preprocessing Functions
# =============================================================================

def normalize_text(text: str, 
                  lowercase: bool = True,
                  remove_punctuation: bool = True,
                  normalize_whitespace: bool = True,
                  expand_contractions: bool = True,
                  remove_disfluencies: bool = False,
                  remove_speaker_labels: bool = True) -> str:
    """
    Normalize transcription text for comparison
    
    Args:
        text: Input text to normalize
        lowercase: Convert to lowercase
        remove_punctuation: Remove punctuation marks
        normalize_whitespace: Normalize whitespace to single spaces
        expand_contractions: Expand contractions (don't -> do not)
        remove_disfluencies: Remove common disfluencies (um, uh, etc.)
        remove_speaker_labels: Remove speaker labels (Doctor:, Patient:, etc.)
        
    Returns:
        Normalized text string
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Unicode normalization
    text = unicodedata.normalize('NFKD', text)
    
    # Remove speaker labels first (before lowercase conversion for better matching)
    if remove_speaker_labels:
        text = _remove_speaker_labels(text)
    
    # Convert to lowercase
    if lowercase:
        text = text.lower()
    
    # Expand contractions
    if expand_contractions:
        text = _expand_contractions(text)
    
    # Remove disfluencies
    if remove_disfluencies:
        text = _remove_disfluencies(text)
    
    # Remove punctuation
    if remove_punctuation:
        # Keep apostrophes in contractions, remove other punctuation
        text = re.sub(r"[^\w\s']", "", text)
        # Remove standalone apostrophes
        text = re.sub(r"\s'\s|\s'$|^'\s", " ", text)
    
    # Normalize whitespace
    if normalize_whitespace:
        text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def _expand_contractions(text: str) -> str:
    """Expand common English contractions"""
    contractions = {
        "don't": "do not", "won't": "will not", "can't": "cannot", "n't": " not",
        "'ll": " will", "'ve": " have", "'re": " are", "'d": " would", "'m": " am",
        "'s": " is", "let's": "let us", "that's": "that is", "what's": "what is",
        "where's": "where is", "when's": "when is", "why's": "why is",
        "how's": "how is", "there's": "there is", "here's": "here is"
    }
    
    for contraction, expansion in contractions.items():
        pattern = r'\b' + re.escape(contraction) + r'\b'
        text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
    
    return text


def _remove_disfluencies(text: str) -> str:
    """Remove common disfluencies and filler words"""
    disfluencies = [
        r'\bum+\b', r'\buh+\b', r'\ber+\b', r'\bah+\b', r'\bmm+\b', r'\bhm+\b',
        r'\bhuh\b', r'\byou know\b', r'\blike\b(?=\s)', r'\bwell\b(?=\s)',
        r'\bso\b(?=\s)', r'\b(ha)+\b'
    ]
    
    for pattern in disfluencies:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    return text


def _remove_speaker_labels(text: str) -> str:
    """Remove speaker labels from transcripts"""
    # Common speaker label patterns
    speaker_patterns = [
        # Standard medical roles
        r'^Doctor:\s*',
        r'^Patient:\s*',
        r'^Physician:\s*',
        r'^Nurse:\s*',
        r'^Clinician:\s*',
        r'^Provider:\s*',
        
        # Variations with titles
        r'^Dr\.?\s+[A-Za-z]+:\s*',
        r'^Doctor\s+[A-Za-z]+:\s*',
        r'^Nurse\s+[A-Za-z]+:\s*',
        
        # Generic speaker labels
        r'^Speaker\s*\d*:\s*',
        r'^Person\s*\d*:\s*',
        r'^Voice\s*\d*:\s*',
        
        # All caps variations
        r'^DOCTOR:\s*',
        r'^PATIENT:\s*',
        r'^PHYSICIAN:\s*',
        r'^NURSE:\s*',
        
        # With brackets or other delimiters
        r'^\[Doctor\]:\s*',
        r'^\[Patient\]:\s*',
        r'^\(Doctor\):\s*',
        r'^\(Patient\):\s*',
        
        # Mid-sentence speaker changes (less common but possible)
        r'\nDoctor:\s*',
        r'\nPatient:\s*',
        r'\nPhysician:\s*',
        r'\nNurse:\s*',
    ]
    
    # Apply all patterns
    for pattern in speaker_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Clean up any remaining line breaks or extra spaces at the beginning
    text = re.sub(r'^\s*\n+\s*', '', text)
    text = re.sub(r'\n+', ' ', text)  # Convert remaining newlines to spaces
    
    return text


def preprocess_transcript(text: str, has_speakers: bool=True, strict_mode: bool = False) -> str:
    """
    Standard preprocessing for transcription comparison
    
    Args:
        text: Input transcription text
        strict_mode: If True, apply more aggressive normalization
        
    Returns:
        Preprocessed text ready for comparison
    """
    return normalize_text(
        text,
        lowercase=True,
        remove_punctuation=True,
        normalize_whitespace=True,
        expand_contractions=True,
        remove_disfluencies=strict_mode,
        remove_speaker_labels=has_speakers
    )


# =============================================================================
# Transcription Result Processing Functions
# =============================================================================

def extract_text_from_result(result: Any) -> str:
    """
    Extract text from various transcription result formats
    
    Args:
        result: Transcription result (string, dict, or object with .text attribute)
        
    Returns:
        Extracted text string
    """
    if result is None:
        return ""
    
    if isinstance(result, str):
        return result
    
    if isinstance(result, dict):
        return result.get("text", "")
    
    if hasattr(result, 'text'):
        return str(result.text)
    
    try:
        return str(result)
    except:
        return ""


def merge_streaming_results(segments: List[Dict[str, Any]], 
                          overlap_threshold: float = 0.1,
                          confidence_threshold: float = 0.0) -> Dict[str, Any]:
    """
    Merge streaming transcription segments into a single result
    
    Args:
        segments: List of transcription segments with text, timing, and confidence
        overlap_threshold: Maximum acceptable overlap between segments (seconds)
        confidence_threshold: Minimum confidence to include segment
        
    Returns:
        Merged transcription result with full text and metadata
    """
    if not segments:
        return {
            "text": "", "confidence": 0.0, "start_time": 0.0, "end_time": 0.0,
            "segment_count": 0, "words": []
        }
    
    # Filter by confidence and sort by start time
    valid_segments = [s for s in segments if s.get("confidence", 0.0) >= confidence_threshold]
    if not valid_segments:
        return {"text": "", "confidence": 0.0, "start_time": 0.0, "end_time": 0.0, 
                "segment_count": 0, "words": []}
    
    sorted_segments = sorted(valid_segments, key=lambda x: x.get("start_time", 0))
    
    # Resolve overlaps
    # merged_segments = _resolve_overlaps(sorted_segments, overlap_threshold)
    merged_segments = sorted_segments
    
    # Combine results
    full_text = " ".join(s["text"].strip() for s in merged_segments if s["text"].strip())
    confidences = [s["confidence"] for s in merged_segments if "confidence" in s]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    
    start_time = min(s.get("start_time", 0) for s in merged_segments)
    end_time = max(s.get("end_time", 0) for s in merged_segments)
    
    all_words = []
    for segment in merged_segments:
        if "words" in segment and segment["words"]:
            all_words.extend(segment["words"])
    
    return {
        "text": full_text,
        "confidence": avg_confidence,
        "start_time": start_time,
        "end_time": end_time,
        "duration": end_time - start_time,
        "segment_count": len(merged_segments),
        "words": all_words,
        "segments": merged_segments
    }


def _resolve_overlaps(segments: List[Dict[str, Any]], 
                     overlap_threshold: float) -> List[Dict[str, Any]]:
    """Resolve overlapping segments by merging or choosing best one"""
    if len(segments) <= 1:
        return segments
    
    merged = []
    current = segments[0].copy()
    
    for next_segment in segments[1:]:
        overlap = current.get("end_time", 0) - next_segment.get("start_time", 0)
        
        if overlap > overlap_threshold:
            current = _merge_two_segments(current, next_segment)
        else:
            merged.append(current)
            current = next_segment.copy()
    
    merged.append(current)
    return merged


def _merge_two_segments(seg1: Dict[str, Any], seg2: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two overlapping segments"""
    text1, text2 = seg1.get("text", "").strip(), seg2.get("text", "").strip()
    combined_text = f"{text1} {text2}".strip()
    
    # Weighted confidence by text length
    conf1, conf2 = seg1.get("confidence", 0.0), seg2.get("confidence", 0.0)
    len1, len2 = len(text1.split()), len(text2.split())
    total_len = len1 + len2
    
    weighted_confidence = (conf1 * len1 + conf2 * len2) / total_len if total_len > 0 else (conf1 + conf2) / 2
    
    return {
        "text": combined_text,
        "confidence": weighted_confidence,
        "start_time": min(seg1.get("start_time", 0), seg2.get("start_time", 0)),
        "end_time": max(seg1.get("end_time", 0), seg2.get("end_time", 0)),
        "words": seg1.get("words", []) + seg2.get("words", [])
    }


# =============================================================================
# Reference Transcript Loading Functions
# =============================================================================

def load_reference_file(file_path: Path, encoding: str = 'utf-8', 
                       format: str = 'auto', clean_text: bool = True) -> str:
    """
    Load reference transcription from file
    
    Args:
        file_path: Path to reference transcription file
        encoding: File encoding
        format: File format ('txt', 'json', 'auto')
        clean_text: Whether to apply reference transcript cleaning
        
    Returns:
        Reference transcription text
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Reference file not found: {file_path}")
    
    if format == 'auto':
        format = file_path.suffix.lower().lstrip('.')
    
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            if format == 'json':
                data = json.load(f)
                text = _extract_text_from_json(data)
            else:
                text = f.read()
        
        # Apply cleaning if requested
        if clean_text:
            text = normalize_text(text)
        
        return text
        
    except Exception as e:
        raise ValueError(f"Failed to load reference file {file_path}: {str(e)}")


def _extract_text_from_json(json_data: Dict[str, Any]) -> str:
    """Extract text from JSON reference file"""
    if isinstance(json_data, dict):
        # Look for common field names
        for field in ['text', 'transcription', 'transcript', 'content']:
            if field in json_data:
                return str(json_data[field])
        
        # Look for segments/chunks
        if 'segments' in json_data:
            segments = json_data['segments']
            if isinstance(segments, list):
                texts = []
                for segment in segments:
                    if isinstance(segment, dict) and 'text' in segment:
                        texts.append(segment['text'])
                    elif isinstance(segment, str):
                        texts.append(segment)
                return " ".join(texts)
    
    return str(json_data)


def create_reference_mapping(reference_dir: Path, audio_files: List[Path], 
                           clean_text: bool = True) -> Dict[str, str]:
    """
    Create mapping between audio files and their reference transcriptions
    
    Args:
        reference_dir: Directory containing reference transcription files
        audio_files: List of audio file paths
        clean_text: Whether to apply reference transcript cleaning
        
    Returns:
        Dictionary mapping audio filename to reference text
    """
    mapping = {}
    reference_extensions = ['.txt', '.json', '.ref']
    
    for audio_file in audio_files:
        audio_stem = audio_file.stem
        reference_file = None
        
        for ext in reference_extensions:
            candidate = reference_dir / f"{audio_stem}{ext}"
            if candidate.exists():
                reference_file = candidate
                break
        
        if reference_file:
            try:
                reference_text = load_reference_file(reference_file, clean_text=clean_text)
                mapping[audio_file.name] = reference_text
            except Exception as e:
                print(f"Warning: Could not load reference for {audio_file.name}: {e}")
        else:
            print(f"Warning: No reference file found for {audio_file.name}")
    
    return mapping


# =============================================================================
# Word Error Rate Calculation Functions
# =============================================================================

def calculate_wer(reference: str, hypothesis: str, normalize: bool = True) -> Dict[str, Any]:
    """
    Calculate Word Error Rate between reference and hypothesis using Levenshtein distance
    
    Args:
        reference: Reference (ground truth) transcription
        hypothesis: Hypothesis (predicted) transcription
        normalize: Whether to preprocess texts before comparison
        
    Returns:
        Dictionary with WER metrics and detailed analysis
    """
    # Preprocess texts if requested
    if normalize:
        reference = preprocess_transcript(reference)
        hypothesis = preprocess_transcript(hypothesis, has_speakers=False)
    
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    # Calculate edit operations using Levenshtein distance
    operations, alignment = _calculate_edit_distance(ref_words, hyp_words)
    
    # Count error types
    substitutions = operations.count('S')
    deletions = operations.count('D')
    insertions = operations.count('I')
    correct = operations.count('C')
    
    # Calculate metrics
    total_ref_words = len(ref_words)
    total_errors = substitutions + deletions + insertions
    
    wer = (total_errors / total_ref_words * 100) if total_ref_words > 0 else 0.0
    accuracy = (correct / total_ref_words * 100) if total_ref_words > 0 else 0.0
    
    return {
        "wer": wer,
        "accuracy": accuracy,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "correct": correct,
        "total_errors": total_errors,
        "reference_words": total_ref_words,
        "hypothesis_words": len(hyp_words),
        "reference_text": reference,
        "hypothesis_text": hypothesis,
        "alignment": alignment,
        "operations": operations
    }


def _calculate_edit_distance(ref_words: List[str], 
                           hyp_words: List[str]) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Calculate edit distance with operation tracking using dynamic programming
    
    Returns:
        Tuple of (operations list, alignment pairs)
    """
    m, n = len(ref_words), len(hyp_words)
    
    # DP table for edit distance and operation tracking
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    ops = [['' for _ in range(n + 1)] for _ in range(m + 1)]
    
    # Initialize
    for i in range(m + 1):
        dp[i][0] = i
        ops[i][0] = 'D'
    for j in range(n + 1):
        dp[0][j] = j
        ops[0][j] = 'I'
    ops[0][0] = 'C'
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i-1].lower() == hyp_words[j-1].lower():
                dp[i][j] = dp[i-1][j-1]
                ops[i][j] = 'C'
            else:
                costs = [
                    (dp[i-1][j-1] + 1, 'S'),  # Substitution
                    (dp[i-1][j] + 1, 'D'),    # Deletion
                    (dp[i][j-1] + 1, 'I')     # Insertion
                ]
                min_cost, operation = min(costs, key=lambda x: x[0])
                dp[i][j] = min_cost
                ops[i][j] = operation
    
    # Backtrack
    operations, alignment = [], []
    i, j = m, n
    
    while i > 0 or j > 0:
        op = ops[i][j]
        
        if op == 'C':
            operations.append('C')
            alignment.append((ref_words[i-1], hyp_words[j-1]))
            i -= 1
            j -= 1
        elif op == 'S':
            operations.append('S')
            alignment.append((ref_words[i-1], hyp_words[j-1]))
            i -= 1
            j -= 1
        elif op == 'D':
            operations.append('D')
            alignment.append((ref_words[i-1], '***'))
            i -= 1
        elif op == 'I':
            operations.append('I')
            alignment.append(('***', hyp_words[j-1]))
            j -= 1
    
    operations.reverse()
    alignment.reverse()
    
    return operations, alignment


def calculate_batch_wer(reference_texts: Dict[str, str], 
                       hypothesis_texts: Dict[str, str]) -> Dict[str, Any]:
    """
    Calculate WER for multiple transcription pairs
    
    Args:
        reference_texts: Dictionary mapping identifiers to reference texts
        hypothesis_texts: Dictionary mapping identifiers to hypothesis texts
        
    Returns:
        Dictionary with overall WER statistics and per-file results
    """
    results = {}
    all_wers = []
    total_substitutions = total_deletions = total_insertions = total_correct = total_ref_words = 0
    
    for identifier in reference_texts:
        if identifier in hypothesis_texts:
            wer_result = calculate_wer(reference_texts[identifier], hypothesis_texts[identifier])
            results[identifier] = wer_result
            
            all_wers.append(wer_result["wer"])
            total_substitutions += wer_result["substitutions"]
            total_deletions += wer_result["deletions"]
            total_insertions += wer_result["insertions"]
            total_correct += wer_result["correct"]
            total_ref_words += wer_result["reference_words"]
    
    # Calculate overall statistics
    if all_wers:
        avg_wer = sum(all_wers) / len(all_wers)
        total_errors = total_substitutions + total_deletions + total_insertions
        overall_wer = (total_errors / total_ref_words * 100) if total_ref_words > 0 else 0.0
        overall_accuracy = (total_correct / total_ref_words * 100) if total_ref_words > 0 else 0.0
    else:
        avg_wer = overall_wer = overall_accuracy = 0.0
    
    return {
        "overall_wer": overall_wer,
        "average_wer": avg_wer,
        "overall_accuracy": overall_accuracy,
        "total_files": len(results),
        "total_substitutions": total_substitutions,
        "total_deletions": total_deletions,
        "total_insertions": total_insertions,
        "total_correct": total_correct,
        "total_reference_words": total_ref_words,
        "per_file_results": results
    }


# =============================================================================
# High-Level Evaluation Functions
# =============================================================================

def evaluate_transcription(hypothesis_result: Any,
                          reference_text: str,
                          merge_segments: bool = False) -> Dict[str, Any]:
    """
    Evaluate a single transcription result against reference
    
    Args:
        hypothesis_result: Transcription result (various formats supported)
        reference_text: Reference transcription text
        merge_segments: Whether to merge streaming segments if applicable
        
    Returns:
        Comprehensive evaluation results
    """
    # Extract and process hypothesis text
    if merge_segments and isinstance(hypothesis_result, list):
        merged_result = merge_streaming_results(hypothesis_result)
        hypothesis_text = merged_result["text"]
    else:
        hypothesis_text = extract_text_from_result(hypothesis_result)
    
    # Calculate WER
    wer_result = calculate_wer(reference_text, hypothesis_text)
    
    # Add additional metrics
    wer_result.update({
        "evaluation_timestamp": time.time(),
        "hypothesis_length_chars": len(hypothesis_text),
        "reference_length_chars": len(reference_text),
        "length_ratio": len(hypothesis_text) / len(reference_text) if reference_text else 0.0
    })
    
    return wer_result


def evaluate_batch_transcriptions(hypothesis_results: Dict[str, Any],
                                 reference_dir: Path,
                                 audio_files: List[Path]) -> Dict[str, Any]:
    """
    Evaluate multiple transcriptions against references
    
    Args:
        hypothesis_results: Dictionary mapping identifiers to transcription results
        reference_dir: Directory containing reference transcription files
        audio_files: List of audio files (used to find references)
        
    Returns:
        Batch evaluation results
    """
    # Load reference texts
    reference_mapping = create_reference_mapping(reference_dir, audio_files)
    
    # Extract hypothesis texts
    hypothesis_texts = {
        identifier: extract_text_from_result(result) 
        for identifier, result in hypothesis_results.items()
    }
    
    # Calculate batch WER
    return calculate_batch_wer(reference_mapping, hypothesis_texts)