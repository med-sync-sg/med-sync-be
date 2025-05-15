import re
from typing import List, Dict
from symspellpy import SymSpell, Verbosity
import requests
import os

dictionary_path = os.path.join("app", "dictionaries", "en-80k.txt")

sym_spell = SymSpell(max_dictionary_edit_distance=4)
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

def normalize_text(text: str) -> str:
    """
    Normalize text by lowering the case and stripping extra whitespace.
    You can add more normalization (e.g., remove punctuation) if needed.
    """
    return re.sub(r'[^\w\s]', '', text.lower().strip())

def clean_transcription(text):
    """Enhanced cleaning: removes extra spaces, duplicate words, and noise."""
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'([.,!?])\1+', r'\1', text)  # Remove repeated punctuation
    text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text)  # Remove duplicate words
    
    # text = re.sub(r'\b[a-zA-Z]\b', '', text)  # Remove single-character words (optional)
    # text = re.sub(r'\d+', '', text)  # Remove numbers if they appear
    text = text.strip()
    return text

def correct_spelling(text):
    words = text.split()
    corrected_words = []
    
    for word in words:
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        if suggestions:
            corrected_words.append(suggestions[0].term)
        else:
            corrected_words.append(word)

    return ' '.join(corrected_words)

def extract_medical_terms(text):
    """
    Extract individual medical terms from a phrase.
    
    Args:
        text (str): Input medical phrase (e.g., "Acute acidosis" or "Anemia, megaloblastic")
        
    Returns:
        list: List of extracted individual terms
    """
    if not text or not isinstance(text, str):
        return []
    
    # Initialize list to hold all terms
    terms = []
    
    # 1. Extract content from parentheses
    parentheses_pattern = r'\(([^)]+)\)'
    parentheses_matches = re.findall(parentheses_pattern, text)
    
    for match in parentheses_matches:
        # Add parenthetical content as separate terms after processing
        inner_terms = re.split(r'[,;/&]|\band\b', match)
        for inner_term in inner_terms:
            inner_clean = clean_term(inner_term)
            if inner_clean:
                terms.append(inner_clean)
    
    # 2. Remove parenthetical content from the original text
    text_without_parentheses = re.sub(parentheses_pattern, '', text)
    
    # 3. Split the main text by common delimiters
    main_terms = re.split(r'[,;/&]|\band\b', text_without_parentheses)
    
    # 4. Clean and add each term
    for term in main_terms:
        cleaned_term = clean_term(term)
        if cleaned_term:
            terms.append(cleaned_term)
    
    # 5. For hyphenated terms, add both the full term and individual parts
    final_terms = []
    for term in terms:
        final_terms.append(term)
        
        # If term contains hyphens, add the individual parts
        if '-' in term and not term.startswith('-') and not term.endswith('-'):
            hyphen_parts = [clean_term(part) for part in term.split('-')]
            final_terms.extend([part for part in hyphen_parts if part])
    
    # 6. Remove duplicates while preserving order
    seen = set()
    unique_terms = [x for x in final_terms if not (x in seen or seen.add(x))]
    
    return unique_terms

def clean_term(term):
    """
    Clean a single term by removing unwanted punctuation and whitespace.
    
    Args:
        term (str): Term to clean
        
    Returns:
        str: Cleaned term
    """
    if not term:
        return ""
    
    # Remove unwanted punctuation but preserve hyphens
    cleaned = re.sub(r'[^\w\s-]', '', term)
    # Normalize whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned