import re
from typing import List, Dict
from symspellpy import SymSpell, Verbosity
import requests
import os

dictionary_path = "frequency_dictionary_en_82_765.txt"
if not os.path.exists(dictionary_path):
    url = "https://raw.githubusercontent.com/wolfgarbe/SymSpell/master/src/SymSpell/frequency_dictionary_en_82_765.txt"
    response = requests.get(url)
    with open(dictionary_path, "wb") as f:
        f.write(response.content)

sym_spell = SymSpell(max_dictionary_edit_distance=2)
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