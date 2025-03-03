import re
from typing import List, Dict

def normalize_text(text: str) -> str:
    """
    Normalize text by lowering the case and stripping extra whitespace.
    You can add more normalization (e.g., remove punctuation) if needed.
    """
    return re.sub(r'[^\w\s]', '', text.lower().strip())