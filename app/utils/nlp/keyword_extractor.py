import spacy
import spacy.tokens
from typing import List, Dict, Any
from app.utils.text_utils import normalize_text
import numpy as np
from sentence_transformers import SentenceTransformer

def extract_keywords_descriptors(doc: spacy.tokens.Doc) -> list:
    """
    Given a spaCy Doc with NER annotations, extract for each entity (keyword):
    - The entity text and label.
    - Any adjectives describing it (using dependency relations).
    - Any numerical modifiers (quantities) related to it.
    
    Returns a list of dictionaries.
    """
    results = []
    for ent in doc.ents:
        adjectives = []
        quantities = []
        
        # Look at the children of the entity's root token.
        for child in ent.root.children:
            # If the child is an adjective, add it.
            if child.pos_ == "ADJ":
                adjectives.append(child.text)
            # If the child is a numerical modifier or looks like a number, add it.
            if child.dep_ in ["nummod", "quantmod"] or child.like_num:
                quantities.append(child.text)
        
        # Optionally, check left siblings for adjectives that might not be direct children.
        # This can catch adjectives preceding the entity in cases where dependency
        # structure might not attach them directly.
        for token in doc[ent.start - 3:ent.start]:  # Look at a window before the entity.
            if token.pos_ == "ADJ" and token.text not in adjectives:
                adjectives.append(token.text)
                
        results.append({
            "term": ent.text,
            "label": ent.label_,
            "adjectives": adjectives,
            "quantities": quantities
        })
    return results

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute the cosine similarity between two vectors.
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def are_semantically_similar(adj1: str, adj2: str, model: SentenceTransformer, threshold: float = 0.95) -> bool:
    """
    Calculate semantic similarity for adjectives.
    In production, replace this with a proper semantic similarity check (e.g., via embeddings).
    For now, we consider them similar if they match exactly.
    """
    first_adj = model.encode(normalize_text(adj1), normalize_embeddings=True)
    second_adj = model.encode(normalize_text(adj2), normalize_embeddings=True)
    return cosine_similarity(first_adj, second_adj) >= threshold

def merge_keyword_dicts(kw1: Dict, kw2: Dict) -> Dict[str, Any]:
    """
    Compare two extracted keyword dictionaries. Direct matching is used for the 'term' and 'quantities' fields.
    Adjectives are compared using semantic similarity (placeholder implementation).
    Any adjectives or quantities that are different from each other are recorded in final_keywords.
    
    Each keyword dictionary is expected to have:
      - "term": e.g. "severe headache"
      - "label": e.g. "SYMPTOM"
      - "adjectives": a list of adjectives (e.g. ["severe", "persistent"])
      - "quantities": a list of quantities (e.g. ["3 days"])
    
    Returns:
      - final_keyword_dict (dict)
    """
    final_keyword_dict = {}
    final_keyword_dict["label"] = kw1["label"]
    
    # Directly compare "term"
    term1 = normalize_text(kw1.get("term", ""))
    term2 = normalize_text(kw2.get("term", ""))
    if term1 != term2:
        final_keyword_dict["term"] = [kw1.get("term", ""), kw2.get("term", "")]
        # If the main term doesn't match, we consider them inconsistent immediately.
        return {}
    
    # Compare "quantities" directly.
    quantities1 = [q for q in kw1.get("quantities", [])]
    quantities2 = [q for q in kw2.get("quantities", [])]
    if set(quantities1) != set(quantities2):
        # Keep differing quantities.
        final_keyword_dict["quantities"] = list(set(quantities1).symmetric_difference(set(quantities2)))
    
    # Compare adjectives semantically.
    adjectives1: List[str] = [normalize_text(adj) for adj in kw1.get("adjectives", [])]
    adjectives2: List[str] = [normalize_text(adj) for adj in kw2.get("adjectives", [])]
    
    differing_adjectives = []
    
    for adj in adjectives1:
        if not any(are_semantically_similar(adj, other) for other in adjectives2):
            consistent_adjectives = False
            differing_adjectives.append(adj)
    for adj in adjectives2:
        if not any(are_semantically_similar(adj, other) for other in adjectives1):
            consistent_adjectives = False
            differing_adjectives.append(adj)
    
    if not consistent_adjectives:
        final_keyword_dict["adjectives"] = list(set(differing_adjectives))
    
    return final_keyword_dict