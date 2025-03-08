import spacy
import spacy.tokens
from typing import List, Dict, Any
from app.utils.text_utils import normalize_text
from app.schemas.section import TextCategoryEnum, SectionCreate
from app.db.iris_session import IrisDataStore
import numpy as np
from sentence_transformers import SentenceTransformer

iris_data_store = IrisDataStore()

def extract_keywords_descriptors(doc: spacy.tokens.Doc) -> list:
    """
    Given a spaCy Doc with NER annotations,
    extract for each entity (keyword):
    - The entity text and label.
    - Any adjectives describing it (using dependency relations).
    - Any numerical modifiers (quantities) related to it.
    
    Only processes spans marked with the 'is_medical_term' attribute.
    Returns a list of dictionaries.
    """
    results = []
    for ent in doc.ents:
        # Only process entities that came from the Aho-Corasick component.
        if not ent._.is_medical_term:
            continue

        adjectives = []
        quantities = []

        # Look at the children of the entity's root token.
        for child in ent.root.children:
            if child.pos_ == "ADJ":
                adjectives.append(child.text)
            if child.dep_ in ["nummod", "quantmod"] or child.like_num:
                quantities.append(child.text)
        
        # Optionally, look at a few tokens before the entity for adjectives.
        for token in doc[max(ent.start - 3, 0):ent.start]:
            if token.pos_ == "ADJ" and token.text not in adjectives:
                adjectives.append(token.text)
        
        results.append({
            "term": ent.text,
            "label": ent.label_,
            "adjectives": adjectives,
            "quantities": quantities
        })
    print(results)
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
      - "quantities": a list of quantities (e.g. ["3 meetings"])
      - "durations": a list of time words
    
    Returns:
      - final_keyword_dict (dict)
    """
    print(f"first: {kw1}")
    print(f"second: {kw2}")
    final_keyword_dict = {
        "term": None,
        "label": None,
        "adjectives": [],
        "quantities": []
    }
    final_keyword_dict["term"] = kw1["term"]
    final_keyword_dict["label"] = kw1["label"]
    
    assert kw1["term"] == kw2["term"]
    
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
            differing_adjectives.append(adj)
    for adj in adjectives2:
        if not any(are_semantically_similar(adj, other) for other in adjectives1):
            differing_adjectives.append(adj)
    
    final_keyword_dict["adjectives"] = list(set(differing_adjectives))
    print(f"final: {final_keyword_dict}")
    return final_keyword_dict

def build_section_create_objects(note_id: int, keyword_dicts: List[Dict]) -> List[SectionCreate]:
    """
    Group extracted keyword dictionaries by category and merge them to create
    SectionCreate objects. For each category (e.g., CHIEF_COMPLAINT, PATIENT_INFORMATION, OTHERS),
    the corresponding content is generated based on a template.
    
    Args:
        note_id (int): The note identifier.
        keyword_dicts (List[Dict]): List of keyword dictionaries extracted from the transcript.
        
    Returns:
        List[SectionCreate]: List of SectionCreate objects ready for database insertion.
    """    
    for kw in keyword_dicts:
        category = iris_data_store.classify_text_category(kw)
        selected_content_dictionary = iris_data_store.find_content_dictionary(keyword_dict=kw, category=category)
        print(selected_content_dictionary) 
        result_content = iris_data_store.merge_keyword_into_template(kw, selected_content_dictionary)
        print(result_content)
    sections = []
    
    
    
    return sections