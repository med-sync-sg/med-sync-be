from typing import List, Dict, Any, Tuple, Optional
from sqlalchemy.orm import Session
import numpy as np
from sentence_transformers import SentenceTransformer
import copy
import json

DEFAULT_MODEL = SentenceTransformer("all-minilm-l6-v2")
# Similarity functions
def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def dot_product_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return np.dot(vec1, vec2)

def embed_text(text: str, model: SentenceTransformer=DEFAULT_MODEL) -> np.ndarray:
    """Generate embedding vector for text"""
    return model.encode(text, normalize_embeddings=True)

def embed_list(input_list: list, model: SentenceTransformer=DEFAULT_MODEL) -> List[Any]:
    """Generate embeddings for a list of texts"""
    embeddings = model.encode(input_list, normalize_embeddings=True)
    return [emb.tolist() for emb in embeddings]

def embed_dictionary(data: Any, model: SentenceTransformer=DEFAULT_MODEL) -> np.ndarray:
    """Generate embedding for a dictionary (serialized to JSON)"""
    return model.encode(json.dumps(data), normalize_embeddings=True)


def clear_template_values(template: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively clear all string values in the template, replacing them with an empty string,
    while preserving the structure.
    """
    new_template = {}
    for key, value in template.items():
        if isinstance(value, str):
            new_template[key] = ""
        elif isinstance(value, dict):
            new_template[key] = clear_template_values(value)
        else:
            new_template[key] = value
    return new_template

def iter_candidate_keys(template: Dict[str, Any]) -> List[Tuple[List[str], str]]:
    """
    Yield candidate key paths and their combined strings from the template.
    """
    candidates = []
    for t_key, t_val in template.items():
        if isinstance(t_val, str):
            candidates.append(([t_key], f"{t_key}: {t_val}"))
        elif isinstance(t_val, dict):
            for sub_key, sub_val in t_val.items():
                if isinstance(sub_val, str):
                    candidates.append(([t_key, sub_key], f"{sub_key}: {sub_val}"))
    return candidates

def set_value_at_path(template: Dict[str, Any], path: List[str], value: Any) -> Dict[str, Any]:
    """
    Set the given value in the template at the specified path.
    """
    current = template
    for key in path[:-1]:
        current = current.setdefault(key, {})
    current[path[-1]] = value
    return template

def get_value_at_path(template: Dict[str, Any], path: List[str]) -> Any:
    """
    Retrieve the value at the specified path in the template.
    """
    current = template
    for key in path:
        current = current.get(key)
        if current is None:
            return None
    return current

def append_value_at_path(template: Dict[str, Any], path: List[str], new_value: Any) -> Dict[str, Any]:
    """
    Append new_value to the field specified by path in the template.
    """
    current = template
    for key in path[:-1]:
        current = current.setdefault(key, {})
    existing = current.get(path[-1], "")
    if existing:
        current[path[-1]] = f"{existing}, {new_value}"
    else:
        current[path[-1]] = str(new_value)
    return template

def merge_flat_keywords_into_template(keywords: Dict[str, Any],
                                      template: Dict[str, Any],
                                      threshold: float = 0.5) -> Dict[str, Any]:
    """
    Merge a keyword dictionary into a template structure.
    Supports the updated keyword dictionary format with semantic_type, cui, and tui.
    
    Args:
        keywords: Dictionary with medical term information
        template: Template dictionary to populate
        threshold: Minimum similarity threshold for field matching
        
    Returns:
        Merged template with populated values
    """
    similarity_template = copy.deepcopy(template)
    working_template = clear_template_values(copy.deepcopy(template))
    extras = {}

    # Build candidate keys (one level deep)
    candidates = []
    for t_key, t_val in similarity_template.items():
        if isinstance(t_val, str):
            candidates.append(([t_key], f"{t_key}: {t_val}"))
        elif isinstance(t_val, dict):
            for sub_key, sub_val in t_val.items():
                if isinstance(sub_val, str):
                    candidates.append(([t_key, sub_key], f"{sub_key}: {sub_val}"))
    
    # Process special fields first - term, semantic_type, cui, tui
    if "term" in keywords and isinstance(keywords["term"], str):
        term = keywords["term"]
        working_template["term"] = term
    
    # Store UMLS metadata
    if "semantic_type" in keywords:
        working_template["semantic_type"] = keywords["semantic_type"]
    if "cui" in keywords:
        working_template["cui"] = keywords["cui"]
    if "tui" in keywords:
        working_template["tui"] = keywords["tui"]
    
    # Process the other fields
    for f_key, f_value in keywords.items():
        # Skip already processed fields
        if f_key in ["term", "semantic_type", "cui", "tui", "label"]:
            continue

        if isinstance(f_value, list):
            for element in f_value:
                element_str = str(element)
                feature_candidate = f"{f_key}: {element_str}"
                best_sim = -1.0
                best_path = []
                
                # Find the best matching field in the template
                for path, candidate in candidates:
                    sim = cosine_similarity(embed_text(feature_candidate), embed_text(candidate))
                    if sim > best_sim:
                        best_sim = sim
                        best_path = path
                
                # Add to the template if a good match is found
                if best_path and best_sim >= threshold:
                    current_value = get_value_at_path(working_template, best_path)
                    if current_value:
                        working_template = append_value_at_path(working_template, best_path, element_str)
                    else:
                        working_template = set_value_at_path(working_template, best_path, element_str)
                else:
                    # Store in extras if no good match
                    extras.setdefault(f_key, []).append(element_str)
        else:
            # Handle non-list values
            f_val_str = str(f_value)
            feature_candidate = f"{f_key}: {f_val_str}"
            best_sim = -1.0
            best_path = []
            
            for path, candidate in candidates:
                sim = cosine_similarity(embed_text(feature_candidate), embed_text(candidate))
                if sim > best_sim:
                    best_sim = sim
                    best_path = path
            
            if best_path and best_sim >= threshold:
                working_template = set_value_at_path(working_template, best_path, f_val_str)
            else:
                extras[f_key] = f_value

    # Add any extra fields that didn't match the template
    if extras:
        if "additional_content" in working_template and isinstance(working_template["additional_content"], dict):
            working_template["additional_content"].update(extras)
        else:
            working_template["additional_content"] = extras
    
    return working_template