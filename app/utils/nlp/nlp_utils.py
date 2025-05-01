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

def merge_flat_keywords_into_template(feature_dict: Dict[str, Any],
                                        template: Dict[str, Any],
                                        threshold: float = 0.5) -> Dict[str, Any]:
    """
    Merge a flat feature dictionary into a nested content template.
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
    
    for f_key, f_value in feature_dict.items():
        if f_key == "label":
            continue

        if isinstance(f_value, list):
            for element in f_value:
                element_str = str(element)
                feature_candidate = f"{f_key}: {element_str}"
                best_sim = -1.0
                best_path = []
                for path, candidate in candidates:
                    sim = cosine_similarity(embed_text(feature_candidate), embed_text(candidate))
                    if sim > best_sim:
                        best_sim = sim
                        best_path = path
                if best_path:
                    current_value = get_value_at_path(working_template, best_path)
                    if current_value:
                        working_template = append_value_at_path(working_template, best_path, element_str)
                    else:
                        working_template = set_value_at_path(working_template, best_path, element_str)
                else:
                    extras.setdefault(f_key, []).append(element_str)
        else:
            if f_key == "term" and isinstance(f_value, str):
                if working_template.get("Main Symptom") is not None and f_value not in ["symptoms", "feverishness", "painful"]:
                    best_path = ["Main Symptom", "name"]
                    working_template = set_value_at_path(working_template, best_path, f_value)
            else:
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

    if extras:
        if "additional_content" in working_template and isinstance(working_template["additional_content"], dict):
            working_template["additional_content"].update(extras)
        else:
            working_template["additional_content"] = extras
            
    print("Final Working Template: ", working_template)
    return working_template