from typing import List, Dict, Any, Tuple, Optional
from sqlalchemy.orm import Session
import numpy as np
from sentence_transformers import SentenceTransformer
import copy
import json
import copy
from datetime import datetime

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

def merge_keywords_into_template(keywords: Dict[str, Any],
                                template_fields: list,
                                template_id: str,
                                threshold: float = 0.5) -> Dict[str, Any]:
    """
    Merge medical keyword dictionary into field values based on template fields.
    
    Args:
        keywords: Dictionary with medical term information (term, modifiers, semantic types, etc.)
        template_fields: List of TemplateFieldRead objects for the section template
        template_id: ID of the SectionTemplate being used
        threshold: Minimum similarity threshold for field matching
        
    Returns:
        Dictionary with populated template values and field_values dictionary
    """
    
    # Create working template
    working_template = {
        "template_id": template_id,
        "field_values": {},  # Will store the field values
        "additional_content": {}  # For unmatched attributes
    }
    
    # Create field candidates for matching
    field_candidates = []
    for field in template_fields:
        field_id = field["id"]
        field_name = field["name"]
        field_desc = field["description"] or ""
        field_text = f"{field_name}: {field_desc}"
        
        field_candidates.append({
            "id": field_id,
            "name": field_name,
            "text": field_text,
            "data_type": field["data_type"] or "any",
            "required": field["required"]
        })
    
    # Process term (primary entity)
    if "term" in keywords and isinstance(keywords["term"], str):
        term = keywords["term"]
        working_template["term"] = term
        
        # Try to find a primary field that could hold the main term
        # Look for fields with names like "diagnosis", "symptom", etc.
        primary_field_names = ["diagnosis", "symptom", "condition", "finding", "chief complaint", "main symptom"]
        primary_field = None
        
        for candidate in field_candidates:
            if any(primary_name in candidate["name"].lower() for primary_name in primary_field_names):
                primary_field = candidate
                break
                
        # If found, assign the term to this field
        if primary_field:
            working_template["field_values"][primary_field["id"]] = {
                "name": primary_field["name"],
                "value": term,
                "updated_at": datetime.now().isoformat()
            }
    
    # Store UMLS metadata if available
    if "semantic_type" in keywords:
        working_template["semantic_type"] = keywords["semantic_type"]
    if "cui" in keywords:
        working_template["cui"] = keywords["cui"]
    if "tui" in keywords:
        working_template["tui"] = keywords["tui"]
    
    # Process the keyword modifiers and attributes
    extras = {}
    
    for attr_key, attr_values in keywords.items():
        # Skip already processed fields and non-modifier attributes
        if attr_key in ["term", "semantic_type", "cui", "tui", "label"]:
            continue
            
        # Convert to list if not already
        attr_list = attr_values if isinstance(attr_values, list) else [attr_values]
        
        for attr_value in attr_list:
            if not attr_value:  # Skip empty values
                continue
                
            attr_str = str(attr_value)
            attr_text = f"{attr_key}: {attr_str}"
            
            # Find best matching field
            best_match = None
            best_score = threshold  # Minimum threshold to consider a match
            
            for candidate in field_candidates:
                # Compute similarity between attribute and field
                sim_score = cosine_similarity(embed_text(attr_text), embed_text(candidate["text"]))
                
                if sim_score > best_score:
                    best_score = sim_score
                    best_match = candidate
            
            # If match found, set the value
            if best_match:
                field_id = best_match["id"]
                
                # Check if this field already has a value
                if field_id in working_template["field_values"]:
                    current_value = working_template["field_values"][field_id]["value"]
                    
                    # Convert to list if not already a list
                    if isinstance(current_value, list):
                        current_value.append(attr_str)
                    else:
                        current_value = [current_value, attr_str]
                    
                    working_template["field_values"][field_id]["value"] = current_value
                else:
                    # Create new field value entry
                    working_template["field_values"][field_id] = {
                        "name": best_match["name"],
                        "value": attr_str,
                        "updated_at": datetime.now().isoformat()
                    }
            else:
                # Store unmatched attributes in extras
                if attr_key not in extras:
                    extras[attr_key] = []
                extras[attr_key].append(attr_str)
    
    # Add extras to additional_content
    if extras:
        working_template["additional_content"] = extras
    
    # Ensure all required fields have at least an empty value
    for field in field_candidates:
        if field["required"] and field["id"] not in working_template["field_values"]:
            working_template["field_values"][field["id"]] = {
                "name": field["name"],
                "value": "",
                "updated_at": datetime.now().isoformat()
            }
    
    # Generate content dictionary that reflects field_values for backward compatibility
    content = {}
    for field_id, field_value in working_template["field_values"].items():
        field_name = field_value["name"]
        content[field_name] = field_value["value"]
    
    # Add content to the template
    working_template["content"] = content
    
    return working_template