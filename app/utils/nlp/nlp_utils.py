from typing import List, Dict, Any, Tuple, Optional
from sqlalchemy.orm import Session
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
import copy
import json
import copy
from datetime import datetime
from app.schemas.section import SectionRead

DEFAULT_MODEL = SentenceTransformer("all-minilm-l6-v2")

logger = logging.getLogger(__name__)

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



# Template management functions
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


def merge_keywords_into_template(
    match_tuple: Tuple[str, Any, List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """
    Create content dictionary from a match tuple, with template field IDs as keys
    and populated template field objects as values.
    
    Args:
        match_tuple: A tuple (keyword_key, keyword_value, [template_fields]) from match_template_fields
        
    Returns:
        Dictionary with template field IDs as keys and populated template field objects as values
    """
    import datetime
    
    # Unpack the tuple
    keyword_key, keyword_value, matching_fields = match_tuple
    
    # Initialize the result dictionary
    result = {
        "content": {},  # Dictionary keyed by field ID with template field objects as values
    }
    
    # Process each matching field
    for field in matching_fields:
        field_id = field.get("id")
        if not field_id:
            continue
            
        field_name = field.get("name", "")
        field_type = field.get("data_type", "string")
        field_description = field.get("description", "")
        field_required = field.get("required", False)
        
        # Prepare value based on field type
        prepared_value = prepare_value_for_field_type(keyword_value, field_type)
        
        # Create complete template field object with value
        template_field = {
            "id": field_id,
            "name": field_name,
            "data_type": field_type,
            "description": field_description,
            "required": field_required,
            "value": prepared_value,
            "source_keyword": keyword_key,
            "updated_at": datetime.datetime.now().isoformat()
        }
        
        # Add to content dictionary, handling potential duplicates
        if field_id in result["content"]:
            # If this field ID already exists, convert to list for multiple entries
            if isinstance(result["content"][field_id], list):
                result["content"][field_id].append(template_field)
            else:
                result["content"][field_id] = [result["content"][field_id], template_field]
        else:
            result["content"][field_id] = template_field
    
    return result

def prepare_value_for_field_type(value: Any, field_type: str) -> Any:
    """
    Prepare a value for insertion into a field based on its data type.
    
    Args:
        value: The value to prepare
        field_type: The target field's data type
        
    Returns:
        The prepared value
    """
    if value is None:
        return None
        
    field_type = field_type.lower()
    
    # Handle different field types
    if field_type in ["integer", "int"]:
        try:
            return int(value)
        except (ValueError, TypeError):
            return None
            
    elif field_type in ["float", "number", "decimal"]:
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
            
    elif field_type in ["boolean", "bool"]:
        if isinstance(value, bool):
            return value
        elif isinstance(value, str):
            return value.lower() in ["true", "yes", "y", "1"]
        else:
            return bool(value)
            
    elif field_type in ["date", "datetime", "time"]:
        # Return as string for date types, can be parsed by frontend
        return str(value)
        
    elif field_type in ["string", "text", "enum"] or not field_type:
        # Default to string representation
        return str(value)
    
    # For unknown types, return as is
    return value