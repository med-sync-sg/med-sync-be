from typing import List, Dict, Any, Tuple
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

def get_semantic_section_type(section_title: str, section_content: Dict[str, Any]) -> str:
    """
    Determine section type based on semantic similarity
    
    Args:
        section_title: Title of the section
        section_content: Content of the section
        
    Returns:
        Predicted section type
    """
    # Reference texts for different section types
    section_types = {
        "CHIEF_COMPLAINT": [
            "primary symptom", "main concern", "reason for visit", 
            "presenting complaint", "chief complaint"
        ],
        "PATIENT_MEDICAL_HISTORY": [
            "medical history", "past medical history", "prior conditions", 
            "previous surgeries", "family history", "medications"
        ],
        "PATIENT_INFORMATION": [
            "patient demographics", "contact information", "personal details", 
            "insurance information", "patient data"
        ],
        "ASSESSMENT": [
            "assessment", "diagnosis", "impression", "clinical impression", 
            "differential diagnosis"
        ],
        "PLAN": [
            "plan", "treatment plan", "recommendations", "follow-up plan", 
            "next steps", "prescribed medications"
        ],
        "PHYSICAL_EXAM": [
            "physical examination", "exam findings", "clinical examination", 
            "vital signs", "physical assessment"
        ]
    }
    
    # Create reference text from section content
    content_text = section_title.lower()
    
    # Add key content terms if available
    if isinstance(section_content, dict):
        # Add main keys as content
        content_text += " " + " ".join(section_content.keys()).lower()
        
        # Look for specific structures like "Main Symptom"
        if "Main Symptom" in section_content and isinstance(section_content["Main Symptom"], dict):
            symptom = section_content["Main Symptom"]
            if "name" in symptom:
                content_text += f" symptom {symptom['name']}"
    
    # Get embedding for content
    content_embedding = embed_text(content_text)
    
    # Calculate similarities with each section type
    best_type = "OTHERS"
    best_score = 0.0
    
    for section_type, reference_texts in section_types.items():
        # Embed each reference text and calculate average similarity
        similarities = []
        
        for ref_text in reference_texts:
            ref_embedding = embed_text(ref_text)
            similarity = cosine_similarity(content_embedding, ref_embedding)
            similarities.append(similarity)
        
        # Get average similarity for this section type
        avg_similarity = sum(similarities) / len(similarities)
        
        # Update best match if better
        if avg_similarity > best_score:
            best_score = avg_similarity
            best_type = section_type
    
    # Only return detected type if similarity is above threshold
    return best_type if best_score > 0.5 else "OTHERS"

def extract_relevant_content(section_content: Dict[str, Any], section_type: str) -> Dict[str, Any]:
    """
    Extract relevant content based on section type
    
    Args:
        section_content: Raw section content
        section_type: Type of section
        
    Returns:
        Extracted content relevant to the section type
    """
    # Structure known patterns to extract
    if section_type == "CHIEF_COMPLAINT":
        # Check for Main Symptom
        if "Main Symptom" in section_content:
            return {"Main Symptom": section_content["Main Symptom"]}
    
    # For other sections, return original content
    return section_content