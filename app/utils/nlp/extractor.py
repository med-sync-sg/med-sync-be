import spacy
import spacy.tokens
from app.schemas.section import TextCategoryEnum, SectionCreate, CHIEF_COMPLAINT_EXAMPLE, PATIENT_MEDICAL_HISTORY_EXAMPLE, PATIENT_INFORMATION_EXAMPLE, OTHER_EXAMPLE
from app.db.session import DataStore
import copy

data_store = DataStore()

def extract_keywords_descriptors(doc: spacy.tokens.Doc):
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

def classify_keyword(keyword_dict: dict):
    """
    Classifies an extracted keyword by computing its embedding and using the Annoy index
    to find the most similar representative term.
    
    Returns:
        assigned_category (str): The category name.
        matched_term (str): The representative term matched.
        distance (float): The angular distance (lower is better).
    """
    # Compute the embedding for the input keyword.
    keyword_emb = data_store.embedding_model.encode(keyword_dict["term"]).astype('float32')
    # Precompute key-to-index mapping

    # Retrieve the nearest neighbor (n=1) along with the distance.
    nearest_indices, distances = data_store.index.get_nns_by_vector(keyword_emb, n=1, include_distances=True)
    idx = nearest_indices[0]
    distance = distances[0]

    assigned_category = data_store.term_categories[idx]
    matched_term = data_store.all_terms[idx]
    return assigned_category, matched_term, distance

def create_section(note_id: int, user_id: int, assigned_category: str, matched_term: str, distance: float, order: int = 1) -> SectionCreate:
    """
    Create a BaseSection object from the (assigned_category, matched_term, distance) tuple.
    Note: The `id` is not set here because it will be assigned by the database upon insertion.
    
    Args:
        assigned_category (str): e.g., "CHIEF_COMPLAINT", "PATIENT_INFORMATION", or "OTHERS"
        matched_term (str): The representative term matched.
        distance (float): The distance from the vector search.
        order (int): The order of the section in the report.
    
    Returns:
        BaseSection: A new BaseSection object without an assigned ID.
    """
    if note_id == -1:
        raise Exception("A Section must belong to a Note object!")
    
    # Prepare default metadata to include the distance value.
    metadata = {"matched_term": matched_term, "distance": distance}

    if assigned_category == TextCategoryEnum.CHIEF_COMPLAINT.name:
        # Use a deep copy to avoid modifying the original template.
        content = copy.deepcopy(CHIEF_COMPLAINT_EXAMPLE)
        # Update the "Main Symptom" with the matched term.
        if "Main Symptom" in content:
            content["Main Symptom"]["name"] = matched_term
        title = "Chief Complaint"
        section_type = TextCategoryEnum.CHIEF_COMPLAINT.name
        section_description = TextCategoryEnum.CHIEF_COMPLAINT.value

    elif assigned_category == TextCategoryEnum.PATIENT_INFORMATION.name:
        content = copy.deepcopy(PATIENT_INFORMATION_EXAMPLE)
        # Append the matched term to Additional Details.
        if "Additional Details" in content:
            current_details = content["Additional Details"]
            content["Additional Details"] = f"{current_details}; {matched_term}" if current_details else matched_term
        else:
            content["Additional Details"] = matched_term
        title = "Patient Information"
        section_type = TextCategoryEnum.PATIENT_INFORMATION.name
        section_description = TextCategoryEnum.PATIENT_INFORMATION.value

    else:  # For OTHERS category
        content = copy.deepcopy(OTHER_EXAMPLE)
        # Add the matched term as a new observation.
        new_observation = {
            "observation": matched_term,
            "notes": f"Distance: {distance:.4f}"
        }
        if "Other Observations" in content:
            content["Other Observations"].append(new_observation)
        else:
            content["Other Observations"] = [new_observation]
        title = "Other Observations"
        section_type = TextCategoryEnum.OTHERS.name
        section_description = TextCategoryEnum.OTHERS.value

    base_section = SectionCreate(
        title=title,
        note_id=note_id,
        user_id=user_id,
        content=content,
        order=order,
        section_type=section_type,
        section_description=section_description
    )
    return base_section