import spacy
import spacy.tokens
from app.schemas.section import TextCategoryEnum, BaseSection
from app.db.session import DataStore

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
    
    # Retrieve the nearest neighbor (n=1) along with the distance.
    nearest_indices, distances = data_store.index.get_nns_by_vector(keyword_emb, n=1, include_distances=True)
    idx = nearest_indices[0]
    distance = distances[0]
    
    assigned_category = data_store.rep_terms.keys()[idx]
    matched_term = data_store.rep_terms.values()[idx]
    
    return assigned_category, matched_term, distance
