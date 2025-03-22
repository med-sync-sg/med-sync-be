import spacy
import spacy.tokens
from spacy.tokens import Span, Token, Doc
from spacy.matcher import DependencyMatcher
from typing import List, Dict, Any
from app.utils.text_utils import normalize_text
from app.schemas.section import TextCategoryEnum, SectionCreate
from app.db.iris_session import IrisDataStore
import numpy as np
from sentence_transformers import SentenceTransformer
import copy

# Initialize IRIS data store (and its embedding model) if needed.
iris_data_store = IrisDataStore()

def extract_pp_object_span(token: Token) -> Span:
    """
    Given a target token, look for a child that is a preposition (ADP),
    then find its prepositional object (child with dep "pobj") and return a
    contiguous span covering the entire subtree of that object.
    If the span begins with a determiner, remove it.
    Returns the Span or None if no such object is found.
    """
    for child in token.children:
        if child.pos_ == "ADP":
            pobj = None
            for sub in child.children:
                if sub.dep_ == "pobj":
                    pobj = sub
                    break
            if pobj is not None:
                subtree_tokens = sorted(list(pobj.subtree), key=lambda t: t.i)
                start = subtree_tokens[0].i
                end = subtree_tokens[-1].i + 1
                span = token.doc[start:end]
                if span[0].pos_ == "DET":
                    span = token.doc[start+1:end]
                return span
    return None

def find_modifiers_for_medical_span(span: Span) -> Dict[str, Any]:
    """
    Given a span, use DependencyMatcher to find all target nouns (POS "NOUN").
    For each target, record a dictionary with:
      - "term": the target token's text,
      - "modifiers": list containing the head of the target (if different from target)
                     plus any adjective children (with DEP "amod" or "attr") attached to the head,
      - "quantities": list of tokens (or extracted PP spans) that indicate numeric/compound modifiers.
    Only tokens in the same sentence as the target are considered.
    Returns a list of dictionaries.
    """
    # Load the model (in practice, load this once externally)
    nlp_en = spacy.load("en_core_web_trf")
    
    matcher = DependencyMatcher(nlp_en.vocab)
    # Our pattern matches any token with POS "NOUN"
    pattern = [
        {"RIGHT_ID": "target", "RIGHT_ATTRS": {"POS": "NOUN"}}
    ]
    matcher.add("FOUNDED", [pattern])
    results = []
    
    # Run the matcher on the span.
    matches = matcher(span)
    for match_id, token_ids in matches:
        result_dict = {
            "term": "",
            "modifiers": [],
            "quantities": []
        }
        target = span[token_ids[0]]
        result_dict["term"] = span.text
        
        # Process direct children of the target to get direct modifiers/quantities.
        for child in target.children:
            if child.pos_ == "ADJ" and child.dep_ in ("amod", "attr"):
                result_dict["modifiers"].append(child.text)
            if child.dep_ in ("nummod", "quantmod", "compound"):
                result_dict["quantities"].append(child.text)

        # Process children of the target's head.
        for child in target.head.children:
            # Only consider tokens in the same sentence.
            if child.sent != target.sent:
                continue
            # If a child is an adjective (and its dependency is "amod" or "attr"), record it.
            if child.pos_ == "ADJ" and child.dep_ in ("amod", "attr"):
                result_dict["modifiers"].append(child.text)
            # If a child is numeric or compound, record it as quantity.
            if child.dep_ in ("nummod", "quantmod", "compound"):
                result_dict["quantities"].append(child.text)
            # If a child is a preposition, extract its prepositional object span.
            if child.pos_ == "ADP":
                pp_span = extract_pp_object_span(target.head)
                if pp_span is not None:
                    # Add the full PP text as a quantity modifier.
                    result_dict["quantities"].append(pp_span.text)
        
        results.append(result_dict)
    print("Extracted dicts: ", results)
    if len(results) == 0:
        return {
            "term": span.text,
            "modifiers": [],
            "quantities": []
        }
    if len(results) == 1:
        return results[0]
    return merge_results_dicts(results)


def find_medical_modifiers(doc: Doc) -> List[Dict[str, Any]]:
    """
    Iterate over all spans in the Doc (here, we assume medical terms appear as entities)
    that have the custom extension 'is_medical' set to True.
    For each such span, find and return its adjective and number modifiers.
    Returns a dictionary mapping each medical span to a list of its modifiers.
    """
    features = []
    for span in doc.ents:
        # Check if the span is marked as medical.
        if span._.is_medical_term == True:
            print("Medical term: ", span.text)
            mods = find_modifiers_for_medical_span(span)
            features.append(mods)
    features = merge_results_dicts(features)
    print("Modifiers result: ", features)
    return features

def merge_results_dicts(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged = {}
    for d in results:
        term = d.get("term", "")
        if term in merged:
            merged[term]["modifiers"].extend(d.get("modifiers", []))
            merged[term]["quantities"].extend(d.get("quantities", []))
        else:
            # Create a new entry copying the lists to avoid modifying the original lists.
            merged[term] = {
                "term": term,
                "modifiers": list(d.get("modifiers", [])),
                "quantities": list(d.get("quantities", []))
            }
    return list(merged.values())