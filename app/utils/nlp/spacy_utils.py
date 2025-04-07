from typing import List, Tuple, Dict, Any
from spacy.matcher import PhraseMatcher
import pandas as pd
import ahocorasick
from spacy import Language
from fastapi import APIRouter
import spacy.tokenizer
from spacy.tokens import Doc, Token
import spacy.tokens
from app.db.data_loader import umls_df_dict
from spacy import displacy
from spacy.matcher import DependencyMatcher
from pathlib import Path
from app.schemas.section import TextCategoryEnum
from sentence_transformers import SentenceTransformer, util
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from os import environ
from spacy.tokens import Span
from Levenshtein import ratio as levenshtein_ratio

# Load a cross-encoder model (e.g., "cross-encoder/ms-marco-MiniLM-L-6-v2" or something domain-specific)
re_ranker_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
re_ranker_tokenizer = AutoTokenizer.from_pretrained(re_ranker_name)
re_ranker_model = AutoModelForSequenceClassification.from_pretrained(re_ranker_name)
HF_TOKEN = environ.get("HF_ACCESS_TOKEN")

router = APIRouter()

df = umls_df_dict["concepts_with_sty_def_df"]
automaton = ahocorasick.Automaton()

@Language.component("ahocorasick")
def AhoCorasickComponent(doc: Doc):
    """
    A custom spaCy component that uses the Aho-Corasick algorithm to find medical terms in text.
    This fixed version handles overlapping entity spans.
    
    Args:
        doc: A spaCy Doc object
        
    Returns:
        doc: The processed spaCy Doc with medical entities added
    """
    # Build the automaton if it's not already built
    if not automaton.get_stats()['total_size']:
        for index, row in df.iterrows():
            term = row["STR"].strip().lower()  # Normalize
            data = {
                'term': term,
                'cui': row['CUI'],
                'semantic_type': row['STY'],
                'tui': row["TUI"]
            }
            automaton.add_word(term, data)
        automaton.make_automaton()
    
    text_lower = doc.text.lower()
    
    # Get all potential matches
    matches = []
    for end_index, data in automaton.iter(text_lower):
        term = data['term']  # The stored term (lowercase)
        start_char = end_index - len(term) + 1
        end_char = end_index + 1
        # Extract the substring from the document.
        found_text = text_lower[start_char:end_char]
        
        span = doc.char_span(start_char, end_char, label=data['semantic_type'], alignment_mode="contract")
        if span is None:
            # Try with expand if contract fails
            span = doc.char_span(start_char, end_char, label=data['semantic_type'], alignment_mode="expand")
            
        if span is not None:
            if found_text.strip().lower() == term.strip().lower():            
                span._.set("is_medical_term", True)
                # Save all information we need for later
                matches.append({
                    'span': span,
                    'score': 1.0,  # Perfect match
                    'type': data['semantic_type'],
                    'term': term
                })
            else:
                # Compute the Levenshtein ratio between the found text and the stored term.
                sim = levenshtein_ratio(found_text, term)
                # Accept the match if similarity is above the threshold.
                if sim >= 0.85:
                    span._.is_medical_term = True
                    matches.append({
                        'span': span,
                        'score': sim,  # Levenshtein similarity
                        'type': data['semantic_type'],
                        'term': term
                    })

    # Sort matches by score (higher is better)
    matches.sort(key=lambda x: x['score'], reverse=True)
    
    # Resolve overlapping spans by keeping only the highest-scored spans
    # that don't overlap with already selected spans
    final_matches = []
    used_tokens = set()
    
    for match in matches:
        span = match['span']
        # Check if any token in this span has already been used
        if not any(token.i in used_tokens for token in span):
            # If not, add this span and mark its tokens as used
            final_matches.append(span)
            used_tokens.update(token.i for token in span)
    
    # Create a new list combining existing non-medical entities with our medical entities
    existing_ents = [ent for ent in doc.ents if not getattr(ent, '_.is_medical_term', False)]
    merged_ents = existing_ents + final_matches
    
    # Filter out any remaining overlaps (should be none after our resolution)
    # This is a safety check in case there are still conflicts
    token_ent_map = {}
    filtered_ents = []
    
    for ent in merged_ents:
        valid = True
        for token in ent:
            if token.i in token_ent_map:
                valid = False
                break
        
        if valid:
            filtered_ents.append(ent)
            for token in ent:
                token_ent_map[token.i] = len(filtered_ents) - 1
    
    # Set the final entities
    doc.ents = filtered_ents
    return doc

def summarize_text():
    pass


nlp_en = spacy.load("en_core_web_trf")

if not Span.has_extension("is_medical_term"):
    Span.set_extension("is_medical_term", default=False)

if "ahocorasick" not in nlp_en.pipe_names:
    nlp_en.add_pipe("ahocorasick", after="ner")

def process_text(text: str) -> Doc:
    # Process the text
    doc = nlp_en(text)
    
    medical_terms = {ent.text: ent.label_ for ent in doc.ents if ent._.is_medical_term}
    print(medical_terms)
    # svg = displacy.render(doc)
    # output_path = Path("./images/dependency_plot.svg")
    # output_path.open("w", encoding="utf-8").write(svg)
    
    # svg = displacy.render(doc, style="ent")
    # output_path = Path("./images/ent_plot.svg")
    # output_path.open("w", encoding="utf-8").write(svg)

    return doc

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
    Returns a dictionary.
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
    that have the custom extension 'is_medical_term' set to True.
    For each such span, find and return its adjective and number modifiers.
    Returns a list of dictionaries, each containing modifier information for a medical term.
    """
    features = []
    for span in doc.ents:
        # Check if the span is marked as medical.
        if span._.is_medical_term == True:
            print("Medical term: ", span.text)
            mods = find_modifiers_for_medical_span(span)
            # Ensure we're adding a dictionary, not a list
            if isinstance(mods, dict):
                features.append(mods)
            elif isinstance(mods, list):
                # If for some reason find_modifiers_for_medical_span returns a list,
                # extend features with the list items
                for item in mods:
                    if isinstance(item, dict):
                        features.append(item)
    
    # Now merge the dictionaries to remove duplicates
    merged_features = merge_results_dicts(features)
    print("Modifiers result: ", merged_features)
    return merged_features

def merge_results_dicts(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge dictionaries with the same 'term' field, combining their modifiers and quantities lists.
    
    Args:
        results: List of dictionaries, each with 'term', 'modifiers', and 'quantities' fields
        
    Returns:
        List of merged dictionaries
    """
    if not results:
        return []
    
    merged = {}
    for item in results:
        # Skip non-dictionary items
        if not isinstance(item, dict):
            print(f"Skipping non-dictionary item: {item}")
            continue
        
        term = item.get("term", "")
        if not term:  # Skip items without a term
            continue
            
        if term in merged:
            # Extend existing modifiers and quantities lists
            modifiers = item.get("modifiers", [])
            quantities = item.get("quantities", [])
            
            if isinstance(modifiers, list):
                merged[term]["modifiers"].extend(modifiers)
            if isinstance(quantities, list):
                merged[term]["quantities"].extend(quantities)
        else:
            # Create a new entry copying the lists to avoid modifying the original lists
            merged[term] = {
                "term": term,
                "modifiers": list(item.get("modifiers", [])),
                "quantities": list(item.get("quantities", []))
            }
    
    return list(merged.values())