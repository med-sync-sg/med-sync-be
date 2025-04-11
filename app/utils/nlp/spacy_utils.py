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

def extract_basic_modifiers(target: Token):
    modifiers = []
    # Expand this list to capture more modifiers
    target_dep_relations = ["amod", "attr", "advmod", "compound", "nmod", "npadvmod"]
    
    # Then use these in your extraction logic
    for child in target.children:
        if child.dep_ in target_dep_relations:
            modifiers.append(child.text)
            
    return modifiers

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

def extract_contextual_modifiers(span, window_size=3):
    """Extract modifiers based on a window around the span"""
    modifiers = []
    doc = span.doc
    
    # Get start and end indices with window
    start_idx = max(0, span[0].i - window_size)
    end_idx = min(len(doc), span[-1].i + window_size + 1)
    
    # Analyze the window
    for token in doc[start_idx:end_idx]:
        # Skip the actual term
        if token.i >= span[0].i and token.i <= span[-1].i:
            continue
            
        # Check if token is an adjective, adverb, or other relevant POS
        if token.pos_ in ["ADJ", "ADV"]:
            modifiers.append(token.text)
            
    return modifiers

def detect_negation(span):
    """Detect if a medical term is negated"""
    doc = span.doc
    
    # Look for common negation words
    negation_tokens = ["no", "not", "n't", "none", "never", "without"]
    
    # Check if any negation token is within a certain distance
    start_idx = max(0, span[0].i - 3)  # Look 3 tokens before
    
    for i in range(start_idx, span[0].i):
        token = doc[i]
        if token.lower_ in negation_tokens or token.lemma_ == "deny":
            return True
            
    return False

def extract_temporal_info(span):
    """Extract temporal information related to a medical term"""
    doc = span.doc
    temporal_info = []
    
    # Common time-related words
    temporal_tokens = ["day", "week", "month", "year", "hour", "minute", "since", "for"]
    
    # Look for time-related information within a window
    window_start = max(0, span[0].i - 5)
    window_end = min(len(doc), span[-1].i + 5)
    
    # Extract phrases containing numbers and temporal tokens
    for i in range(window_start, window_end):
        token = doc[i]
        if token.like_num or token.lower_ in temporal_tokens:
            # Extract the phrase containing the temporal information
            # This is simplified and would need refinement
            if token.like_num and i+1 < len(doc) and doc[i+1].lower_ in temporal_tokens:
                temporal_info.append(f"{token.text} {doc[i+1].text}")
    
    return temporal_info

def find_modifiers_for_medical_span(span: Span) -> Dict[str, Any]:
    """
    Enhanced function to find modifiers for medical terms using both
    dependency parsing and NER.
    """
    result_dict = {
        "term": span.text,
        "modifiers": [],
        "quantities": [],
        "temporal": []
    }
    
    doc = span.doc
    sent = span.sent
    
    # 1. Process dependency-based modifiers
    target_tokens = [span[0]]  # Start with the first token of the span
    
    # If span has a head outside itself, include that too
    if span.root.head.i < span.start or span.root.head.i >= span.end:
        target_tokens.append(span.root.head)
    
    # Go through each relevant token and its children
    for token in target_tokens:
        # Check for adjective modifiers
        for child in token.children:
            # Adjectives, adverbs, and compounds as modifiers
            if child.pos_ in ["ADJ", "ADV"] or child.dep_ in ["amod", "advmod", "compound", "nmod"]:
                result_dict["modifiers"].append(child.text)
            
            # Find numerical modifiers and classifiers
            if child.dep_ in ["nummod", "quantmod"] or child.pos_ == "NUM":
                # Check for compound quantities like "three days"
                quantity_phrase = []
                quantity_phrase.append(child.text)
                
                # Look for nouns or time units following the number
                for gchild in child.children:
                    if gchild.pos_ in ["NOUN"] or gchild.text in ["days", "weeks", "months", "years"]:
                        quantity_phrase.append(gchild.text)
                
                result_dict["quantities"].append(" ".join(quantity_phrase))
    
    # 2. Use NER to extract quantities and temporal information
    for ent in doc.ents:
        # Skip if the entity is part of the medical term itself
        if (ent.start >= span.start and ent.start < span.end) or \
           (ent.end > span.start and ent.end <= span.end):
            continue
        
        # Check if entity appears in the same sentence or nearby context
        if ent.sent == sent or abs(ent.start - span.end) <= 10 or abs(span.start - ent.end) <= 10:
            # Temporal information
            if ent.label_ in ["DATE", "TIME", "DURATION"]:
                # Check if this entity is related to our term using dependency path
                if is_entity_related_to_span(ent, span):
                    result_dict["temporal"].append(ent.text)
            
            # Quantities
            elif ent.label_ in ["CARDINAL", "QUANTITY", "ORDINAL", "PERCENT"]:
                if is_entity_related_to_span(ent, span):
                    result_dict["quantities"].append(ent.text)
    
    # 3. Handle special patterns like "X feels Y"
    extract_special_patterns(span, result_dict)
    
    # Remove duplicates while preserving order
    result_dict["modifiers"] = list(dict.fromkeys(result_dict["modifiers"]))
    result_dict["quantities"] = list(dict.fromkeys(result_dict["quantities"]))
    result_dict["temporal"] = list(dict.fromkeys(result_dict["temporal"]))
    
    return result_dict

def is_entity_related_to_span(ent: Span, span: Span) -> bool:
    """
    Determine if an entity is related to a span using dependency path analysis.
    
    Args:
        ent: Named entity span
        span: Medical term span
        
    Returns:
        True if related, False otherwise
    """
    # Simple proximity check
    if abs(ent.start - span.end) <= 5 or abs(span.start - ent.end) <= 5:
        return True
    
    # Check for specific patterns that indicate relation
    # For example: "pain for 3 days" - "3 days" is related to "pain"
    
    # Check for preposition relationship
    for token in span:
        # Look for prepositions after the span
        for child in token.children:
            if child.pos_ == "ADP" and child.i > span.end:
                # Check if entity appears after the preposition
                if child.i < ent.start:
                    return True
    
    # More complex path analysis could be added here
    
    return False

def extract_special_patterns(span: Span, result_dict: Dict[str, Any]) -> None:
    """
    Extract modifiers using common symptom description patterns.
    
    Args:
        span: Medical term span
        result_dict: Dictionary to update with extracted information
    """
    doc = span.doc
    sent = span.sent
    
    # Pattern: "X feels Y" where X is the term and Y is a modifier
    for token in sent:
        if token.lemma_ in ["feel", "seem", "appear", "be"] and is_token_related_to_span(token, span):
            for child in token.children:
                if child.pos_ == "ADJ":
                    result_dict["modifiers"].append(child.text)
    
    # Pattern: "X for Y" where Y is temporal
    for token in sent:
        if token.text == "for" and is_token_related_to_span(token, span):
            for child in token.children:
                if child.pos_ == "NUM" or child.dep_ == "nummod":
                    temporal_phrase = [t.text for t in child.subtree]
                    result_dict["temporal"].append(" ".join(temporal_phrase))

def is_token_related_to_span(token: Token, span: Span) -> bool:
    """Check if a token is directly related to a span via dependency path"""
    for sp_token in span:
        # Check direct path
        if token.head == sp_token or sp_token.head == token:
            return True
        
        # Check one level up
        if token.head == sp_token.head:
            return True
    
    return False

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