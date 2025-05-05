from typing import List, Dict, Any, Optional
import spacy
from spacy import Language
from spacy.tokens import Doc, Token, Span
import ahocorasick

# Initialize automaton for Aho-Corasick algorithm
automaton = ahocorasick.Automaton()

# Register custom extensions
if not Span.has_extension("is_medical_term"):
    Span.set_extension("is_medical_term", default=False)
    
if not Span.has_extension("cui"):
    Span.set_extension("cui", default="N/A")

if not Span.has_extension("tui"):
    Span.set_extension("tui", default="N/A")
    

# Load model
nlp_en = spacy.load("en_core_web_trf")

@Language.component("ahocorasick")
def AhoCorasickComponent(doc: Doc):
    """
    A custom spaCy component that uses the Aho-Corasick algorithm to find medical terms in text.
    
    Args:
        doc: A spaCy Doc object
        
    Returns:
        doc: The processed spaCy Doc with medical entities added
    """
    from app.db.data_loader import umls_df_dict
    df = umls_df_dict["concepts_with_sty_def_df"]
    
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
        
        span = doc.char_span(start_char, end_char, label=data['semantic_type'], alignment_mode="contract")
        if span is None:
            # Try with expand if contract fails
            span = doc.char_span(start_char, end_char, label=data['semantic_type'], alignment_mode="expand")
            
        if span is not None:
            # Add UMLS metadata to the span
            span._.set("is_medical_term", True)
            span._.set("cui", data['cui'])
            span._.set("tui", data['tui'])
            
            matches.append({
                'span': span,
                'score': 1.0,
                'type': data['semantic_type'],
                'cui': data['cui'],
                'tui': data['tui'],
                'term': term
            })

    # Sort matches by score (higher is better)
    matches.sort(key=lambda x: x['score'], reverse=True)
    
    # Resolve overlapping spans by keeping only the highest-scored spans
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
    
    # Filter out any remaining overlaps
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

def process_text(text: str) -> Doc:
    """
    Process text with the NLP pipeline and medical entity recognition
    
    Args:
        text: Input text to process
    
    Returns:
        doc: Processed spaCy Doc with medical entities
    """
    # Make sure the medical entity recognition component is added
    if "ahocorasick" not in nlp_en.pipe_names:
        nlp_en.add_pipe("ahocorasick", after="ner")
    
    # Process the text
    doc = nlp_en(text)
    return doc

def find_medical_modifiers(doc: Doc) -> List[Dict[str, Any]]:
    """
    Extract medical terms and their modifiers from the document.
    Uses spaCy's native NER functionality to process spans and incorporate standard
    entity types like DATE, TIME, CARDINAL, etc.
    
    Args:
        doc: Processed spaCy Doc
    
    Returns:
        List of dictionaries with medical terms and their modifiers
    """
    results = []
    
    # First collect all standard spaCy entities for context
    standard_entities = {}
    for ent in doc.ents:
        # Skip irrelevant entity types
        if ent.label_ in ["PRODUCT", "FAC", "ORG", "EVENT"]:
            continue
            
        standard_entities[ent.start] = {
            "text": ent.text,
            "type": ent.label_,
            "start": ent.start,
            "end": ent.end
        }
    
    # Process medical entities
    for ent in doc.ents:
        # Only process entities marked as medical terms
        if getattr(ent._, "is_medical_term", False):
            # Create result dictionary with required fields
            result = {
                "term": ent.text,
                "semantic_type": ent.label_,
                "cui": getattr(ent._, "cui", ""),
                "tui": getattr(ent._, "tui", ""),
                "modifiers": [],
                "quantities": [],
                "locations": [],
                "temporal": [],
                "negations": []  # Store complete negated phrases
            }
            
            # Get the sentence containing the entity (safely)
            try:
                sent = ent.sent
                sent_start = sent.start
                sent_end = sent.end
            except ValueError:
                # Create a safe window around the entity if it spans sentences
                sent_start = max(0, ent.start - 5)
                sent_end = min(len(doc), ent.end + 5)
                sent = doc[sent_start:sent_end]
            
            # Check for negation
            term_negation = check_term_negation(ent, sent, doc)
            if term_negation:
                result["negations"].append({
                    "negated_item": "term",
                    "negated_value": ent.text,
                    "negation_phrase": term_negation
                })
            
            # Process modifiers using dependency parsing
            for token in ent:
                # Safety check
                if token.i < 0 or token.i >= len(doc):
                    continue
                    
                # Use token.children directly - it's usually safe in spaCy
                for child in token.children:
                    # Safety check
                    if child.i < 0 or child.i >= len(doc):
                        continue
                    
                    # Adjective modifiers
                    if child.pos_ in ["ADJ", "ADV"] or child.dep_ in ["amod", "advmod"]:
                        modifier = child.text
                        result["modifiers"].append(modifier)
                    
                    # Numerical modifiers 
                    elif child.dep_ in ["nummod", "quantmod"] or child.pos_ == "NUM":
                        # Get the subtree safely
                        try:
                            # Sort subtree by position to maintain correct word order
                            subtree_tokens = sorted([t for t in child.subtree if 0 <= t.i < len(doc)], key=lambda t: t.i)
                            if subtree_tokens:
                                quantity_phrase = " ".join(t.text for t in subtree_tokens)
                                
                                # Determine if this is a temporal expression
                                time_words = ["day", "week", "month", "year", "hour", "minute", "second", "ago"]
                                if any(word in quantity_phrase.lower() for word in time_words):
                                    result["temporal"].append(quantity_phrase)
                                else:
                                    result["quantities"].append(quantity_phrase)
                        except Exception:
                            # Fallback to just the token text if subtree fails
                            result["quantities"].append(child.text)
                    
                    # Prepositions for location and temporal expressions
                    elif child.pos_ == "ADP" and child.dep_ == "prep":
                        try:
                            # Get and sort the subtree safely
                            subtree_tokens = sorted([t for t in child.subtree if 0 <= t.i < len(doc)], key=lambda t: t.i)
                            if subtree_tokens:
                                prep_phrase = " ".join(t.text for t in subtree_tokens)
                                
                                # Classify by preposition type
                                if child.text.lower() in ["in", "on", "at", "near", "around", "throughout"]:
                                    result["locations"].append(prep_phrase)
                                elif child.text.lower() in ["for", "since", "during", "after", "before", "ago"]:
                                    result["temporal"].append(prep_phrase)
                                else:
                                    result["modifiers"].append(prep_phrase)
                        except Exception:
                            # Fallback if subtree fails
                            pass
            
            # Now look for standard NER entities in the sentence that might relate to our medical term
            for idx in range(sent_start, sent_end):
                if idx in standard_entities:
                    std_ent = standard_entities[idx]
                    
                    # Skip if the entity overlaps with our medical term
                    if (std_ent["start"] <= ent.end and std_ent["end"] >= ent.start):
                        continue
                    
                    # Check if this standard entity is close to our medical term
                    distance = min(abs(std_ent["start"] - ent.end), abs(ent.start - std_ent["end"]))
                    if distance <= 10:  # Within 10 tokens
                        entity_type = std_ent["type"]
                        entity_text = std_ent["text"]
                        
                        # Process based on entity type
                        if entity_type in ["DATE", "TIME"]:
                            result["temporal"].append(entity_text)
                        elif entity_type in ["CARDINAL", "QUANTITY", "PERCENT", "MONEY"]:
                            result["quantities"].append(entity_text)
                        elif entity_type in ["GPE", "LOC"]:
                            result["locations"].append(entity_text)
                        elif entity_type == "ORDINAL":
                            result["modifiers"].append(entity_text)
            
            # Add to results
            results.append(result)
    
    # Process and deduplicate results
    unique_results = {}
    for result in results:
        term = result["term"]
        if term not in unique_results:
            unique_results[term] = result
        else:
            # Merge fields
            for field in ["modifiers", "quantities", "locations", "temporal", "negations"]:
                if field in result and field in unique_results[term]:
                    unique_results[term][field].extend(result[field])
    
    # Remove duplicates while preserving order
    for term, data in unique_results.items():
        for field in ["modifiers", "quantities", "locations", "temporal"]:
            if field in data:
                data[field] = list(dict.fromkeys(data[field]))
        
        # Special handling for negations
        if "negations" in data:
            unique_negations = []
            seen = set()
            for neg in data["negations"]:
                key = f"{neg.get('negated_item', '')}:{neg.get('negated_value', '')}"
                if key not in seen:
                    seen.add(key)
                    unique_negations.append(neg)
            data["negations"] = unique_negations
    
    return list(unique_results.values())

def check_term_negation(entity: Span, sentence: Span, doc: Doc) -> Optional[str]:
    """
    Check if a medical term is negated and return the negation phrase
    
    Args:
        entity: The medical entity
        sentence: The sentence containing the entity
        doc: The full document
        
    Returns:
        Negation phrase or None if not negated
    """
    # Safety check for entity bounds
    if entity.start < 0 or entity.end > len(doc):
        return None
        
    negation_words = ["no", "not", "n't", "never", "without", "deny", "denies", "denied"]
    
    # Look for negation words within a window before the entity
    start_idx = max(0, entity.start - 3)
    for i in range(start_idx, entity.start):
        if i >= len(doc):
            continue
            
        token = doc[i]
        if token.lower_ in negation_words or token.lemma_ in negation_words:
            # Example: "no fever" or "denies headache"
            return f"{token.text} {entity.text}"
    
    # Check for negation as a child of the entity head
    for token in entity:
        if token.i >= len(doc):
            continue
            
        for child in token.children:
            if child.i >= len(doc):
                continue
                
            if child.dep_ == "neg":
                # Example: "pain is not present"
                return f"{entity.text} {child.text}"
    
    return None