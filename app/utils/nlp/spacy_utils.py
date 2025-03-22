from typing import List, Tuple, Dict, Any
from spacy.matcher import PhraseMatcher
import pandas as pd
import ahocorasick
from spacy import Language
from fastapi import APIRouter
import spacy.tokenizer
from spacy.tokens import Doc
import spacy.tokens
from app.db.umls_data_loader import umls_df_dict
from spacy import displacy
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

    matches = []
    text_lower = doc.text.lower()
    
    # Iterate over matches from the automaton.
    for end_index, data in automaton.iter(doc.text.lower()):
        term = data['term']  # The stored term (lowercase)
        start_char = end_index - len(term) + 1
        end_char = end_index + 1
        # Extract the substring from the document.
        found_text = text_lower[start_char:end_char]
        span = doc.char_span(start_char, end_char, label=data['semantic_type'], alignment_mode="expand")
        if found_text.strip().lower() == term.strip().lower():            
            span._.is_medical_term = True
            matches.append(span)
        else:
            # Compute the Levenshtein ratio between the found text and the stored term.
            sim = levenshtein_ratio(found_text, term)
            # Accept the match if similarity is above the threshold.
            if sim >= 0.85:
                span = doc.char_span(start_char, end_char, label=data["semantic_type"], alignment_mode="expand")
                if span is not None:
                    # Mark this span as coming from Aho-Corasick.
                    span._.is_medical_term = True
                    matches.append(span)

    # Replace doc.ents entirely with our matches.
    doc.ents = list(doc.ents) + matches
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
