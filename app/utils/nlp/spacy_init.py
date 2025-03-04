from typing import List, Tuple, Dict, Any
from spacy.matcher import PhraseMatcher
import pandas as pd
import ahocorasick
from spacy import Language
from fastapi import APIRouter
import spacy.tokenizer
from spacy.tokens import Doc
import spacy.tokens
from app.db.umls_data import umls_df_dict
from spacy import displacy
from pathlib import Path
from app.schemas.section import TextCategoryEnum
from sentence_transformers import SentenceTransformer, util
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from os import environ
from spacy.tokens import Span

# Load a cross-encoder model (e.g., "cross-encoder/ms-marco-MiniLM-L-6-v2" or something domain-specific)
re_ranker_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
re_ranker_tokenizer = AutoTokenizer.from_pretrained(re_ranker_name)
re_ranker_model = AutoModelForSequenceClassification.from_pretrained(re_ranker_name)
HF_TOKEN = environ.get("HF_ACCESS_TOKEN")

router = APIRouter()

class AhoCorasickComponent:
    def __init__(self):
        self.automaton = None

    def initialize(self, get_examples=None, nlp: Language=None, config: dict={}):
        """
        spaCy calls this at `nlp.initialize()` time.
        Here, load your DataFrame and build the Aho-Corasick automaton once.
        """
        df : pd.DataFrame = umls_df_dict["combined_df"] 
        self.automaton = ahocorasick.Automaton()
        print(df.head())
        for index, row in df.iterrows():
            term = row['STR']
            data = {
                'term': term.lower(),
                'cui': row['CUI'],
                'semantic_type': row['STY'],
                'tui': row["TUI"]
            }
            self.automaton.add_word(term.lower(), data)

        self.automaton.make_automaton()
        
    def __call__(self, doc: spacy.tokens.Doc) -> spacy.tokens.Doc:
        if not self.automaton:
            return doc

        matches = []
        # Iterate over matches from the automaton.
        for end_index, data in self.automaton.iter(doc.text.lower()):
            term = data['term']
            start_char = end_index - len(term) + 1
            end_char = end_index + 1
            span = doc.char_span(start_char, end_char, label=data['semantic_type'])
            if span is not None:
                # Mark the span as coming from Aho-Corasick.
                span._.is_medical_term = True
                matches.append(span)

        # Replace doc.ents entirely with our matches.
        doc.ents = matches
        return doc

@Language.factory("ahocorasick")
def create_ahocorasick_component(nlp: Language, name: str, config: dict={}):
    """
    This factory function returns an instance of AhoCorasickComponent.
    spaCy will call its initialize() method when you run nlp.initialize().
    """
    return AhoCorasickComponent()

def summarize_text():
    pass


nlp_en = spacy.load("en_core_web_trf")

if not Span.has_extension("is_medical_term"):
    Span.set_extension("is_medical_term", default=False)

if "ahocorasick" not in nlp_en.pipe_names:
    nlp_en.add_pipe("ahocorasick", last=True)

def process_text(text: str) -> Doc:
    # Process the text
    doc = nlp_en(text)
    return doc
