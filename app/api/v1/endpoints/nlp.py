from spacy.matcher import PhraseMatcher
import pandas as pd
import ahocorasick
import spacy
from spacy import Language
from fastapi import APIRouter
from spacy.tokens import Doc

router = APIRouter()

# Define the custom component
@Language.component(name="ahocorasick")
def ahocorasick_ner(doc: Doc):
    matches = []
    A = build_automaton()
    for end_index, data in A.iter(doc.text.lower()):
        term = data['term']
        start_char = end_index - len(term) + 1
        end_char = end_index + 1
        span = doc.char_span(start_char, end_char, label=data['cui'])
        if span is not None:
            matches.append(span)
    # Set the entities in the doc
    doc.ents = list(doc.ents) + matches
    return doc

@router.get("/nlp/ner-tag") 
def process_text(text: str) -> Doc:
    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("ahocorasick", last=True)

    # Process the text
    doc = nlp(text)
    
    for ent in doc.ents:
        print(f"Entity: {ent.text}, Label: {ent.label_}")
    
    return doc


def build_automaton(df: pd.DataFrame):
    A = ahocorasick.Automaton()

    for index, row in df.iterrows():
        term = row['STR']
        data = {
            'term': term,
            'cui': row['CUI'],
            'semantic_type': row['STY']
        }
        A.add_word(term, data)

    A.make_automaton()
    
    return A