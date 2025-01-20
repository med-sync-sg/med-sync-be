from spacy.matcher import PhraseMatcher
import pandas as pd
import ahocorasick
import spacy
from spacy import Language
from fastapi import APIRouter
from spacy.tokens import Doc
from app.utils.import_umls import DataStore

router = APIRouter()

@Language.factory("ahocorasick")
def create_ahocorasick_component(nlp: Language, name: str):
    """
    This factory function returns an instance of AhoCorasickComponent.
    spaCy will call its initialize() method when you run nlp.initialize().
    """
    return AhoCorasickComponent()

class AhoCorasickComponent:
    def __init__(self):
        # The automaton will be built in initialize(), so leave it None for now.
        self.automaton = None

    def initialize(self, nlp: Language, config: dict):
        """
        spaCy calls this at `nlp.initialize()` time.
        Here, load your DataFrame and build the Aho-Corasick automaton once.
        """
        df = DataStore().get_combined_df()  # or however you load your data
        self.automaton = ahocorasick.Automaton()

        # Populate the automaton
        for _, row in df.iterrows():
            term = row['STR']
            data = {
                'term': term,
                'cui': row['CUI'],
                'semantic_type': row['STY']
            }
            self.automaton.add_word(term.lower(), data)

        self.automaton.make_automaton()

    def __call__(self, doc: Doc) -> Doc:
        """
        The main function that spaCy calls for each Doc.
        Use the automaton to find matches in the text.
        """
        if not self.automaton:
            # If for some reason initialize() wasn't called,
            # or automaton is still None, raise an error or do nothing
            return doc

        matches = []
        # We do doc.text.lower() for case-insensitive matching
        for end_index, data in self.automaton.iter(doc.text.lower()):
            term = data['term']
            start_char = end_index - len(term) + 1
            end_char = end_index + 1
            span = doc.char_span(start_char, end_char, label=data['cui'])
            if span is not None:
                matches.append(span)

        # Merge new entities with existing ones
        doc.ents = list(doc.ents) + matches
        return doc





def process_text(text: str) -> Doc:
    # Process the text
    nlp = spacy.load("en_core_web_trf")
    if "ahocorasick" not in nlp.pipe_names:
        nlp.add_pipe("ahocorasick", last=True)
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

def summarize_text():
    pass

def classify_text():
    pass