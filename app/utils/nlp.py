from spacy.matcher import PhraseMatcher
import pandas as pd
import ahocorasick
import spacy
from spacy import Language
from fastapi import APIRouter
from spacy.tokens import Doc
from app.utils.import_umls import DataStore
from spacy import displacy
from pathlib import Path

router = APIRouter()


class AhoCorasickComponent:
    def __init__(self):
        self.automaton = None

    def initialize(self, get_examples=None, nlp: Language=None, config: dict={}):
        """
        spaCy calls this at `nlp.initialize()` time.
        Here, load your DataFrame and build the Aho-Corasick automaton once.
        """
        df : pd.DataFrame = DataStore().get_combined_df() 
        self.automaton = ahocorasick.Automaton()
        print(df.head())
        for index, row in df.iterrows():
            term = row['STR']
            data = {
                'term': term.lower(),
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
        doc.ents = matches
        return doc

@Language.factory("ahocorasick")
def create_ahocorasick_component(nlp: Language, name: str, config: dict={}):
    """
    This factory function returns an instance of AhoCorasickComponent.
    spaCy will call its initialize() method when you run nlp.initialize().
    """
    return AhoCorasickComponent()


nlp_en = spacy.load("en_core_web_trf")
if "ahocorasick" not in nlp_en.pipe_names:
    nlp_en.add_pipe("ahocorasick", last=True)
nlp_en.initialize()

def process_text(text: str) -> Doc:
    # Process the text
    doc = nlp_en(text)
    rendered = displacy.render(doc, style="ent")
    output_path = Path("./images/sentence.html")
    output_path.open("w").write(rendered)
    return doc


def summarize_text():
    pass

def classify_text():
    pass