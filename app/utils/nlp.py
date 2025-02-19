from typing import List, Tuple, Dict, Any
from spacy.matcher import PhraseMatcher
import pandas as pd
import ahocorasick
from spacy import Language
from fastapi import APIRouter
import spacy.tokenizer
from spacy.tokens import Doc
import spacy.tokens
from app.utils.import_umls import DataStore
from spacy import displacy
from pathlib import Path
from app.schemas.schemas import Section, TextCategoryEnum, ChiefComplaintSection, PatientInformationSection
from sentence_transformers import SentenceTransformer, util
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from os import environ

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


def summarize_text():
    pass

def load_labse_model() -> tuple[SentenceTransformer, dict]:
    # Load LaBSE (bi-encoder model)
    # Note: "sentence-transformers/LaBSE" is a popular checkpoint on Hugging Face
    labse_model = SentenceTransformer("sentence-transformers/LaBSE", token=HF_TOKEN)

    # Precompute topic embeddings
    topic_embeddings = {}
    for category in TextCategoryEnum:
        # Convert definition to an embedding
        emb = labse_model.encode(category.value, convert_to_tensor=True)
        topic_embeddings[category.name] = emb
    
    return (labse_model, topic_embeddings)
        

def get_chunk_embedding(chunk_text: str):
    return labse_model.encode(chunk_text, convert_to_tensor=True)

def rank_candidates_bi_encoder(chunk_emb, topic_embeddings: Dict[str, Any], top_k=2):
    """Return the top_k candidate topics based on cosine similarity."""
    scores = []
    for label, topic_emb in topic_embeddings.items():
        score = util.cos_sim(chunk_emb, topic_emb).item()  # or dot score
        scores.append((label, score))

    # Sort by descending score
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return the top K
    return scores[:top_k]

def cross_encoder_score(text_a, text_b):
    """Return the relevance score from a cross-encoder."""
    inputs = re_ranker_tokenizer.encode_plus(
        text_a, text_b, return_tensors="pt", truncation=True, max_length=128
    )
    with torch.no_grad():
        outputs = re_ranker_model(**inputs)
        logits = outputs.logits
        # Typically for a cross-encoder trained on a ranking task, the final dimension is 1 or 2
        # Suppose it's 1-dimensional. Higher = more relevant
        score = logits.squeeze().item()  
    return score

def re_rank_candidates(chunk_text, candidates):
    """Given a chunk and top candidates, re-rank with a cross-encoder."""
    # candidates is a list of (label, bi_encoder_score)
    results = []
    for label, _ in candidates:
        label_def = TextCategoryEnum[label].value  # label definition
        ce_score = cross_encoder_score(chunk_text, label_def)
        results.append((label, ce_score))
    # Sort by descending re-ranker score
    results.sort(key=lambda x: x[1], reverse=True)
    print("Re-ranked candidates:", results)
    return results

def identify_text_topic(text: str, entities: List[str]) -> Tuple[str, float]:
    # Step 1: Compute the chunk embedding using the augmented text.
    chunk_emb = get_chunk_embedding(text)
    
    # Step 2: Rank candidate topics using the bi-encoder (LaBSE) step.
    candidates = rank_candidates_bi_encoder(chunk_emb, topic_embeddings, top_k=10)
    print("Top candidates from LaBSE:", candidates)
    
    # Step 3: Re-rank the candidates using your re-ranker.
    final_ranking = re_rank_candidates(text, candidates)
    print("Re-ranked candidates:", final_ranking)
    
    # Select the best candidate.
    best_label, best_score = final_ranking[0]
    print("Chosen label:", best_label, best_score)
    
    return (best_label, best_score)

def categorize_doc(doc: spacy.tokens.Doc) -> Dict[str, List[str]]:
    """
    Identify topic segments for a transcript by combining the raw text with
    NER tags from the Spacy Doc object.
    
    :param doc: A Spacy Doc object of the full transcript with NER annotations.
    :return: A list of Section objects (or prints the chosen label for the chunk).
    """
    result : Dict[str, List[str]] = {}
    for category in TextCategoryEnum:
        # Populate the result dict with using TextCategoryEnum names as keys
        # and an empty list as the value
        # Allows the next part to categorize text
        result[category.name] = []

    full_text = doc.text
    lines = full_text.split("\n")    

    offset = 0  
    for line in lines:
        start_offset = offset
        end_offset = offset + len(line)
        
        # Find entities whose character span falls within the current line.
        line_entities = [
            ent.label_ for ent in doc.ents
            if ent.start_char >= start_offset and ent.end_char <= end_offset
        ]
        # Remove duplicate entity labels.
        line_entities = list(set(line_entities))
        
        # Update the offset; assume a newline is one character.
        offset = end_offset + 1
        best_label, score = identify_text_topic(text=line, entities=line_entities)
        result[best_label].append(line)

    return result

        
nlp_en = spacy.load("en_core_web_trf")
if "ahocorasick" not in nlp_en.pipe_names:
    nlp_en.add_pipe("ahocorasick", last=True)
labse_model, topic_embeddings = load_labse_model()

def process_text(text: str) -> Doc:
    # Process the text
    doc = nlp_en(text)
    return doc
