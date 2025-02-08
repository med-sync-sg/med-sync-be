import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

nltk.download('punkt')

def split_sentences(text: str) -> list:
    return sent_tokenize(text)

def get_sentence_vectors(sentences: list):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)
    return tfidf_matrix.toarray()  # Each row corresponds to a sentence vector

def build_similarity_matrix(sentence_vectors: np.ndarray) -> np.ndarray:
    # Compute cosine similarity between all pairs of sentence vectors
    sim_matrix = cosine_similarity(sentence_vectors)
    # Optionally, zero-out the diagonal (self-similarity)
    np.fill_diagonal(sim_matrix, 0)
    return sim_matrix

def rank_sentences(sim_matrix: np.ndarray) -> list:
    # Create a graph where each sentence is a node
    nx_graph = nx.from_numpy_array(sim_matrix)
    # Apply the PageRank algorithm
    scores = nx.pagerank_numpy(nx_graph)
    # Return the sentences ranked by their PageRank score
    ranked_sentences = sorted(((score, idx) for idx, score in scores.items()), reverse=True)
    return ranked_sentences  # List of tuples (score, sentence_index)

def generate_summary(text: str, top_n: int = 3) -> str:
    sentences = split_sentences(text)
    if len(sentences) <= top_n:
        return text  # If there are fewer sentences than top_n, return full text
    
    sentence_vectors = get_sentence_vectors(sentences)
    sim_matrix = build_similarity_matrix(sentence_vectors)
    ranked = rank_sentences(sim_matrix)
    
    # Select the indices of the top ranked sentences
    top_sentence_indices = sorted([idx for _, idx in ranked[:top_n]])
    # Reorder them in the original order
    summary = " ".join([sentences[i] for i in top_sentence_indices])
    return summary
