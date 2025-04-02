import json
import requests
from os import environ
from typing import Dict, Any, Tuple, List
import pandas as pd
import pyarrow as pa
from pyarrow.feather import read_feather
from io import BytesIO
import copy
import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker
from app.schemas.section import TextCategoryEnum
from Levenshtein import ratio
from app.utils.nlp.nlp_utils import embed_text, DEFAULT_MODEL

# Set up database connection (adjust DATABASE_URL as needed)
# Database settings for PostgreSQL
DB_USER = environ.get('DB_USER', "root")
DB_PASSWORD = environ.get('DB_PASSWORD', "medsync!")
DB_HOST = environ.get('DB_HOST', "localhost")
DB_PORT = environ.get('DB_PORT', "9000")
DB_NAME = environ.get('DB_NAME', "medsync_db")
DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL)
SessionMaker = sessionmaker(bind=engine)

# Other constants
DATA_LOADER_URL = "http://127.0.0.1:8002"
schema_name = "med_sync"

print("Loading UMLS data...")

umls_df_dict: dict = {
    "concepts_with_sty_def_df": None,
    # "patient_information_df": None,
}

# Load UMLS concepts_with_sty_def data from the data loader URL
concepts_with_sty_def = BytesIO(requests.get(f"{DATA_LOADER_URL}/umls-data/symptoms-and-diseases").content)
umls_df_dict["concepts_with_sty_def_df"] = read_feather(concepts_with_sty_def)
concepts_with_sty_def.close()


def classify_text_category(input_text: str, threshold: float = 0.5) -> str:
    """
    Classify the input text using PostgreSQL session-based query with pgvector.
    """
    # Get the embedding as a NumPy array
    input_embedding = embed_text(input_text)
    
    # Convert NumPy array to a Python list, then to a string representation
    input_embedding_list = input_embedding.tolist()
    input_embedding_str = str(input_embedding_list)
    
    # For pgvector, use the '<->' operator for L2 distance or '<=>''for cosine distance
    sql = f"""
        SELECT category, term, term_embeddings <=> vector(:input_embedding) AS score
        FROM {schema_name}.TextCategoryEmbeddings
        ORDER BY score ASC
        LIMIT 1
    """
    
    with SessionMaker() as session:
        result = session.execute(text(sql), {"input_embedding": input_embedding_str}).fetchone()
        
    if result:
        category, db_term, score = result
        print(f"Vector result - Category: {category}, Term: {db_term}, Score: {score}")
        
        # Calculate string similarity
        string_sim = ratio(input_text.lower(), db_term.lower())
        print(f"String similarity: {string_sim:.4f}")
        
        if string_sim > 0.85 or float(score) <= threshold:
            return TextCategoryEnum[category].name
        else:
            return TextCategoryEnum.OTHERS.name
    else:
        return TextCategoryEnum.OTHERS.name
    
    
def find_content_dictionary(keyword_dict: dict, category: str) -> dict:
    """
    Query the PostgreSQL database using pgvector to find the closest matching
    content dictionary entry for a given keyword dictionary and category.
    """
    table_name = "ContentDictionaryJson"
    # Step 1: Convert the keyword_dict to an embedding.
    input_embedding = DEFAULT_MODEL.encode(json.dumps(keyword_dict), normalize_embeddings=True)
    input_embedding_list = input_embedding.tolist()
    input_embedding_str = str(input_embedding_list)
    # Use PostgreSQL syntax with pgvector operator (<=>) to measure distance.
    sql = f"""
        SELECT content_dictionary
        FROM {schema_name}.{table_name}
        WHERE category = :category
        ORDER BY description_embeddings <=> :input_embedding
        LIMIT 1
    """
    with SessionMaker() as session:
        result = session.execute(text(sql), {"category": category.upper(), "input_embedding": input_embedding_str}).fetchone()
    if result is not None:
        result_content = result[0]
        print(result_content)
        return json.loads(result_content)
    else:
        print("No matching content dictionary found.")
        return {}