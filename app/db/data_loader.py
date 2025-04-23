import json
import requests
from os import environ
from pyarrow.feather import read_feather
from io import BytesIO
import copy
import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker
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