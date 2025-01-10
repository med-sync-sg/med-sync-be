import os
from app import app
import pandas as pd
from sqlalchemy import create_engine
from app.utils import constants
from os import environ 

DB_USER = environ.get('DB_USER')
DB_PASSWORD = environ.get('DB_PASSWORD')
DB_HOST = environ.get('DB_HOST')
DB_PORT = environ.get('DB_PORT')
DB_NAME = environ.get('DB_NAME')
# Path to MRCONSO.RRF
UMLS_ROOT_DIRECTORY = os.path.join("umls", "2024AB", "META")

# Define column names as per UMLS documentation
columns = ["CUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF",
            "AUI", "SAUI", "SCUI", "SDUI", "SAB", "TTY", "CODE",
            "STR", "SRL", "SUPPRESS", "CVF"]

def load_concepts():
    # MRCONSO.RRF
    concepts = pd.read_csv(os.path.join(UMLS_ROOT_DIRECTORY, "MRCONSO.RRF"), sep='|', names=columns, index_col=False)
    # Remove the last empty column caused by trailing delimiter
    concepts = concepts.drop(concepts.columns[-1], axis=1)
    concepts = concepts.loc[concepts["LAT"] == "ENG"]
    # Keep only unique CUI-STR term pairs
    concepts = concepts.drop_duplicates(subset=["CUI", "STR"])
    print("UMLS English Concepts Loaded.")
    return concepts
        
def load_semantic_types():
    # MRSTY.RRF
    cols = ["CUI", "TUI", "STN", "STY", "ATUI", "CVF"]
    semantic_types = pd.read_csv(os.path.join(UMLS_ROOT_DIRECTORY, "MRSTY.RRF"), sep="|", names=cols, index_col=False)
    semantic_types = semantic_types[semantic_types["TUI"].isin(constants.SYMPTOMS_AND_DISEASES_TUI)]
    print("UMLS Semantic Types Loaded.")
    return semantic_types

def load_relationships():
    cols = ["CUI1", "AUI1", "STYPE1", "REL", "CUI2", "AUI2", "STYPE2", "RELA"]
    relationships = pd.read_csv(os.path.join(UMLS_ROOT_DIRECTORY, "MRREL.RRF"), sep="|", names=cols, usecols=[0,3,4,7], index_col=False)
    print("UMLS Relations Loaded.")
    return relationships

def combine_data(concepts, semantic_types, relationships):
    # MRREL.RRF
    concepts_with_types = pd.merge(concepts, semantic_types, on='CUI')

    # concepts_with_types = concepts_with_types[concepts_with_types['TUI'].isin(relevant_types)]
    concepts_with_types = concepts_with_types.drop_duplicates(subset=["CUI"])

    relationships_df = relationships.merge(concepts_with_types, left_on="CUI1", right_on="CUI")
    
    wanted_rela_labels = ["diagnostic_criteria_of", "defining_characteristic_of"]
    print(f'Relevant labels for RELA columns: {wanted_rela_labels}')
    
    filtered_relationships = relationships[relationships["RELA"].isin(wanted_rela_labels)]
    
    relationships_df = filtered_relationships[['CUI1', 'RELA', 'CUI2']]
    return relationships_df

def connect_to_docker_psql():
    engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
    return engine