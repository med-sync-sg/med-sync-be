import os
from os import environ 
import pandas as pd
import json
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Engine
from typing import List
from sqlalchemy.orm import sessionmaker

DB_USER = environ.get('DB_USER')
DB_PASSWORD = environ.get('DB_PASSWORD')
DB_HOST = environ.get('DB_HOST')
DB_PORT = environ.get('DB_PORT')
DB_NAME = environ.get('DB_NAME')

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

def create_session():
    engine: Engine = create_engine(DATABASE_URL)
    SessionMaker: sessionmaker = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    return SessionMaker
    

# Path to MRCONSO.RRF
UMLS_ROOT_DIRECTORY = os.path.join("umls", "2024AB", "META")

SYMPTOMS_AND_DISEASES_TUI = [
    'T047',  # Disease or Syndrome
    'T184',  # Sign or Symptom
]

DRUGS_AND_MEDICINES_TUI = [
    'T195',
    'T200',
]

PATIENT_INFORMATION_TUI = [
    'T98',
    'T99',
    'T100'
]

class DataStore:
    _instance = None
    engine = create_engine(DATABASE_URL)
    SessionMaker = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    concepts_df: pd.DataFrame
    definitions_df: pd.DataFrame
    relationships_df: pd.DataFrame
    semantic_df: pd.DataFrame
    concepts_with_sty_def_df: pd.DataFrame
    
    class Config:
        arbitrary_types_allowed = True
        
    def __new__(cls):
        if cls._instance is None:
            print("Loading DataFrame...")
            with cls.engine.connect() as conn:               
                # SQL DB loading
                cls._instance = super(DataStore, cls).__new__(cls)
                cls._instance.concepts_df = cls._instance.load_concepts(conn)
                cls._instance.definitions_df = cls._instance.load_definitions(conn)
                cls._instance.relationships_df = cls._instance.load_relationships(conn)
                cls._instance.semantic_df = cls._instance.load_semantic_types(conn)
                cls._instance.concepts_with_sty_def_df = cls._instance.get_concepts_with_sty_def(SYMPTOMS_AND_DISEASES_TUI, conn)
                print("Session loading completed.")
        return cls._instance

    def load_concepts(self, connection: Engine):
        columns = ["CUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF",
            "AUI", "SAUI", "SCUI", "SDUI", "SAB", "TTY", "CODE",
            "STR", "SRL", "SUPPRESS", "CVF"]
        
        # MRCONSO.RRF
        inspector = inspect(connection)
        table_name = "umls_concepts"
        if inspector.has_table(table_name):
            return pd.read_sql_table(table_name=table_name, con=connection)
        else:        
            concepts = pd.read_csv(os.path.join(UMLS_ROOT_DIRECTORY, "MRCONSO.RRF"), sep='|', names=columns, index_col=False)        
            # Remove the last empty column caused by trailing delimiter
            concepts = concepts.drop(concepts.columns[-1], axis=1)
            concepts = concepts.loc[concepts["LAT"] == "ENG"]
            # Keep only unique CUI-STR term pairs
            concepts = concepts.drop_duplicates(subset=["CUI", "STR"])
            print("UMLS English Concepts Loaded.")
            # print("Uploading UMLS concepts to the db...")
            # concepts.to_sql(table_name, con=connection)
            return concepts[["CUI", "LAT", "STR"]]

    def load_definitions(self, connection: Engine):
        # MRDEF.RRF
        cols = ["CUI", "AUI", "ATUI", "SATUI", "SAB", "DEF"]
        inspector = inspect(connection)
        table_name = "umls_definitions"
        if inspector.has_table(table_name):
            df = pd.read_sql_table(table_name=table_name, con=connection, columns=cols)
            return df
        else:        
            df = pd.read_csv(os.path.join(UMLS_ROOT_DIRECTORY, "MRDEF.RRF"), sep="|", names=cols, index_col=False)
            df = df[["CUI", "DEF"]]
            print("UMLS Definitions Loaded.")
            # print("Uploading UMLS definitions to the db...")
            # df.to_sql(table_name, con=connection)
            return df

    def load_semantic_types(self, connection: Engine):
        # MRSTY.RRF
        cols = ["CUI", "TUI", "STN", "STY", "ATUI", "CVF"]
        inspector = inspect(connection)
        table_name = "umls_semantic_types"
        if inspector.has_table(table_name):
            df = pd.read_sql_table(table_name=table_name, con=connection, columns=cols)
            return df
        else:        
            df = pd.read_csv(os.path.join(UMLS_ROOT_DIRECTORY, "MRSTY.RRF"), sep="|", names=cols, index_col=False)
            print("UMLS Semantic Types Loaded.")
            # print("Uploading UMLS semantic types to the db...")
            # df.to_sql(table_name, con=connection)

            return df[["CUI", "TUI", "STY"]]
    
    def load_relationships(self, connection: Engine):
        inspector = inspect(connection)
        cols = ["CUI1", "AUI1", "STYPE1", "REL", "CUI2", "AUI2", "STYPE2", "RELA"]
        table_name = "umls_relationships"
        if inspector.has_table(table_name):
            df = pd.read_sql_table(table_name=table_name, con=connection, columns=cols)
            return df
        else:
            df = pd.read_csv(os.path.join(UMLS_ROOT_DIRECTORY, "MRREL.RRF"), sep="|", names=cols, usecols=[0,3,4,7], index_col=False)
            print("UMLS Relationships Loaded.")
            # print("Uploading UMLS relationships to the db...")
            # df.to_sql(table_name, con=connection)
            return df[["CUI1", "REL", "CUI2", "RELA"]]

    
    def get_concepts_with_sty_def(self, target_tuis: List[str], connection: Engine):
        inspector = inspect(connection)
        table_name="concepts_def_sty"
        
        if inspector.has_table(table_name):
            df = pd.read_sql_table(table_name=table_name, con=connection)
            return df
        else:
            concepts_with_sty_def_df = self.concepts_df.merge(self.semantic_df, on="CUI", how="inner")
            concepts_with_sty_def_df = concepts_with_sty_def_df[concepts_with_sty_def_df["TUI"].isin(target_tuis)]
            concepts_with_sty_def_df = concepts_with_sty_def_df.merge(self.definitions_df, on="CUI", how="inner")
            
            concepts_with_sty_def_df["DEF"] = concepts_with_sty_def_df["DEF"].astype('string')
            concepts_with_sty_def_df = concepts_with_sty_def_df.drop_duplicates("CUI")
            concepts_with_sty_def_df = concepts_with_sty_def_df.dropna(subset=["CUI", "STR", "DEF", "STY"])
            concepts_with_sty_def_df.to_sql(table_name, con=connection)
            return concepts_with_sty_def_df
