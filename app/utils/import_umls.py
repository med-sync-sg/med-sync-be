import os
from app import app
import pandas as pd
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Engine
from app.utils import constants
from os import environ 
import pandas as pd
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

def upload_umls(engine: Engine):
    load_concepts(engine)
    load_semantic_types(engine)
    load_relationships(engine)

def load_concepts(connection: Engine):
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
        return concepts
        
def load_semantic_types(connection: Engine):
    # MRSTY.RRF
    cols = ["CUI", "TUI", "STN", "STY", "ATUI", "CVF"]
    inspector = inspect(connection)
    table_name = "umls_semantic_types"
    if inspector.has_table(table_name):
        df = pd.read_sql_table(table_name=table_name, con=connection, columns=cols)
        return df[df["TUI"].isin(constants.SYMPTOMS_AND_DISEASES_TUI)]
    else:        
        df = pd.read_csv(os.path.join(UMLS_ROOT_DIRECTORY, "MRSTY.RRF"), sep="|", names=cols, index_col=False)
        print("UMLS Semantic Types Loaded.")
        # print("Uploading UMLS semantic types to the db...")
        # df.to_sql(table_name, con=connection)
        df = df[df["TUI"].isin(constants.SYMPTOMS_AND_DISEASES_TUI)]
        return df

def load_relationships(connection: Engine):
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
        return df

def combine_data(concepts: pd.DataFrame, semantic_types: pd.DataFrame, relationships: pd.DataFrame, connection: Engine):
    inspector = inspect(connection)
    table_name = "combined_table"
    if inspector.has_table(table_name):
        df = pd.read_sql_table(table_name=table_name, con=connection)
        return df
    else:
        concepts_with_types = pd.merge(concepts, semantic_types, on='CUI')

        # concepts_with_types = concepts_with_types[concepts_with_types['TUI'].isin(relevant_types)]
        concepts_with_types = concepts_with_types.drop_duplicates(subset=["CUI"])

        relationships_df = relationships.merge(concepts_with_types, left_on="CUI1", right_on="CUI")
        
        wanted_rela_labels = ["diagnostic_criteria_of", "defining_characteristic_of"]
        print(f'Relevant labels for RELA columns: {wanted_rela_labels}')
        
        filtered_relationships_df = relationships_df[relationships_df["RELA"].isin(wanted_rela_labels)]
        
        result_df = filtered_relationships_df[['CUI1', 'RELA', 'CUI2']]
        print("Uploading combined dataframe to the db...")
        filtered_relationships_df.to_sql(table_name, con=connection, if_exists="replace")
        return filtered_relationships_df

class DataStore:
    _instance = None
    connection = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
    def __new__(cls):
        if cls._instance is None:
            print("Loading UMLS DataFrame...")
            cls._instance = super(DataStore, cls).__new__(cls)
            cls._instance.concepts_df = load_concepts(cls._instance.connection)
            cls._instance.relationships_df = load_relationships(cls._instance.connection)
            cls._instance.semantic_df = load_semantic_types(cls._instance.connection)
            cls._instance.combined_df = combine_data(cls._instance.concepts_df, cls._instance.semantic_df, cls._instance.relationships_df, cls._instance.connection)
            print("UMLS DataFrame loading completed.")
        return cls._instance
    
    def get_combined_df(self):
        return self.combined_df