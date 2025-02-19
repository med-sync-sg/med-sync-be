import os
import pandas as pd
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker
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

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

from app.models.models import Base

columns = ["CUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF",
            "AUI", "SAUI", "SCUI", "SDUI", "SAB", "TTY", "CODE",
            "STR", "SRL", "SUPPRESS", "CVF"]

class DataStore:
    _instance = None
    engine = create_engine(DATABASE_URL, echo=True)  # Logs SQL to console
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def __new__(cls):
        if cls._instance is None:
            print("Loading DataFrame...")
            with cls.engine.connect() as conn:
                Base.metadata.create_all(bind=cls.engine)

                cls._instance = super(DataStore, cls).__new__(cls)
                cls._instance.concepts_df = cls._instance.load_concepts(conn)
                cls._instance.relationships_df = cls._instance.load_relationships(conn)
                cls._instance.semantic_df = cls._instance.load_semantic_types(conn)
                cls._instance.combined_df = cls._instance.combine_data(cls._instance.concepts_df, cls._instance.semantic_df, cls._instance.relationships_df, conn)
        return cls._instance
    
    # Define column names as per UMLS documentation

    def upload_umls(self, engine: Engine):
        self.load_concepts(engine)
        self.load_semantic_types(engine)
        self.load_relationships(engine)

    def load_concepts(self, connection: Engine):
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
            
    def load_semantic_types(self, connection: Engine):
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
            return df

    def combine_data(self, concepts: pd.DataFrame, semantic_types: pd.DataFrame, relationships: pd.DataFrame, connection: Engine):
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

    
    def get_combined_df(self):
        return self.combined_df
    
    def get_db(self):
        """
        Provides a DB session for frameworks like FastAPI.
        Usage example (with FastAPI):
            @app.get("/items")
            def read_items(db: Session = Depends(data_store.get_db)):
                ...
        """
        db = self.__class__.SessionLocal()
        try:
            yield db
        finally:
            db.close()