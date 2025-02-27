import os
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker
from os import environ 
import pandas as pd
import json
from annoy import AnnoyIndex
import numpy as np
from sentence_transformers import SentenceTransformer

DB_USER = environ.get('DB_USER')
DB_PASSWORD = environ.get('DB_PASSWORD')
DB_HOST = environ.get('DB_HOST')
DB_PORT = environ.get('DB_PORT')
DB_NAME = environ.get('DB_NAME')
# Path to MRCONSO.RRF
UMLS_ROOT_DIRECTORY = os.path.join("umls", "2024AB", "META")

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

from app.models.models import Base

SYMPTOMS_AND_DISEASES_TUI = [
    'T047',  # Disease or Syndrome
    'T184',  # Sign or Symptom
]

class DataStore:
    _instance = None
    current_note_id : int = -1
    engine = create_engine(DATABASE_URL, echo=True)  # Logs SQL to console
    SessionMaker = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    concepts_df: pd.DataFrame
    relationships_df: pd.DataFrame
    semantic_df: pd.DataFrame
    combined_df: pd.DataFrame
    index: AnnoyIndex
    rep_terms: dict[str, str]
    embedding_model: SentenceTransformer
    
    def __new__(cls):
        if cls._instance is None:
            print("Loading DataFrame...")
            with cls.engine.connect() as conn:
                Base.metadata.create_all(bind=cls.engine)
                
                # SQL DB loading
                cls._instance = super(DataStore, cls).__new__(cls)
                cls._instance.concepts_df = cls._instance.load_concepts(conn)
                cls._instance.relationships_df = cls._instance.load_relationships(conn)
                cls._instance.semantic_df = cls._instance.load_semantic_types(conn)
                cls._instance.combined_df = cls._instance.combine_data(conn)
                
                # Embedding model for categorization
                cls._instance.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                cls._instance.rep_terms = cls._instance.load_section_rep_terms(conn)
                index, all_terms, term_categories = cls._instance.build_index()
                cls._instance.index = index
                cls._instance.all_terms = all_terms
                cls._instance.term_categories = term_categories
                print("Session loading completed.")
        return cls._instance
    # Define column names as per UMLS documentation

    def build_index(self):
        representative_terms = self.rep_terms

        all_terms = []
        term_categories = []
        for category, terms in representative_terms.items():
            for term in terms:
                all_terms.append(term)
                term_categories.append(category)

        print("Terms ready.")
        # Compute embeddings for each representative term.
        embeddings = self.embedding_model.encode(all_terms, convert_to_tensor=False, show_progress_bar=True)
        embeddings = np.vstack(embeddings).astype('float32')
    
        num_elements, dim = embeddings.shape
        print("Embeddings ready.")
        # Create an Annoy index.
        # 'angular' distance approximates cosine similarity.
        index = AnnoyIndex(dim, 'angular')
        for i in range(num_elements):
            index.add_item(i, embeddings[i])
        # Build the index with a chosen number of trees (more trees = higher accuracy, slower build time).
        index.build(10)
        print("Index built.")
        return index, all_terms, term_categories

    def upload_umls(self, engine: Engine):
        self.load_concepts(engine)
        self.load_semantic_types(engine)
        self.load_relationships(engine)

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
            return concepts
            
    def load_semantic_types(self, connection: Engine):
        # MRSTY.RRF
        cols = ["CUI", "TUI", "STN", "STY", "ATUI", "CVF"]
        inspector = inspect(connection)
        table_name = "umls_semantic_types"
        if inspector.has_table(table_name):
            df = pd.read_sql_table(table_name=table_name, con=connection, columns=cols)
            return df[df["TUI"].isin(SYMPTOMS_AND_DISEASES_TUI)]
        else:        
            df = pd.read_csv(os.path.join(UMLS_ROOT_DIRECTORY, "MRSTY.RRF"), sep="|", names=cols, index_col=False)
            print("UMLS Semantic Types Loaded.")
            # print("Uploading UMLS semantic types to the db...")
            # df.to_sql(table_name, con=connection)
            df = df[df["TUI"].isin(SYMPTOMS_AND_DISEASES_TUI)]
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

    def combine_data(self, connection: Engine):
        inspector = inspect(connection)
        table_name = "combined_table"
        if inspector.has_table(table_name):
            df = pd.read_sql_table(table_name=table_name, con=connection)
            return df
        else:
            concepts_with_types = pd.merge(self.concepts_df, self.semantic_df, on='CUI')

            # concepts_with_types = concepts_with_types[concepts_with_types['TUI'].isin(relevant_types)]
            concepts_with_types = concepts_with_types.drop_duplicates(subset=["CUI"])

            relationships_df = self.relationships_df.merge(concepts_with_types, left_on="CUI1", right_on="CUI")
            
            wanted_rela_labels = ["diagnostic_criteria_of", "defining_characteristic_of"]
            print(f'Relevant labels for RELA columns: {wanted_rela_labels}')
            
            filtered_relationships_df = relationships_df[relationships_df["RELA"].isin(wanted_rela_labels)]
            
            result_df = filtered_relationships_df[['CUI1', 'RELA', 'CUI2']]
            print("Uploading combined dataframe to the db...")
            filtered_relationships_df.to_sql(table_name, con=connection, if_exists="replace")
            return filtered_relationships_df

    def load_section_rep_terms(self, connection: Engine):
        """
        Loads UMLS concepts and semantic types, then filters the concepts
        to keep only those that belong to the TUI group defined in SYMPTOMS_AND_DISEASES_TUI.
        
        Returns:
            pd.DataFrame: Filtered concepts DataFrame.
        """
        inspector = inspect(connection)
        table_name = "section_representative_terms"

        if inspector.has_table(table_name):
            query = "SELECT category, terms FROM section_representative_terms"
            df = pd.read_sql(query, connection)
            print("Loaded representative terms.")
            # If the 'terms' column is stored as a JSON string, convert it to a list.
            # (If it's already parsed into Python objects, this step might not be needed.)
            df['terms'] = df['terms'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)            
            # Build the dictionary
            rep_terms_dict = pd.Series(df["terms"].values, index=df["categories"]).to_dict()
            return rep_terms_dict
        else:
            # Load concepts and semantic types using the provided functions.
            concepts = self.concepts_df
            semantic_types = self.semantic_df
        
            # Filter concepts: keep only rows where the CUI is present in the semantic types.
            filtered_concepts = concepts.merge(semantic_types[['CUI']], on="CUI", how="inner")
                
            rep_terms = {
                "CHIEF_COMPLAINT": [],
                "PATIENT_INFORMATION": [],
                "OTHERS": []
            }
            rep_terms["CHIEF_COMPLAINT"] = filtered_concepts["STR"].to_list()
            # Remove duplicates.
            rep_terms["CHIEF_COMPLAINT"] = list(set(rep_terms["CHIEF_COMPLAINT"]))
            
            # For PATIENT_INFORMATION, use a manually defined set of representative terms.
            rep_terms["PATIENT_INFORMATION"] = ["age", "gender", "occupation", "address", "name"]
            
            # For OTHERS, we add a generic term.
            rep_terms["OTHERS"] = ["miscellaneous", "other"]
            # Convert the dictionary into a list of tuples, then create a DataFrame.
            data = [(category, terms) for category, terms in rep_terms.items()]
            df = pd.DataFrame(data, columns=['Category', 'Terms'])
            df.to_sql(table_name, connection, if_exists="replace")
            print("Uploaded representative terms.")
            return rep_terms
    
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
        db = self.__class__.SessionMaker()
        try:
            yield db
        finally:
            db.close()