import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from enum import Enum
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.engine import Engine
from Levenshtein import ratio
from os import environ
from typing import List, Any, Tuple, Dict, Optional
from sentence_transformers import SentenceTransformer

# Configure logger
logger = logging.getLogger(__name__)

# Database settings for PostgreSQL
DB_USER = environ.get('DB_USER', "root")
DB_PASSWORD = environ.get('DB_PASSWORD', "medsync!")
DB_HOST = environ.get('DB_HOST', "localhost")
DB_PORT = environ.get('DB_PORT', "9000")
DB_NAME = environ.get('DB_NAME', "medsync_db")
DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

def create_session() -> sessionmaker:
    """Create SQLAlchemy session maker"""
    engine: Engine = create_engine(DATABASE_URL)
    SessionMaker: sessionmaker = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionMaker

def resource_path(relative_path):
    """Get absolute path to resource, works in dev and for PyInstaller"""
    try:
        base_path = sys._MEIPASS  # for PyInstaller onefile bundles
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class TextCategoryEnum(str, Enum):
    """Text categories for classification"""
    CHIEF_COMPLAINT = "This text describes the patient's primary symptoms..."
    PATIENT_INFORMATION = "This text describes the demographic and personal details..."
    PATIENT_MEDICAL_HISTORY = "This text describes the patient's medical history..."
    OTHERS = "This text refers to all other contents not classified above."

# TUI lists
SYMPTOMS_AND_DISEASES_TUI = ['T047', 'T184']
DRUGS_AND_MEDICINES_TUI = ['T195', 'T200']
PATIENT_INFORMATION_TUI = ['T98', 'T99', 'T100']

class DataStore:
    """
    Singleton class for database operations and embeddings management.
    Handles database setup, vector embeddings, and text classification.
    """
    _instance = None
    engine = create_engine(DATABASE_URL)
    SessionMaker = create_session()
    concepts_with_sty_def_df: pd.DataFrame
    schema_name = "med_sync"
    model = None  # Will be initialized on demand
    
    def __new__(cls):
        if cls._instance is None:
            logger.info("Initializing DataStore singleton")
            cls._instance = super(DataStore, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the DataStore (only runs once due to singleton pattern)"""
        if not getattr(self, "_initialized", False):
            try:
                # Load the sentence transformer model
                self.model = SentenceTransformer("all-minilm-l6-v2")
                logger.info("Loaded SentenceTransformer model")
                
                # Ensure schema exists
                with self.SessionMaker() as session:
                    session.execute(text(f"CREATE SCHEMA IF NOT EXISTS {self.schema_name};"))
                    session.commit()
                    logger.info(f"Ensured schema '{self.schema_name}' exists")
                
                # Load concepts data
                self.concepts_with_sty_def_df = self.get_concepts_with_sty_def(self.engine)
                logger.info(f"Loaded concepts data with {len(self.concepts_with_sty_def_df)} rows")
                
                # Set up database tables if they don't exist
                self.initialize_database()
                
                self._initialized = True
                logger.info("DataStore initialization complete")
            except Exception as e:
                logger.error(f"Error initializing DataStore: {str(e)}")
                raise
    
    def initialize_database(self):
        """Initialize database tables and data if needed"""
        # Check which tables need to be created
        inspector = inspect(self.engine)
        tables_to_create = []
        
        for table_name in ["TextCategories", "TextCategoryEmbeddings", "ContentDictionaryJson"]:
            if not inspector.has_table(table_name, schema=self.schema_name):
                tables_to_create.append(table_name)
        
        if not tables_to_create:
            logger.info("All required tables already exist, skipping database initialization")
            return
        
        logger.info(f"Tables to create: {tables_to_create}")
        
        # Create tables and load data where needed
        if "TextCategories" in tables_to_create:
            self.set_up_categories()
        
        if "TextCategoryEmbeddings" in tables_to_create:
            self.set_up_category_embeddings()
        
        if "ContentDictionaryJson" in tables_to_create:
            self.set_up_content_dictionary_embeddings()

    def set_up_categories(self):
        """Set up the text categories table"""
        table_name = "TextCategories"
        logger.info(f"Creating table {self.schema_name}.{table_name}")
        
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.schema_name}.{table_name} (
            category VARCHAR(1000), 
            description VARCHAR(20000), 
            description_embeddings vector(384)
        )
        """
        
        with self.SessionMaker() as session:
            session.execute(text(create_table_sql))
            
            # Prepare data for all categories
            data = {
                "category": [],
                "description": [],
                "description_embeddings": [],
            }
            
            for text_category in TextCategoryEnum:
                data["category"].append(text_category.name)
                data["description"].append(text_category.value)
                # Get the embedding as a list of floats
                emb = self.model.encode(text_category.value, normalize_embeddings=True).tolist()
                data["description_embeddings"].append(emb)
            
            # Create rows for insertion
            rows = [
                {
                    "category": row['category'],
                    "description": row['description'],
                    "description_embeddings": row["description_embeddings"]
                }
                for index, row in pd.DataFrame(data=data).iterrows()
            ]
            
            # Insert data
            insert_sql = f"""
                INSERT INTO {self.schema_name}.{table_name}
                (category, description, description_embeddings)
                VALUES (:category, :description, :description_embeddings)
            """
            
            for row in rows:
                session.execute(text(insert_sql), row)
            
            session.commit()
            logger.info(f"Inserted {len(rows)} categories into {self.schema_name}.{table_name}")

    def _get_category(self, tui: str) -> str:
        """Map TUI code to category name"""
        if tui in SYMPTOMS_AND_DISEASES_TUI:
            return TextCategoryEnum.CHIEF_COMPLAINT.name
        if tui in PATIENT_INFORMATION_TUI:
            return TextCategoryEnum.PATIENT_INFORMATION.name
        return TextCategoryEnum.OTHERS.name

    def set_up_category_embeddings(self):
        """Set up the text category embeddings table"""
        table_name = "TextCategoryEmbeddings"
        logger.info(f"Creating table {self.schema_name}.{table_name}")
        
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.schema_name}.{table_name} (
            cui VARCHAR(100), 
            term VARCHAR(1000), 
            term_embeddings vector(384), 
            category VARCHAR(1000), 
            semantic_type VARCHAR(1000), 
            description VARCHAR(20000), 
            description_embeddings vector(384)
        )
        """
        
        with self.SessionMaker() as session:
            session.execute(text(create_table_sql))
            session.commit()
            
            # Get the data
            full_df = self.concepts_with_sty_def_df
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(full_df)} terms")
            term_embeddings = self.model.encode(full_df["STR"].tolist(), normalize_embeddings=True).tolist()
            description_embeddings = self.model.encode(full_df["DEF"].tolist(), normalize_embeddings=True).tolist()
            
            # Generate categories
            categories = full_df["TUI"].apply(lambda element: self._get_category(tui=element))
            
            # Batch insert rows
            batch_size = 1000
            total_rows = len(full_df)
            
            insert_sql = f"""
                INSERT INTO {self.schema_name}.{table_name}
                (cui, term, term_embeddings, category, semantic_type, description, description_embeddings)
                VALUES (:cui, :term, :term_embeddings, :category, :semantic_type, :description, :description_embeddings)
            """
            
            for i in range(0, total_rows, batch_size):
                end_idx = min(i + batch_size, total_rows)
                logger.info(f"Inserting batch {i//batch_size + 1}/{(total_rows + batch_size - 1)//batch_size}: rows {i} to {end_idx}")
                
                # Create batch of rows
                batch = [
                    {
                        "cui": full_df["CUI"].iloc[j],
                        "term": full_df["STR"].iloc[j],
                        "term_embeddings": term_embeddings[j],
                        "category": categories.iloc[j],
                        "semantic_type": full_df["STY"].iloc[j],
                        "description": full_df["DEF"].iloc[j],
                        "description_embeddings": description_embeddings[j]
                    }
                    for j in range(i, end_idx)
                ]
                
                # Execute batch insert
                session.execute(text(insert_sql), batch)
                session.commit()
            
            logger.info(f"Successfully uploaded {total_rows} embeddings to {self.schema_name}.{table_name}")
            
    def set_up_content_dictionary_embeddings(self):
        """Set up the content dictionary JSON table"""
        table_name = "ContentDictionaryJson"
        logger.info(f"Creating table {self.schema_name}.{table_name}")
        
        create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {self.schema_name}.{table_name} (
                content_dictionary VARCHAR(32000), 
                category VARCHAR(1000), 
                description VARCHAR(20000), 
                embeddings vector(384), 
                description_embeddings vector(384)
            )
        """
        
        with self.SessionMaker() as session:
            session.execute(text(create_table_sql))
            session.commit()
            
            # Define dictionary templates
            chief_complaint_example = {
                "Main Symptom": {
                    "name": "The name of the primary symptom (e.g., headache, diarrhea)",
                    "duration": "The duration of the symptom (e.g., 3 days)",
                    "severity": "The severity level of the symptom (e.g., mild, moderate, severe)",
                    "additional_details": "Additional details regarding the symptom, such as frequency, triggers, or associated features"
                }
            }
            
            patient_information_example = {
                "Demographics": {
                    "name": "Patient's full name",
                    "age": "Patient's age in years",
                    "gender": "Patient's gender (e.g., Male, Female)"
                },
                "Contact Information": {
                    "address": "Patient's address (e.g., street, city, country, postal code)",
                    "phone": "Patient's contact phone number"
                },
                "Occupation": "Patient's occupation (e.g., teacher, pharmaceutical manager)",
                "Additional Details": "Other relevant personal details (e.g., marital status, living conditions, recent travel history)"
            }
            
            medical_history_example = {
                "Chronic Conditions": [
                    {
                        "name": "The name of a chronic condition (e.g., eczema)",
                        "diagnosed_date": "The date the condition was diagnosed (format: YYYY-MM-DD)",
                        "severity": "The severity of the condition (e.g., mild, moderate, severe)",
                        "notes": "Additional notes regarding the condition (e.g., treatment response, frequency of flare-ups)"
                    }
                ],
                "Past Diagnoses": [
                    {
                        "name": "The name of a past diagnosis (e.g., asthma)",
                        "diagnosed_date": "The date of diagnosis (format: YYYY-MM-DD)",
                        "notes": "Additional details regarding the diagnosis (e.g., management, history)"
                    }
                ],
                "Medications": [
                    {
                        "name": "The name of a medication (e.g., Ibuprofen)",
                        "dosage": "The dosage of the medication (e.g., 200 mg)",
                        "frequency": "The frequency at which the medication is taken (e.g., as needed)",
                        "notes": "Additional notes (e.g., response, side effects)"
                    }
                ],
                "Allergies": [
                    {
                        "substance": "The allergen or substance causing an allergic reaction (e.g., penicillin)",
                        "notes": "Details about the allergy or any additional notes (e.g., NKDA if none)"
                    }
                ]
            }
            
            others_examples = {
                "Other Observations": [
                    {
                        "observation": "A miscellaneous observation not covered by the primary categories",
                        "notes": "Additional details or context regarding the observation"
                    }
                ]
            }
    
            examples = [
                (chief_complaint_example, TextCategoryEnum.CHIEF_COMPLAINT.name, "Contains chief complaints or main symptoms of the patient."), 
                (patient_information_example, TextCategoryEnum.PATIENT_INFORMATION.name, "Contains patient information such as demographics."), 
                (medical_history_example, TextCategoryEnum.PATIENT_MEDICAL_HISTORY.name, "Contains the medical history information of the patient such as current medications or chronic illnesses."), 
                (others_examples, TextCategoryEnum.OTHERS.name, "Any other information belongs to this section.")
            ]
    
            # Generate embeddings and build rows for insertion
            insert_sql = f"""
                INSERT INTO {self.schema_name}.{table_name}
                (content_dictionary, category, description, embeddings, description_embeddings)
                VALUES (:content_dictionary, :category, :description, :embeddings, :description_embeddings)
            """
    
            for example, category, description in examples:
                row = {
                    "content_dictionary": json.dumps(example),
                    "category": category,
                    "description": description,
                    "embeddings": self.embed_dictionary(example).tolist(),
                    "description_embeddings": self.embed_text(description).tolist()
                }
                session.execute(text(insert_sql), row)
    
            session.commit()
            logger.info(f"Successfully set up {len(examples)} dictionary templates in {self.schema_name}.{table_name}")

    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding vector for text"""
        return self.model.encode(text, normalize_embeddings=True)

    def embed_list(self, input_list: list) -> List[Any]:
        """Generate embeddings for a list of texts"""
        embeddings = self.model.encode(input_list, normalize_embeddings=True)
        return [emb.tolist() for emb in embeddings]

    def embed_dictionary(self, data: Any) -> np.ndarray:
        """Generate embedding for a dictionary (serialized to JSON)"""
        return self.model.encode(json.dumps(data), normalize_embeddings=True)

    def get_concepts_with_sty_def(self, connection: Engine) -> pd.DataFrame:
        """Get concepts dataframe from database or fallback to CSV file"""
        inspector = inspect(connection)
        table_name = "concepts_def_sty"
        
        if inspector.has_table(table_name):
            logger.info(f"Loading concepts from database table {table_name}")
            return pd.read_sql_table(table_name=table_name, con=connection)
        else:
            csv_path = resource_path(os.path.join("db_app", "data", "concepts_def_sty.csv"))
            logger.info(f"Loading concepts from CSV file {csv_path}")
            return pd.read_csv(csv_path, header=0)
            
    def classify_text_category(self, input_text: str, threshold: float = 0.5) -> str:
        """
        Classify the input text by performing a vector search against stored category embeddings.
        
        Args:
            input_text: The text to classify
            threshold: Similarity threshold for classification
            
        Returns:
            The name of the matching category or OTHERS if no match
        """
        # Get the embedding as a list, then convert to string (for PostgreSQL pgvector)
        input_embedding = self.embed_text(input_text).tolist()
        
        # Query using cosine distance operator <=> for pgvector
        sql = f"""
            SELECT category, term, term_embeddings <=> :input_embedding AS score
            FROM {self.schema_name}.TextCategoryEmbeddings
            ORDER BY score ASC
            LIMIT 1
        """
        
        with self.SessionMaker() as session:
            result = session.execute(
                text(sql), 
                {"input_embedding": input_embedding}
            ).fetchone()
        
        if result:
            category, db_term, score = result
            logger.debug(f"Vector result - Category: {category}, Term: {db_term}, Score: {score}")
            
            # Also check string similarity as an additional signal
            string_sim = ratio(input_text.lower(), db_term.lower())
            logger.debug(f"String similarity: {string_sim:.4f}")
            
            if string_sim > 0.85 or float(score) <= threshold:
                return TextCategoryEnum[category].name
        
        # Default to OTHERS if no good match found
        return TextCategoryEnum.OTHERS.name
            
    def find_content_dictionary(self, keyword_dict: dict, category: str) -> dict:
        """
        Find a matching content dictionary template for the given keywords and category.
        
        Args:
            keyword_dict: Dictionary of extracted keywords
            category: Text category name
            
        Returns:
            The best matching content dictionary template
        """
        table_name = "ContentDictionaryJson"
        
        # Convert the keyword dictionary to an embedding
        input_embedding = self.embed_dictionary(keyword_dict).tolist()
        
        # Query for the best matching template
        sql = f"""
            SELECT content_dictionary
            FROM {self.schema_name}.{table_name}
            WHERE category = :category
            ORDER BY description_embeddings <=> :input_embedding ASC
            LIMIT 1
        """
        
        with self.SessionMaker() as session:
            result = session.execute(
                text(sql),
                {
                    "category": category.upper(),
                    "input_embedding": input_embedding
                }
            ).fetchone()
            
            if result:
                return json.loads(result[0])
        
        # If no template found, return a minimal default template
        return {"Main Content": {"text": ""}}