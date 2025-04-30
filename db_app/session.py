import os
import sys
import json
import logging
import threading
import pandas as pd
import numpy as np
from enum import Enum
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.engine import Engine
from sqlalchemy.exc import ProgrammingError, SQLAlchemyError
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
    engine: Engine = create_engine(
        DATABASE_URL,
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_pre_ping=True  # Verify connections before using them
    )
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
    _lock = threading.Lock()  # Thread safety for singleton pattern
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                logger.info("Initializing DataStore singleton")
                cls._instance = super(DataStore, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        """Initialize the DataStore (only runs once due to singleton pattern)"""
        if not self._initialized:
            try:
                # Create engine with proper connection pooling
                self.engine = create_engine(
                    DATABASE_URL,
                    pool_size=5,
                    max_overflow=10,
                    pool_timeout=30,
                    pool_pre_ping=True
                )
                self.SessionMaker = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
                self.schema_name = "med_sync"
                
                # Load the sentence transformer model
                logger.info("Loading SentenceTransformer model...")
                self.model = SentenceTransformer("all-minilm-l6-v2")
                logger.info("SentenceTransformer model loaded successfully")
                
                # Setup database schema and extensions
                if not self._setup_database_prerequisites():
                    logger.error("Failed to set up database prerequisites")
                    raise RuntimeError("Database prerequisites setup failed")
                
                # Load concepts data
                logger.info("Loading concepts data...")
                self.concepts_with_sty_def_df = self.get_concepts_with_sty_def(self.engine)
                logger.info(f"Loaded concepts data with {len(self.concepts_with_sty_def_df)} rows")
                
                # Set up database tables if they don't exist
                self.initialize_database()
                
                self._initialized = True
                logger.info("DataStore initialization complete")
                
            except Exception as e:
                logger.error(f"Error initializing DataStore: {str(e)}")
                raise
    
    def _setup_database_prerequisites(self) -> bool:
        """Set up database schema and required extensions"""
        try:
            with self.SessionMaker() as session:
                # Create schema if it doesn't exist
                session.execute(text(f"CREATE SCHEMA IF NOT EXISTS {self.schema_name};"))
                
                # Ensure the pgvector extension is installed
                try:
                    session.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                    logger.info("Pgvector extension is available")
                except SQLAlchemyError as e:
                    logger.error(f"Error creating pgvector extension: {str(e)}")
                    logger.error("Make sure the pgvector extension is installed in PostgreSQL")
                    return False
                    
                session.commit()
                return True
                
        except Exception as e:
            logger.error(f"Error setting up database prerequisites: {str(e)}")
            return False
    
    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the schema using SQL
        
        Args:
            table_name: Name of the table to check
            
        Returns:
            True if the table exists, False otherwise
        """
        sql = """
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = :schema_name
                AND table_name = :table_name
            );
        """
        try:
            with self.SessionMaker() as session:
                result = session.execute(
                    text(sql), 
                    {"schema_name": self.schema_name, "table_name": table_name}
                ).scalar()
                return bool(result)
        except Exception as e:
            logger.error(f"Error checking if table {table_name} exists: {str(e)}")
            return False
            
    def get_row_count(self, table_name: str) -> int:
        """
        Get the number of rows in a table
        
        Args:
            table_name: Name of the table
            
        Returns:
            Number of rows or 0 if table doesn't exist or error occurs
        """
        try:
            with self.SessionMaker() as session:
                count = session.execute(
                    text(f"SELECT COUNT(*) FROM {self.schema_name}.{table_name}")
                ).scalar()
                return int(count)
        except Exception as e:
            logger.error(f"Error getting row count for {table_name}: {str(e)}")
            return 0
    
    def initialize_database(self):
        """Initialize database tables and data if needed"""
        logger.info("Checking database tables...")

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
        try:
            inspector = inspect(connection)
            table_name = "concepts_def_sty"
            
            if inspector.has_table(table_name):
                logger.info(f"Loading concepts from database table {table_name}")
                return pd.read_sql_table(table_name=table_name, con=connection)
            else:
                # Look for CSV file in both the app directory and the db_app directory
                possible_paths = [
                    resource_path(os.path.join("db_app", "data", "concepts_def_sty.csv")),
                    resource_path(os.path.join("data", "concepts_def_sty.csv")),
                    "concepts_def_sty.csv"  # Try current directory as last resort
                ]
                
                for csv_path in possible_paths:
                    if os.path.exists(csv_path):
                        logger.info(f"Loading concepts from CSV file {csv_path}")
                        return pd.read_csv(csv_path, header=0)
                
                # If no file found, create an empty dataframe with required columns
                logger.warning("No concepts data found. Creating empty dataframe.")
                return pd.DataFrame(columns=["CUI", "STR", "TUI", "STY", "DEF"])
                
        except Exception as e:
            logger.error(f"Error loading concepts data: {str(e)}")
            # Return empty dataframe with required columns
            return pd.DataFrame(columns=["CUI", "STR", "TUI", "STY", "DEF"])
            