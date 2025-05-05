from neo4j import GraphDatabase
import logging
import os
from typing import Dict, List, Any, Optional, Union
from contextlib import contextmanager
from sentence_transformers import SentenceTransformer
from fastapi import Depends

# Configure logger
logger = logging.getLogger(__name__)

class Neo4jSession:
    """
    Singleton class for managing Neo4j database connections in the app
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Neo4jSession, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the Neo4j connection (runs only once)"""
        if not self._initialized:
            # Get connection parameters from environment
            self.uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
            self.user = os.environ.get("NEO4J_USER", "neo4j")
            self.password = os.environ.get("NEO4J_PASSWORD", "medsync!")
            
            # Initialize driver
            self.driver = None
            self._connect()
            
            # Initialize embedding model
            try:
                self.model = SentenceTransformer("BAAI/bge-small-en-v1.5")
                self.embedding_dimension = 384  # all-minilm-l6-v2 outputs 384-dimensional embeddings
                logger.info("Loaded SentenceTransformer model for embeddings")
            except Exception as e:
                logger.error(f"Error loading SentenceTransformer model: {str(e)}")
                self.model = None
            
            self._initialized = True
            logger.info(f"Neo4jSession initialized with connection to {self.uri}")
    
    def _connect(self):
        """Establish connection to Neo4j"""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            logger.info(f"Connected to Neo4j at {self.uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            self.driver = None
    
    def close(self):
        """Close the Neo4j connection"""
        if self.driver:
            self.driver.close()
            self.driver = None
            logger.info("Neo4j connection closed")
    
    @contextmanager
    def get_session(self):
        """
        Context manager for Neo4j sessions
        
        Usage:
            with neo4j_session.get_session() as session:
                result = session.run("MATCH (n) RETURN count(n)")
        """
        if not self.driver:
            logger.error("No active Neo4j connection")
            raise ConnectionError("No active Neo4j connection")
            
        session = None
        try:
            session = self.driver.session()
            yield session
        finally:
            if session:
                session.close()
    
    def run_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Run a Cypher query with parameters
        
        Args:
            query: Cypher query string
            parameters: Query parameters (optional)
            
        Returns:
            List of records as dictionaries
        """
        if not self.driver:
            logger.error("No active Neo4j connection")
            return []
        
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                records = [record.data() for record in result]
                return records
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            return []
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text
        
        Args:
            text: Text to encode
            
        Returns:
            Vector embedding as list of floats
        """
        if not self.model:
            logger.error("SentenceTransformer model not available")
            return [0.0] * self.embedding_dimension  # Return zero vector
            
        try:
            embedding = self.model.encode(text).tolist()
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return [0.0] * self.embedding_dimension  # Return zero vector
    
    def run_vector_search(self, label: str, embedding_field: str, 
                         text: str = "", vector: List[float] = None,
                         similarity_threshold: float = 0.7, 
                         limit: int = 10,
                         is_doctor: bool = True,
                         additional_filters: str = None) -> List[Dict[str, Any]]:
        """
        Run a vector similarity search
        
        Args:
            label: Node label to search (e.g., "SectionTemplate")
            embedding_field: Field containing vector embedding
            text: Text to search for (will be converted to embedding)
            vector: Vector embedding to search with (alternative to text)
            similarity_threshold: Minimum similarity score (0-1)
            limit: Maximum number of results
            additional_filters: Additional Cypher WHERE clause
            
        Returns:
            List of matching records with similarity scores
        """
        
        if not is_doctor:
            text = text + "; patient-reported, subjective"
        else:
            text = text + "; doctor's speech"
        # Get embedding vector
        if vector is None and text is not None:
            vector = self.get_embedding(text)
        elif vector is None and text is None:
            logger.error("Either text or vector must be provided for vector search")
            return []
        
        # Build the query
        base_query = f"""
        MATCH (n:{label})
        WHERE n.{embedding_field} IS NOT NULL
        """
        
        # Add additional filters if provided
        if additional_filters:
            base_query += f"AND {additional_filters}\n"
        
        # Add similarity calculation and filtering
        query = base_query + f"""
        WITH n, vector.similarity.cosine(n.{embedding_field}, $query_embedding) AS similarity
        WHERE similarity > $threshold
        RETURN n, similarity
        ORDER BY similarity DESC
        LIMIT $limit
        """
        
        parameters = {
            "query_embedding": vector,
            "threshold": similarity_threshold,
            "limit": limit
        }
        
        try:
            return self.run_query(query, parameters)
        except Exception as e:
            logger.error(f"Error in vector search: {str(e)}")
            return []
    
    def get_connection_status(self) -> Dict[str, Any]:
        """
        Check connection status
        
        Returns:
            Dictionary with connection status
        """
        if not self.driver:
            return {"connected": False, "message": "No active driver"}
        
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test").single()
                if result and result.get("test") == 1:
                    return {"connected": True, "message": "Connected to Neo4j"}
                return {"connected": False, "message": "Failed connection test"}
        except Exception as e:
            return {"connected": False, "message": f"Error: {str(e)}"}

# Create singleton instance
neo4j_session = Neo4jSession()

# Dependency for FastAPI
def get_neo4j_session():
    """
    Dependency for FastAPI to get the Neo4j session
    
    Usage:
        @app.get("/templates")
        def get_templates(neo4j: Neo4jSession = Depends(get_neo4j_session)):
            ...
    """
    return neo4j_session