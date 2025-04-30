import logging
import time
from typing import List, Dict, Any, Optional, Callable
from neo4j import GraphDatabase, Driver, Session, Transaction
from neo4j.exceptions import ServiceUnavailable, AuthError, DatabaseError

class Neo4jConnectionManager:
    """
    Manager for Neo4j database connections with connection pooling,
    error handling, and retry mechanisms.
    """
    
    def __init__(self, uri: str, user: str, password: str, 
                 max_connection_pool_size: int = 50, 
                 max_transaction_retry_time: int = 30):
        """
        Initialize the Neo4j connection manager
        
        Args:
            uri: Neo4j connection URI
            user: Neo4j username
            password: Neo4j password
            max_connection_pool_size: Maximum size of the connection pool
            max_transaction_retry_time: Maximum time to retry transactions
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.logger = logging.getLogger(__name__)
        
        # Configure connection with proper pooling
        self.driver = GraphDatabase.driver(
            uri, 
            auth=(user, password),
            max_connection_pool_size=max_connection_pool_size,
            max_transaction_retry_time=max_transaction_retry_time
        )
        
        # Track connection state
        self._connected = False
        self._connection_error = None
        self._last_health_check = 0
        self._health_check_interval = 60  # seconds
        
        # Try to establish initial connection
        self._verify_connection()
        
        self.logger.info(f"Neo4j connection manager initialized with URI: {uri}")
    
    def close(self):
        """Close the Neo4j connection driver"""
        if hasattr(self, 'driver') and self.driver:
            self.driver.close()
            self._connected = False
            self.logger.info("Neo4j connection closed")
    
    def _verify_connection(self) -> bool:
        """
        Verify the Neo4j connection is working
        
        Returns:
            True if connected, False otherwise
        """
        try:
            # Run a simple query to verify connection
            with self.driver.session() as session:
                result = session.run("RETURN 1 as n")
                record = result.single()
                if record and record["n"] == 1:
                    self._connected = True
                    self._connection_error = None
                    self._last_health_check = time.time()
                    return True
                else:
                    self._connected = False
                    self._connection_error = "Unexpected result from test query"
                    return False
                    
        except (ServiceUnavailable, AuthError, DatabaseError) as e:
            self._connected = False
            self._connection_error = str(e)
            self.logger.error(f"Neo4j connection verification failed: {str(e)}")
            return False
        except Exception as e:
            self._connected = False
            self._connection_error = str(e)
            self.logger.error(f"Unexpected error during Neo4j connection verification: {str(e)}")
            return False
    
    def is_connected(self, force_check: bool = False) -> bool:
        """
        Check if connected to Neo4j
        
        Args:
            force_check: Force a new connection check even if recently checked
            
        Returns:
            True if connected, False otherwise
        """
        current_time = time.time()
        
        # Use cached status if recent check exists and not forced
        if (not force_check and 
            self._last_health_check > 0 and 
            current_time - self._last_health_check < self._health_check_interval):
            return self._connected
            
        # Otherwise verify actual connection
        return self._verify_connection()
    
    def get_connection_status(self) -> Dict[str, Any]:
        """
        Get detailed connection status
        
        Returns:
            Dictionary with connection status details
        """
        is_connected = self.is_connected()
        
        return {
            "connected": is_connected,
            "uri": self.uri,
            "last_check": self._last_health_check,
            "error": self._connection_error if not is_connected else None,
        }
    
    def run_query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query with automatic reconnection attempts
        
        Args:
            query: Cypher query string
            parameters: Optional query parameters
            
        Returns:
            List of records as dictionaries
            
        Raises:
            Exception: If query execution fails after retries
        """
        if not self.is_connected():
            self.logger.warning("Attempting to run query while disconnected, trying to reconnect...")
            if not self._verify_connection():
                raise ConnectionError(f"Neo4j is not connected: {self._connection_error}")
        
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                return [dict(record) for record in result]
                
        except (ServiceUnavailable, DatabaseError) as e:
            # Connection might have been lost, try to reconnect
            self.logger.warning(f"Neo4j connection issue, attempting to reconnect: {str(e)}")
            self._connected = False
            
            # Try to reconnect once
            if self._verify_connection():
                # If reconnected, retry the query
                with self.driver.session() as session:
                    result = session.run(query, parameters or {})
                    return [dict(record) for record in result]
            
            # If still fails, raise the exception
            raise Exception()
    
    def run_transaction(self, transaction_function: Callable[[Transaction], Any], 
                       max_retries: int = 3) -> Any:
        """
        Run a function within a Neo4j transaction with retry logic
        
        Args:
            transaction_function: Function that takes a transaction and returns a result
            max_retries: Maximum number of retries if the transaction fails
            
        Returns:
            Result of the transaction function
            
        Raises:
            Exception: If transaction execution fails after retries
        """
        retry_count = 0
        last_error = None
        
        while retry_count < max_retries:
            try:
                with self.driver.session() as session:
                    return session.execute_write(transaction_function)
            
            except (ServiceUnavailable, DatabaseError) as e:
                retry_count += 1
                last_error = e
                self.logger.warning(f"Neo4j transaction failed (attempt {retry_count}/{max_retries}): {str(e)}")
                
                # Exponential backoff for retries
                if retry_count < max_retries:
                    wait_time = 0.5 * (2 ** retry_count)  # 1, 2, 4, 8... seconds
                    time.sleep(wait_time)
                    
                    # Try to reconnect
                    self._verify_connection()
            
            except Exception as e:
                # For non-connection errors, don't retry
                self.logger.error(f"Neo4j transaction failed with non-retryable error: {str(e)}")
                raise
        
        # If we get here, all retries failed
        raise last_error or Exception("Transaction failed after maximum retries")
    
    def get_session(self) -> Session:
        """
        Get a Neo4j session for manual transaction management
        
        Returns:
            Neo4j session
            
        Note:
            Caller is responsible for closing the session
        """
        if not self.is_connected():
            if not self._verify_connection():
                raise ConnectionError(f"Neo4j is not connected: {self._connection_error}")
                
        return self.driver.session()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with driver cleanup"""
        self.close()