from db_app.db_app import app, UMLSKnowledgeGraph, get_db

import logging
import sys
import argparse
from os import environ
from typing import Optional
import uvicorn

def configure_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Configure global logging settings
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to a log file
    """
    # Parse log level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Base configuration
    config = {
        'level': numeric_level,
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'datefmt': '%Y-%m-%d %H:%M:%S',
        'handlers': [logging.StreamHandler(sys.stdout)]
    }
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(config['format']))
        config['handlers'].append(file_handler)
    
    # Apply configuration
    logging.basicConfig(**config)
    
    # Set third-party loggers to be less verbose
    logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('graphql').setLevel(logging.WARNING)
    
    # Log the configuration
    logging.info(f"Logging configured with level: {log_level}")
    if log_file:
        logging.info(f"Logs will be saved to: {log_file}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="MedSync Data Service")
    parser.add_argument("--host", default=environ.get("DB_HOST", "127.0.0.1"), help="Host address")
    parser.add_argument("--port", type=int, default=8002, help="Port number")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument("--log-file", help="Optional log file path")
    parser.add_argument("--preload-graph", action="store_true", help="Preload UMLS knowledge graph at startup")
    parser.add_argument("--graph-limit", type=int, default=10000, help="Limit for preloaded graph size")
    return parser.parse_args()

def preload_umls_graph(limit: int = 10000):
    """Preload the UMLS knowledge graph at startup for faster queries"""
    try:
        logging.info(f"Preloading UMLS knowledge graph with limit={limit}...")
        
        # Get database session
        db = next(get_db())
        
        # Build and cache the graph
        graph = UMLSKnowledgeGraph(db)
        graph.build_from_db(limit=limit)
        
        # Store in a global variable or cache
        app.cached_graph = graph
        
        stats = graph.get_stats()
        logging.info(f"UMLS graph preloaded with {stats['nodes']} nodes and {stats['edges']} edges")
        
    except Exception as e:
        logging.error(f"Error preloading UMLS graph: {str(e)}")

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup logging
    configure_logging(log_level=args.log_level, log_file=args.log_file)
    
    # Preload graph if requested
    if args.preload_graph:
        preload_umls_graph(args.graph_limit)
    
    # Log startup configuration
    logging.info(f"Starting Data Service on {args.host}:{args.port}")
    
    # Run the API without hot-reload to preserve the heavy data loading
    uvicorn.run(
        app, 
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
        reload=False  # Disable reload to prevent duplicate data loading
    )