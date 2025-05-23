from db_app.db_app import app
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
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup logging
    configure_logging(log_level=args.log_level, log_file=args.log_file)
    
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