from db_app.db_app import app
import logging
import sys
import argparse
from os import environ
from typing import Optional

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

if __name__ == "__main__":
    import uvicorn
    configure_logging()
    
    def run_db():
        """Runs the DataStore service without hot-reload (to persist heavy-loaded data)."""
        uvicorn.run(app, host="127.0.0.1", port=8002)
        
    run_db()
