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

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="MedSync API Server")
    parser.add_argument("--host", default=environ.get("API_HOST", "127.0.0.1"), help="Host address")
    parser.add_argument("--port", type=int, default=int(environ.get("API_PORT", "8001")), help="Port number")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    return parser.parse_args()

if __name__ == "__main__":
    import uvicorn
    
    # Setup logging
    configure_logging()
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Log startup configuration
    logging.info(f"Starting API server on {args.host}:{args.port}")
    
    # Run the API
    uvicorn.run(
        "app.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )