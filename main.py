import logging
import sys
import argparse
from os import environ

def configure_logging():
    """Configure logging format and level"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

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