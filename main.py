from app.app import app  # Import the FastAPI app instance
import argparse

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
    parser = argparse.ArgumentParser(
        prog="MedSyncBE",
        description="Runs the backend server locally for MedSync"
    )
    
    parser.add_argument("load_umls", help="Choose whether to load UMLS files (for testing)", action="store_true", required=False)
    
    args = parser.parse_args()
    
    load_umls = True
    
    if args.load_umls:
        load_umls = False
