import sys
import subprocess
import time
import os
import argparse
import uvicorn

def parse_arguments():
    parser = argparse.ArgumentParser(description="MedSync Unified Launcher")
    parser.add_argument("--db-only", action="store_true", help="Start only the database service")
    parser.add_argument("--db-host", default="127.0.0.1", help="Database host address")
    parser.add_argument("--db-port", type=int, default=8002, help="Database port")
    parser.add_argument("--main-host", default="127.0.0.1", help="Main server host address")
    parser.add_argument("--main-port", type=int, default=8001, help="Main server port")
    return parser.parse_args()

def main():
    args = parse_arguments()
    print(args)
    # Start the appropriate service(s) based on arguments
    if args.db_only:
        start_db_service(args.db_host, args.db_port)
    else:
        start_main_service(args.main_host, args.main_port)

def start_db_service(host, port):
    print(f"Starting MedSync Database Service on {host}:{port}...")
    from db_app.db_app import app
    
    uvicorn.run(app, host=host, port=port)
    
def start_main_service(host, port):
    print(f"Starting MedSync Main Service on {host}:{port}...")
    from app.app import app  
    
    # Run in the current process
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()