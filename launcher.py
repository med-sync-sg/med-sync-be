import sys
import subprocess
import time
import os
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="MedSync Unified Launcher")
    parser.add_argument("--db-only", action="store_true", help="Start only the database service")
    parser.add_argument("--main-only", action="store_true", help="Start only the main application")
    parser.add_argument("--db-host", default="127.0.0.1", help="Database host address")
    parser.add_argument("--db-port", type=int, default=8002, help="Database port")
    parser.add_argument("--main-host", default="127.0.0.1", help="Main server host address")
    parser.add_argument("--main-port", type=int, default=8001, help="Main server port")
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Start the appropriate service(s) based on arguments
    if args.db_only:
        start_db_service(args.db_host, args.db_port)
    elif args.main_only:
        start_main_service(args.main_host, args.main_port)
    else:
        # Start both with db first, then main
        db_process = start_db_service(args.db_host, args.db_port)
        print(f"Waiting for database to initialize...")
        time.sleep(5)  # Give the DB time to start
        start_main_service(args.main_host, args.main_port)

def start_db_service(host, port):
    print(f"Starting MedSync Database Service on {host}:{port}...")
    from db_app.db_app import app
    import uvicorn
    
    # Start in subprocess to avoid blocking
    return subprocess.Popen([
        sys.executable,
        "-c",
        f"import uvicorn; uvicorn.run('db_app.db_app:app', host='{host}', port={port})"
    ])

def start_main_service(host, port):
    print(f"Starting MedSync Main Service on {host}:{port}...")
    from app.app import app  
    import uvicorn
    
    # Run in the current process
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()