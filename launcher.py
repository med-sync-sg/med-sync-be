import subprocess
import time
import os
import sys

# Configuration variables
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Define the relative paths for your executables.
DB_MAIN_PATH = os.path.join(BASE_DIR, "Python", "db_main.exe")
MAIN_EXE_PATH = os.path.join(BASE_DIR, "Python", "main.exe")
MED_SYNC_PATH = os.path.join(BASE_DIR, "Release", "med_sync.exe")

# Delays (in seconds) between launching processes
DB_MAIN_DELAY = 5    # Wait 5 seconds after launching db_main.exe
MAIN_EXE_DELAY = 5   # Wait 5 seconds after launching main.exe

def launch_process(exe_path, delay=0, wait_for_termination=False):
    """Launch a process and optionally wait for it or delay execution."""
    try:
        print(f"Launching: {exe_path}")
        proc = subprocess.Popen([exe_path])
        if delay > 0:
            time.sleep(delay)
        if wait_for_termination:
            proc.wait()
        return proc
    except Exception as e:
        print(f"Error launching {exe_path}: {e}")
        sys.exit(1)

def main():
    # Step 1: Launch the backend that sets up the database.
    launch_process(DB_MAIN_PATH, delay=DB_MAIN_DELAY, wait_for_termination=False)
    
    # Step 2: Launch the main backend process.
    launch_process(MAIN_EXE_PATH, delay=MAIN_EXE_DELAY, wait_for_termination=False)
    
    # Step 3: Launch the Flutter UI.
    launch_process(MED_SYNC_PATH, delay=0, wait_for_termination=False)
    
    print("All processes launched successfully.")

if __name__ == '__main__':
    main()