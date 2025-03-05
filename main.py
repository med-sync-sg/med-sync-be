import logging
import sys

# Configure logging to print DEBUG level and above messages to the console
if __name__ == "__main__":
    import uvicorn
    def run_api():
        """Runs the main API with hot-reload enabled (watching selected directories)."""
        uvicorn.run(
            "app.app:app",
            host="127.0.0.1",
            port=8001,
            reload=True,
            reload_dirs=["app"],
        )
        
    run_api()