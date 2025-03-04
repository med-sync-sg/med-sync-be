from db_app.db_app import app
if __name__ == "__main__":
    import uvicorn
    def run_db():
        """Runs the DataStore service without hot-reload (to persist heavy-loaded data)."""
        uvicorn.run(app, host="127.0.0.1", port=8002)
        
    run_db()
