def test_imports():
    modules_to_test = [
        # Main app
        # "app",
        "db_app",
        # Web framework
        "fastapi", "uvicorn", "starlette",
        # Database
        "sqlalchemy", "alembic", "psycopg2",
        # NLP
        "spacy", "sentence_transformers", "Levenshtein",
        # Data processing
        "numpy", "librosa",
        # Other
        "jinja2", "websockets", "python_multipart"
    ]
    
    results = {}
    for module in modules_to_test:
        try:
            __import__(module)
            results[module] = "Success"
        except ImportError as e:
            results[module] = f"Failed: {str(e)}"
    
    # Print results
    for module, status in results.items():
        print(f"{module}: {status}")

if __name__ == "__main__":
    test_imports()