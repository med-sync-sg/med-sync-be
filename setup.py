import sys
import os
from cx_Freeze import setup, Executable

# Set recursion limit higher
# This is needed because cx_Freeze can hit Python's recursion limit when analyzing imports
# sys.setrecursionlimit(3000)  # Default is 1000

# Determine base based on platform
base = None
if sys.platform == "win32":
    base = "Console"  # Use "Win32GUI" for a GUI app without console window

# Determine target extension based on platform
target_ext = ".exe" if sys.platform == "win32" else ""

# List of packages to include - CRITICAL FOR FIXING RECURSION ISSUES
# First, include top-level packages explicitly
packages = [
    # Your application
    "app",
    
    # Web framework and API
    "fastapi", "uvicorn", "starlette", "pydantic", "websockets",
    
    # Database
    "sqlalchemy", "alembic", "psycopg2",
    
    # NLP core
    "spacy", "spacy_alignments", "spacy_loggers", "spacy_legacy",
    "sentence_transformers", "python_levenshtein",
    
    # Data processing
    "numpy", "librosa", "scipy",
    
    # SpaCy dependencies
    "thinc", "confection", "wasabi", "smart_open", "preshed", 
    "cymem", "murmurhash", "srsly", "catalogue", "blis",
    
    # Utilities
    "jinja2", "typing_extensions", "multiprocessing",
    "asyncio", "importlib", "json", "logging", "datetime",
    
    # Additional libraries from your requirements
    "tokenizers", "transformers", "torch", "pyahocorasick",
]

# Specific modules that need explicit inclusion
includes = [
    # Basic Python modules
    "collections.abc", "importlib.metadata", "importlib.resources",
    
    # Multiprocessing
    "multiprocessing.pool", "multiprocessing.managers", 
    "multiprocessing.synchronize", "concurrent.futures.process",
    
    # Asyncio
    "asyncio.base_events", "asyncio.proactor_events", "asyncio.selector_events",
    
    # SpaCy specific modules
    "spacy.training", "spacy.ml", "spacy.vocab", "spacy.tokens",
    
    # NumPy core
    "numpy.core._multiarray_umath", "numpy.random", "numpy.linalg",
    
    # Common modules that may be missed
    "zlib", "encodings.idna", "encodings.utf_8", "encodings.ascii"
]

# Packages to EXCLUDE to prevent recursion
excludes = [
    "tkinter",
    "matplotlib",
    "PyQt5",
    "PyQt6",
    "PySide6",
    "IPython",
    "notebook",
    "jupyter",
    "test",
    "unittest",
    "curated_transformers",
    "doctest",
    "sphinx", 
]

# Files to include
include_files = [
    # Add configuration files
    # ("config.ini", "config.ini") if os.path.exists("config.ini") else None,
    (".env", ".env") if os.path.exists(".env") else None,
    
    # Add NLP templates
    ("app/utils/nlp/report_templates", "app/utils/nlp/report_templates") 
    if os.path.exists("app/utils/nlp/report_templates") else None,
]

# Clean up None values from include_files
include_files = [item for item in include_files if item is not None]

# Define build options
build_exe_options = {
    "packages": packages,
    "includes": includes,
    "excludes": excludes,
    "include_files": include_files,
    "include_msvcr": True,  # Include Microsoft Visual C++ runtime (Windows)
    "build_exe": "build/exe",
    "optimize": 1,  # Less aggressive optimization to avoid issues
    "silent": False,  # Show build progress
    
    # CRITICAL FOR FIXING RECURSION: Break circular imports by excluding problematic packages
    "zip_include_packages": ["*"],
    "zip_exclude_packages": ["numpy.core"],    
}

# Create list of executables
executables = [
    Executable(
        # Your entry point script
        "main.py",
        base=base,
        # Output name of the executable
        target_name=f"medsync_server{target_ext}",
        # Icon file (optional)
        icon="app_icon.ico" if os.path.exists("app_icon.ico") else None,
        # Copyright information
        copyright="Copyright © 2024 MedSync Team",
    )
]

# Setup function
setup(
    name="MedSync Server",
    version="1.0.0",
    description="Medical Transcription and Notes Application",
    options={"build_exe": build_exe_options},
    executables=executables,
    author="MedSync Team",
)