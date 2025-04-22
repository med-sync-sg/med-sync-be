@echo off
echo MedSync Server Build Script
echo ========================

REM Check if Python is installed
python --version > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python is not installed or not in PATH. Please install Python first.
    exit /b 1
)

REM Check if pip is installed
pip --version > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo pip is not installed. Please install pip first.
    exit /b 1
)

REM Check if Nuitka is installed
pip show nuitka > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Nuitka is not installed. Installing Nuitka...
    pip install nuitka
    if %ERRORLEVEL% NEQ 0 (
        echo Failed to install Nuitka. Please install manually: pip install nuitka
        exit /b 1
    )
)

echo Creating build directories...
if not exist "build" mkdir build
if not exist "dist" mkdir dist

echo Installing required packages...
pip install -r requirements.txt

echo Compiling MedSync server with Nuitka...
python -m nuitka ^
    --follow-imports ^
    --include-package=app ^
    --include-package=db_app ^
    --include-package=uvicorn ^
    --include-package=fastapi ^
    --include-package=sqlalchemy ^
    --include-package=pydantic ^
    --include-package=psycopg2 ^
    --include-package=websockets ^
    --include-package=numpy ^
    --include-package=spacy ^
    --include-package=jinja2 ^
    --include-package=librosa ^
    --include-package=alembic ^
    --include-package=Levenshtein ^
    --include-data-dir=app/utils/nlp/report_templates=report_templates ^
    --include-distribution-metadata=Levenshtein ^
    --include-module=socket ^
    --include-module=_socket ^
    --include-module=asyncio ^
    --standalone ^
    --output-dir=dist ^
    --output-file=med_sync_server.exe ^
    launcher.py

echo Copying configuration files...
copy .env dist\.env > nul 2>&1

echo Build complete! Executable is in the dist folder.
pause