@echo off
echo MedSync Server Build Script using cx_Freeze
echo ==========================================

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

REM Check if cx_Freeze is installed
pip show cx_freeze > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo cx_Freeze is not installed. Installing cx_Freeze...
    pip install cx_freeze
    if %ERRORLEVEL% NEQ 0 (
        echo Failed to install cx_Freeze. Please install manually: pip install cx_freeze
        exit /b 1
    )
)

echo Installing required packages...
pip install -r requirements.txt

echo Creating build directories...
if not exist "build" mkdir build
if not exist "dist" mkdir dist

echo Building MedSync server with cx_Freeze...
python setup.py build

echo Build complete!
echo The executable is in the build\exe.* directory.

echo Creating ZIP package...
if exist "dist\medsync_server.zip" del "dist\medsync_server.zip"
powershell Compress-Archive -Path "build\exe.*\*" -DestinationPath "dist\medsync_server.zip"
echo Package created at dist\medsync_server.zip

echo To run the server, navigate to the build\exe.* directory and run medsync_server.exe
pause