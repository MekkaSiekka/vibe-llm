@echo off
echo Starting Local LLM Service...
echo.

REM Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Docker is not running. Please start Docker Desktop.
    pause
    exit /b 1
)

REM Create directories
if not exist "models_cache" mkdir models_cache
if not exist "logs" mkdir logs

REM Start the service
echo Starting development environment...
docker-compose up -d

if %errorlevel% equ 0 (
    echo.
    echo ✅ Service started successfully!
    echo.
    echo 🌐 API: http://localhost:8000
    echo 🔌 WebSocket: ws://localhost:8000/ws
    echo.
    echo 📋 To view logs: docker-compose logs -f
    echo 🛑 To stop: docker-compose down
    echo.
    echo 🧪 Testing service...
    timeout /t 10 /nobreak >nul
    python test_service.py
) else (
    echo ❌ Failed to start service
    pause
    exit /b 1
)

pause

