from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, status
import logging

from app.utils.websocket_manager import WebSocketManager
from app.api.v1.endpoints import auth, notes, users, reports, tests

# Configure logger
logger = logging.getLogger(__name__)

def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    app = FastAPI(
        title="MedSync API",
        description="Backend API for medical transcription and analysis",
        version="1.0.0"
    )
    
    # Register routers
    app.include_router(auth.router, prefix="/auth", tags=["auth"])
    app.include_router(notes.router, prefix="/notes", tags=["note"])
    app.include_router(users.router, prefix="/users", tags=["user"])
    app.include_router(reports.router, prefix="/reports", tags=["report"])
    app.include_router(tests.router, prefix="/tests", tags=["test"])

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure this with proper origins in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return app

# Create FastAPI application
app = create_app()

# Initialize WebSocket manager
websocket_manager = WebSocketManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for receiving audio chunks from clients.
    
    Expected format of incoming messages:
    { "data": "<base64-encoded audio chunk>" }
    """
    await websocket_manager.handle_client(websocket)