from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, status
from sqlalchemy.orm import Session
import logging
import json

from app.api.v1.endpoints import auth, notes, users, reports, tests, calibration
from app.utils.websocket_handler import websocket_endpoint

# Configure logger
logger = logging.getLogger(__name__)

def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    logger.info("Creating FastAPI application")
    
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
    app.include_router(calibration.router, prefix="/calibration", tags=["calibration"])

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure with proper origins in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add WebSocket endpoint
    app.add_api_websocket_route("/ws", websocket_endpoint)
    
    return app

# Create FastAPI application
app = create_app()