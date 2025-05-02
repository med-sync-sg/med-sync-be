# Update the app.py file to integrate Neo4j routes

# Original imports
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, status
from sqlalchemy.orm import Session
import logging
import json

from app.db.local_session import DatabaseManager
from app.models.models import SOAPCategory
from app.api.v1.endpoints import auth, notes, users, reports, tests, calibration, templates, umls
from app.utils.websocket_handler import websocket_endpoint
from app.db.neo4j_session import neo4j_session
from contextlib import asynccontextmanager

# Configure logger
logger = logging.getLogger(__name__)

def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    logger.info("Creating FastAPI application")
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Initializing Neo4j connection...")
        # Neo4j session is a singleton, so it's already initialized
        status = neo4j_session.get_connection_status()
        logger.info(f"Neo4j connection status: {status}")
        yield
        # Add startup and shutdown events for Neo4j
        logger.info("Closing Neo4j connection...")
        neo4j_session.close()
    
    app = FastAPI(
        title="MedSync API",
        description="Backend API for medical transcription and analysis",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Register routers
    app.include_router(auth.router, prefix="/auth", tags=["auth"])
    app.include_router(notes.router, prefix="/notes", tags=["note"])
    app.include_router(users.router, prefix="/users", tags=["user"])
    app.include_router(reports.router, prefix="/reports", tags=["report"])
    app.include_router(tests.router, prefix="/tests", tags=["test"])
    app.include_router(calibration.router, prefix="/calibration", tags=["calibration"])
    
    # Add Neo4j-based routers
    app.include_router(templates.router, prefix="/templates", tags=["templates"])
    app.include_router(umls.router, prefix="/umls", tags=["umls"])

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