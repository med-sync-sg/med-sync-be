from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, status
from sqlalchemy.orm import Session
import logging

from app.api.v1.endpoints import auth, notes, users, reports, tests
from app.db.local_session import DatabaseManager
from app.services.audio_service import AudioService
from app.services.transcription_service import TranscriptionService
from app.services.nlp_service import KeywordService
from app.services.note_service import NoteService

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
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure with proper origins in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return app

# Create FastAPI application
app = create_app()

# Initialize service singletons
audio_service = AudioService()
transcription_service = TranscriptionService(audio_service)
keyword_service = KeywordService()

# Get DB session
get_session = DatabaseManager().get_session

# Active WebSocket connections
active_connections = {}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, db: Session = Depends(get_session)):
    """
    WebSocket endpoint for audio streaming and transcription
    
    Expected message format:
    { "data": "<base64-encoded audio chunk>" }
    """
    # Connection parameters
    connection_id = None
    user_id = None
    note_id = None
    
    try:
        # Get connection parameters
        params = websocket.query_params
        token = params.get("token")
        user_id = params.get("user_id")
        note_id = params.get("note_id")
        
        # Validate parameters
        if not all([token, user_id, note_id]):
            logger.warning("Missing required WebSocket parameters")
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
            
        # Accept the connection
        await websocket.accept()
        
        # Generate unique connection ID
        connection_id = f"{user_id}_{note_id}_{id(websocket)}"
        active_connections[connection_id] = websocket
        
        logger.info(f"WebSocket connected: user_id={user_id}, note_id={note_id}")
        
        # Initialize note service for this connection
        note_service = NoteService(db)
        
        # Main message loop
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            # Process audio chunk if present
            if data and "data" in data and data["data"]:
                await process_audio_chunk(
                    data["data"], 
                    int(user_id), 
                    int(note_id), 
                    websocket,
                    note_service
                )
            else:
                logger.info("Received empty data, closing connection")
                break
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: user_id={user_id}, note_id={note_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        # Clean up
        if connection_id and connection_id in active_connections:
            del active_connections[connection_id]
            
        # Reset state
        transcription_service.reset()
        keyword_service.clear()
        
        logger.info(f"WebSocket resources cleaned up for user_id={user_id}, note_id={note_id}")

async def process_audio_chunk(chunk_base64: str, user_id: int, note_id: int, 
                             websocket: WebSocket, note_service: NoteService):
    """
    Process an audio chunk using the service layer
    
    Args:
        chunk_base64: Base64-encoded audio data
        user_id: User ID for the session
        note_id: Note ID for the session
        websocket: WebSocket connection
        note_service: Note service instance
    """
    try:
        import base64
        
        # Decode base64 data
        audio_bytes = base64.b64decode(chunk_base64)
        
        # Add to audio service
        audio_service.add_chunk(audio_bytes)
        
        # Process with transcription service
        did_transcribe = transcription_service.process_audio_segment(user_id, note_id)
        
        if did_transcribe:
            # Get current transcript
            transcript_info = transcription_service.get_current_transcript()
            
            # Extract keywords
            keywords = transcription_service.extract_keywords()
            
            # Process keywords
            keyword_service.process_and_buffer_keywords(keywords)
            keyword_service.merge_keywords()
            
            # Create sections
            sections = keyword_service.create_sections(user_id, note_id)
            
            # Send transcript update to client
            await websocket.send_json({
                'text': transcript_info['text']
            })
            
            # Add sections to note and send to client
            sections_json = []
            for section in sections:
                # Save section to database
                db_section = note_service.add_section_to_note(note_id, section)
                if db_section:
                    # Convert to JSON for websocket response
                    sections_json.append({
                        'id': db_section.id,
                        'title': db_section.title,
                        'content': db_section.content,
                        'section_type': db_section.section_type
                    })
            
            # Send sections to client
            if sections_json:
                await websocket.send_json({
                    'sections': sections_json
                })
                
            logger.info(f"Processed audio chunk: transcribed={did_transcribe}, sections={len(sections_json)}")
            
    except Exception as e:
        logger.error(f"Error processing audio chunk: {str(e)}")
        # Don't re-raise; allow connection to continue