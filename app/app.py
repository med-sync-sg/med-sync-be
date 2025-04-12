from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, status
from sqlalchemy.orm import Session
import logging

from app.api.v1.endpoints import auth, notes, users, reports, tests, calibration
from app.db.local_session import DatabaseManager
from app.services.audio_service import AudioService
from app.services.transcription_service import TranscriptionService
from app.services.nlp.keyword_extract_service import KeywordExtractService
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
    app.include_router(calibration.router, prefix="/calibration", tags=["calibration"])  # Add new calibration router

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
transcription_service = TranscriptionService()
keyword_service = KeywordExtractService()

# Get DB session
get_session = DatabaseManager().get_session

# Active WebSocket connections
active_connections = {}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, db: Session = Depends(get_session)):
    """
    WebSocket endpoint for audio streaming and transcription
    
    Expected message format:
    { "data": "<base64-encoded audio chunk>" } or
    { "data": "<text>", "type": "text" }
    """
    # Connection parameters
    connection_id = None
    user_id = None
    note_id = None
    
    try:
        # Get connection parameters
        params = websocket.query_params
        token = params.get("token")
        user_id = params.get("user_id", "0")
        note_id = params.get("note_id", "0")
        
        # Accept the connection (even for anonymous/test users)
        await websocket.accept()
        
        # Generate unique connection ID
        connection_id = f"{user_id}_{note_id}_{id(websocket)}"
        active_connections[connection_id] = websocket
        
        logger.info(f"WebSocket connected: user_id={user_id}, note_id={note_id}")
        
        # For anonymous/test mode, use default IDs
        try:
            user_id = int(user_id)
            note_id = int(note_id)
        except ValueError:
            user_id = 0
            note_id = 0
            logger.warning("Using anonymous mode (user_id=0, note_id=0)")
        
        # Main message loop
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            # Handle the message using our refactored handler
            await handle_websocket_message(data, user_id, note_id, websocket, db)
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: user_id={user_id}, note_id={note_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        # logger.error(traceback.format_exc())
    finally:
        # Clean up
        if connection_id and connection_id in active_connections:
            del active_connections[connection_id]
            
        # Reset state in services
        transcription_service.reset()
        keyword_service.clear()
        
        logger.info(f"WebSocket resources cleaned up for user_id={user_id}, note_id={note_id}")

async def handle_websocket_message(data, user_id, note_id, websocket, db):
    """
    Process a message from WebSocket (either audio or text)
    This function delegates to the appropriate service layer methods
    
    Args:
        data: Message data (dict from JSON)
        user_id: User ID for the session
        note_id: Note ID for the session
        websocket: WebSocket connection object
        db: Database session
    """
    # Initialize services (use existing singletons when possible)
    note_service = NoteService(db)
    
    try:
        # Check if this is a text message
        if data.get("type") == "text" and "data" in data:
            # Process text directly
            text_content = data["data"]
            
            # Use the transcription service to process text
            transcription_service.full_transcript = text_content
            transcription_service.transcript_segments.append(text_content)
            
            # Extract keywords 
            keywords = transcription_service.extract_keywords()
            
            # Send transcript update to client
            await websocket.send_json({
                'text': text_content
            })
            
        elif "data" in data and data["data"]:
            # This is audio data - process with AudioService
            audio_data = data["data"]
            
            # Add to audio service if it's base64 data
            import base64
            try:
                audio_bytes = base64.b64decode(audio_data)
                audio_service = transcription_service.audio_service
                audio_service.add_chunk(audio_bytes)
            except:
                logger.warning("Failed to decode audio data")
                return
            
            # Let the transcription service process the audio
            did_transcribe = transcription_service.process_audio_segment(user_id, note_id)
            if not did_transcribe:
                return
                
            # Get the transcript
            transcript_info = transcription_service.get_current_transcript()
            
            # Extract keywords
            keywords = transcription_service.extract_keywords()
            
            # Send transcript to client
            await websocket.send_json({
                'text': transcript_info['text']
            })
        else:
            # Empty data or heartbeat - just acknowledge
            await websocket.send_json({
                'status': 'connected'
            })
            return
        
        # Process keywords
        keyword_service.process_and_buffer_keywords(keywords)
        keyword_service.merge_keywords()
        
        # Create sections
        sections = keyword_service.create_sections(user_id, note_id)
        
        # Add sections to the note and prepare for response
        sections_json = []
        for section in sections:
            db_section = note_service.add_section_to_note(note_id, section)
            if db_section:
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
        
    except Exception as e:
        logger.error(f"Error handling WebSocket message: {str(e)}")
        # logger.error(traceback.format_exc())
        # Send error to client
        await websocket.send_json({
            'error': str(e)
        })