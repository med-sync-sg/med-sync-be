from fastapi import WebSocket, WebSocketDisconnect, Depends, status
from sqlalchemy.orm import Session
import logging
import json
import base64
import numpy as np
from typing import Optional, Dict, Any, Union
import time

from app.db.local_session import DatabaseManager
from app.services.audio_service import AudioService
from app.services.transcription_service import TranscriptionService
from app.services.nlp.keyword_extract_service import KeywordExtractService
from app.services.note_service import NoteService
from app.utils.speech_processor import SpeechProcessor
from app.api.v1.endpoints.calibration import calibration_service

# Configure logger
logger = logging.getLogger(__name__)

# Initialize services
get_session = DatabaseManager().get_session

# Create speech processor singleton
speech_processor = SpeechProcessor()

# Active WebSocket connections
active_connections = {}

async def websocket_endpoint(websocket: WebSocket, db: Session = Depends(get_session)):
    """
    WebSocket endpoint for audio streaming with voice calibration support
    
    Expected message format:
    { 
        "data": "<base64-encoded audio chunk>",
        "use_adaptation": true/false,
        "user_id": 123
    }
    
    Or for config updates:
    {
        "config_update": {
            "use_adaptation": true/false,
            "user_id": 123
        }
    }
    """
    # Connection parameters
    connection_id = None
    user_id = None
    note_id = None
    
    # Voice adaptation settings
    use_adaptation = False
    adaptation_user_id = None
    
    try:
        # Get connection parameters from query params
        params = websocket.query_params
        token = params.get("token")
        user_id = params.get("user_id")
        note_id = params.get("note_id")
        
        # Check if adaptation is requested in query params
        use_adaptation_param = params.get("use_adaptation", "false").lower()
        use_adaptation = use_adaptation_param == "true"
        adaptation_user_id = int(user_id) if user_id and user_id.isdigit() else None
        
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
        
        logger.info(f"WebSocket connected: user_id={user_id}, note_id={note_id}, use_adaptation={use_adaptation}")
        
        # Initialize services for this connection
        note_service = NoteService(db)
        audio_service = AudioService()
        transcription_service = TranscriptionService(
            audio_service=audio_service,
            speech_processor=speech_processor
        )
        keyword_service = KeywordExtractService()
        
        # Verify adaptation is possible
        if use_adaptation and adaptation_user_id is not None:
            status = calibration_service.get_calibration_status(adaptation_user_id, db)
            if not status.calibration_complete:
                use_adaptation = False
                # Always send as a JSON string
                await websocket.send_text(json.dumps({
                    "warning": "No completed voice calibration found",
                    "adaptation_enabled": False
                }))
                logger.warning(f"No completed calibration for user {adaptation_user_id}")
        
        # Main message loop
        while True:
            # Receive message
            raw_data = await websocket.receive_text()
            data = json.loads(raw_data)
            
            # Check if this is a configuration update
            if "config_update" in data:
                config = data["config_update"]
                use_adaptation = config.get("use_adaptation", use_adaptation)
                adaptation_user_id = config.get("user_id", adaptation_user_id)
                
                logger.info(f"Updated configuration: use_adaptation={use_adaptation}, user_id={adaptation_user_id}")
                
                # Verify adaptation is possible
                if use_adaptation and adaptation_user_id is not None:
                    status = calibration_service.get_calibration_status(adaptation_user_id, db)
                    if not status.calibration_complete:
                        use_adaptation = False
                        # Always send as a JSON string
                        await websocket.send_text(json.dumps({
                            "warning": "No completed voice calibration found",
                            "adaptation_enabled": False
                        }))
                        logger.warning(f"No completed calibration for user {adaptation_user_id}")
                    else:
                        # Always send as a JSON string
                        await websocket.send_text(json.dumps({
                            "info": "Using voice calibration",
                            "adaptation_enabled": True,
                            "profile_id": status.profile_id
                        }))
                
                continue
            
            # Process audio chunk if present
            if data and "data" in data:
                try:
                    # Extract audio data
                    audio_base64 = data["data"]
                    if not audio_base64:
                        logger.info("Received end of stream signal.")
                        break
                    
                    audio_bytes = base64.b64decode(audio_base64)
                    
                    # Check for adaptation settings in the message
                    message_use_adaptation = data.get("use_adaptation", use_adaptation)
                    message_user_id = data.get("user_id", adaptation_user_id)
                    
                    # Update settings if changed
                    if message_use_adaptation != use_adaptation or message_user_id != adaptation_user_id:
                        use_adaptation = message_use_adaptation
                        adaptation_user_id = message_user_id
                        logger.info(f"Updated settings from message: use_adaptation={use_adaptation}, user_id={adaptation_user_id}")
                    
                    # Process the audio
                    await process_audio_chunk(
                        audio_bytes, 
                        int(user_id), 
                        int(note_id), 
                        websocket,
                        note_service,
                        transcription_service,
                        keyword_service,
                        use_adaptation,
                        adaptation_user_id,
                        db
                    )
                except Exception as e:
                    logger.error(f"Error processing audio chunk: {str(e)}")
                    # Always send as a JSON string
                    await websocket.send_text(json.dumps({
                        "error": f"Error processing audio: {str(e)}"
                    }))
            else:
                logger.info("Received malformed data or control message")
                continue
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: user_id={user_id}, note_id={note_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        # Clean up
        if connection_id and connection_id in active_connections:
            del active_connections[connection_id]
            
        # Reset state
        if 'transcription_service' in locals():
            transcription_service.reset()
        if 'keyword_service' in locals():
            keyword_service.clear()
        
        logger.info(f"WebSocket resources cleaned up for user_id={user_id}, note_id={note_id}")

async def process_audio_chunk(
    chunk_bytes: bytes, 
    user_id: int, 
    note_id: int, 
    websocket: WebSocket, 
    note_service: NoteService,
    transcription_service: TranscriptionService,
    keyword_service: KeywordExtractService,
    use_adaptation: bool = False,
    adaptation_user_id: Optional[int] = None,
    db: Session = None
) -> None:
    """
    Process an audio chunk using the service layer with database access
    
    Args:
        chunk_bytes: Raw audio bytes
        user_id: User ID for the session
        note_id: Note ID for the session
        websocket: WebSocket connection
        note_service: Note service instance
        transcription_service: Transcription service instance
        keyword_service: Keyword extraction service instance
        use_adaptation: Whether to use voice adaptation
        adaptation_user_id: User ID for adaptation profile
        db: Database session
    """
    try:
        audio_service = transcription_service.audio_service
        
        # Add to audio service
        audio_service.add_chunk(chunk_bytes)
        
        # Check for minimum audio duration
        if not audio_service.has_minimum_audio():
            return  # Not enough audio yet
        
        # Check for silence (indicating end of utterance)
        if not audio_service.detect_silence():
            return  # No silence detected yet
            
        logger.info(f"Processing audio segment for user {user_id}, note {note_id}, adaptation={use_adaptation}")
        
        # Get audio data
        audio_samples = audio_service.get_wave_data()
        
        # Perform transcription
        start_time = time.time()
        if use_adaptation and adaptation_user_id is not None:
            # Use speaker-adapted transcription
            transcription = speech_processor.transcribe_with_adaptation(
                audio_samples, 
                adaptation_user_id,
                db
            )
        else:
            # Use standard transcription
            transcription = speech_processor.transcribe(audio_samples)
        
        processing_time = time.time() - start_time
        
        # Process transcription result
        if not transcription:
            logger.warning("Transcription produced empty result")
            audio_service.reset_current_buffer()
            return
            
        # Update transcript in transcription service
        transcription_service._process_transcription_text(transcription)
        
        # Get current transcript state
        transcript_info = transcription_service.get_current_transcript()
        
        # Extract keywords
        keywords = transcription_service.extract_keywords()
        
        # Process keywords
        keyword_service.process_and_buffer_keywords(keywords)
        keyword_service.merge_keywords()
        
        # Create sections
        sections = keyword_service.create_sections(user_id, note_id)
        
        # Reset buffer for next segment
        audio_service.reset_current_buffer()
        
        # Send transcript update to client - ALWAYS AS JSON STRING
        await websocket.send_text(json.dumps({
            'text': transcript_info['text'],
            'using_adaptation': use_adaptation,
            'processing_time_ms': round(processing_time * 1000, 2)
        }))
        
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
        
        # Send sections to client - ALWAYS AS JSON STRING
        if sections_json:
            await websocket.send_text(json.dumps({
                'sections': sections_json
            }))
            
        logger.info(f"Processed audio chunk: transcription_length={len(transcription)}, sections={len(sections_json)}, adaptation={use_adaptation}")
        
    except Exception as e:
        logger.error(f"Error processing audio chunk: {str(e)}")
        # Always send as a JSON string
        await websocket.send_text(json.dumps({
            "error": f"Error processing audio: {str(e)}"
        }))

def get_active_connections() -> Dict[str, WebSocket]:
    """Get all active WebSocket connections"""
    return active_connections

async def broadcast_message(message: Union[str, Dict[str, Any]]) -> None:
    """
    Broadcast a message to all connected clients
    
    Args:
        message: Message to broadcast (string or JSON-serializable dict)
    """
    for connection_id, websocket in active_connections.items():
        try:
            if isinstance(message, str):
                await websocket.send_text(message)
            else:
                # Always send as a JSON string
                await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error broadcasting to {connection_id}: {str(e)}")