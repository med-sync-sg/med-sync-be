from fastapi import WebSocket, WebSocketDisconnect, Depends, status
from sqlalchemy.orm import Session
import logging
import json
import base64
import numpy as np
from typing import Optional, Dict, Any, Union, Tuple, List
import time
import asyncio
from contextlib import asynccontextmanager

from app.db.local_session import DatabaseManager
from app.services.audio_service import AudioService
from app.services.transcription_service import TranscriptionService
from app.services.nlp.keyword_extract_service import KeywordExtractService
from app.services.note_service import NoteService
from app.services.report_generation.report_service import ReportService
from app.models.models import ReportTemplate, Section
from app.utils.speech_processor import SpeechProcessor
from app.api.v1.endpoints.calibration import calibration_service

# Configure logger
logger = logging.getLogger(__name__)

# Initialize services
get_session = DatabaseManager().get_session

# Create speech processor singleton
speech_processor = SpeechProcessor()

# Active WebSocket connections with cleanup management
active_connections: Dict[str, Dict[str, Any]] = {}

# Constants
MAX_BUFFER_SIZE = 1024 * 1024 * 5  # 5MB maximum buffer size
HEARTBEAT_INTERVAL = 30  # Seconds between heartbeats

@asynccontextmanager
async def managed_websocket_connection(websocket: WebSocket, connection_id: str):
    """
    Context manager for WebSocket connections that ensures proper cleanup
    """
    try:
        await websocket.accept()
        logger.info(f"WebSocket connected: {connection_id}")
        yield websocket
    except Exception as e:
        logger.error(f"WebSocket error in {connection_id}: {str(e)}")
    finally:
        # Clean up resources
        if connection_id in active_connections:
            connection_data = active_connections[connection_id]
            # Clear services
            if 'transcription_service' in connection_data:
                connection_data['transcription_service'].reset()
            if 'keyword_service' in connection_data:
                connection_data['keyword_service'].clear()
            if 'audio_service' in connection_data:
                # Clear audio buffers
                connection_data['audio_service'].clear_session()
            
            # Remove from active connections
            del active_connections[connection_id]
            logger.info(f"WebSocket resources cleaned up for {connection_id}")

async def handle_report_request(
    ws: WebSocket, 
    message: Dict[str, Any], 
    db: Session,
    note_id: int,
    user_id: int
) -> None:
    """Handle report generation requests through websocket"""
    try:
        report_type = message.get("report_type", "doctor")  # Default to doctor report
        template_id = message.get("template_id")  # Optional template ID
        
        # Initialize report service
        report_service = ReportService(db)
        
        # Generate appropriate report based on request
        if template_id:
            # Get template and verify access
            template = db.query(ReportTemplate).filter(
                ReportTemplate.id == template_id
            ).first()
            
            if not template:
                await ws.send_text(json.dumps({
                    "error": "Template not found",
                    "success": False
                }))
                return
                
            # Generate report with custom template
            report_html = report_service.generate_report_from_template(note_id, template)
        elif report_type == "patient":
            # Generate default patient report
            report_html = report_service.generate_patient_report(note_id)
        else:
            # Generate default doctor report
            report_html = report_service.generate_doctor_report(note_id)
        
        if not report_html:
            await ws.send_text(json.dumps({
                "error": "Failed to generate report",
                "success": False
            }))
            return
            
        # Send the report back to the client
        await ws.send_text(json.dumps({
            "report_html": report_html,
            "report_type": report_type,
            "success": True
        }))
        
        logger.info(f"Generated {report_type} report for note {note_id}")
        
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        await ws.send_text(json.dumps({
            "error": f"Error generating report: {str(e)}",
            "success": False
        }))

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
            
        # Generate unique connection ID
        connection_id = f"{user_id}_{note_id}_{id(websocket)}"
        
        # Set up services within the managed context
        async with managed_websocket_connection(websocket, connection_id) as ws:
            # Initialize services for this connection
            note_service = NoteService(db)
            audio_service = AudioService()
            transcription_service = TranscriptionService(
                audio_service=audio_service,
                speech_processor=speech_processor
            )
            keyword_service = KeywordExtractService()
            
            # Store services for cleanup in case of unexpected disconnection
            active_connections[connection_id] = {
                'websocket': ws,
                'audio_service': audio_service,
                'transcription_service': transcription_service,
                'keyword_service': keyword_service,
                'last_heartbeat': time.time()
            }
            
            # Verify adaptation is possible
            if use_adaptation and adaptation_user_id is not None:
                status = calibration_service.get_calibration_status(adaptation_user_id, db)
                if not status.calibration_complete:
                    use_adaptation = False
                    await ws.send_text(json.dumps({
                        "warning": "No completed voice calibration found",
                        "adaptation_enabled": False
                    }))
                    logger.warning(f"No completed calibration for user {adaptation_user_id}")
            
            # Start heartbeat task
            heartbeat_task = asyncio.create_task(send_heartbeats(connection_id))
            
            # Main message loop
            while True:
                # Receive message with timeout
                try:
                    raw_data = await asyncio.wait_for(
                        ws.receive_text(),
                        timeout=HEARTBEAT_INTERVAL * 1.5
                    )
                    data = json.loads(raw_data)
                    
                    # Update last heartbeat time
                    active_connections[connection_id]['last_heartbeat'] = time.time()
                except asyncio.TimeoutError:
                    # Check if connection is still active
                    if time.time() - active_connections[connection_id]['last_heartbeat'] > HEARTBEAT_INTERVAL * 2:
                        logger.warning(f"Connection {connection_id} timed out")
                        break
                    continue
                except WebSocketDisconnect:
                    logger.info(f"WebSocket disconnected: {connection_id}")
                    break
                
                # Check if this is a configuration update
                if "config_update" in data:
                    await handle_config_update(
                        ws, data["config_update"], 
                        connection_id, db, 
                        use_adaptation, adaptation_user_id
                    )
                    continue
                
                if "report_request" in data:
                    await handle_report_request(
                        ws, 
                        data["report_request"], 
                        db,
                        int(note_id),
                        int(user_id)
                    )
                    continue
                                
                # Process audio chunk if present
                if data and "data" in data:
                    await handle_audio_data(
                        ws, data, db,
                        connection_id, int(user_id), int(note_id),
                        use_adaptation, adaptation_user_id,
                        note_service, transcription_service, keyword_service, audio_service
                    )
                else:
                    logger.info("Received malformed data or control message")
                    continue
                    
            # Cancel heartbeat task
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: user_id={user_id}, note_id={note_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        if websocket.client_state == WebSocket.application_state.CONNECTED:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
    finally:
        # Clean up will be handled by the context manager
        pass

async def send_heartbeats(connection_id: str):
    """Send periodic heartbeats to keep the connection alive"""
    try:
        while connection_id in active_connections:
            # Send heartbeat
            websocket = active_connections[connection_id]['websocket']
            await websocket.send_text(json.dumps({"heartbeat": time.time()}))
            
            # Update timestamp
            active_connections[connection_id]['last_heartbeat'] = time.time()
            
            # Wait for next interval
            await asyncio.sleep(HEARTBEAT_INTERVAL)
    except asyncio.CancelledError:
        # Task was cancelled, exit gracefully
        pass
    except Exception as e:
        logger.error(f"Error in heartbeat for {connection_id}: {str(e)}")

async def handle_config_update(
    websocket: WebSocket,
    config: Dict[str, Any],
    connection_id: str,
    db: Session,
    use_adaptation: bool,
    adaptation_user_id: Optional[int]
) -> Tuple[bool, Optional[int]]:
    """Handle configuration update messages"""
    try:
        new_use_adaptation = config.get("use_adaptation", use_adaptation)
        new_adaptation_user_id = config.get("user_id", adaptation_user_id)
        
        logger.info(f"Updated configuration: use_adaptation={new_use_adaptation}, user_id={new_adaptation_user_id}")
        
        # Verify adaptation is possible
        if new_use_adaptation and new_adaptation_user_id is not None:
            status = calibration_service.get_calibration_status(new_adaptation_user_id, db)
            if not status.calibration_complete:
                new_use_adaptation = False
                await websocket.send_text(json.dumps({
                    "warning": "No completed voice calibration found",
                    "adaptation_enabled": False
                }))
                logger.warning(f"No completed calibration for user {new_adaptation_user_id}")
            else:
                await websocket.send_text(json.dumps({
                    "info": "Using voice calibration",
                    "adaptation_enabled": True,
                    "profile_id": status.profile_id
                }))
        
        return new_use_adaptation, new_adaptation_user_id
    except Exception as e:
        logger.error(f"Error processing config update: {str(e)}")
        await websocket.send_text(json.dumps({
            "error": f"Error updating configuration: {str(e)}"
        }))
        return use_adaptation, adaptation_user_id

async def handle_audio_data(
    websocket: WebSocket,
    data: Dict[str, Any],
    db: Session,
    connection_id: str,
    user_id: int,
    note_id: int,
    use_adaptation: bool,
    adaptation_user_id: Optional[int],
    note_service: NoteService,
    transcription_service: TranscriptionService,
    keyword_service: KeywordExtractService,
    audio_service: AudioService
) -> None:
    """Handle incoming audio data"""
    try:
        # Extract audio data
        audio_base64 = data["data"]
        if not audio_base64:
            logger.info("Received end of stream signal.")
            return
        
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
            user_id, 
            note_id, 
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
        await websocket.send_text(json.dumps({
            "error": f"Error processing audio: {str(e)}"
        }))

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
        
        # Check buffer size before adding
        current_size = len(audio_service.current_buffer) + len(audio_service.session_buffer)
        if current_size + len(chunk_bytes) > MAX_BUFFER_SIZE:
            # Buffer too large, send warning and clear partial data
            await websocket.send_text(json.dumps({
                "warning": "Audio buffer too large, clearing oldest data"
            }))
            # Clear oldest 25% of the session buffer to make room
            trim_size = len(audio_service.session_buffer) // 4
            audio_service.session_buffer = audio_service.session_buffer[trim_size:]
        
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
        
        # Create sections
        sections = keyword_service.create_section_from_keywords()
        
        # Reset buffer for next segment
        audio_service.reset_current_buffer()
        
        # Send transcript update to client
        await websocket.send_text(json.dumps({
            'text': transcript_info['text'],
            'using_adaptation': use_adaptation,
            'processing_time_ms': round(processing_time * 1000, 2)
        }))
        
        # Add sections to note and send to client
        sections_json: List[Section] = []
        for section in sections:
            try:
                # Validate section data before saving
                if not section.content:
                    logger.warning(f"Skipping empty section for note {note_id}")
                    continue
                    
                # Save section to database
                db_section = note_service.add_section_to_note(note_id, section)
                if db_section != None:
                    # Convert to JSON for websocket response
                    sections_json.append(db_section)
            except Exception as section_error:
                logger.error(f"Error adding section: {str(section_error)}")
        
        # Send sections to client
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

def get_active_connections() -> Dict[str, Dict[str, Any]]:
    """Get all active WebSocket connections"""
    return active_connections

async def broadcast_message(message: Union[str, Dict[str, Any]]) -> None:
    """
    Broadcast a message to all connected clients
    
    Args:
        message: Message to broadcast (string or JSON-serializable dict)
    """
    for connection_id, connection_data in active_connections.items():
        try:
            websocket = connection_data['websocket']
            if isinstance(message, str):
                await websocket.send_text(message)
            else:
                # Always send as a JSON string
                await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error broadcasting to {connection_id}: {str(e)}")