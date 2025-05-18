from fastapi import WebSocket, WebSocketDisconnect, Depends, status
from fastapi.websockets import WebSocketState
from sqlalchemy.orm import Session
import logging
import json
import base64
import numpy as np
from typing import Optional, Dict, Any, Union, Tuple, List
import time
import asyncio
from contextlib import asynccontextmanager
import traceback
from app.db.local_session import DatabaseManager
from app.services.audio_service import AudioService
from app.services.transcription_service import TranscriptionService
from app.services.nlp.keyword_extract_service import KeywordExtractService
from app.services.note_service import NoteService
from app.models.models import ReportTemplate, Section
from app.schemas.section import SectionCreate
from app.services.diarization_service import DiarizationService
from app.models.models import ReportTemplate
from app.utils.speech_processor import SpeechProcessor
from app.utils.nlp.spacy_utils import process_text
from app.api.v1.endpoints.calibration import calibration_service
import datetime

# Configure logger
logger = logging.getLogger(__name__)

# For JSON serialization
def serialize_for_json(obj):
    """Convert NumPy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime.datetime):
        return obj.isoformat()
    elif isinstance(obj, set):
        return list(obj)
    return obj

# Initialize services
get_session = DatabaseManager().get_session

# Create speech processor singleton
speech_processor = SpeechProcessor()

# Active WebSocket connections with cleanup management
active_connections: Dict[str, Dict[str, Any]] = {}

# Constants
MAX_BUFFER_SIZE = 1024 * 1024 * 5  # 5MB maximum buffer size
HEARTBEAT_INTERVAL = 60  # Seconds between heartbeats

async def send_heartbeats(connection_id: str) -> None:
    """Send periodic heartbeats to keep the connection alive"""
    try:
        while connection_id in active_connections:
            # Send heartbeat
            websocket = active_connections[connection_id]['websocket']
            await websocket.send_text(json.dumps({
                "heartbeat": time.time(),
                "connection_id": connection_id
            }, default=serialize_for_json))
            
            # Update timestamp
            active_connections[connection_id]['last_heartbeat'] = time.time()
            
            # Wait for next interval
            await asyncio.sleep(HEARTBEAT_INTERVAL)
    except asyncio.CancelledError:
        # Task was cancelled, exit gracefully
        pass
    except Exception as e:
        logger.error(f"Error in heartbeat for {connection_id}: {str(e)}")

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
            websocket.close(reason="Successfully closed websocket")
            logger.info(f"WebSocket resources cleaned up for {connection_id}")

async def process_audio_with_transcription_first(
    chunk_bytes: bytes, 
    user_id: int, 
    note_id: int, 
    websocket: WebSocket, 
    note_service: NoteService,
    transcription_service: TranscriptionService,
    keyword_service: KeywordExtractService,
    diarization_service: DiarizationService,
    use_adaptation: bool = False,
    adaptation_user_id: Optional[int] = None,
    doctor_id: Optional[int] = None,
    db: Session = None
) -> None:
    """
    Process an audio chunk with transcription-first approach, then diarization when possible
    
    Args:
        chunk_bytes: Raw audio bytes
        user_id: User ID for the session
        note_id: Note ID for the session
        websocket: WebSocket connection
        note_service: Note service instance
        transcription_service: Transcription service instance
        keyword_service: Keyword extraction service instance
        diarization_service: Diarization service instance
        use_adaptation: Whether to use voice adaptation
        adaptation_user_id: User ID for adaptation profile
        doctor_id: User ID of the doctor for diarization
        db: Database session
    """
    try:
        # Check buffer size before adding
        if chunk_bytes == None:
            return

        # Add to audio service
        transcription_service.audio_service.add_chunk(chunk_bytes)
        
        # Function to serialize NumPy types for JSON
        def serialize_for_json(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, datetime.datetime):
                return obj.isoformat()
            elif isinstance(obj, set):
                return list(obj)
            return obj

        # Step 1: Always perform immediate transcription
        min_transcription_duration_ms = 1000  # 1 second minimum for transcription
        if transcription_service.audio_service.has_minimum_audio(min_duration_ms=min_transcription_duration_ms):
            # Check for silence to know when to process
            if transcription_service.audio_service.detect_silence():
                logger.info(f"Processing immediate transcription for user {user_id}")
                
                # Get current audio data for transcription
                current_audio = transcription_service.audio_service.get_wave_data()
                
                # Perform transcription with or without adaptation
                transcription_start = time.time()
                if use_adaptation and adaptation_user_id is not None:
                    transcription_text = transcription_service.speech_processor.transcribe_with_adaptation(
                        current_audio, adaptation_user_id, db
                    )
                    logger.info(f"Used speaker adaptation for user {adaptation_user_id}")
                else:
                    transcription_text = transcription_service.speech_processor.transcribe(current_audio)
                    logger.info("Used standard transcription")
                
                transcription_time = time.time() - transcription_start
                
                # Send immediate transcription result
                if transcription_text:
                    await websocket.send_text(json.dumps({
                        'text': transcription_text,
                        'speaker': 'unknown',  # Will be updated by diarization if successful
                        'processing_time_ms': round(transcription_time * 1000, 2),
                        'using_adaptation': use_adaptation,
                        'type': 'immediate_transcription'
                    }, default=serialize_for_json))
                    
                    # Process transcription for keywords and sections
                    await process_transcription_for_sections(
                        transcription_text, 
                        note_id, 
                        websocket, 
                        transcription_service, 
                        keyword_service, 
                        note_service
                    )
                
                # Reset current buffer after processing
                transcription_service.audio_service.reset_current_buffer()

        # Step 2: Attempt diarization if there's enough accumulated audio
        min_diarization_duration_ms = 3000  # 3 seconds minimum for diarization
        session_audio = transcription_service.audio_service.get_wave_data(use_session_buffer=True)
        session_duration_ms = len(session_audio) / transcription_service.audio_service.DEFAULT_SAMPLE_RATE * 1000
        
        if session_duration_ms >= min_diarization_duration_ms:
            # Only process diarization at intervals to avoid excessive processing
            current_time = time.time()
            if current_time - diarization_service.last_diarization_time > diarization_service.diarization_interval_seconds:
                logger.info(f"Attempting diarization with {session_duration_ms:.1f}ms of audio")
                
                # Process diarization
                diarization_start = time.time()
                diarization_results = diarization_service.diarize_buffered_audio(doctor_id)
                diarization_time = time.time() - diarization_start
                
                # Update last processing time
                diarization_service.last_diarization_time = current_time
                
                # Check diarization results
                status = diarization_results.get("status", "failed")
                if status == "success":
                    logger.info("Diarization successful, processing segments")
                    
                    # Process diarization results
                    await process_diarization_results(
                        diarization_results, 
                        transcription_service, 
                        websocket, 
                        note_service, 
                        keyword_service,
                        note_id,
                        use_adaptation,
                        adaptation_user_id,
                        doctor_id,
                        db,
                        diarization_time
                    )
                else:
                    logger.info(f"Diarization not successful (status: {status}), continuing with transcription-only approach")
                    
                    # Send status update to client
                    await websocket.send_text(json.dumps({
                        'diarization_status': status,
                        'message': 'Using transcription-only mode',
                        'buffer_duration_ms': session_duration_ms
                    }, default=serialize_for_json))
        else:
            logger.debug(f"Not enough audio for diarization yet ({session_duration_ms:.1f}ms < {min_diarization_duration_ms}ms)")
            
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        logger.error(traceback.format_exc())
        await websocket.send_text(json.dumps({
            "error": f"Error processing audio: {str(e)}"
        }))


async def process_transcription_for_sections(
    transcription_text: str,
    note_id: int,
    websocket: WebSocket,
    transcription_service: TranscriptionService,
    keyword_service: KeywordExtractService,
    note_service: NoteService
) -> None:
    """
    Process transcription text to extract keywords and create sections
    
    Args:
        transcription_text: The transcribed text
        note_id: Note ID for creating sections
        websocket: WebSocket connection for sending updates
        transcription_service: Transcription service instance
        keyword_service: Keyword extraction service instance
        note_service: Note service instance
    """
    try:
        logger.info("Extracting keywords from the transcript...")
        # Update transcription service state
        transcription_service.full_transcript = (
            transcription_service.full_transcript + " " + transcription_text
        ).strip()
        transcription_service.transcript_segments.append(transcription_text)
        
        # Process through NLP pipeline
        doc = process_text(transcription_text)
        
        # Extract keywords
        keywords = keyword_service.extract_keywords_from_doc(doc)
        
        if keywords:
            # Process keywords
            keyword_service.process_and_buffer_keywords(keywords)
            
            logger.info("Creating sections from keywords...")

            # Create sections from keywords
            templates, sections = keyword_service.create_section_from_keywords()
            
            # Add sections to note
            sections_json = []
            for section in sections:
                db_section = note_service.add_section_to_note(note_id, section)
                if db_section:
                    sections_json.append({
                        'id': db_section.id,
                        'title': db_section.title,
                        'template_id': db_section.template_id,
                        'soap_category': db_section.soap_category,
                        'content': db_section.content
                    })
            
            # Send sections to client if any were created
            if sections_json:
                def serialize_for_json(obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, datetime.datetime):
                        return obj.isoformat()
                    elif isinstance(obj, set):
                        return list(obj)
                    return obj
                
                await websocket.send_text(json.dumps({
                    'sections': sections_json,
                    'keywords_processed': len(keywords)
                }, default=serialize_for_json))
                
                logger.info(f"Created {len(sections_json)} sections from transcription")
                
    except Exception as e:
        logger.error(f"Error processing transcription for sections: {str(e)}")


async def process_diarization_results(
    diarization_results: Dict[str, Any],
    transcription_service: TranscriptionService,
    websocket: WebSocket,
    note_service: NoteService,
    keyword_service: KeywordExtractService,
    note_id: int,
    use_adaptation: bool,
    adaptation_user_id: Optional[int],
    doctor_id: Optional[int],
    db: Session,
    processing_time: float
) -> None:
    """
    Process successful diarization results to create speaker-labeled transcripts
    
    Args:
        diarization_results: Results from diarization service
        transcription_service: Transcription service instance
        websocket: WebSocket connection
        note_service: Note service instance
        keyword_service: Keyword extraction service instance
        note_id: Note ID for creating sections
        use_adaptation: Whether to use voice adaptation
        adaptation_user_id: User ID for adaptation profile
        doctor_id: User ID of the doctor
        db: Database session
        processing_time: Time taken for diarization processing
    """
    try:
        # Function to serialize NumPy types for JSON
        def serialize_for_json(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, datetime.datetime):
                return obj.isoformat()
            elif isinstance(obj, set):
                return list(obj)
            return obj
        
        # Extract diarization data
        segments = diarization_results.get("segments", [])
        speaker_mapping = diarization_results.get("speaker_mapping", {})
        confidence_scores = diarization_results.get("confidence_scores", {})
        
        logger.info(f"Processing {len(segments)} diarized segments")
        
        # Store segments by speaker role
        doctor_segments = []
        patient_segments = []
        
        # Get session audio for segment extraction
        session_audio = transcription_service.audio_service.get_wave_data(use_session_buffer=True)
        sample_rate = transcription_service.audio_service.DEFAULT_SAMPLE_RATE
        
        # Process each segment with appropriate transcription
        for i, (start_sec, end_sec) in enumerate(segments):
            # Skip if segment has no speaker mapping
            if i not in speaker_mapping:
                continue
                
            # Get speaker role for this segment
            role = speaker_mapping.get(i)
            
            # Extract segment audio
            start_sample = int(start_sec * sample_rate)
            end_sample = int(end_sec * sample_rate)
            
            # Check index bounds
            if start_sample >= len(session_audio) or end_sample > len(session_audio):
                logger.warning(f"Segment {i} indices out of bounds")
                continue
            
            segment_audio = session_audio[start_sample:end_sample]
            
            # Skip very short segments
            if len(segment_audio) < 0.5 * sample_rate:  # Less than 0.5 seconds
                continue
            
            # Determine transcription method based on speaker role
            if role == "doctor" and doctor_id is not None and use_adaptation:
                segment_text = transcription_service.speech_processor.transcribe_with_adaptation(
                    segment_audio, doctor_id, db
                )
                is_doctor = True
            elif role == "patient" and adaptation_user_id is not None and use_adaptation:
                segment_text = transcription_service.speech_processor.transcribe_with_adaptation(
                    segment_audio, adaptation_user_id, db
                )
                is_doctor = False
            else:
                segment_text = transcription_service.speech_processor.transcribe(segment_audio)
                is_doctor = (role == "doctor")
            
            # Skip empty transcriptions
            if not segment_text:
                continue
            
            # Get confidence score for this segment
            confidence = confidence_scores.get(i, 0.7)
            
            # Store segment based on speaker role
            segment_data = {
                "start": float(start_sec),
                "end": float(end_sec),
                "text": segment_text,
                "is_doctor": is_doctor,
                "confidence": float(confidence)
            }
            
            if is_doctor:
                doctor_segments.append(segment_data)
            else:
                patient_segments.append(segment_data)
        
        # Generate formatted transcript
        full_transcript = format_diarized_transcript(doctor_segments, patient_segments)
        
        # Process doctor's speech for sections (only if we have doctor segments)
        doctor_text = " ".join([seg["text"] for seg in doctor_segments])
        if doctor_text:
            await process_transcription_for_sections(
                doctor_text, 
                note_id, 
                websocket, 
                transcription_service, 
                keyword_service, 
                note_service
            )
        
        # Send diarization results to client
        await websocket.send_text(json.dumps({
            'diarization': True,
            'transcript': full_transcript,
            'doctor_segments': doctor_segments,
            'patient_segments': patient_segments,
            'processing_time_ms': round(processing_time * 1000, 2),
            'doctor_speaking_time': float(sum(seg["end"] - seg["start"] for seg in doctor_segments)),
            'patient_speaking_time': float(sum(seg["end"] - seg["start"] for seg in patient_segments)),
            'total_segments': len(doctor_segments) + len(patient_segments),
            'confidence_scores': {int(k): float(v) for k, v in confidence_scores.items()},
            'status': 'diarization_complete'
        }, default=serialize_for_json))
        
        logger.info(f"Sent diarization results: {len(doctor_segments)} doctor, {len(patient_segments)} patient segments")
        
    except Exception as e:
        logger.error(f"Error processing diarization results: {str(e)}")
        logger.error(traceback.format_exc())


# Updated main websocket message handling loop
async def websocket_endpoint(websocket: WebSocket, db: Session = Depends(get_session)):
    """
    WebSocket endpoint for audio streaming with transcription-first approach
    """
    # Connection parameters
    connection_id = None
    user_id = None
    note_id = None
    doctor_id = None
    
    # Voice adaptation settings
    use_adaptation = False
    adaptation_user_id = None
    
    # For JSON serialization
    def serialize_for_json(obj):
        """Convert NumPy types to Python native types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime.datetime):
            return obj.isoformat()
        elif isinstance(obj, set):
            return list(obj)
        return obj
    
    try:
        # Get connection parameters from query params
        params = websocket.query_params
        token = params.get("token")
        user_id = params.get("user_id")
        note_id = params.get("note_id")
        doctor_id = params.get("doctor_id", user_id)  # Default to user_id if not specified
        
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
            
            # Initialize diarization service
            diarization_service = DiarizationService(db_session=db)
            
            # Store services for cleanup
            active_connections[connection_id] = {
                'websocket': ws,
                'audio_service': audio_service,
                'transcription_service': transcription_service,
                'keyword_service': keyword_service,
                'diarization_service': diarization_service,
                'last_heartbeat': time.time()
            }
            
            # Convert doctor_id to int if provided
            if doctor_id and doctor_id.isdigit():
                doctor_id = int(doctor_id)
            else:
                doctor_id = adaptation_user_id
            
            # Verify adaptation capabilities and send initial status
            if use_adaptation and adaptation_user_id is not None:
                calibration_status = calibration_service.get_calibration_status(adaptation_user_id, db)
                if not calibration_status.calibration_complete:
                    use_adaptation = False
                    await ws.send_text(json.dumps({
                        "warning": "No completed voice calibration found",
                        "adaptation_enabled": False,
                        "mode": "transcription_first"
                    }, default=serialize_for_json))
                else:
                    await ws.send_text(json.dumps({
                        "info": "Voice calibration profile found",
                        "adaptation_enabled": True,
                        "profile_id": calibration_status.profile_id,
                        "mode": "transcription_first_with_diarization"
                    }, default=serialize_for_json))
            else:
                await ws.send_text(json.dumps({
                    "info": "Starting transcription-first mode",
                    "adaptation_enabled": False,
                    "mode": "transcription_first"
                }, default=serialize_for_json))
            
            # Start heartbeat task
            heartbeat_task = asyncio.create_task(send_heartbeats(connection_id))
            
            # Main message loop
            while True:
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
                
                # Handle configuration updates
                if "config_update" in data:
                    updated_config = await handle_config_update(
                        ws, data["config_update"], 
                        connection_id, db, 
                        use_adaptation, adaptation_user_id
                    )
                    
                    if updated_config:
                        use_adaptation, adaptation_user_id = updated_config
                        
                    if "doctor_id" in data["config_update"]:
                        new_doctor_id = data["config_update"]["doctor_id"]
                        if new_doctor_id and isinstance(new_doctor_id, int):
                            doctor_id = new_doctor_id
                            logger.info(f"Updated doctor_id to {doctor_id}")
                    
                    continue
                                
                # Process audio chunk if present
                if data and "data" in data:
                    logger.debug("Received audio data for transcription-first processing")
                    
                    # Check for updated settings in the message
                    message_use_adaptation = data.get("use_adaptation", use_adaptation)
                    message_user_id = data.get("user_id", adaptation_user_id)
                    message_doctor_id = data.get("doctor_id", doctor_id)
                    
                    # Update settings if changed
                    if message_use_adaptation != use_adaptation or message_user_id != adaptation_user_id:
                        use_adaptation = message_use_adaptation
                        adaptation_user_id = message_user_id
                        logger.info(f"Updated settings: adaptation={use_adaptation}, user_id={adaptation_user_id}")
                    
                    if message_doctor_id != doctor_id:
                        doctor_id = message_doctor_id
                        logger.info(f"Updated doctor_id: {doctor_id}")
                    
                    # Process audio with transcription-first approach
                    await process_audio_with_transcription_first(
                        chunk_bytes=base64.b64decode(data["data"]) if data["data"] else None,
                        user_id=int(user_id), 
                        note_id=int(note_id), 
                        websocket=ws,
                        note_service=note_service,
                        transcription_service=transcription_service,
                        keyword_service=keyword_service,
                        diarization_service=diarization_service,
                        use_adaptation=use_adaptation,
                        adaptation_user_id=adaptation_user_id,
                        doctor_id=doctor_id,
                        db=db
                    )
                    
                elif data and "ping" in data:
                    # Send pong response
                    await ws.send_text(json.dumps({
                        "pong": time.time(),
                        "connection_id": connection_id,
                        "mode": "transcription_first"
                    }, default=serialize_for_json))
                else:
                    logger.info("Received unsupported message type")
                    await ws.send_text(json.dumps({
                        "warning": "Unsupported message format",
                        "received": str(type(data))
                    }, default=serialize_for_json))
                    continue
                    
            # Cancel heartbeat task
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: user_id={user_id}, note_id={note_id}")
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {str(e)}")
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_text(json.dumps({
                "error": "Invalid JSON message format",
                "details": str(e)
            }))
            await websocket.close(code=status.WS_1003_UNSUPPORTED_DATA)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        logger.error(traceback.format_exc())
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.send_text(json.dumps({
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }, default=serialize_for_json))
                await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
            except:
                await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
    finally:
        # Cleanup handled by context manager
        pass
            

def format_diarized_transcript(doctor_segments, patient_segments):
    """
    Format the transcript with speaker labels
    
    Args:
        doctor_segments: List of doctor segment dictionaries
        patient_segments: List of patient segment dictionaries
        
    Returns:
        Formatted transcript string
    """
    # Combine all segments
    all_segments = []
    
    for segment in doctor_segments:
        all_segments.append({
            "role": "Doctor",
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"]
        })
    
    for segment in patient_segments:
        all_segments.append({
            "role": "Patient",
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"]
        })
    
    # Sort by start time
    all_segments.sort(key=lambda x: x["start"])
    
    # Format as readable transcript
    lines = []
    for segment in all_segments:
        lines.append(f"[{segment['role']}] {segment['text']}")
    
    return "\n".join(lines)

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
            calibration_status = calibration_service.get_calibration_status(new_adaptation_user_id, db)
            if not calibration_status.calibration_complete:
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
                    "profile_id": calibration_status.profile_id
                }))
        
        return new_use_adaptation, new_adaptation_user_id
    except Exception as e:
        logger.error(f"Error processing config update: {str(e)}")
        await websocket.send_text(json.dumps({
            "error": f"Error updating configuration: {str(e)}"
        }))
        return use_adaptation, adaptation_user_id

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