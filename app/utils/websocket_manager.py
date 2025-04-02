import logging
import base64
from typing import List, Dict, Any, Optional, Tuple
from fastapi import WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel

from app.utils.speech_processor import AudioCollector
from app.utils.auth_utils import decode_access_token

# Configure logger
logger = logging.getLogger(__name__)

class WebSocketManager:
    """
    Manager for WebSocket connections handling audio transcription
    
    This class:
    1. Manages active WebSocket connections
    2. Validates connection parameters and tokens
    3. Processes incoming audio data
    4. Sends transcription results back to clients
    """
    
    def __init__(self):
        """Initialize the WebSocket manager"""
        self.active_connections: List[WebSocket] = []
        self.audio_collector = AudioCollector()
        logger.info("WebSocket manager initialized")
    
    async def connect(self, websocket: WebSocket) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Connect a WebSocket client after validating connection parameters
        
        Args:
            websocket: The WebSocket connection
            
        Returns:
            (success, connection_info): Whether connection was successful and connection info
        """
        # Extract connection parameters
        connection_info = await self._validate_connection_params(websocket)
        
        if not connection_info:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return False, None
        
        # Accept the connection
        await websocket.accept()
        self.active_connections.append(websocket)
        
        logger.info(f"WebSocket client connected: user_id={connection_info['user_id']}, "
                   f"note_id={connection_info['note_id']}")
        
        return True, connection_info
    
    async def disconnect(self, websocket: WebSocket, connection_info: Optional[Dict[str, Any]] = None):
        """
        Disconnect a WebSocket client and clean up resources
        
        Args:
            websocket: The WebSocket connection
            connection_info: Optional connection information
        """
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        
        user_id = connection_info.get('user_id', 'unknown') if connection_info else 'unknown'
        
        # Reset audio collector state for this session
        self.audio_collector.session_audio = bytearray()
        self.audio_collector.full_transcript_text = ""
        
        logger.info(f"WebSocket client disconnected: user_id={user_id}")
    
    async def _validate_connection_params(self, websocket: WebSocket) -> Optional[Dict[str, Any]]:
        """
        Validate connection parameters from the WebSocket query params
        
        Args:
            websocket: The WebSocket connection
            
        Returns:
            Dictionary of validated connection parameters or None if invalid
        """
        token = websocket.query_params.get("token")
        note_id = websocket.query_params.get("note_id")
        user_id = websocket.query_params.get("user_id")
        
        if not all([token, note_id, user_id]):
            logger.warning("Missing required WebSocket connection parameters")
            return None
        
        try:
            # Uncomment for production use
            # payload = decode_access_token(token)
            # if str(payload.get("sub")) != str(user_id):
            #    logger.warning(f"Token subject mismatch: {payload.get('sub')} != {user_id}")
            #    return None
            
            return {
                "token": token,
                "note_id": note_id,
                "user_id": user_id
            }
        except Exception as e:
            logger.error(f"Token validation error: {str(e)}")
            return None
    
    async def process_audio_chunk(self, chunk: str, connection_info: Dict[str, Any], 
                                websocket: WebSocket) -> None:
        """
        Process an audio chunk, transcribe if appropriate, and send results
        
        Args:
            chunk: Base64-encoded audio chunk
            connection_info: Connection information including user_id and note_id
            websocket: The WebSocket connection to send results to
        """
        try:
            # Extract connection info
            user_id = connection_info["user_id"]
            note_id = connection_info["note_id"]
            
            # Decode base64 audio chunk
            audio_bytes = base64.b64decode(chunk)
            
            # Add to the audio collector
            self.audio_collector.add_chunk(audio_bytes)
            
            # Attempt transcription
            did_transcribe = self.audio_collector.transcribe_audio_segment(
                int(user_id), int(note_id)
            )
            
            # If transcription occurred, send results back to client
            if did_transcribe:
                # Send transcribed text
                await websocket.send_json({
                    'text': self.audio_collector.full_transcript_text
                })
                
                # Create and send sections
                sections = self.audio_collector.make_sections(int(user_id), int(note_id))
                sections_json = [section.model_dump_json() for section in sections]
                
                await websocket.send_json({
                    'sections': sections_json
                })
                
                logger.debug(f"Sent transcription results for user {user_id}, note {note_id}")
        except Exception as e:
            logger.error(f"Error processing audio chunk: {str(e)}")
            # Continue processing despite errors
    
    async def handle_client(self, websocket: WebSocket):
        """
        Main handler for WebSocket client interactions
        
        Args:
            websocket: The WebSocket connection
        """
        # Connect and validate the client
        success, connection_info = await self.connect(websocket)
        
        if not success:
            return
        
        try:
            # Main reception loop
            while True:
                # Receive JSON data
                data = await websocket.receive_json()
                
                # Process audio chunk if valid
                if data and "data" in data and data["data"] is not None:
                    await self.process_audio_chunk(data["data"], connection_info, websocket)
                else:
                    logger.info("Received end-of-stream or empty data. Ending connection.")
                    break
                    
        except WebSocketDisconnect:
            logger.info(f"WebSocket client disconnected: user_id={connection_info.get('user_id', 'unknown')}")
        except Exception as e:
            logger.error(f"WebSocket error: {str(e)}")
        finally:
            # Clean up connection resources
            await self.disconnect(websocket, connection_info)