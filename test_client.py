"""
Test Client for MedSync Application

This script tests the entire workflow of the MedSync application:
1. Connection to the database service
2. Retrieval of UMLS data
3. WebSocket connection for audio processing
4. Complete transcription and processing workflow

Usage:
    python test_client.py [--db-url DB_URL] [--app-url APP_URL] [--audio-file AUDIO_FILE]
"""
import argparse
import asyncio
import base64
import json
import logging
import os
import sys
import time
from typing import Dict, Any, List, Optional
from pprint import pprint
import aiohttp
import requests
import numpy as np
import websockets
import wave
import datetime
import traceback
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("test_client")

# Default URLs
DEFAULT_DB_URL = "http://127.0.0.1:8002"
DEFAULT_APP_URL = "http://127.0.0.1:8001"
DEFAULT_AUDIO_FILE = os.path.join("test_audios", "test_30sec.wav")

SAMPLE_TRANSCRIPT = """
Patient: Doctor, I've had a sore throat for the past three days, and it's getting worse. It feels scratchy, and swallowing is uncomfortable.
Doctor: I see. Has it been painful enough to affect eating or drinking?
Patient: No, but I also have a mild cough and keep sneezing a lot. My nose has been running non-stop.
Doctor: Sounds like you're experiencing some nasal irritation. Have you noticed any thick or discolored mucus?
Patient: Yeah, sometimes I feel some mucus at the back of my throat.
Doctor: Alright. Do you feel any tightness in your chest or shortness of breath when coughing?
Patient: No, but I've been feeling a bit feverish since last night. I haven't checked my temperature though. My body feels tired too.
Doctor: Fatigue and feverishness can be common with viral infections. Any chills or sweating?
Patient: No.
Doctor: Understood. Based on your symptoms, it looks like an upper respiratory tract infection, likely viral. Let me examine your throat to confirm.
Doctor: I will give you acetaminophen for the fever. Take two at a time three times a day.
"""

class WebSocketTester:
    """Helper class for testing WebSocket functionality of the application"""
    
    def __init__(self, base_url: str, user_id: int = 1, token: str = "test_token"):
        """
        Initialize the WebSocket tester
        
        Args:
            base_url: Base URL of the application (e.g., 'http://localhost:8001')
            user_id: User ID for testing
            token: Authentication token (optional for testing)
        """
        self.base_url = base_url
        self.user_id = user_id
        self.token = token
        
    async def test_connection(self, 
                             note_id: int = 1, 
                             timeout: float = 5.0) -> Dict[str, Any]:
        """
        Test basic WebSocket connection and heartbeat
        
        Args:
            note_id: ID of the note to use
            timeout: Connection timeout in seconds
            
        Returns:
            Test result dictionary
        """
        # Extract websocket URL from base URL
        ws_server = self.base_url.split('://')[-1]
        ws_url = f"ws://{ws_server}/ws?token={self.token}&user_id={self.user_id}&note_id={note_id}"
        
        try:
            logger.info(f"Testing WebSocket connection to {ws_url}")
            
            async with websockets.connect(ws_url, ping_interval=None) as websocket:
                logger.info("WebSocket connection established")
                
                # Wait for initial message (usually a welcome or heartbeat)
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=timeout)
                    try:
                        response_data = json.loads(response)
                        logger.info(f"Received initial message: {json.dumps(response_data)}")
                        return {
                            "success": True,
                            "message": "Connection successful",
                            "response": response_data
                        }
                    except json.JSONDecodeError:
                        logger.warning(f"Received non-JSON response: {response}")
                        return {
                            "success": True,
                            "message": "Connection successful but unexpected response format",
                            "response": response
                        }
                except asyncio.TimeoutError:
                    logger.warning("No initial message received within timeout")
                    return {
                        "success": True,
                        "message": "Connection successful but no response received",
                        "response": None
                    }
        except Exception as e:
            logger.error(f"WebSocket connection error: {str(e)}")
            return {
                "success": False,
                "message": f"Connection failed: {str(e)}",
                "error": str(e)
            }

    async def send_text(self, 
                      text: str, 
                      note_id: int = 1, 
                      timeout: float = 5.0) -> Dict[str, Any]:
        """
        Send text to the WebSocket for processing
        
        Args:
            text: Text to send
            note_id: ID of the note to use
            timeout: Response timeout in seconds
            
        Returns:
            Test result dictionary with processed sections
        """
        # Extract websocket URL from base URL
        ws_server = self.base_url.split('://')[-1]
        ws_url = f"ws://{ws_server}/ws?token={self.token}&user_id={self.user_id}&note_id={note_id}"
        
        try:
            logger.info(f"Connecting to WebSocket for text processing")
            received_sections = []
            
            async with websockets.connect(ws_url, ping_interval=None) as websocket:
                logger.info("WebSocket connection established")
                
                # Send text message
                message = {
                    "text": text
                }
                await websocket.send(json.dumps(message))
                logger.info(f"Sent text message: {text[:50]}...")
                
                # Wait for response(s)
                start_time = time.time()
                while time.time() - start_time < timeout:
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        try:
                            response_data = json.loads(response)
                            
                            # Process sections if present
                            if "sections" in response_data:
                                sections = response_data["sections"]
                                received_sections.extend(sections)
                                logger.info(f"Received {len(sections)} sections")
                            
                            # Check for processing completion
                            if "processing_complete" in response_data and response_data["processing_complete"]:
                                logger.info("Processing complete signal received")
                                break
                                
                        except json.JSONDecodeError:
                            logger.warning(f"Received non-JSON response: {response[:100]}...")
                    except asyncio.TimeoutError:
                        # No response in this check, continue waiting
                        pass
                
                # Gracefully close the connection
                close_msg = {"type": "close"}
                await websocket.send(json.dumps(close_msg))
                
                return {
                    "success": True,
                    "message": f"Processed text with {len(received_sections)} sections",
                    "sections": received_sections
                }
                
        except Exception as e:
            logger.error(f"WebSocket text processing error: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "message": f"Text processing failed: {str(e)}",
                "error": str(e)
            }
    
    async def stream_audio(self, 
                         audio_file: str,
                         note_id: int = 1,
                         chunk_size: int = 4096,
                         use_adaptation: bool = False) -> Dict[str, Any]:
        """
        Stream audio data from a file to the WebSocket endpoint for real-time processing
        
        Args:
            audio_file: Path to WAV audio file
            note_id: ID of the note to use
            chunk_size: Audio chunk size in bytes
            use_adaptation: Whether to use voice adaptation
            
        Returns:
            Test result dictionary with processed transcriptions and sections
        """
        if not os.path.exists(audio_file):
            return {
                "success": False,
                "message": f"Audio file not found: {audio_file}",
                "error": "File not found"
            }
            
        # Extract websocket URL from base URL
        ws_server = self.base_url.split('://')[-1]
        ws_url = f"ws://{ws_server}/ws?token={self.token}&user_id={self.user_id}&note_id={note_id}"
        
        try:
            # Read audio file and prepare chunks
            audio_chunks = self._read_audio_chunks(audio_file, chunk_size)
            logger.info(f"Audio file loaded: {len(audio_chunks)} chunks")
            
            # Store received data
            received_transcriptions = []
            received_sections = []
            
            # Connect to WebSocket
            async with websockets.connect(ws_url, ping_interval=None) as websocket:
                logger.info("WebSocket connection established")
                
                # Send audio chunks
                for i, chunk in enumerate(audio_chunks):
                    chunk_b64 = base64.b64encode(chunk).decode('utf-8')
                    message = {
                        "data": chunk_b64,
                        "use_adaptation": use_adaptation,
                        "user_id": self.user_id
                    }
                    await websocket.send(json.dumps(message))
                    logger.info(f"Sent audio chunk {i+1}/{len(audio_chunks)}")
                    
                    # Process any responses without blocking too long
                    await self._process_responses(
                        websocket, 
                        received_transcriptions, 
                        received_sections, 
                        timeout=0.3
                    )
                
                # Send end-of-stream signal
                await websocket.send(json.dumps({"data": None}))
                logger.info("Sent end-of-stream signal")
                
                # Process final responses with longer timeout
                await self._process_responses(
                    websocket, 
                    received_transcriptions, 
                    received_sections, 
                    timeout=10.0,
                    is_final=True
                )
                
                # Validate the most recent section
                validation_result = self._validate_sections(received_sections)
                
                return {
                    "success": True,
                    "message": f"Processed audio with {len(received_transcriptions)} transcriptions and {len(received_sections)} sections",
                    "transcription_count": len(received_transcriptions),
                    "section_count": len(received_sections),
                    "transcriptions": received_transcriptions,
                    "sections": received_sections,
                    "validation": validation_result
                }
                
        except Exception as e:
            logger.error(f"WebSocket audio streaming error: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "message": f"Audio streaming failed: {str(e)}",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    async def _process_responses(self, 
                              websocket, 
                              received_transcriptions: List[str], 
                              received_sections: List[Dict[str, Any]],
                              timeout: float = 1.0,
                              is_final: bool = False) -> None:
        """
        Process WebSocket responses
        
        Args:
            websocket: Active WebSocket connection
            received_transcriptions: List to append transcriptions to
            received_sections: List to append sections to
            timeout: Timeout for receiving messages
            is_final: Whether this is final response processing
        """
        start_time = time.time()
        max_duration = 10.0 if is_final else timeout
        received_count = 0
        
        # Process responses for the specified duration
        while time.time() - start_time < max_duration:
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=0.5)
                received_count += 1
                
                try:
                    response_data = json.loads(response)
                    
                    # Process transcription
                    if "text" in response_data:
                        transcript = response_data["text"]
                        received_transcriptions.append(transcript)
                        adaptation_status = response_data.get("using_adaptation", False)
                        processing_time = response_data.get("processing_time_ms", None)
                        
                        logger.info(f"Received transcription ({len(transcript)} chars)")
                        if processing_time:
                            logger.info(f"Processing time: {processing_time} ms")
                        if adaptation_status:
                            logger.info(f"Using adaptation: {adaptation_status}")
                    
                    # Process sections
                    if "sections" in response_data:
                        sections = response_data["sections"]
                        received_sections.extend(sections)
                        logger.info(f"Received {len(sections)} sections")
                        
                        for section in sections:
                            title = section.get('title', 'Untitled')
                            soap_category = section.get('soap_category', 'Unknown')
                            logger.info(f"  Section: {title} ({soap_category})")
                    
                    # Check for errors
                    if "error" in response_data:
                        logger.error(f"Server error: {response_data['error']}")
                        
                except json.JSONDecodeError:
                    logger.warning(f"Received non-JSON response: {response[:100]}...")
                    
            except asyncio.TimeoutError:
                # No response in this check
                if is_final and received_count == 0 and time.time() - start_time > 3.0:
                    # Break early if no responses received for a while during final processing
                    logger.info("No responses during final processing, finishing early")
                    break
        
        if is_final:
            logger.info(f"Finished processing {received_count} final responses")
    
    def _read_audio_chunks(self, audio_file: str, chunk_size: int = 4096) -> List[bytes]:
        """
        Read audio file as chunks for streaming
        
        Args:
            audio_file: Path to WAV audio file
            chunk_size: Size of each chunk in bytes
            
        Returns:
            List of audio data chunks
        """
        import wave
        
        try:
            with wave.open(audio_file, 'rb') as wav_file:
                # Get audio parameters
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                frame_rate = wav_file.getframerate()
                logger.info(f"Audio file: {channels} channels, {sample_width} bytes/sample, {frame_rate} Hz")
                
                # Read chunks
                chunks = []
                data = wav_file.readframes(chunk_size)
                while data:
                    chunks.append(data)
                    data = wav_file.readframes(chunk_size)
                
                return chunks
        except Exception as e:
            logger.error(f"Error reading audio file: {str(e)}")
            return []
    
    def _validate_sections(self, sections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate section structure based on expected fields
        
        Args:
            sections: List of section dictionaries
            
        Returns:
            Validation result dictionary
        """
        if not sections:
            return {
                "valid": False,
                "message": "No sections to validate"
            }
            
        # Use the last section for validation
        section = sections[-1]
        
        # Check required fields
        required_fields = ['id', 'title', 'content', 'soap_category']
        missing_fields = [f for f in required_fields if f not in section]
        
        if missing_fields:
            return {
                "valid": False,
                "message": f"Section missing required fields: {missing_fields}",
                "missing_fields": missing_fields
            }
        
        # Validate content structure
        content = section.get('content', {})
        if not content:
            return {
                "valid": False,
                "message": "Section has empty content dictionary"
            }
        
        # Check field structure
        field_validation = {}
        for field_id, field_data in content.items():
            if isinstance(field_data, dict):
                # Check required field properties
                field_required = ['name', 'value', 'data_type']
                field_missing = [f for f in field_required if f not in field_data]
                field_validation[field_id] = {
                    "valid": len(field_missing) == 0,
                    "missing": field_missing if field_missing else None,
                    "type": "dictionary"
                }
            elif isinstance(field_data, list):
                # Check list of field objects
                field_validation[field_id] = {
                    "valid": True,
                    "type": "list",
                    "count": len(field_data)
                }
            else:
                field_validation[field_id] = {
                    "valid": False,
                    "type": str(type(field_data)),
                    "message": "Unexpected field data type"
                }
        
        # Determine overall validity
        valid_fields = [v["valid"] for v in field_validation.values()]
        is_valid = all(valid_fields) if valid_fields else False
        
        return {
            "valid": is_valid,
            "message": "Section structure validation completed",
            "field_count": len(content),
            "field_validation": field_validation
        }


class TestClient:
    """Test client for MedSync application"""
    
    def __init__(self, db_url: str, app_url: str, audio_file: str):
        """
        Initialize test client
        
        Args:
            db_url: URL for the database service
            app_url: URL for the main application
            audio_file: Path to an audio file for testing
        """
        self.db_url = db_url
        self.app_url = app_url
        self.audio_file = audio_file
        
        # Authentication info
        self.token = None
        self.user_id = None
        self.note_id = None
        
        logger.info(f"Test client initialized with DB URL: {db_url}, App URL: {app_url}")
        
    def run_tests(self):
        """Run all tests"""
        try:
            # Test database service
            # self.test_db_connection()
            # self.test_umls_data()
            
            # Test main application
            # self.authenticate()
            # self.create_test_note()
            
            # Test WebSocket (needs to run in an async context)
            # asyncio.run(self.test_websocket())
            
            # self.generate_test_report_doctor(user_id=2, note_id=12, template_type="doctor")
            
            results = self.test_text_processing()
            self.generate_doctor_report_from_text_processing(results)
            # self.test_adaptation_feature()
            logger.info("All tests completed.")
            return True
            
        except Exception as e:
            logger.error(f"Test failed: {str(e)}")
            return False

    def test_adaptation_feature(self):
        # API endpoint
        url = f"{self.app_url}/tests/test-adaptation"
        
        # User ID with calibration data
        user_id = 1  # Replace with an actual user ID that has calibration data
        
        # Prepare the request
        with open(DEFAULT_AUDIO_FILE, "rb") as f:
            files = {"audio_file": f}
            data = {"user_id": user_id, "use_adaptation": True}
            
            # Make the request
            response = requests.post(url, files=files, data=data)
            
        # Print the results
        print("Response status:", response.status_code)
        if response.status_code == 200:
            result = response.json()
            print("\nStandard transcription:", result["standard_transcription"])
            print("Standard processing time:", result["standard_processing_time_ms"], "ms")
            
            if result.get("adapted_transcription"):
                print("\nAdapted transcription:", result["adapted_transcription"])
                print("Adaptation processing time:", result["adaptation_processing_time_ms"], "ms")
                print("Adaptation info:", result["adaptation_info"])
            else:
                print("\nNo adaptation results available")
        else:
            print("Error:", response.text)

    def generate_doctor_report_from_note(self, note_id: int):
        """
        Generate a SOAP-structured doctor's report from an existing note in the database
        
        Args:
            note_id: ID of the note to generate report for
            
        Returns:
            HTML content of the report or None if failed
        """
        logger.info(f"Generating SOAP-structured doctor report for note {note_id}")
        
        try:
            # Call the doctor report generation endpoint
            response = requests.get(
                f"{self.app_url}/reports/doctor/{note_id}", 
                headers={"Authorization": f"Bearer {self.token}"} if self.token else None
            )
            
            if response.status_code == 200:
                # Return HTML content
                report_html = response.text
                logger.info(f"Successfully generated SOAP-structured doctor report for note {note_id} ({len(report_html)} bytes)")
                
                # Save the report to a file for inspection
                timestamp = int(time.time())
                filename = f"soap_doctor_report_{note_id}_{timestamp}.html"
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(report_html)
                logger.info(f"Saved SOAP-structured doctor report to {filename}")
                
                return report_html
            else:
                logger.error(f"Failed to generate doctor report: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error generating doctor report: {str(e)}")
            return None

    def generate_doctor_report_from_text_processing(self, processing_result):
        """
        Generate a SOAP-structured doctor's report from text processing results
        
        Args:
            processing_result: Result from test_text_processing method
            
        Returns:
            Doctor report HTML or None if failed
        """
        if not processing_result or not processing_result.get("success", False):
            logger.error("No valid text processing result provided")
            return None
        
        try:
            # Extract entities and sections from the processing result
            entities = processing_result.get("entities", [])
            processed_content = processing_result.get("processed_content", [])
            
            # Create a note structure from the processing result
            note_data = {
                "title": "SOAP Report from Processed Text",
                "patient_id": None,  # No patient ID for demonstration
                "user_id": self.user_id if hasattr(self, "user_id") else 1,
                "encounter_date": datetime.datetime.now().strftime("%Y-%m-%d"),
                "sections": []
            }
            
            # Convert processed content to sections with SOAP categories
            for section in processed_content:
                # Handle different section formats
                if isinstance(section, dict):
                    # Extract the first key as the title if present
                    title = next(iter(section.keys())) if section else "Untitled Section"
                    
                    # Determine SOAP category based on content
                    soap_category = section.get("soap_category", "N/A")
                    
                    # Add to sections
                    note_data["sections"].append({
                        "title": title,
                        "soap_category": soap_category,
                        "content": section
                    })
            
            # Generate doctor report
            logger.info(f"Generating SOAP-structured doctor report from text processing data")
            
            response = requests.post(
                f"{self.app_url}/tests/doctor/generate",
                json=note_data,
                headers={"Authorization": f"Bearer {self.token}"} if self.token else None
            )
            
            if response.status_code == 200:
                # Get HTML content
                report_html = response.text
                
                # Save the report to a file for inspection
                timestamp = int(time.time())
                filename = f"soap_doctor_report_direct_{timestamp}.html"
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(report_html)
                logger.info(f"Saved SOAP-structured doctor report to {filename}")
                
                return report_html
            else:
                logger.error(f"Failed to generate doctor report: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error generating doctor report from text processing: {str(e)}")
            return None
        
    def test_db_connection(self):
        """Test connection to the database service"""
        logger.info("Testing database service connection...")
        
        try:
            response = requests.get(f"{self.db_url}/status", timeout=5)
            if response.status_code == 200:
                status_data = response.json()
                if status_data.get('status') == 'ready':
                    logger.info("Database service connection successful!")
                    logger.info(f"Service status: {json.dumps(status_data, indent=2)}")
                else:
                    logger.warning(f"Database service not ready: {status_data.get('message', 'Unknown reason')}")
            else:
                raise Exception(f"Database service returned status code {response.status_code}")
        except requests.RequestException as e:
            raise Exception(f"Could not connect to database service: {str(e)}")
    
    def test_umls_data(self):
        """Test fetching UMLS data from the database service"""
        logger.info("Testing UMLS data retrieval...")
        
        try:
            # Test symptoms and diseases endpoint
            response = requests.get(f"{self.db_url}/umls-data/symptoms-and-diseases", timeout=10)
            if response.status_code != 200:
                raise Exception(f"Failed to fetch symptoms data: {response.status_code}")
            logger.info(f"Successfully retrieved symptoms data ({len(response.content)} bytes)")
            
            # Test drugs endpoint
            response = requests.get(f"{self.db_url}/umls-data/drugs", timeout=10)
            if response.status_code != 200:
                raise Exception(f"Failed to fetch drugs data: {response.status_code}")
            logger.info(f"Successfully retrieved drugs data ({len(response.content)} bytes)")
            
            logger.info("UMLS data retrieval test passed!")
        except requests.RequestException as e:
            raise Exception(f"Error fetching UMLS data: {str(e)}")
    
    def authenticate(self):
        """Authenticate with the main application"""
        logger.info("Authenticating with the main application...")
        
        try:
            # Create a test user if needed
            credentials = {
                "username": "test_user",
                "password": "test_password"
            }
            
            # Try to sign up first (this might fail if user exists, which is OK)
            try:
                signup_data = {
                    "username": "test_user",
                    "password": "test_password",
                    "first_name": "Test",
                    "last_name": "User",
                    "email": "test@example.com",
                    "age": 30,
                    "notes": []
                }
                requests.post(f"{self.app_url}/auth/sign-up", json=signup_data)
            except:
                logger.info("User may already exist, proceeding to login")
            
            # Login
            response = requests.post(f"{self.app_url}/auth/login", json=credentials)
            if response.status_code != 200:
                raise Exception(f"Authentication failed: {response.status_code}")
            
            auth_data = response.json()
            self.token = auth_data.get("access_token")
            if not self.token:
                raise Exception("No token received after authentication")
            
            # Get user ID
            headers = {"Authorization": f"Bearer {self.token}"}
            response = requests.get(f"{self.app_url}/users/", headers=headers)
            if response.status_code != 200:
                raise Exception(f"Failed to get user info: {response.status_code}")
            
            users = response.json()
            if users and len(users) > 0:
                self.user_id = users[0].get("id")
                logger.info(f"Authentication successful! User ID: {self.user_id}")
            else:
                raise Exception("No user found after authentication")
            
        except requests.RequestException as e:
            raise Exception(f"Authentication error: {str(e)}")
    
    def create_test_note(self):
        """Create a test note for the authenticated user"""
        logger.info("Creating a test note...")
        
        try:
            if not self.token or not self.user_id:
                raise Exception("Not authenticated")
            
            headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
            
            # Create a note
            note_data = {
                "title": f"Test Note {int(time.time())}",
                "user_id": self.user_id,
                "patient_id": 12345,
                "encounter_date": "2023-04-03",
                "sections": []
            }
            response = requests.post(f"{self.app_url}/notes/create", json=note_data, headers=headers)

            if response.status_code != 201:
                raise Exception(f"Failed to create note: {response.status_code}")
            
            note = response.json()
            self.note_id = note.get("id")
            if not self.note_id:
                raise Exception("No note ID received after creation")
            
            logger.info(f"Test note created successfully! Note ID: {self.note_id}")
            
        except requests.RequestException as e:
            raise Exception(f"Error creating test note: {str(e)}")
    
    def test_text_processing(self):
        """
        Test direct text processing without audio transcription
        
        This sends a text file directly to the backend for processing
        """
        logger.info("Testing direct text processing...")
        
        try:
            # if not self.token or not self.user_id or not self.note_id:
                # raise Exception("Authentication or note creation failed, cannot test text processing")
            
            # if not self.text_file or not os.path.exists(self.text_file):
                # raise Exception(f"Text file not found: {self.text_file}")
            
            # Read the text file
            # with open(self.text_file, 'r', encoding='utf-8') as file:
                # text_content = file.read()
            text_content = SAMPLE_TRANSCRIPT
                
            logger.info(f"Loaded text file ({len(text_content)} characters)")
            
            headers = {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json"
            }
            
            # Prepare the request data
            request_data = {
                "transcript": text_content
            }
            
            # Send the text to the backend for processing
            response = requests.post(
                f"{self.app_url}/tests/text-transcript", 
                json=request_data, 
                headers=headers
            )
            
            if response.status_code != 200:
                logger.error(f"Text processing failed: {response.status_code}")
                logger.error(f"Response: {response.text}")
                raise Exception(f"Failed to process text: {response.status_code}")
            
            # Process response
            result = response.json()
            # Log the results
            logger.info("Text processing successful!")
        
            print_formatted_transcript_results(result)
            return result
            
        except Exception as e:
            raise Exception(f"Text processing test error: {str(e)}")
    
    async def test_websocket(self):
        tester = WebSocketTester(base_url="http://localhost:8001")
        connection_result = await tester.test_connection()
        print(f"Connection test: {'SUCCESS' if connection_result['success'] else 'FAILED'}")
        
        # Test audio streaming
        audio_result = await tester.stream_audio(
            audio_file=self.audio_file,
            note_id=1,
            use_adaptation=False
        )
        
        print(f"Audio test: {'SUCCESS' if audio_result['success'] else 'FAILED'}")
        print(f"Transcription count: {audio_result.get('transcription_count', 0)}")
        print(f"Section count: {audio_result.get('section_count', 0)}")
        
        
        print(audio_result)
        # # Test text processing
        # text_result = await tester.send_text(
        #     text="Patient reports a sore throat and fever lasting three days.",
        #     note_id=1
        # )
        
        # print(f"Text test: {'SUCCESS' if text_result['success'] else 'FAILED'}")
        # print(f"Sections: {len(text_result.get('sections', []))}")
        
        return tester
    
    def _read_audio_chunks(self, audio_file: str, chunk_size: int = 4096) -> List[bytes]:
        """
        Read audio file as chunks for streaming
        
        Args:
            audio_file: Path to WAV audio file
            chunk_size: Size of each chunk in bytes
            
        Returns:
            List of audio data chunks
        """
        try:
            with wave.open(audio_file, 'rb') as wav_file:
                # Get audio parameters
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                frame_rate = wav_file.getframerate()
                logger.info(f"Audio file: {channels} channels, {sample_width} bytes/sample, {frame_rate} Hz")
                
                # Read chunks
                chunks = []
                data = wav_file.readframes(chunk_size)
                while data:
                    chunks.append(data)
                    data = wav_file.readframes(chunk_size)
                
                return chunks
        except Exception as e:
            logger.error(f"Error reading audio file: {str(e)}")
            return []

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="MedSync Test Client")
    parser.add_argument("--db-url", default=DEFAULT_DB_URL, help="URL for database service")
    parser.add_argument("--app-url", default=DEFAULT_APP_URL, help="URL for main application")
    parser.add_argument("--audio-file", default=DEFAULT_AUDIO_FILE, help="Path to audio file for testing")
    return parser.parse_args()

def print_formatted_transcript_results(result: dict):
    """
    Prints the results from the process_text_transcript endpoint in a nicely formatted way
    
    Args:
        result: Response dictionary from the text-transcript endpoint
    """
    if not result.get("success", False):
        print("\n❌ TRANSCRIPT PROCESSING FAILED")
        print(f"Error: {result.get('error', 'Unknown error')}")
        return

    print("\n✅ TRANSCRIPT PROCESSING SUCCESSFUL")
    print("=" * 80)
    
    # Print summary
    transcript = result.get("transcription", "")
    print(f"📝 TRANSCRIPT ({len(transcript)} chars):")
    print("-" * 80)
    print(transcript[:200] + "..." if len(transcript) > 200 else transcript)
    print("-" * 80)
    
    # Print extracted keywords
    keywords = result.get("entities", [])
    print(f"\n🔑 EXTRACTED KEYWORDS: {len(keywords)}")
    print("-" * 80)
    for i, kw in enumerate(keywords):  # Show first 5 keywords
        if isinstance(kw, dict) and "term" in kw:
            print(f"  {i+1}. Term: {kw.get('term')}")
            
            for key, value in kw.items():
                print(f"     {key}: {', '.join(value)}")
        else:
            print(f"  {i+1}. {kw}")
    print("-" * 80)
    
    # Print template suggestions
    templates = result.get("template_suggestions", [])
    print(f"\n📋 TEMPLATE SUGGESTIONS: {len(templates)}")
    print("-" * 80)
    for i, template in enumerate(templates):  # Show first 3 templates
        print(f"  {i+1}. {template.get('name', 'Unnamed template')}")
        print(f"     Description: {template.get('description', 'N/A')}")
        print(f"     Similarity score: {template.get('similarity_score', 0):.2f}")
        print(f"     Template ID: {template.get('id', 'N/A')}")
        if i < len(templates) - 1:  # Add separator between templates
            print("  " + "-" * 30)
    print("-" * 80)
    
    # Print processed content (sections)
    sections = result.get("processed_content", [])
    print(sections[0])
    print(f"\n📊 PROCESSED SECTIONS: {len(sections)}")
    print("-" * 80)
    
    for i, section in enumerate(sections):  # Show first 3 sections
        # Handle different section formats
        print(f"{i + 1}. Section {section['title']} using template \"{section['template_id']}\"")
        print(f"   Section Content:")
        pprint(section['content'])
        print(f"   ALL KEYS: ")
        pprint(section)    
        if i < len(sections) - 1:  # Add separator between sections
            print("  " + "-" * 30)
    
    print("=" * 80)
    print("PROCESSING COMPLETE\n")

if __name__ == "__main__":
    args = parse_arguments()
    client = TestClient(args.db_url, args.app_url, args.audio_file)
    success = client.run_tests()
    sys.exit(0 if success else 1)