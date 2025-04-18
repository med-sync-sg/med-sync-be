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

import aiohttp
import requests
import numpy as np
import websockets
import wave

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
Doctor: Good morning, how can I help you today?

Patient: Hello doctor. I've been having a persistent cough for the past two weeks. It started as just a tickle in my throat but now it's getting worse.

Doctor: I'm sorry to hear that. Is your cough dry or are you coughing up any phlegm?

Patient: It's mostly dry, but sometimes in the morning I cough up some clear phlegm. Not much though.

Doctor: I see. And have you been experiencing any other symptoms? Fever, chills, shortness of breath?

Patient: I had a slight fever for a couple of days when it first started, maybe 99.5°F. No chills. I do feel a bit short of breath when I climb the stairs, which isn't normal for me.

Doctor: Are you having any chest pain or tightness?

Patient: Not really pain, but sometimes I feel some tightness in my chest, especially after coughing a lot.

Doctor: Have you been exposed to anyone with similar symptoms or who's been sick recently?

Patient: Yes, my coworker had a bad cold last month. Several people in the office got sick afterward.

Doctor: I understand. What about your medical history? Do you have any conditions like asthma, allergies, or COPD?

Patient: I have mild seasonal allergies, usually in the spring, but no asthma or anything like that. I'm generally healthy.

Doctor: Do you smoke or vape?

Patient: No, never have.

Doctor: And are you taking any medications currently?

Patient: Just over-the-counter stuff for the cough - some Mucinex and occasionally Tylenol for headaches.

Doctor: Based on what you've told me, this sounds like a post-viral cough, possibly from a respiratory infection. I'd like to listen to your lungs and check your throat to make sure.

Patient: That makes sense. I was wondering if it might be bronchitis or something.

Doctor: It's possible. Let me examine you and we'll figure out the best treatment approach.
"""

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
            self.test_db_connection()
            self.test_umls_data()
            
            # Test main application
            self.authenticate()
            self.create_test_note()
            
            # Test WebSocket (needs to run in an async context)
            # asyncio.run(self.test_websocket())
            
            
            self.test_text_processing()
            logger.info("All tests completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Test failed: {str(e)}")
            return False
    
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
            if not self.token or not self.user_id or not self.note_id:
                raise Exception("Authentication or note creation failed, cannot test text processing")
            
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
            
            if isinstance(result, list):
                logger.info(f"Received {len(result)} processed sections")
                for i, section in enumerate(result):
                    if isinstance(section, str):
                        # If it's a serialized JSON string, try to parse it
                        try:
                            section_obj = json.loads(section)
                            logger.info(f"Section {i+1}:")
                            # Check for common patterns in the returned data
                            if "Main Symptom" in section_obj:
                                symptom_name = section_obj["Main Symptom"].get("name", "Unknown")
                                logger.info(f"  - Main symptom: {symptom_name}")
                            # Log the full object if it's small enough
                            if len(section) < 500:
                                logger.info(f"  - Full content: {section}")
                            else:
                                logger.info(f"  - Full content omitted (too large)")
                        except json.JSONDecodeError:
                            logger.info(f"Section {i+1}: {section[:100]}...")
                    else:
                        logger.info(f"Section {i+1}: {section}")
            else:
                logger.info(f"Result: {result}")
                
            return result
            
        except Exception as e:
            raise Exception(f"Text processing test error: {str(e)}")
    
    async def test_websocket(self):
        """Test WebSocket connection and audio processing"""
        if not os.path.exists(self.audio_file):
            logger.warning(f"Audio file {self.audio_file} not found, skipping WebSocket test")
            return
        
        if not self.token or not self.user_id or not self.note_id:
            raise Exception("Authentication or note creation failed, cannot test WebSocket")
        
        logger.info("Testing WebSocket connection and audio processing...")
        
        try:
            # Prepare WebSocket URL with query parameters
            ws_url = f"ws://{self.app_url.split('://')[-1]}/ws?token={self.token}&user_id={self.user_id}&note_id={self.note_id}"
            
            # Read audio file and encode in chunks for sending
            audio_chunks = self._read_audio_chunks(self.audio_file)
            logger.info(f"Audio file loaded: {len(audio_chunks)} chunks")
            
            # Connect to WebSocket
            async with websockets.connect(ws_url) as websocket:
                logger.info("WebSocket connection established")
                
                # Send audio chunks
                for i, chunk in enumerate(audio_chunks):
                    chunk_b64 = base64.b64encode(chunk).decode('utf-8')
                    message = {
                        "data": chunk_b64
                    }
                    await websocket.send(json.dumps(message))
                    logger.info(f"Sent audio chunk {i+1}/{len(audio_chunks)}")
                    
                    # Give the server some time to process
                    await asyncio.sleep(0.1)
                    
                    # Check for any responses
                    try:
                        # Set a timeout for receiving
                        response = await asyncio.wait_for(websocket.recv(), timeout=0.2)
                        response_data = json.loads(response)
                        
                        if "text" in response_data:
                            logger.info(f"Received transcription: {response_data['text']}")
                        
                        if "sections" in response_data:
                            logger.info(f"Received sections: {len(response_data['sections'])} sections")
                            for section in response_data['sections']:
                                logger.info(f"  - Section: {section.get('title', 'Untitled')}")
                    except asyncio.TimeoutError:
                        # No response yet, continue
                        pass
                
                # Send empty data to indicate end of stream
                await websocket.send(json.dumps({"data": None}))
                logger.info("Sent end-of-stream signal")
                
                # Wait for final processing
                try:
                    # Wait for final response with a longer timeout
                    final_response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    response_data = json.loads(final_response)
                    logger.info(f"Final response received: {json.dumps(response_data, indent=2)}")
                except asyncio.TimeoutError:
                    logger.info("No final response received within timeout")
                
                logger.info("WebSocket test completed")
                
        except Exception as e:
            raise Exception(f"WebSocket test error: {str(e)}")
    
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

if __name__ == "__main__":
    args = parse_arguments()
    client = TestClient(args.db_url, args.app_url, args.audio_file)
    success = client.run_tests()
    sys.exit(0 if success else 1)