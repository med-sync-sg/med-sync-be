
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import List
from app.utils.import_umls import DataStore
from app.utils.nlp import process_text
import queue
import threading
import torch
import whisper
import numpy as np
import argparse
from spacy import displacy
from api.v1 import endpoints as v1_endpoints

def create_app() -> FastAPI:
    app = FastAPI(title="Backend Connection", version="1.0.0")
    app.include_router(v1_endpoints.router, prefix="/v1", tags=["v1"])

    return app

connected_clients: List[WebSocket] = []

app = create_app()

load_umls = True

if load_umls == True:
    data_store = DataStore()  # first import triggers load

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = whisper.load_model("medium")
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    shared_transcript = []
    audio_queue = queue.Queue()
    def preprocess_audio(buffer):
        audio_data = np.frombuffer(buffer, dtype=np.int16).astype(np.float32)
        audio_data /= 32768.0  # Now the range is [-1.0, +1.0]
        return audio_data

    def transcribe_stream(audio_queue: queue.Queue, websocket: WebSocket, shared_transcript: list):
        buffer = bytes()
        while True:
            data = audio_queue.get()
            if data is None:  # End signal
                break

            buffer += bytes(data)

            # Process audio every ~1 second
            if len(buffer) > 32000:  # ~1 second of 16kHz 16-bit mono audio
                audio_data = preprocess_audio(buffer)
                print(f"Processing buffer with size: {len(buffer)} bytes")
                
                try:
                    result = model.transcribe(audio_data, language="en")
                    text = result["text"]
                    if text.strip():
                        print("TRANSCRIBED TEXT:", text)
                        # Append to our shared transcript list
                        shared_transcript.append(text)
                except Exception as e:
                    print("Error during transcription:", e)
                
                # Reset buffer
                buffer = bytes()

    await websocket.accept()
    connected_clients.append(websocket)
    print("Client connected")
    
    transcription_thread = threading.Thread(
        target=transcribe_stream,
        daemon=True,
        args=(audio_queue, websocket, shared_transcript)
    )
    transcription_thread.start()
    
    try:
        while True:
            data = await websocket.receive_json()
            if data:
                try:
                    if data["data"]:
                        audio_chunk = data["data"]
                        if audio_chunk == None:
                            audio_queue.put(None)
                        audio_queue.put(audio_chunk)
                    elif data["data"] == None: # End of audio data stream
                        full_transcript = ""
                        for text in shared_transcript:
                            full_transcript = full_transcript + text
                        doc = process_text(full_transcript)
                        print(displacy.render(doc, style="ent"))
                except Exception as e:
                    print(f"Error processing audio chunk: {e}")
            else:
                print("Non-audio data received")

    except WebSocketDisconnect:
        print("Client disconnected")
        audio_queue.put(None)
        connected_clients.remove(websocket)
    except Exception as e:
        print(f"WebSocket connection error: {e}")
        


# Add routes or other configurations here
@app.get("/")
async def read_root():
    return {"message": "Hello, FastAPI from app.py!"}

@app.post("/audio-test")
async def process_transcript_audio(data):
   pass

@app.post("/text-test")
async def process_transcript_text(data: str):
    result_doc = process_text(data)
    return result_doc