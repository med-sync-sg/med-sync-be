
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import List
from app.utils.import_umls import DataStore
from app.utils.nlp import process_text, categorize_doc
from app.utils.speech_processor import AudioCollector
import numpy as np
from pyannote.audio import Pipeline, Inference
import torch
import numpy as np
import argparse
from spacy import displacy
import io
import soundfile as sf
from os import environ
import base64

HF_TOKEN = environ.get("HF_ACCESS_TOKEN")
def create_app() -> FastAPI:
    app = FastAPI(title="Backend Connection", version="1.0.0")
    # app.include_router(v1_endpoints.router, prefix="/v1", tags=["v1"])

    return app

connected_clients: List[WebSocket] = []

app = create_app()

load_umls = False

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

# Global collector instance
audio_collector = AudioCollector()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Receives audio chunks (raw PCM bytes) from the frontend as JSON: 
    { "data": bytes }
    We accumulate them in AudioCollector for near real-time or offline usage.
    """
    await websocket.accept()
    print("Client connected")

    try:
        while True:
            data = await websocket.receive_json()
            if data and "data" in data:
                chunk = data["data"]  # raw PCM bytes (sent as a string)
                if chunk is None:
                    print("End-of-stream or no 'data' field. Breaking loop.")
                    break
                # Convert the received string to bytes (adjust this conversion as needed)
                audio_collector.add_chunk(base64.b64decode(chunk))
                audio_collector.transcribe_audio_segment()
            else:
                print("End-of-stream or no 'data' field. Breaking loop.")
                break

    except WebSocketDisconnect:
        print("WebSocket disconnected")

    # For diarization, use the full session audio rather than the (possibly cleared) real-time buffer.
    wave_data = audio_collector.get_session_wave_data()
    print(f"Total wave_data shape: {wave_data.shape}")
    
    # Convert wave_data to an in-memory WAV for Pyannote
    buffer_wav = io.BytesIO()
    sf.write(buffer_wav, wave_data, 16000, format="WAV")
    buffer_wav.seek(0)  # rewind

    # Use the diarization pipeline (some require HF token if private)
    diar_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)
    diarization_result = diar_pipeline(
        buffer_wav, 
        min_speakers=2, 
        max_speakers=2
    )
    
    with open("audio.rttm", "w") as rttm:
        diarization_result.write_rttm(rttm)
    
    # Optionally, clear the session data for the next session.
    audio_collector.session_audio = bytearray()
    audio_collector.full_transcript.clear()

    print("All done.")
        


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