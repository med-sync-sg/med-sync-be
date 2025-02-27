
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import List
from app.db.session import DataStore
from app.utils.speech_processor import AudioCollector
import numpy as np
from pyannote.audio import Pipeline, Inference
import torch
import numpy as np
import argparse
from spacy import displacy
from app.api.v1.endpoints import auth, notes, users, templates, reports
import io
import soundfile as sf
from os import environ
import base64

HF_TOKEN = environ.get("HF_ACCESS_TOKEN")
def create_app() -> FastAPI:
    app = FastAPI(title="Backend Connection", version="1.0.0")
    app.include_router(auth.router, prefix="/auth", tags=["auth"])
    app.include_router(notes.router, prefix="/notes", tags=["note"])
    app.include_router(users.router, prefix="/users", tags=["user"])
    app.include_router(templates.router, prefix="/templates", tags=["template"])
    app.include_router(reports.router, prefix="/reports", tags=["report"])

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
                transcribed_text = audio_collector.transcribe_audio_segment()
                await websocket.send_json({'text': transcribed_text})
            else:
                print("End-of-stream or no 'data' field. Breaking loop.")
                break
        audio_collector.get_tagged_doc_and_upload_sections()
        # Optionally, clear the session data for the next session.
        audio_collector.session_audio = bytearray()
        audio_collector.full_transcript.clear()

    except WebSocketDisconnect:
        print("WebSocket disconnected")



    print("All done.")
