
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, status
from typing import List

from app.utils.speech_processor import AudioCollector
import numpy as np
from pyannote.audio import Pipeline, Inference
import torch
import numpy as np
import argparse
import io
import soundfile as sf
from os import environ
import base64
import json
import requests
from app.api.v1.endpoints import auth, notes, users, templates, reports, tests
from app.db.iris_session import IrisDataStore
from app.utils.text_utils import clean_transcription, correct_spelling
HF_TOKEN = environ.get("HF_ACCESS_TOKEN")
def create_app() -> FastAPI:

    app = FastAPI(title="Backend Connection", version="1.0.0")
    app.include_router(auth.router, prefix="/auth", tags=["auth"])
    app.include_router(notes.router, prefix="/notes", tags=["note"])
    app.include_router(users.router, prefix="/users", tags=["user"])
    app.include_router(templates.router, prefix="/templates", tags=["template"])
    app.include_router(reports.router, prefix="/reports", tags=["report"])
    app.include_router(tests.router, prefix="/tests", tags=["test"])

    return app

connected_clients: List[WebSocket] = []

IrisDataStore()

app = create_app()

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
    token = websocket.query_params.get("token")
    note_id = websocket.query_params.get("note_id")
    user_id = websocket.query_params.get("user_id")

    if not token or not note_id or not user_id:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    
    # Validate token (this should raise an exception if invalid).
    # try:
    #     payload = decode_access_token(token)
    #     # Check that the token's subject matches the provided user_id.
    #     if payload.get("sub") != user_id:
    #         await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
    #         return
    # except Exception as e:
    #     await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
    #     return
    
    
    await websocket.accept()
    print("Client authentication verified and connected")

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
                did_transcribe = audio_collector.transcribe_audio_segment(user_id, note_id)
                if did_transcribe:
                    transcribed_text = audio_collector.full_transcript_text
                    cleaned_text = clean_transcription(transcribed_text)
                    corrected_text = correct_spelling(cleaned_text)
                    print(corrected_text)
                    await websocket.send_json({'text': corrected_text})
                    # sections = audio_collector.make_sections(user_id, note_id)
                    # sections_json = [section.model_dump_json() for section in sections]
                    # print(sections_json)
                    # await websocket.send_json({'sections': sections_json})
                else:
                    print("No transcript was created.")
            else:
                print("End-of-stream or no 'data' field. Breaking loop.")
                sections = audio_collector.make_sections(user_id, note_id)
                sections_json = [section.model_dump_json() for section in sections]
                print(sections_json)
                await websocket.send_json({'sections': sections_json})
                break
        # Optionally, clear the session data for the next session.
        audio_collector.session_audio = bytearray()
        audio_collector.full_transcript_text = ""
    except WebSocketDisconnect:
        print("WebSocket disconnected")

    print("All done.")
