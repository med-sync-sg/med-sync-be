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
import webrtcvad
from resemblyzer import VoiceEncoder
from spacy import displacy
import uvicorn
import argparse
from collections import deque

def create_app() -> FastAPI:
    app = FastAPI(title="Backend Connection", version="1.0.0")
    return app

app = create_app()
connected_clients: List[WebSocket] = []

parser = argparse.ArgumentParser(
    prog="MedSyncBE",
    description="Runs the backend server locally for MedSync"
)

load_umls = True
if load_umls:
    data_store = DataStore()  # First import triggers load

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = whisper.load_model("medium")
encoder = VoiceEncoder()
vad = webrtcvad.Vad(3)

RATE = 16000
CHUNK = 1024
audio_buffer = deque(maxlen=RATE // CHUNK * 3)

speaker_db = {}
speaker_id = 0

def detect_speaker(embedding):
    global speaker_id
    for spk, ref_embedding in speaker_db.items():
        similarity = np.inner(embedding, ref_embedding)
        if similarity > 0.75:
            return spk
    speaker_id += 1
    speaker_db[f"Speaker {speaker_id}"] = embedding
    return f"Speaker {speaker_id}"

def process_audio(audio_queue: queue.Queue, websocket: WebSocket):
    buffer = bytes()
    BUFFER_SIZE = 64000

    while True:
        data = audio_queue.get()
        if data is None:
            break

        buffer += data

        if len(buffer) >= BUFFER_SIZE:
            audio_data = np.frombuffer(buffer, dtype=np.int16).astype(np.float32) / 32768

            try:
                result = model.transcribe(audio_data, language="en")
                text = result["text"]

                if text.strip():
                    print("TRANSCRIBED TEXT:", text)

                    embedding = encoder.embed_utterance(audio_data)
                    speaker = detect_speaker(embedding)
                    print(f"[Diarization] Detected: {speaker}")

                    import asyncio
                    asyncio.run(websocket.send_text(f"{speaker}: {text}"))  # Fixed async sending

            except Exception as e:
                print("Error during processing:", e)

            buffer = bytes()  # Clear buffer

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    print("Client connected")

    audio_queue = queue.Queue()
    processing_thread = threading.Thread(
        target=process_audio,
        daemon=True,
        args=(audio_queue, websocket),
    )
    processing_thread.start()

    try:
        while True:
            data = await websocket.receive_bytes()
            audio = np.frombuffer(data, dtype=np.int16)

            if vad.is_speech(data, RATE):
                audio_queue.put(data)

    except WebSocketDisconnect:
        print("Client disconnected")
        audio_queue.put(None)
        connected_clients.remove(websocket)
    except Exception as e:
        print(f"WebSocket connection error: {e}")

@app.get("/")
async def read_root():
    return {"message": "Hello, FastAPI from app.py!"}

@app.post("/audio-test")
async def process_transcript_audio(data):
    return {"status": "Audio processing not implemented yet."}

@app.post("/text-test")
async def process_transcript_text(data: str):
    result_doc = process_text(data)
    return result_doc