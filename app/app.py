
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import List
from app.utils.import_umls import connect_to_docker_psql, load_concepts, load_relationships, load_semantic_types, combine_data
from app.utils.nlp import process_text
import queue
import threading
import torch
import whisper
import numpy as np
from main import load_umls

def create_app() -> FastAPI:
    app = FastAPI(title="Backend Connection", version="1.0.0")
    
    return app

connected_clients: List[WebSocket] = []

app = create_app()
engine = connect_to_docker_psql()

if load_umls == True:
    concepts_df = load_concepts()
    relationships_df = load_relationships()
    semantic_df = load_semantic_types()
    combined_df = combine_data(concepts_df, semantic_df, relationships_df)

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
    full_transcript = ""
    audio_queue = queue.Queue()
    def preprocess_audio(buffer):
        audio_data = np.frombuffer(buffer, dtype=np.int16).astype(np.float32)
        audio_data /= 32768.0  # Now the range is [-1.0, +1.0]
        return audio_data

    async def transcribe_stream(audio_queue: queue.Queue, websocket: WebSocket):
        buffer = bytes()
        while True:
            data = audio_queue.get()
            if data is None:  # End signal
                break

            buffer += data

            # Process audio every ~1 second
            if len(buffer) > 32000:  # ~1 second of 16kHz 16-bit mono audio
                audio_data = preprocess_audio(buffer)
                print(f"Processing buffer with size: {len(buffer)} bytes")
                
                try:
                    result = model.transcribe(audio_data, language="en")
                    text = result["text"]
                    if text.strip():  # Only print non-empty transcriptions
                        print("TRANSCRIBED TEXT:", text)
                        full_transcript = full_transcript + text.strip()
                        doc = process_text(text.strip())
                        await websocket.send_json({"transcript": full_transcript, "doc": doc})

                except Exception as e:
                    print("Error during transcription:", e)
                
                # Reset buffer
                buffer = bytes()

    await websocket.accept()
    connected_clients.append(websocket)
    print("Client connected")
    
    transcription_thread = threading.Thread(target=transcribe_stream, daemon=True, args=(audio_queue, websocket))
    transcription_thread.start()
    
    try:
        while True:
            data = await websocket.receive_bytes()
            if data:
                try:
                    audio_chunk = data                    
                    if audio_chunk == None:
                        audio_queue.put(None)
                    audio_queue.put(audio_chunk)
                except Exception as e:
                    print(f"Error processing audio chunk: {e}")
            else:
                doc = process_text(full_transcript)
                await websocket.send_json(doc.to_json())
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