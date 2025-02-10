
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

def create_app() -> FastAPI:
    app = FastAPI(title="Backend Connection", version="1.0.0")
    # app.include_router(v1_endpoints.router, prefix="/v1", tags=["v1"])

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
                chunk = data["data"]  # raw PCM bytes
                audio_collector.add_chunk(chunk)

                # If you want near real-time partial transcription, you can call:
                # audio_collector.transcribe_audio_buffer(min_bytes=32000)
                #
                # This will transcribe every time the buffer exceeds ~1 second
                #
                # Or you can comment it out if you only want offline transcription.
                #
                audio_collector.transcribe_audio_buffer()
            else:
                print("End-of-stream or no 'data' field. Breaking loop.")
                break

    except WebSocketDisconnect:
        print("WebSocket disconnected")

    wave_data = audio_collector.get_wave_data()
    print(f"Total wave_data shape: {wave_data.shape}")
    
    # Convert wave_data to an in-memory WAV for Pyannote
    buffer_wav = io.BytesIO()
    sf.write(buffer_wav, wave_data, 16000, format="WAV")
    buffer_wav.seek(0)  # rewind

    # Use the diarization pipeline (some require HF token if private)
    diar_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
    diarization_result = diar_pipeline(
        buffer_wav, 
        min_speakers=2, 
        max_speakers=2
    )
    
    print("Diarization result:")
    for seg, spk in diarization_result.itertracks(yield_label=True):
        print(f"Segment {seg.start:.2f}-{seg.end:.2f}: speaker={spk}")

    # Reset the collector for next session or keep as needed
    audio_collector.reset_buffer()
    audio_collector.full_transcript.clear()

    print("All done.")

# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     shared_transcript = []
#     audio_queue = queue.Queue()
#     def preprocess_audio(buffer):
#         audio_data = np.frombuffer(buffer, dtype=np.int16).astype(np.float32)
#         audio_data /= 32768.0  # Now the range is [-1.0, +1.0]
#         return audio_data

#     def transcribe_stream(audio_queue: queue.Queue, websocket: WebSocket, shared_transcript: list):
#         buffer = bytes()
#         while True:
#             data = audio_queue.get()
#             if data is None:  # End signal
#                 break

#             buffer += bytes(data)

#             # Process audio every ~1 second
#             if len(buffer) > 32000:  # ~1 second of 16kHz 16-bit mono audio
#                 audio_data = preprocess_audio(buffer)
#                 print(f"Processing buffer with size: {len(buffer)} bytes")
                
#                 try:
#                     result = model.transcribe(audio_data, language="en")
#                     text = result["text"]
#                     if text.strip():
#                         print("TRANSCRIBED TEXT:", text)
#                         # Append to our shared transcript list
#                         shared_transcript.append(text)
#                 except Exception as e:
#                     print("Error during transcription:", e)
                
#                 # Reset buffer
#                 buffer = bytes()

#     await websocket.accept()
#     connected_clients.append(websocket)
#     print("Client connected")
    
#     transcription_thread = threading.Thread(
#         target=transcribe_stream,
#         daemon=True,
#         args=(audio_queue, websocket, shared_transcript)
#     )
#     transcription_thread.start()
    
#     try:
#         while True:
#             data = await websocket.receive_json()
#             if data:
#                 try:
#                     if data["data"]:
#                         audio_chunk = data["data"]
#                         if audio_chunk == None:
#                             audio_queue.put(None)
#                         audio_queue.put(audio_chunk)
#                     elif data["data"] == None: # End of audio data stream
#                         full_transcript = ""
#                         for text in shared_transcript:
#                             full_transcript = full_transcript + text
#                         doc = process_text(full_transcript)
#                         categorized = categorize_doc(doc)
#                         print(categorized)
#                         print(displacy.render(doc, style="ent"))
#                 except Exception as e:
#                     print(f"Error processing audio chunk: {e}")
#             else:
#                 print("Non-audio data received")

#     except WebSocketDisconnect:
#         print("Client disconnected")
#         audio_queue.put(None)
#         connected_clients.remove(websocket)
#     except Exception as e:
#         print(f"WebSocket connection error: {e}")
        


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