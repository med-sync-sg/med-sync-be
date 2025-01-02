
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import List

def create_app() -> FastAPI:
    app = FastAPI(title="Backend Connection", version="1.0.0")
    return app

connected_clients: List[WebSocket] = []

app = create_app()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    print("Client connected")
    await websocket.send_json({"msg": "MedSync Local Server Connected"})

    try:
        while True:
            data = await websocket.receive_json()
            if "audio" in data:
                print(f"Received audio data: {data['audio']}")
                # Simulate transcription processing
                transcription_result = f"Processed: {data['audio']}"
                await websocket.send_json({"transcription": transcription_result})
    except WebSocketDisconnect:
        print("Client disconnected")
        connected_clients.remove(websocket)
    except Exception as e:
        print(f"WebSocket connection error: {e}")
        


# Add routes or other configurations here
@app.get("/")
async def read_root():
    return {"message": "Hello, FastAPI from app.py!"}

@app.post("/transcribe")
async def transcribe_audio(consultation_text: str):
    # Do transcription
    transcription_text = "Completed!"
    return {"message": {transcription_text} }