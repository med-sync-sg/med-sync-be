
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import List

def create_app() -> FastAPI:
    app = FastAPI(title="Backend Connection", version="1.0.0")

    connected_clients: List[WebSocket] = []

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        connected_clients.append(websocket)
        print("Client connected")
        await websocket.send_json({"msg": "Welcome to the server!"})

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

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add routes or other configurations here
    @app.get("/")
    async def read_root():
        return {"message": "Hello, FastAPI from app.py!"}

    return app

app = create_app()