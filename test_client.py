import asyncio
import os
import json
import wave
import base64
import logging
import argparse
import websockets

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("audio_stream_tester")

async def stream_audio(uri, audio_file_path, chunk_size=1024):
    """
    Stream an audio file over a WebSocket connection, simulating real-time audio.
    
    Args:
        uri (str): WebSocket server URI.
        audio_file_path (str): Path to the audio file.
        chunk_size (int): Number of frames to read per iteration.
    """
    try:
        async with websockets.connect(uri) as websocket:
            logger.info("WebSocket connected to %s", uri)
            with wave.open(audio_file_path, 'rb') as wf:
                framerate = wf.getframerate()
                logger.info("Opened audio file %s with frame rate: %d", audio_file_path, framerate)
                while True:
                    data = wf.readframes(chunk_size * 10)
                    if not data:
                        logger.info("End of audio file reached.")
                        break
                    # Encode the binary data as base64 to embed in JSON.
                    payload = {
                        "data": base64.b64encode(data).decode("utf-8")
                    }
                    message = json.dumps(payload)
                    
                    # Send the JSON payload over the WebSocket.
                    await websocket.send(message)
                    logger.info("Sent chunk of size %d bytes", len(data))
                    
                    # Simulate real-time delay: chunk_duration = frames / framerate
                    delay = chunk_size / framerate
                    
    except websockets.exceptions.ConnectionClosed as e:
        logger.error("WebSocket connection closed: %s", e)
    except Exception as e:
        logger.exception("Unexpected error during audio streaming: %s", e)

def main():
    parser = argparse.ArgumentParser(description="Test Audio Streaming Pipeline")
    parser.add_argument('--uri', type=str, default="ws://127.0.0.1:8001/ws",
                        help="WebSocket URI")
    parser.add_argument('--file', type=str, default=os.path.join("test_audios", "day1_consultation01_doctor.wav"),
                        help="Path to the audio file")
    parser.add_argument('--chunk', type=int, default=1024,
                        help="Chunk size in frames")
    args = parser.parse_args()
    
    asyncio.run(stream_audio(args.uri, args.file, args.chunk))

if __name__ == "__main__":
    main()