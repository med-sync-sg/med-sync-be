import queue
import threading
import numpy as np
from pyannote.audio import Pipeline, Inference
from pyannote.audio.utils.signal import InMemoryAudioFile
from fastapi import WebSocket, WebSocketDisconnect
import whisper

class AudioCollector:
    """
    A singleton-like class that:
    1. Accumulates raw PCM bytes in self.audio_buffer
    2. Allows near real-time transcription with Whisper
    3. After the session, we can do diarization on the entire buffer
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            print("Initializing AudioCollector (loading Whisper).")
            cls._instance = super(AudioCollector, cls).__new__(cls)
            cls._instance.audio_buffer = bytes()
            cls._instance.model = whisper.load_model("medium")  # or "base" for faster
            cls._instance.full_transcript = []
        return cls._instance

    def add_chunk(self, chunk: bytes):
        """Add new PCM data to the buffer."""
        self.audio_buffer += chunk

    def reset_buffer(self):
        """Reset the entire audio buffer."""
        self.audio_buffer = bytes()

    def get_wave_data(self) -> np.ndarray:
        """
        Convert self.audio_buffer (16-bit PCM) into float32 samples [-1..1].
        Return shape: (num_samples,)
        """
        samples = np.frombuffer(self.audio_buffer, dtype=np.int16).astype(np.float32)
        samples /= 32768.0
        return samples

    def transcribe_audio_buffer(self, min_bytes=32000):
        """
        Optional: Near real-time transcription if buffer > min_bytes (~1 second).
        Clears the buffer after transcribing to start fresh.
        """
        if len(self.audio_buffer) > min_bytes:
            print(f"Transcribing buffer of size {len(self.audio_buffer)} bytes ...")
            audio_data = self.get_wave_data()
            try:
                result = self.model.transcribe(audio_data, language="en")
                text = result["text"]
                if text.strip():
                    print("TRANSCRIBED TEXT:", text)
                    self.full_transcript.append(text)
            except Exception as e:
                print("Error during transcription:", e)
            self.reset_buffer()
        else:
            # Not enough audio yet to do a decent transcription
            print(f"Buffer too small ({len(self.audio_buffer)} bytes). Not transcribing.")
