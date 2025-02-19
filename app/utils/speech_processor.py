import queue
import threading
import numpy as np
from pyannote.audio import Pipeline, Inference
from fastapi import WebSocket, WebSocketDisconnect
import whisper
import pyaudio
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import soundfile as sf
from os import environ

import torch
import soundfile as sf
import librosa

def play_raw_audio(audio_buffer: bytearray, sample_rate=16000, sample_width=2, channels=1):
    p = pyaudio.PyAudio()
    stream = p.open(
        format=p.get_format_from_width(sample_width),
        channels=channels,
        rate=sample_rate,
        output=True
    )
    stream.write(bytes(audio_buffer))
    stream.stop_stream()
    stream.close()
    p.terminate()

class AudioCollector:
    """
    A singleton-like class that:
    1. Accumulates raw PCM bytes in a real-time buffer (audio_buffer) for near real-time transcription.
    2. Accumulates all PCM bytes in a persistent session buffer (session_audio) for offline processing (e.g., diarization).
    3. Provides methods to convert buffers to wave data.
    """
    _instance = None

    def __new__(cls):
        MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
        if cls._instance is None:
            print("Initializing AudioCollector.")
            cls._instance = super(AudioCollector, cls).__new__(cls)
            cls._instance.audio_buffer = bytearray()      # used for near real-time transcription
            cls._instance.session_audio = bytearray()       # accumulates full session audio for later processing
            # cls._instance.model = whisper.load_model("medium")  # or "base" for faster processing
            cls._instance.processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
            cls._instance.model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
            cls._instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            cls._instance.model.to(cls._instance.device)

            
            cls._instance.full_transcript = []
        return cls._instance

    def add_chunk(self, chunk: bytes):
        """Add new PCM data to both the real-time and session buffers."""
        self.audio_buffer.extend(chunk)
        self.session_audio.extend(chunk)

    def reset_buffer(self):
        """Reset only the real-time audio buffer (keeping session audio intact)."""
        self.audio_buffer = bytearray()

    @staticmethod
    def detect_silence_adaptive(audio_frames, sample_rate, frame_duration_ms=20, offset=100, silence_ratio_threshold=0.7):
        """
        Adaptive silence detection using RMS energy:
          - Computes the RMS energy for each 20-ms frame.
          - Estimates the noise floor as the 10th percentile of the frame energies.
          - Sets an adaptive threshold = noise_floor + offset.
          - Returns True if more than 'silence_ratio_threshold' (default 70%)
            of frames have energies below this threshold.
        """
        energies = []
        for frame in audio_frames:
            samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
            energy = np.sqrt(np.mean(samples ** 2))
            energies.append(energy)
        if not energies:
            return False
        noise_floor = np.percentile(energies, 10)
        threshold = noise_floor + offset
        silent_count = sum(1 for energy in energies if energy < threshold)
        ratio = silent_count / len(energies)
        return ratio > silence_ratio_threshold

    def get_wave_data(self) -> np.ndarray:
        """
        Convert the real-time audio_buffer (16-bit PCM) into float32 samples normalized to [-1, 1].
        Returns a numpy array of shape (num_samples,).
        """
        buf = self.audio_buffer
        if len(buf) % 2 != 0:
            # Trim the last byte if the buffer length is odd
            buf = buf[:-1]
        samples = np.frombuffer(buf, dtype=np.int16).astype(np.float32)
        samples /= 32768.0
        return samples

    def get_session_wave_data(self) -> np.ndarray:
        """
        Convert the full session audio (16-bit PCM) into float32 samples normalized to [-1, 1].
        Returns a numpy array of shape (num_samples,).
        """
        buf = self.session_audio
        if len(buf) % 2 != 0:
            # Trim the last byte if the buffer length is odd
            buf = buf[:-1]
        samples = np.frombuffer(buf, dtype=np.int16).astype(np.float32)
        samples /= 32768.0
        return samples

    def transcribe_audio_segment(self):
        """
        Preprocesses the audio and runs inference using the XLS-R model.
        Returns the transcription as text.
        """
        sample_rate = 16000
        sample_width = 2  # bytes per sample
        frame_duration_ms = 20
        frame_size = int(sample_rate * (frame_duration_ms / 1000.0) * sample_width)  # ~640 bytes
        
        window_size = sample_rate * sample_width  # 1 second of audio = 32000 bytes
        if len(self.audio_buffer) < window_size:
            return

        audio_window = self.audio_buffer[-window_size:]
        frames = [
            audio_window[i:i+frame_size]
            for i in range(0, len(audio_window), frame_size)
            if len(audio_window[i:i+frame_size]) == frame_size
        ]
        transcription = ""
        if self.detect_silence_adaptive(frames, sample_rate, frame_duration_ms):
            print(f"Silence detected. Transcribing buffer of size {len(self.audio_buffer)} bytes ...")
            
            # Preprocess the raw audio bytes.
            audio_samples = self.get_wave_data()
            
            input_values = self.processor(audio_samples, sampling_rate=sample_rate, return_tensors="pt").input_values
            input_values = input_values.to(self.device)
            # play_raw_audio(self.audio_buffer)
            
            try:
                with torch.no_grad():
                    logits = self.model(input_values).logits
                    
                # Greedy decoding: get the predicted token IDs
                predicted_ids = torch.argmax(logits, dim=-1)

                # Decode the token IDs to text
                decoded = self.processor.batch_decode(predicted_ids)
                transcription = decoded[0]
                if transcription.strip():
                    print("TRANSCRIBED TEXT:", transcription)
                    self.full_transcript.append(transcription)
                self.reset_buffer()
            except Exception as e:
                print("Error during transcription:", e)
        else:
            pass
        
        # Decode token IDs to text using the model’s dictionary.
        # The target dictionary is available via task.target_dictionary.
        return transcription

    def transcribe_audio_buffer(self):
        """
        Transcribes the real-time audio buffer only when a pause is detected using adaptive thresholds.
        The silence detection analyzes the last 1 second of audio (assuming 16 kHz, 16-bit PCM)
        by splitting it into 20-ms frames.
        After transcription, the real-time buffer is cleared, while the full session audio remains.
        """
        sample_rate = 16000
        sample_width = 2  # bytes per sample
        frame_duration_ms = 20
        frame_size = int(sample_rate * (frame_duration_ms / 1000.0) * sample_width)  # ~640 bytes
        
        window_size = sample_rate * sample_width  # 1 second of audio = 32000 bytes
        if len(self.audio_buffer) < window_size:
            return

        audio_window = self.audio_buffer[-window_size:]
        frames = [
            audio_window[i:i+frame_size]
            for i in range(0, len(audio_window), frame_size)
            if len(audio_window[i:i+frame_size]) == frame_size
        ]

        if self.detect_silence_adaptive(frames, sample_rate, frame_duration_ms):
            print(f"Silence detected. Transcribing buffer of size {len(self.audio_buffer)} bytes ...")
            # play_raw_audio(self.audio_buffer)
            audio_data = self.get_wave_data()
            try:
                result = self.model.transcribe(audio_data, language="en")
                text = result["text"]
                if text.strip():
                    print("TRANSCRIBED TEXT:", text)
                    self.full_transcript.append(text)
                self.reset_buffer()
            except Exception as e:
                print("Error during transcription:", e)
        else:
            pass