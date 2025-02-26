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
import os
from pyctcdecode import build_ctcdecoder
import torch
from app.utils.nlp.spacy_init import process_text
from app.utils.nlp.extractor import extract_keywords_descriptors, classify_keyword


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
    full_transcript = []
    def __new__(cls):
        MODEL_ID = "facebook/wav2vec2-base-960h"
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
            # Build the decoder using the vocabulary from the processor
            # Remove any special tokens as needed (like [UNK], [PAD])
            vocab_dict = cls._instance.processor.tokenizer.get_vocab()
            # Create an ordered list of tokens (the order must match the model training)
            # Often you might sort by the tokenizer's indices or use a predefined list
            vocab = [None] * len(vocab_dict)
            for token, idx in vocab_dict.items():
                vocab[idx] = token
                
            # Build a CTC decoder without a language model (or add a kenlm model if you have one)
            cls._instance.decoder = build_ctcdecoder(
                vocab, kenlm_model_path=os.path.join(".", "training", "umls_corpus.binary"),
                alpha=0.3,
                beta=1.0
            )
            
            cls._instance.full_transcript_text = ""
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
                print(logits.shape)
                logits = logits.squeeze(0)
                print(logits.shape)
                # After obtaining logits from the model:
                logits_np = logits.cpu().numpy()
                # Use beam search decoder
                transcription = self.decoder.decode(logits_np)
                print("Beam search transcription:", transcription)
                
                if transcription.strip():
                    self.full_transcript_text += transcription
                    self.full_transcript.append(process_text(transcription))
                self.reset_buffer()
            except Exception as e:
                print("Error during transcription:", e)
        else:
            pass
        
        # Decode token IDs to text using the model’s dictionary.
        # The target dictionary is available via task.target_dictionary.
        return transcription
    
    def get_tagged_full_transcript(self):
        doc = process_text(''.join(self.full_transcript_text))
        keyword_list = extract_keywords_descriptors(doc)
        print(keyword_list)
        for keyword_dict in keyword_list:
            print(classify_keyword(keyword_dict))
        return doc