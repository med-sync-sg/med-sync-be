import numpy as np
import pyaudio
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from os import environ
import os
from pyctcdecode import build_ctcdecoder
import torch
from app.utils.nlp.spacy_utils import process_text
from app.utils.nlp.keyword_extractor import find_medical_modifiers
from app.models.models import post_section
from app.schemas.section import SectionCreate, TextCategoryEnum
from app.db.iris_session import IrisDataStore
import pandas as pd
from typing import List, Dict, Any
from spacy.tokens.doc import Doc
from app.utils.text_utils import clean_transcription, correct_spelling

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
    buffer_keyword_dicts: List[dict] = []
    final_keyword_dicts: List[dict] = []
    current_node_id: int = -1
    buffer_sections: List[SectionCreate] = []
    iris_data_store: IrisDataStore = IrisDataStore()
    full_transcript_text: str = ""
    full_transcript_segments: List[str] = []
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
            # Build a CTC decoder with a language model (kenlm)
            cls._instance.decoder = build_ctcdecoder(
                vocab, kenlm_model_path=os.path.join(".", "training", "umls_corpus.binary"),
                alpha=0.3,
                beta=1.0
            )

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

    def make_sections(self, user_id: int, note_id: int) -> List[SectionCreate]:
        sections = []
        contents = self.fill_content_dictionary()
        print("Contents:", contents)
        for index in range(len(contents)):
            term = self.final_keyword_dicts[index]["term"]
            category = self.iris_data_store.classify_text_category(term) # embed term
            section = SectionCreate(user_id=user_id, note_id=note_id, title=self.final_keyword_dicts[index]["label"], content=contents[index], section_type=category, section_description=TextCategoryEnum[category].value)
            sections.append(section)
        for section in sections:
            print(section)
        return sections

    def fill_content_dictionary(self) -> List[Dict[str, Any]]:
        result = []
        for result_keyword_dict in self.final_keyword_dicts:
            category = self.iris_data_store.classify_text_category(result_keyword_dict["term"])
            template = self.iris_data_store.find_content_dictionary(result_keyword_dict, category)
            result.append(self.iris_data_store.recursive_fill_content_dictionary(result_keyword_dict, template))
            
        return result

    def transcribe_audio_segment(self, user_id: int, note_id: int) -> bool:
        """
        Preprocesses the audio and runs inference.
        Returns False if transcription did not occur as no silence was detected yet and True if the function did transcribe.
        """
        sample_rate = 16000
        sample_width = 2  # bytes per sample
        frame_duration_ms = 20
        frame_size = int(sample_rate * (frame_duration_ms / 1000.0) * sample_width)  # ~640 bytes
        
        window_size = sample_rate * sample_width  # 1 second of audio = 32000 bytes
        if len(self.audio_buffer) < window_size:
            return False

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
                logits = logits.squeeze(0)
                # After obtaining logits from the model:
                logits_np = logits.cpu().numpy()
                # Use beam search decoder
                transcription = self.decoder.decode(logits_np)
                print("Beam search transcription:", transcription)
                
                transcription = transcription.lower()
                cleaned_text = clean_transcription(transcription)
                corrected_text = correct_spelling(cleaned_text)
                print("Corrected Text: ", corrected_text)
                
                if len(cleaned_text) > 0:
                    self.full_transcript_text = self.full_transcript_text + ". " + transcription
                    self.full_transcript_segments.append(transcription)
                    transcription_doc = process_text(self.full_transcript_text)
                    print(transcription_doc.ents)
                    # keyword_dicts = extract_prototype_features(doc=transcription_doc)
                    # print("Keyword Dicts: ",  keyword_dicts)
                    # self.buffer_keyword_dicts.extend(keyword_dicts)
                    
                for keyword_dict in self.buffer_keyword_dicts:
                    found = False
                    print(f"Keyword Dict: {keyword_dict}")
                    # Search for an existing entry with the same term.
                    for i, existing_dict in enumerate(self.final_keyword_dicts):
                        if keyword_dict["term"] == existing_dict["term"]:
                            self.final_keyword_dicts[i] = self.iris_data_store.merge_flat_keywords_into_template(existing_dict, keyword_dict)
                            print(f"Merged Dicts: {self.final_keyword_dicts}")
                            found = True
                            break
                    if not found:
                        self.final_keyword_dicts.append(keyword_dict)
                self.buffer_keyword_dicts = [] # Clear buffer for next iterations
                
                self.reset_buffer()
            except Exception as e:
                print("Error during transcription:", e)
            return True
        else:
            for keyword_dict in self.buffer_keyword_dicts:
                found = False
                print(f"Keyword Dict: {keyword_dict}")
                # Search for an existing entry with the same term.
                for i, existing_dict in enumerate(self.final_keyword_dicts):
                    if keyword_dict["term"] == existing_dict["term"]:
                        self.final_keyword_dicts[i] = self.iris_data_store.merge_flat_keywords_into_template(existing_dict, keyword_dict)
                        print(f"Merged Dicts: {self.final_keyword_dicts}")
                        found = True
                        break
                if not found:
                    self.final_keyword_dicts.append(keyword_dict)
            self.buffer_keyword_dicts = [] # Clear buffer for next iterations
            return False