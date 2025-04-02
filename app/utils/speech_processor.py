import numpy as np
import pyaudio
import torch
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from threading import Lock

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pyctcdecode import build_ctcdecoder

from app.utils.nlp.spacy_utils import process_text
from app.models.models import post_section
from app.schemas.section import SectionCreate, TextCategoryEnum
from app.utils.text_utils import clean_transcription, correct_spelling
from app.db.data_loader import classify_text_category, find_content_dictionary, merge_flat_keywords_into_template
# Configure logger
logger = logging.getLogger(__name__)

def play_raw_audio(audio_buffer: bytearray, sample_rate=16000, sample_width=2, channels=1):
    """Play raw audio buffer through system audio"""
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


class SpeechProcessor:
    """Base class for speech processing functionality"""
    
    def __init__(self, model_id: str = "facebook/wav2vec2-base-960h"):
        """Initialize the speech processor with the specified model"""
        self.model_id = model_id
        self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Build the CTC decoder
        self._initialize_decoder()
        
        logger.info(f"Speech processor initialized with model {model_id} on {self.device}")
    
    def _initialize_decoder(self):
        """Initialize the CTC decoder with language model"""
        # Get vocabulary from the processor
        vocab_dict = self.processor.tokenizer.get_vocab()
        
        # Create ordered vocabulary list
        vocab = [None] * len(vocab_dict)
        for token, idx in vocab_dict.items():
            vocab[idx] = token
        
        # Build decoder with language model
        kenlm_path = os.path.join(".", "training", "umls_corpus.binary")
        if os.path.exists(kenlm_path):
            self.decoder = build_ctcdecoder(
                vocab, 
                kenlm_model_path=kenlm_path,
                alpha=0.3,
                beta=1.0
            )
            logger.info("CTC decoder initialized with language model")
        else:
            self.decoder = build_ctcdecoder(vocab)
            logger.warning(f"Language model not found at {kenlm_path}, using basic CTC decoder")
    
    def transcribe(self, audio_samples: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe audio samples to text
        
        Args:
            audio_samples: Normalized audio samples (float32 [-1.0, 1.0])
            sample_rate: Sample rate of the audio
            
        Returns:
            Transcribed text
        """
        try:
            # Preprocess audio
            input_values = self.processor(
                audio_samples, 
                sampling_rate=sample_rate, 
                return_tensors="pt"
            ).input_values
            input_values = input_values.to(self.device)
            
            # Run model inference
            with torch.no_grad():
                logits = self.model(input_values).logits
            
            # Decode with beam search
            logits_np = logits.squeeze(0).cpu().numpy()
            transcription = self.decoder.decode(logits_np)
            
            return transcription.lower()
            
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            return ""


class AudioCollector:
    """
    Singleton class for collecting and transcribing audio in real-time.
    
    This class:
    1. Accumulates raw PCM bytes in a real-time buffer for near real-time transcription
    2. Accumulates all PCM bytes in a persistent session buffer for offline processing
    3. Manages transcription results and extracted medical keywords
    """
    _instance = None
    _lock = Lock()  # Thread safety for singleton pattern
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                logger.info("Initializing AudioCollector singleton")
                cls._instance = super(AudioCollector, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        """Initialize the AudioCollector (only runs once due to singleton pattern)"""
        if not getattr(self, "_initialized", False):
            # Audio buffers
            self.audio_buffer = bytearray()      # For near real-time transcription
            self.session_audio = bytearray()     # Accumulates full session audio
            
            # Transcription state
            self.full_transcript_text = ""
            self.full_transcript_segments = []
            
            # Keyword extraction state
            self.buffer_keyword_dicts = []
            self.final_keyword_dicts = []
            
            # Section generation state
            self.current_node_id = -1
            self.buffer_sections = []
            
            # Dependencies
            self.speech_processor = SpeechProcessor()
            
            self._initialized = True
            logger.info("AudioCollector initialization complete")
    
    def add_chunk(self, chunk: bytes):
        """Add new PCM data to both the real-time and session buffers"""
        self.audio_buffer.extend(chunk)
        self.session_audio.extend(chunk)
    
    def reset_buffer(self):
        """Reset only the real-time audio buffer (keeping session audio intact)"""
        self.audio_buffer = bytearray()
    
    def get_wave_data(self) -> np.ndarray:
        """
        Convert the real-time audio_buffer (16-bit PCM) into float32 samples normalized to [-1, 1]
        
        Returns:
            numpy array of shape (num_samples,)
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
        Convert the full session audio (16-bit PCM) into float32 samples normalized to [-1, 1]
        
        Returns:
            numpy array of shape (num_samples,)
        """
        buf = self.session_audio
        if len(buf) % 2 != 0:
            # Trim the last byte if the buffer length is odd
            buf = buf[:-1]
        samples = np.frombuffer(buf, dtype=np.int16).astype(np.float32)
        samples /= 32768.0
        return samples
    
    @staticmethod
    def detect_silence_adaptive(audio_frames, sample_rate, frame_duration_ms=20, 
                               offset=100, silence_ratio_threshold=0.7):
        """
        Adaptive silence detection using RMS energy:
          - Computes the RMS energy for each frame
          - Estimates the noise floor as the 10th percentile of frame energies
          - Sets an adaptive threshold = noise_floor + offset
          - Returns True if more than silence_ratio_threshold of frames have 
            energies below this threshold
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
    
    def make_sections(self, user_id: int, note_id: int) -> List[SectionCreate]:
        """
        Create structured sections from the extracted keywords
        
        Args:
            user_id: The ID of the user
            note_id: The ID of the note
            
        Returns:
            List of SectionCreate objects
        """
        sections = []
        contents = self.fill_content_dictionary()
        logger.debug(f"Generated content dictionaries: {contents}")
        
        # Create a section for each content dictionary
        for index in range(len(contents)):
            if index >= len(self.final_keyword_dicts):
                logger.warning(f"No keyword dict found for content {index}")
                continue
                
            term = self.final_keyword_dicts[index]["term"]
            category = classify_text_category(term)
            
            section = SectionCreate(
                user_id=user_id,
                note_id=note_id,
                title=self.final_keyword_dicts[index].get("label", "Section"),
                content=contents[index],
                section_type=category,
                section_description=TextCategoryEnum[category].value
            )
            sections.append(section)
            
        logger.info(f"Created {len(sections)} sections")
        return sections
    
    def fill_content_dictionary(self) -> List[Dict[str, Any]]:
        """
        Fill content dictionaries based on the extracted keywords
        
        Returns:
            List of content dictionaries
        """
        result = []
        for result_keyword_dict in self.final_keyword_dicts:
            category = classify_text_category(result_keyword_dict["term"])
            template = find_content_dictionary(result_keyword_dict, category)
            content = merge_flat_keywords_into_template(
                result_keyword_dict, template
            )
            result.append(content)
            
        return result
    
    def transcribe_audio_segment(self, user_id: int, note_id: int) -> bool:
        """
        Preprocesses the audio and runs inference if silence is detected
        
        Returns:
            True if transcription occurred, False otherwise
        """
        sample_rate = 16000
        sample_width = 2  # bytes per sample
        frame_duration_ms = 20
        frame_size = int(sample_rate * (frame_duration_ms / 1000.0) * sample_width)
        
        # Ensure we have enough audio for processing (1 second window)
        window_size = sample_rate * sample_width
        if len(self.audio_buffer) < window_size:
            return False
        
        # Create frames from the audio window
        audio_window = self.audio_buffer[-window_size:]
        frames = [
            audio_window[i:i+frame_size]
            for i in range(0, len(audio_window), frame_size)
            if len(audio_window[i:i+frame_size]) == frame_size
        ]
        
        # Process only if silence is detected, indicating end of utterance
        if self.detect_silence_adaptive(frames, sample_rate, frame_duration_ms):
            logger.info(f"Silence detected. Transcribing buffer of size {len(self.audio_buffer)} bytes")
            
            # Get audio samples and transcribe
            audio_samples = self.get_wave_data()
            transcription = self.speech_processor.transcribe(audio_samples, sample_rate)
            
            if transcription:
                # Clean and correct transcription
                cleaned_text = clean_transcription(transcription)
                corrected_text = correct_spelling(cleaned_text)
                logger.debug(f"Transcription: '{transcription}' -> '{corrected_text}'")
                
                # Update transcript
                if len(cleaned_text) > 0:
                    self.full_transcript_text = (self.full_transcript_text + ". " + transcription).strip()
                    self.full_transcript_segments.append(transcription)
                    
                    # Process text with NLP pipeline
                    transcription_doc = process_text(self.full_transcript_text)
                    logger.debug(f"Entities detected: {transcription_doc.ents}")
                    
                # Process extracted keywords
                self._process_keywords()
                
                # Reset buffer for next transcription
                self.reset_buffer()
                return True
            else:
                logger.warning("Transcription produced empty result")
        
        # Even if we didn't transcribe, still process any pending keywords
        self._process_keywords()
        return False
    
    def _process_keywords(self):
        """Process and merge extracted keywords"""
        for keyword_dict in self.buffer_keyword_dicts:
            found = False
            
            # Search for existing entry with the same term
            for i, existing_dict in enumerate(self.final_keyword_dicts):
                if keyword_dict["term"] == existing_dict["term"]:
                    # Merge with existing entry
                    self.final_keyword_dicts[i] = merge_flat_keywords_into_template(
                        existing_dict, keyword_dict
                    )
                    logger.debug(f"Merged keyword dict: {self.final_keyword_dicts[i]}")
                    found = True
                    break
                    
            # Add as new entry if not found
            if not found:
                self.final_keyword_dicts.append(keyword_dict)
                
        # Clear buffer for next iteration
        self.buffer_keyword_dicts = []