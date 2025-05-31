import numpy as np
import os
import torch
import logging
import time
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
import librosa

from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from sqlalchemy.orm import Session

from app.db.local_session import DatabaseManager
from app.models.models import SpeakerProfile
from app.utils.voice_adaptation_utils import AdaptationTransformer, preprocess_audio_for_speaker, get_base_model_stats
from app.utils.text_utils import sym_spell, clean_transcription

# Configure logger
logger = logging.getLogger(__name__)

# Data models for transcription results
@dataclass
class WordTiming:
    """Represents a single word with timing information"""
    word: str
    start_time: float
    end_time: float
    confidence: float = 1.0
    speaker_id: Optional[str] = None

@dataclass
class TranscriptionSegment:
    """Represents a segment of transcription"""
    text: str
    start_time: float
    end_time: float
    words: List[WordTiming] = field(default_factory=list)
    speaker_id: Optional[str] = None
    confidence: float = 1.0

@dataclass
class TranscriptionResult:
    """Complete transcription result with all metadata"""
    text: str
    words: List[WordTiming] = field(default_factory=list)
    segments: List[TranscriptionSegment] = field(default_factory=list)
    language: str = "en"
    duration: float = 0.0
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TranscriptionConfig:
    """Configuration for transcription"""
    backend: str = "whisper_onnx"  # or "whisper_onnx"
    model_path: str = f"./training/whisper_onnx"
    model_size: str = "small"
    language: str = "en"
    task: str = "transcribe"
    enable_speaker_adaptation: bool = True
    enable_medical_postprocessing: bool = True
    enable_word_timing: bool = True
    confidence_threshold: float = 0.7
    use_gpu: bool = True
    beam_size: int = 5
    temperature: float = 0.0
    compression_ratio_threshold: float = 2.4
    no_speech_threshold: float = 0.6

# Abstract base class for transcription backends
class TranscriptionBackend(ABC):
    """Abstract base class for transcription backends"""
    
    @abstractmethod
    def transcribe(self, audio_samples: np.ndarray, sample_rate: int = 16000) -> TranscriptionResult:
        """Transcribe audio and return structured result"""
        pass
    
    @abstractmethod
    def transcribe_with_timing(self, audio_samples: np.ndarray, sample_rate: int = 16000) -> TranscriptionResult:
        """Transcribe audio with word-level timing"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        pass

class WhisperTransformersBackend(TranscriptionBackend):
    """Whisper backend using HuggingFace Transformers"""
    
    def __init__(self, config: TranscriptionConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() and config.use_gpu else "cpu"
        
        # Load model and processor
        model_id = f"openai/whisper-{config.model_size}"
        logger.info(f"Loading Whisper model: {model_id} on {self.device}")
        
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id)
        self.model.to(self.device)
        
        # Set model to eval mode
        self.model.eval()
        
    def transcribe(self, audio_samples: np.ndarray, sample_rate: int = 16000) -> TranscriptionResult:
        """Basic transcription without timing"""
        try:
            if len(audio_samples) == 0:
                return TranscriptionResult(text="", duration=0.0)
            
            # Process audio
            inputs = self.processor(
                audio_samples, 
                sampling_rate=sample_rate, 
                return_tensors="pt"
            ).input_features.to(self.device)
            
            # Generate transcription
            with torch.no_grad():
                generated_ids = self.model.generate(
                    inputs,
                    language=self.config.language,
                    task=self.config.task,
                    num_beams=self.config.beam_size,
                    temperature=self.config.temperature
                )
            
            # Decode
            transcription = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            
            # Calculate duration
            duration = len(audio_samples) / sample_rate
            
            return TranscriptionResult(
                text=transcription.strip(),
                duration=duration,
                language=self.config.language
            )
            
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            return TranscriptionResult(text="", duration=0.0)
    
    def transcribe_with_timing(self, audio_samples: np.ndarray, sample_rate: int = 16000) -> TranscriptionResult:
        """Direct approach using Whisper's token timestamps"""
        try:
            if len(audio_samples) == 0:
                return TranscriptionResult(text="", duration=0.0)
            
            # Process audio
            inputs = self.processor(
                audio_samples, 
                sampling_rate=sample_rate, 
                return_tensors="pt"
            ).input_features.to(self.device)
            
            # Generate with specific timestamp configuration
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    return_dict_in_generate=True,
                    output_scores=True,
                    return_timestamps=True
                )
            
            # Decode and extract timestamps
            transcription = self.processor.decode(outputs.sequences[0], skip_special_tokens=False)
            
            # Parse timestamps from the transcription
            words_with_timing = self._parse_whisper_timestamps(transcription, outputs, audio_samples, sample_rate)
            
            return TranscriptionResult(
                text=transcription,
                words=words_with_timing,
                duration=len(audio_samples) / sample_rate
            )
            
        except Exception as e:
            logger.error(f"Error in direct timestamp approach: {str(e)}")
            return self.transcribe(audio_samples, sample_rate)

    def _parse_whisper_timestamps(self, transcription: str, outputs, audio_samples: np.ndarray, sample_rate: int) -> List[WordTiming]:
        """Parse Whisper's timestamp tokens to extract word timings"""
        # Whisper uses special tokens like <|0.00|> for timestamps
        import re
        
        words = []
        pattern = r'<\|(\d+\.\d+)\|>([^<]+)'
        matches = re.findall(pattern, transcription)
        
        for i, (timestamp, text) in enumerate(matches):
            start_time = float(timestamp)
            # Estimate end time based on next timestamp or audio duration
            if i < len(matches) - 1:
                end_time = float(matches[i + 1][0])
            else:
                end_time = len(audio_samples) / sample_rate
            
            # Split text into words
            word_list = text.strip().split()
            if word_list:
                # Distribute time evenly among words
                word_duration = (end_time - start_time) / len(word_list)
                for j, word in enumerate(word_list):
                    words.append(WordTiming(
                        word=word,
                        start_time=start_time + j * word_duration,
                        end_time=start_time + (j + 1) * word_duration,
                        confidence=1.0
                    ))
        
        return words
    
    def _process_pipeline_output(self, pipeline_output: Dict[str, Any], audio_samples: np.ndarray, sample_rate: int) -> TranscriptionResult:
        """Process pipeline output to extract word timing"""
        text = pipeline_output.get("text", "")
        chunks = pipeline_output.get("chunks", [])
        
        words = []
        segments = []
        
        if chunks:
            # Process chunks into words
            for chunk in chunks:
                word_text = chunk.get("text", "").strip()
                if not word_text:
                    continue
                    
                timestamp = chunk.get("timestamp")
                if timestamp and isinstance(timestamp, (list, tuple)) and len(timestamp) >= 2:
                    start_time = float(timestamp[0]) if timestamp[0] is not None else 0.0
                    end_time = float(timestamp[1]) if timestamp[1] is not None else start_time + 0.1
                else:
                    # No timing info, estimate based on position
                    start_time = 0.0
                    end_time = 0.1
                
                word = WordTiming(
                    word=word_text,
                    start_time=start_time,
                    end_time=end_time,
                    confidence=1.0  # Pipeline doesn't provide confidence
                )
                words.append(word)
            
            # Create a single segment for the whole transcription
            if words:
                segment = TranscriptionSegment(
                    text=text,
                    start_time=words[0].start_time,
                    end_time=words[-1].end_time,
                    words=words
                )
                segments.append(segment)
        
        return TranscriptionResult(
            text=text,
            words=words,
            segments=segments,
            duration=len(audio_samples) / sample_rate,
            language=self.config.language
        )
    
    def _process_whisper_output(self, outputs: Dict[str, Any], audio_samples: np.ndarray, sample_rate: int) -> TranscriptionResult:
        """Process Whisper output to extract word timing"""
        # This is a simplified version - in practice, you'd need to parse
        # Whisper's token timestamps and map them to words
        
        if isinstance(outputs, str):
            # Simple output without timing
            return TranscriptionResult(
                text=outputs.strip(),
                duration=len(audio_samples) / sample_rate
            )
        
        # Extract text and timing information
        text = outputs.get("text", "")
        segments = outputs.get("segments", [])
        
        words = []
        transcription_segments = []
        
        for segment in segments:
            segment_text = segment.get("text", "")
            start_time = segment.get("start", 0.0)
            end_time = segment.get("end", 0.0)
            
            # Extract words from segment
            segment_words = []
            if "words" in segment:
                for word_info in segment["words"]:
                    word = WordTiming(
                        word=word_info["word"],
                        start_time=word_info["start"],
                        end_time=word_info["end"],
                        confidence=word_info.get("probability", 1.0)
                    )
                    words.append(word)
                    segment_words.append(word)
            
            transcription_segment = TranscriptionSegment(
                text=segment_text,
                start_time=start_time,
                end_time=end_time,
                words=segment_words
            )
            transcription_segments.append(transcription_segment)
        
        return TranscriptionResult(
            text=text,
            words=words,
            segments=transcription_segments,
            duration=len(audio_samples) / sample_rate,
            language=self.config.language
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "backend": "whisper_transformers",
            "model_size": self.config.model_size,
            "device": self.device,
            "language": self.config.language
        }

class WhisperONNXBackend(TranscriptionBackend):
    """Whisper backend using ONNX Runtime for improved performance via optimum library"""
    
    def __init__(self, config: TranscriptionConfig):
        self.config = config
        
        # Import required libraries
        try:

            self.WhisperProcessor = WhisperProcessor
            self.ORTModelForSpeechSeq2Seq = ORTModelForSpeechSeq2Seq
        except ImportError as e:
            raise ImportError(
                "Please install required packages: "
                "pip install transformers optimum[onnxruntime] onnxruntime-gpu"
            ) from e
        
        # Load ONNX model
        self._load_onnx_model()
        
    def _load_onnx_model(self):
        """Load ONNX model and processor from exported directory"""
        # Determine model path
        if self.config.model_path:
            model_path = self.config.model_path
        else:
            # Default path structure from whisper_to_onnx.py
            model_path = f"./training/whisper_onnx"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"ONNX model not found at {model_path}. "
                f"Please run whisper_to_onnx.py first to export the model."
            )
        
        # Load processor
        processor_path = os.path.join(model_path, "processor")
        if os.path.exists(processor_path):
            self.processor = self.WhisperProcessor.from_pretrained(processor_path)
        else:
            # Fallback to loading from main directory
            self.processor = self.WhisperProcessor.from_pretrained(model_path)
        
        # Load ONNX model with optimum
        providers = (
            "CPUExecutionProvider" 
            if self.config.use_gpu 
            else "CPUExecutionProvider"
        )
        
        self.model = self.ORTModelForSpeechSeq2Seq.from_pretrained(
            model_path,
            provider=providers
        )
        
        print(f"Loaded ONNX Whisper model from: {model_path}")
        print(f"Using providers: {self.model.providers}")
        
    def transcribe(self, audio_samples: np.ndarray, sample_rate: int = 16000) -> TranscriptionResult:
        """Transcribe using ONNX model via optimum library"""
        # Ensure audio is in the right format
        if sample_rate != 16000:
            audio_samples = librosa.resample(audio_samples, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        
        # Preprocess audio
        inputs = self.processor(
            audio_samples,
            sampling_rate=sample_rate,
            return_tensors="pt"
        )
        
        # Set generation parameters
        generation_kwargs = {
            "max_length": 448,
            "num_beams": 1,
            "do_sample": False,
        }
        
        # Add language if specified
        if self.config.language:
            generation_kwargs["language"] = self.config.language
        
        # Run inference
        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs.input_features,
                **generation_kwargs
            )
        
        # Decode to text
        transcription = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )[0]
        
        return TranscriptionResult(
            text=transcription,
            duration=len(audio_samples) / sample_rate,
            language=self.config.language
        )
    
    def transcribe_with_timing(self, audio_samples: np.ndarray, sample_rate: int = 16000) -> TranscriptionResult:
        """ONNX transcription with word-level timestamps"""
        # Ensure audio is in the right format
        if sample_rate != 16000:
            audio_samples = librosa.resample(audio_samples, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        
        # Preprocess audio
        inputs = self.processor(
            audio_samples,
            sampling_rate=sample_rate,
            return_tensors="pt"
        )
        
        # Set generation parameters with timestamps
        generation_kwargs = {
            "max_length": 448,
            "num_beams": 1,
            "do_sample": False,
            "return_timestamps": True,
        }
        
        # Add language if specified
        if self.config.language:
            generation_kwargs["language"] = self.config.language
        
        # Run inference
        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs.input_features,
                **generation_kwargs
            )
        
        # Decode with timestamps
        transcription = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True,
            decode_with_timestamps=True
        )[0]
        
        # Parse segments if timestamps are available
        segments = self._parse_segments(transcription) if isinstance(transcription, dict) else None
        text = transcription.get('text', transcription) if isinstance(transcription, dict) else transcription
        
        return TranscriptionResult(
            text=text,
            duration=len(audio_samples) / sample_rate,
            language=self.config.language,
            segments=segments
        )
    
    def _parse_segments(self, transcription_result):
        """Parse segment information from transcription result"""
        # This would need to be implemented based on the actual format
        # returned by the processor when timestamps are enabled
        if isinstance(transcription_result, dict) and 'segments' in transcription_result:
            return transcription_result['segments']
        return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "backend": "whisper_onnx",
            "model_size": getattr(self.config, 'model_size', 'unknown'),
            "providers": self.model.providers if hasattr(self.model, 'providers') else [],
            "model_path": getattr(self.config, 'model_path', 'default'),
            "language": self.config.language,
            "use_gpu": self.config.use_gpu
        }
    
    def cleanup(self):
        """Clean up resources"""
        # ONNX Runtime handles cleanup automatically
        pass

class MedicalPostProcessor:
    """Post-processor for medical transcriptions"""
    
    def __init__(self, enable_spell_check: bool = True):
        self.enable_spell_check = enable_spell_check
        
    def process(self, result: TranscriptionResult) -> TranscriptionResult:
        """Process transcription result for medical accuracy"""
        # Process the main text
        processed_text = self._process_text(result.text)
        
        # Process each word
        processed_words = []
        for word in result.words:
            processed_word = self._process_word(word)
            processed_words.append(processed_word)
        
        # Update result
        result.text = processed_text
        result.words = processed_words
        
        return result
    
    def _process_text(self, text: str) -> str:
        """Process complete text"""
        if not text:
            return text
            
        # Clean transcription
        text = clean_transcription(text)
        
        # Medical spell checking could go here
        # For now, keep it simple
        
        return text
    
    def _process_word(self, word: WordTiming) -> WordTiming:
        """Process individual word"""
        # Could apply medical dictionary validation here
        # For now, just clean the word
        word.word = word.word.strip()
        return word

class SpeechProcessor:
    """
    Refactored speech processor with pluggable backends and medical focus
    """
    
    def __init__(self, config: Optional[TranscriptionConfig] = None):
        """Initialize with configuration"""
        self.config = config or TranscriptionConfig()
        
        # Initialize backend
        self._init_backend()
        
        # Initialize medical post-processor
        self.medical_processor = MedicalPostProcessor(
            enable_spell_check=self.config.enable_medical_postprocessing
        )
        
        # Database and caching
        self.db_manager = DatabaseManager()
        self.adaptation_cache = {}
        self.profile_cache = {}
        self.profile_cache_ttl = 300
        self.profile_cache_timestamps = {}
        
        logger.info(f"SpeechProcessor initialized with backend: {self.config.backend}")
    
    def _init_backend(self):
        """Initialize the transcription backend"""
        if self.config.backend == "whisper_onnx":
            try:
                self.backend = WhisperONNXBackend(self.config)
            except ImportError:
                logger.warning("ONNX not available, falling back to transformers")
                self.config.backend = "whisper_transformers"
                self.backend = WhisperTransformersBackend(self.config)
        else:
            self.backend = WhisperTransformersBackend(self.config)
    
    def transcribe(self, audio_samples: np.ndarray, sample_rate: int = 16000, 
                   with_timing: bool = None) -> Union[str, TranscriptionResult]:
        """
        Transcribe audio - returns string for backward compatibility
        
        Args:
            audio_samples: Audio samples
            sample_rate: Sample rate
            with_timing: Whether to include word timing
            
        Returns:
            String transcription or TranscriptionResult if with_timing=True
        """
        # Determine if we need timing
        if with_timing is None:
            with_timing = self.config.enable_word_timing
        
        # Transcribe
        if with_timing:
            result = self.backend.transcribe_with_timing(audio_samples, sample_rate)
        else:
            result = self.backend.transcribe(audio_samples, sample_rate)
        
        # Post-process if enabled
        if self.config.enable_medical_postprocessing:
            result = self.medical_processor.process(result)
        
        # Return string for backward compatibility when timing not requested
        if not with_timing:
            return result.text
        
        return result
    
    def transcribe_with_adaptation(self, audio_samples: np.ndarray, user_id: int,
                                   db: Optional[Session] = None, sample_rate: int = 16000,
                                   with_timing: bool = None) -> Union[str, TranscriptionResult]:
        """
        Transcribe with speaker adaptation
        
        Returns string for backward compatibility, TranscriptionResult if with_timing=True
        """
        session_created = False
        if db is None:
            db = next(self.db_manager.get_session())
            session_created = True
        
        try:
            # Get speaker profile
            profile = self._get_speaker_profile(user_id, db)
            
            if profile is None:
                logger.warning(f"No speaker profile found for user {user_id}")
                return self.transcribe(audio_samples, sample_rate, with_timing)
            
            # Apply speaker-specific preprocessing
            processed_audio = preprocess_audio_for_speaker(audio_samples, profile)
            
            # Transcribe
            result = self.transcribe(processed_audio, sample_rate, with_timing)
            
            # Add adaptation metadata
            if isinstance(result, TranscriptionResult):
                result.metadata["speaker_adapted"] = True
                result.metadata["user_id"] = user_id
            
            logger.info(f"Applied speaker adaptation for user {user_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error in adaptive transcription: {str(e)}")
            return self.transcribe(audio_samples, sample_rate, with_timing)
            
        finally:
            if session_created and db is not None:
                db.close()
    
    def _get_speaker_profile(self, user_id: int, db: Session) -> Optional[Dict[str, Any]]:
        """Get speaker profile from database or cache"""
        current_time = time.time()
        
        # Check cache first
        if user_id in self.profile_cache:
            cache_time = self.profile_cache_timestamps.get(user_id, 0)
            if current_time - cache_time < self.profile_cache_ttl:
                return self.profile_cache[user_id]
        
        try:
            # Get from database
            profile = db.query(SpeakerProfile)\
                .filter(SpeakerProfile.user_id == user_id, SpeakerProfile.is_active == True)\
                .first()
            
            if not profile:
                return None
            
            profile_dict = profile.get_profile_dict()
            
            # Cache it
            self.profile_cache[user_id] = profile_dict
            self.profile_cache_timestamps[user_id] = current_time
            
            return profile_dict
            
        except Exception as e:
            logger.error(f"Error getting speaker profile: {str(e)}")
            return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        info = self.backend.get_model_info()
        info.update({
            "medical_postprocessing": self.config.enable_medical_postprocessing,
            "speaker_adaptation_enabled": self.config.enable_speaker_adaptation,
            "profile_cache_size": len(self.profile_cache)
        })
        return info
    
    def clear_cache(self, user_id: Optional[int] = None):
        """Clear cache for a specific user or all users"""
        if user_id is not None:
            self.profile_cache.pop(user_id, None)
            self.profile_cache_timestamps.pop(user_id, None)
            self.adaptation_cache.pop(user_id, None)
        else:
            self.profile_cache.clear()
            self.profile_cache_timestamps.clear()
            self.adaptation_cache.clear()