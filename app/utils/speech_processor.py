import numpy as np
import os
import torch
import logging
import pickle
from typing import Optional, Dict, Any, Tuple
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pyctcdecode import build_ctcdecoder
from sqlalchemy.orm import Session

from app.db.local_session import DatabaseManager
from app.models.models import SpeakerProfile
from app.utils.voice_adaptation_utils import AdaptationTransformer, preprocess_audio_for_speaker, get_base_model_stats

# Configure logger
logger = logging.getLogger(__name__)

class SpeechProcessor:
    """
    Unified speech processor with standard and adaptive transcription capabilities.
    
    This class handles:
    1. Model initialization and configuration
    2. Audio preprocessing
    3. Standard transcription
    4. Speaker-adapted transcription using database profiles
    5. Profile caching for performance
    """
    
    def __init__(self, model_id: str = "facebook/wav2vec2-base-960h", 
                 use_gpu: bool = True,
                 language_model_path: Optional[str] = None):
        """
        Initialize the speech processor with the specified model
        
        Args:
            model_id: HuggingFace model ID for Wav2Vec2
            use_gpu: Whether to use GPU if available
            language_model_path: Optional path to a KenLM language model binary
        """
        try:
            logger.info(f"Initializing SpeechProcessor with model {model_id}")
            
            # Initialize tokenizer/processor
            self.processor = Wav2Vec2Processor.from_pretrained(model_id)
            
            # Initialize model
            self.model = Wav2Vec2ForCTC.from_pretrained(model_id)
            
            # Set device (CUDA if available and requested, otherwise CPU)
            self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
            self.model.to(self.device)
            logger.info(f"Model loaded on {self.device}")
            
            # Initialize the decoder
            self._initialize_decoder(language_model_path)
            
            # Database connection and adaptation cache
            self.db_manager = DatabaseManager()
            self.adaptation_cache = {}  # Cache transformers by user_id
            self.profile_cache = {}     # Cache profiles by user_id
            self.profile_cache_ttl = 300  # Cache TTL in seconds
            self.profile_cache_timestamps = {}  # When profiles were cached
            
        except Exception as e:
            logger.error(f"Error initializing SpeechProcessor: {str(e)}")
            raise RuntimeError(f"Failed to initialize speech processor: {str(e)}")
    
    def _initialize_decoder(self, language_model_path: Optional[str] = None):
        """
        Initialize the CTC decoder with optional language model
        
        Args:
            language_model_path: Optional path to KenLM language model binary
        """
        try:
            # Get vocabulary from the processor
            vocab_dict = self.processor.tokenizer.get_vocab()
            
            # Create ordered vocabulary list
            vocab = [None] * len(vocab_dict)
            for token, idx in vocab_dict.items():
                vocab[idx] = token
            
            # Default path for language model if not specified
            if language_model_path is None:
                language_model_path = os.path.join(".", "training", "umls_corpus.binary")
            
            # Build decoder with language model if available
            if language_model_path and os.path.exists(language_model_path):
                logger.info(f"Initializing CTC decoder with language model: {language_model_path}")
                self.decoder = build_ctcdecoder(
                    vocab, 
                    kenlm_model_path=language_model_path,
                    alpha=0.3,  # Language model weight
                    beta=1.0    # Word insertion bonus
                )
            else:
                logger.warning("Initializing CTC decoder without language model")
                self.decoder = build_ctcdecoder(vocab)
                
        except Exception as e:
            logger.error(f"Error initializing decoder: {str(e)}")
            # Fallback to simple decoder without language model
            self.decoder = build_ctcdecoder(vocab if 'vocab' in locals() else [])
    
    def preprocess_audio(self, audio_samples: np.ndarray, sample_rate: int = 16000) -> torch.Tensor:
        """
        Preprocess audio samples for the model
        
        Args:
            audio_samples: Normalized audio samples (float32 [-1.0, 1.0])
            sample_rate: Sample rate of the audio
            
        Returns:
            Tensor of processed input values
        """
        # Check if audio is empty
        if len(audio_samples) == 0:
            logger.warning("Empty audio provided for preprocessing")
            return torch.zeros((1, 0), device=self.device)
        
        # Process audio with the Wav2Vec2 processor
        inputs = self.processor(
            audio_samples, 
            sampling_rate=sample_rate, 
            return_tensors="pt"
        ).input_values
        
        # Move to the correct device
        return inputs.to(self.device)
    
    def transcribe(self, audio_samples: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Standard transcription without speaker adaptation
        
        Args:
            audio_samples: Normalized audio samples (float32 [-1.0, 1.0])
            sample_rate: Sample rate of the audio
            
        Returns:
            Transcribed text as string
        """
        try:
            # Check if audio is empty
            if len(audio_samples) == 0:
                logger.warning("Empty audio provided for transcription")
                return ""
            
            # Preprocess audio
            input_values = self.preprocess_audio(audio_samples, sample_rate)
            
            # Run model inference
            with torch.no_grad():
                logits = self.model(input_values).logits
            
            # Move logits to CPU and convert to numpy for decoder
            logits_np = logits.squeeze(0).cpu().numpy()
            
            # Decode with beam search
            transcription = self.decoder.decode(logits_np)
            
            return transcription.lower()
            
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            return ""
    
    def transcribe_with_adaptation(self, audio_samples: np.ndarray, user_id: int, 
                                  db: Optional[Session] = None, sample_rate: int = 16000) -> str:
        """
        Transcribe audio using speaker adaptation from database
        
        Args:
            audio_samples: Normalized audio samples (float32 [-1.0, 1.0])
            user_id: User ID for speaker adaptation profile
            db: Optional database session (will create one if not provided)
            sample_rate: Sample rate of the audio
            
        Returns:
            Transcribed text with speaker adaptation applied
        """
        # Create database session if not provided
        session_created = False
        if db is None:
            db = next(self.db_manager.get_session())
            session_created = True
        
        try:
            # Get speaker profile
            profile = self._get_speaker_profile(user_id, db)
            
            if profile is None:
                logger.warning(f"No speaker profile found for user {user_id}, using standard transcription")
                return self.transcribe(audio_samples, sample_rate)
            
            # Apply speaker-specific preprocessing
            processed_audio = preprocess_audio_for_speaker(audio_samples, profile)
            
            # Get adaptation transformer
            transformer = self._get_adaptation_transformer(user_id, profile)
            
            # Standard transcription with preprocessed audio
            # In a full implementation, we would use the transformer to modify 
            # features inside the model, but for simplicity we just use preprocessing
            transcription = self.transcribe(processed_audio, sample_rate)
            
            logger.info(f"Applied speaker adaptation for user {user_id}")
            return transcription
            
        except Exception as e:
            logger.error(f"Error in adaptive transcription for user {user_id}: {str(e)}")
            # Fallback to standard transcription on error
            return self.transcribe(audio_samples, sample_rate)
            
        finally:
            # Close session if we created it
            if session_created and db is not None:
                db.close()
    
    def _get_speaker_profile(self, user_id: int, db: Session) -> Optional[Dict[str, Any]]:
        """
        Get speaker profile from database or cache
        
        Args:
            user_id: User ID
            db: Database session
            
        Returns:
            Speaker profile dictionary or None if not found
        """
        import time
        current_time = time.time()
        
        # Check cache first
        if user_id in self.profile_cache:
            cache_time = self.profile_cache_timestamps.get(user_id, 0)
            if current_time - cache_time < self.profile_cache_ttl:
                logger.debug(f"Using cached profile for user {user_id}")
                return self.profile_cache[user_id]
        
        try:
            # Get active profile from database
            profile = db.query(SpeakerProfile)\
                .filter(SpeakerProfile.user_id == user_id, SpeakerProfile.is_active == True)\
                .first()
            
            if not profile:
                logger.warning(f"No active speaker profile found for user {user_id}")
                return None
                
            # Get profile dictionary
            profile_dict = profile.get_profile_dict()
            
            # Cache the profile
            self.profile_cache[user_id] = profile_dict
            self.profile_cache_timestamps[user_id] = current_time
            
            return profile_dict
                
        except Exception as e:
            logger.error(f"Error getting speaker profile for user {user_id}: {str(e)}")
            return None
    
    def _get_adaptation_transformer(self, user_id: int, profile: Dict[str, Any]) -> Optional[AdaptationTransformer]:
        """
        Get or create adaptation transformer for a user
        
        Args:
            user_id: User ID
            profile: Speaker profile dictionary
            
        Returns:
            AdaptationTransformer or None if creation fails
        """
        # Check cache first
        if user_id in self.adaptation_cache:
            return self.adaptation_cache[user_id]
            
        try:
            # Extract mean vector and covariance matrix
            mean_vector = profile.get("mean_vector")
            covariance_matrix = profile.get("covariance_matrix")
            
            if mean_vector is None or covariance_matrix is None:
                logger.warning(f"Invalid profile data for user {user_id}")
                return None
                
            # Create transformer
            transformer = AdaptationTransformer(mean_vector, covariance_matrix)
            
            # Estimate transform from base model
            base_stats = get_base_model_stats()
            transformer.estimate_transform(base_stats)
            
            # Cache for future use
            self.adaptation_cache[user_id] = transformer
            
            return transformer
            
        except Exception as e:
            logger.error(f"Error creating adaptation transformer for user {user_id}: {str(e)}")
            return None
    
    def clear_cache(self, user_id: Optional[int] = None):
        """
        Clear cache for a specific user or all users
        
        Args:
            user_id: Specific user ID to clear, or None to clear all
        """
        if user_id is not None:
            # Clear specific user
            if user_id in self.adaptation_cache:
                del self.adaptation_cache[user_id]
            if user_id in self.profile_cache:
                del self.profile_cache[user_id]
            if user_id in self.profile_cache_timestamps:
                del self.profile_cache_timestamps[user_id]
            logger.info(f"Cleared cache for user {user_id}")
        else:
            # Clear all caches
            self.adaptation_cache.clear()
            self.profile_cache.clear()
            self.profile_cache_timestamps.clear()
            logger.info("Cleared all caches")
    
    def transcribe_with_timestamps(self, audio_samples: np.ndarray, 
                                  sample_rate: int = 16000,
                                  user_id: Optional[int] = None,
                                  db: Optional[Session] = None) -> Dict[str, Any]:
        """
        Transcribe audio with word-level timestamps, optionally using speaker adaptation
        
        Args:
            audio_samples: Normalized audio samples (float32 [-1.0, 1.0])
            sample_rate: Sample rate of the audio
            user_id: Optional user ID for adaptation
            db: Optional database session
            
        Returns:
            Dictionary with transcription text and timestamps
        """
        try:
            # Apply adaptation preprocessing if requested
            if user_id is not None and db is not None:
                profile = self._get_speaker_profile(user_id, db)
                if profile:
                    audio_samples = preprocess_audio_for_speaker(audio_samples, profile)
                    logger.info(f"Applied speaker adaptation preprocessing for user {user_id}")
            
            # Preprocess audio
            input_values = self.preprocess_audio(audio_samples, sample_rate)
            
            # Run model inference
            with torch.no_grad():
                logits = self.model(input_values).logits
            
            # Move logits to CPU and convert to numpy for decoder
            logits_np = logits.squeeze(0).cpu().numpy()
            
            # Decode with timestamps
            beams = self.decoder.decode_beams(logits_np)
            best_beam = beams[0]
            
            return {
                "text": best_beam[0].lower(),
                "timestamps": best_beam[2],
                "words": [{"word": w, "start": s, "end": e} 
                          for w, (s, e) in zip(best_beam[0].split(), best_beam[2])],
                "used_adaptation": user_id is not None and profile is not None
            }
            
        except Exception as e:
            logger.error(f"Timestamp transcription error: {str(e)}")
            return {
                "text": "",
                "timestamps": [],
                "words": [],
                "used_adaptation": False,
                "error": str(e)
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_id": self.model.config._name_or_path,
            "device": str(self.device),
            "vocab_size": len(self.processor.tokenizer.get_vocab()),
            "has_language_model": hasattr(self.decoder, "kenlm_model"),
            "model_parameters": sum(p.numel() for p in self.model.parameters()),
            "adaptation_cache_size": len(self.adaptation_cache),
            "profile_cache_size": len(self.profile_cache)
        }