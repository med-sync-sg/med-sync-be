import numpy as np
import os
import torch
import logging
import pickle
import time

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
    
    def __init__(self, model_id: str = "facebook/wav2vec2-large-960h-lv60", # "facebook/wav2vec2-base-960h"
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
                    kenlm_model_path=language_model_path, # currently there is no LM
                    alpha=2.5,  # 0.3 Language model weight 1.5~2.5
                    beta=0.0,    # 1.0 Word insertion bonus 0.0~1.0
                    beam_width=100  # 기존 default 보다 크게
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
    
    def preprocess_audio_for_speaker(self, audio_samples: np.ndarray, speaker_profile: Dict[str, Any]) -> np.ndarray:
        """
        Preprocess audio with speaker-specific settings
        
        Args:
            audio_samples: Raw audio samples
            speaker_profile: Speaker profile from calibration
            
        Returns:
            Processed audio samples
        """
        try:
            # Extract mean vector and covariance matrix
            mean_vector = speaker_profile.get('mean_vector')
            covariance_matrix = speaker_profile.get('covariance_matrix')
            
            if mean_vector is None or covariance_matrix is None:
                logger.warning("Missing required statistics in speaker profile")
                return audio_samples
            
            # Volume normalization - consistent input level is important
            if np.max(np.abs(audio_samples)) > 0:
                normalized = audio_samples / np.max(np.abs(audio_samples)) * 0.9
            else:
                normalized = audio_samples
                
            # Extract MFCC features from the audio
            mfccs = self.extract_mfcc_features(normalized, sr=16000)
            
            # Apply CMVN normalization using speaker statistics
            # This aligns the feature distribution with the speaker's profile
            base_stats = get_base_model_stats()
            base_mean = base_stats.get('mean_vector')
            base_std = np.sqrt(np.diag(base_stats.get('covariance_matrix')))
            speaker_std = np.sqrt(np.diag(covariance_matrix))
            
            # Calculate scaling factors
            scale_factors = speaker_std / (base_std + 1e-10)
            
            # Adjust audio based on feature comparison (simplified frequency warping)
            # This is a basic approach - more sophisticated methods would implement 
            # vocal tract length normalization or similar techniques
            adjusted = normalized.copy()
            
            # Apply a simple filter based on the MFCC difference
            if len(adjusted) > 320:  # At least 20ms of audio
                # We're simplifying here - in a production system, you would apply
                # proper feature-space transformations and feature-to-audio conversion
                adjusted = self.apply_simple_spectral_shaping(adjusted, scale_factors)
                
            return adjusted
            
        except Exception as e:
            logger.error(f"Error preprocessing audio for speaker: {e}")
            return audio_samples

    def apply_simple_spectral_shaping(audio: np.ndarray, scale_factors: np.ndarray) -> np.ndarray:
        """
        Apply simple spectral shaping based on scale factors
        
        Args:
            audio: Input audio samples
            scale_factors: Scaling factors derived from MFCC comparison
            
        Returns:
            Shaped audio samples
        """
        import scipy.signal as signal
        
        # Create a simple filter based on the first few scale factors
        # This is very simplified - real applications would use more sophisticated methods
        factors = scale_factors[:8]  # Use first 8 factors
        
        # Normalize factors to create filter coefficients
        b = factors / np.sum(factors)
        a = [1.0]
        
        # Apply filter
        return signal.lfilter(b, a, audio)

    def extract_mfcc_features(samples: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        Extract MFCC features from audio samples
        
        Args:
            samples: Audio samples
            sr: Sample rate
            
        Returns:
            MFCC features
        """
        import librosa
        
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(
            y=samples, 
            sr=sr, 
            n_mfcc=13,
            hop_length=int(sr * 0.01),  # 10ms hop
            n_fft=int(sr * 0.025)       # 25ms window
        )
        
        return mfccs
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
            
            # Preprocess audio for model
            input_values = self.preprocess_audio(processed_audio, sample_rate)
            
            # Apply feature-space adaptation if transformer exists
            if transformer:
                # Run model inference with transformation
                with torch.no_grad():
                    # Get the features from the model's feature extractor
                    features = self.model.wav2vec2.feature_extractor(input_values)
                    
                    # In a complete implementation, you would apply the transformer to these features
                    # For now, use the standard features since we've preprocessed the audio
                    
                    # Continue with the model pipeline
                    logits = self.model(input_values).logits
                    
                # Move logits to CPU and convert to numpy for decoder
                logits_np = logits.squeeze(0).cpu().numpy()
                
                # Apply language model decoding with adaptation biases
                # Custom bias to improve recognition of medical terms if relevant
                medical_bias = 1.2  # Slight bias for medical terms
                
                # Decode with these settings
                transcription = self.decoder.decode(logits_np)
                
            else:
                # Fallback to standard transcription with preprocessed audio
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
            
            # If VTLN warping factor is not present, estimate it
            if 'vtln_warp_factor' not in profile_dict and 'mean_vector' in profile_dict:
                # Get base model stats
                base_stats = get_base_model_stats()
                base_mean = base_stats.get('mean_vector')
                
                if base_mean is not None and len(base_mean) == len(profile_dict['mean_vector']):
                    # Estimate VTLN warping factor
                    start_idx = 1
                    end_idx = min(6, len(profile_dict['mean_vector']))
                    
                    if end_idx > start_idx:
                        speaker_spectral_mean = np.mean(profile_dict['mean_vector'][start_idx:end_idx])
                        base_spectral_mean = np.mean(base_mean[start_idx:end_idx])
                        
                        if abs(base_spectral_mean) > 1e-6:  # Avoid division by zero
                            ratio = base_spectral_mean / speaker_spectral_mean
                            vtln_warp_factor = max(0.8, min(1.2, ratio))
                            profile_dict['vtln_warp_factor'] = vtln_warp_factor
                            logger.info(f"Estimated VTLN warp factor for user {user_id}: {vtln_warp_factor}")
            
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
            
            # Log successful creation
            logger.info(f"Created adaptation transformer for user {user_id} " + 
                        f"with feature dimension {len(mean_vector)}")
            
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
            
            # Get adaptation transformer
            transformer = self._get_adaptation_transformer(user_id, profile)
            
            # Check if we should apply VTLN directly to audio
            vtln_warp_factor = None
            if transformer:
                vtln_warp_factor = transformer.get_vtln_warp_factor()
                
            # Apply speaker-specific preprocessing (including VTLN)
            processed_audio = preprocess_audio_for_speaker(audio_samples, profile)
            
            # Apply additional feature-space adaptation
            if transformer:
                # Preprocess audio for model
                input_values = self.preprocess_audio(processed_audio, sample_rate)
                
                # Run model inference with transformation
                with torch.no_grad():
                    # Get the features from the model's feature extractor
                    features = self.model.wav2vec2.feature_extractor(input_values)
                    
                    # Apply feature-space transformation
                    features_np = features.cpu().numpy()
                    transformed_features = transformer.transform_features(features_np)
                    
                    # Convert back to tensor on the correct device
                    if hasattr(torch, 'as_tensor'):  # PyTorch 1.5+
                        transformed_features = torch.as_tensor(
                            transformed_features, 
                            device=features.device, 
                            dtype=features.dtype
                        )
                    else:
                        transformed_features = torch.tensor(
                            transformed_features, 
                            device=features.device, 
                            dtype=features.dtype
                        )
                    
                    # Continue with the model pipeline
                    # Many models don't have a way to inject modified features directly,
                    # so we may need to continue with the standard pipeline
                    logits = self.model(input_values).logits
                    
                # Move logits to CPU and convert to numpy for decoder
                logits_np = logits.squeeze(0).cpu().numpy()
                
                # Apply language model decoding with medical terminology biasing
                # For medical terminology, we can apply a slight bias to improve recognition
                from app.utils.constants import SYMPTOMS_AND_DISEASES_TUI
                
                # This is a placeholder - a more complete implementation would 
                # use the decoder's biasing capabilities if available
                medical_bias = 1.2  # Slight bias for medical terms
                
                # Decode with these settings
                transcription = self.decoder.decode(logits_np)
                
            else:
                # Fallback to standard transcription with preprocessed audio
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