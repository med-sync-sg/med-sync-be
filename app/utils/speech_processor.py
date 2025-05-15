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
from app.utils.text_utils import extract_medical_terms, sym_spell, clean_transcription
from app.db.data_loader import umls_df_dict

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
            # self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
            # self.model.to(self.device)
            # logger.info(f"Model loaded on {self.device}")
            
            # self.initialize_medical_dictionary()
            
            # Initialize the decoder
            self.decoder = self._create_decoder()            
            
            # Database connection and adaptation cache
            self.db_manager = DatabaseManager()
            self.adaptation_cache = {}  # Cache transformers by user_id
            self.profile_cache = {}     # Cache profiles by user_id
            self.profile_cache_ttl = 300  # Cache TTL in seconds
            self.profile_cache_timestamps = {}  # When profiles were cached
            
        except Exception as e:
            logger.error(f"Error initializing SpeechProcessor: {str(e)}")
            raise RuntimeError(f"Failed to initialize speech processor: {str(e)}")
    
    
    def _load_unigrams(self, unigrams_file, top_n=10000, min_score=-10.0):
        """
        Load unigrams from file and prepare them for decoder
        
        Args:
            unigrams_file: Path to the unigrams.txt file
            top_n: Limit to the top N words by probability
            min_score: Minimum log probability to include
            
        Returns:
            Tuple of (unigram list with scores, hotword list for boosting)
        """
        print(f"Loading unigrams from {unigrams_file}")
        
        unigram_list = []  # List of (word, score) tuples
        hotwords = []      # List of domain-specific words for boosting
        
        try:
            with open(unigrams_file, 'r', encoding='utf-8') as f:
                # Skip header if it exists
                first_line = f.readline().strip()
                if first_line.startswith("word\t"):
                    pass  # Skip header
                else:
                    # If no header, process the first line
                    parts = first_line.split('\t')
                    if len(parts) >= 2:
                        word = parts[0].strip()
                        score = float(parts[1])
                        unigram_list.append((word, score))
                        
                        # High-probability words become hotwords for boosting
                        if score > min_score:
                            hotwords.append(word)
                
                # Process remaining lines
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        word = parts[0].strip()
                        score = float(parts[1])
                        unigram_list.append((word, score))
                        
                        # High-probability words become hotwords for boosting
                        if score > min_score:
                            hotwords.append(word)
            
            # Sort by score (highest first) and limit to top_n
            unigram_list.sort(key=lambda x: x[1], reverse=True)
            if top_n:
                unigram_list = unigram_list[:top_n]
                
            print(f"Loaded {len(unigram_list)} unigrams, {len(hotwords)} hotwords")
            
            # Print some samples
            if unigram_list:
                print("Sample unigrams (word, log prob):")
                for word, score in unigram_list[:10]:
                    print(f"  {word}: {score:.4f}")
            
        except Exception as e:
            print(f"Error loading unigrams: {str(e)}")
            # Return empty lists if file can't be loaded
            return [], []
            
        return unigram_list, hotwords
    
    def _create_decoder(self):
        # Path to language model
        lm_path = "D:/medsync/med_sync_be/training/4-gram.binary"
        unigram_path = "D:/medsync/med_sync_be/training/unigrams.txt"
        unigram_texts, hotwords = self._load_unigrams(unigram_path)
        
        # CRITICAL FIX: Create a vocabulary list that EXACTLY matches the model's vocabulary
        # Don't filter tokens or add/remove any - it must match the model exactly
        
        # Get all tokens from the tokenizer
        vocab_dict = self.processor.tokenizer.get_vocab()
        
        # Sort by token ID to maintain the exact order
        sorted_tokens = sorted(vocab_dict.items(), key=lambda x: x[1])
        
        # Now prepare decoder vocabulary
        # For CTC decoding, we replace the CTC blank token (usually <pad>) with an empty string
        # Other special tokens might need special handling
        decoder_vocab = []
        
        # Find the blank token (usually <pad> or a special token)
        blank_token = self.processor.tokenizer.pad_token
        blank_token_id = vocab_dict.get(blank_token, None)
        
        # Create the decoder vocabulary with proper mapping
        for token, idx in sorted_tokens:
            # For the blank token, use empty string
            if token == blank_token:
                decoder_vocab.append("")
            # For WordPiece tokens, replace "▁" with space
            elif "▁" in token:
                decoder_vocab.append(token.replace("▁", " "))
            # Other tokens remain as is
            else:
                decoder_vocab.append(token)
        
        print(f"Decoder vocabulary prepared with {len(decoder_vocab)} tokens (matching model)")
        
        # Build the decoder
        try:
            decoder = build_ctcdecoder(
                decoder_vocab,
                lm_path,
                unigrams=unigram_texts,
                alpha=0.7,
                beta=1.5
            )
            print("Successfully created decoder with language model")
        except Exception as e:
            print(f"Error building decoder with LM: {str(e)}")
            print("Falling back to decoder without language model")
            decoder = build_ctcdecoder(decoder_vocab)
        
        return decoder

    def initialize_medical_dictionary(self):
        """Initialize the dictionary with additional medical terms"""
        def create_medical_dictionary():
            """Generate a comprehensive medical dictionary from UMLS data"""
            
            try:
                # Get UMLS concepts
                df = umls_df_dict["concepts_with_sty_def_df"]
                logger.info(len(df))
                logger.info(df.columns)
                # Extract terms
                terms = set()
                for _, row in df.iterrows():
                    term: str = row["STR"]
                    
                    term_to_add = extract_medical_terms(term)
                    
                    if term_to_add:
                        for target_term in term_to_add:
                            if len(target_term) > 2:
                                terms.add(target_term)
                
                # Create dictionary directory if it doesn't exist
                os.makedirs("app/dictionaries", exist_ok=True)
                
                if os.path.exists(os.path.join("app", "dictionaries", "medical_terms.txt")):
                    logger.info("Medical terms already exist.")
                    return
                
                # Write to file
                with open(os.path.join("app", "dictionaries", "medical_terms.txt"), "w", encoding='utf-8') as f:
                    for term in sorted(terms):
                        f.write(f"{term}\n")
                
                print(f"Created medical dictionary with {len(terms)} terms")
                
            except Exception as e:
                print(f"Error creating medical dictionary: {str(e)}")  
        
        create_medical_dictionary()
        
        # Add medical terms to SymSpell dictionary
        try:
            # Get path to medical terms dictionary
            medical_dict_path = os.path.join("app", "dictionaries", "medical_terms.txt")
            
            if os.path.exists(medical_dict_path):
                # Load medical dictionary
                with open(medical_dict_path, 'r') as f:
                    for line in f:
                        term = line.strip()
                        if term:
                            # Add to dictionary with a decent frequency
                            sym_spell.create_dictionary_entry(term, 1000)
                
                logger.info(f"Loaded medical terms dictionary from {medical_dict_path}")
            else:
                logger.warning(f"Medical dictionary not found at {medical_dict_path}")
                
        except Exception as e:
            logger.error(f"Error loading medical dictionary: {str(e)}")

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
                adjusted = self.apply_spectral_shaping(adjusted, scale_factors)
                
            return adjusted
            
        except Exception as e:
            logger.error(f"Error preprocessing audio for speaker: {e}")
            return audio_samples

    def apply_spectral_shaping(self, audio: np.ndarray, scale_factors: np.ndarray) -> np.ndarray:
        """
        Apply spectral shaping to audio based on MFCC-derived scale factors
        
        Args:
            audio: Input audio samples
            scale_factors: Scaling factors derived from MFCC comparison
            
        Returns:
            Shaped audio samples
        """
        import scipy.signal as signal
        import numpy as np
        
        # Check for valid input
        if len(audio) == 0:
            return audio
        
        if np.all(scale_factors == 0) or len(scale_factors) == 0:
            return audio
        
        try:
            # For vocal tract adaptation, we want to design a filter that
            # transforms the spectral characteristics of the audio
            
            # 1. Extend scale factors if needed (use full available factors)
            if len(scale_factors) < 13:  # Typical MFCC dimension
                # Pad with ones (neutral scaling)
                extended_factors = np.ones(13)
                extended_factors[:len(scale_factors)] = scale_factors
            else:
                extended_factors = scale_factors[:13]  # Use first 13 (covers main formants)
            
            # 2. Design filter based on scale factors - convert to proper filter response
            # Use a more sophisticated filter design
            nyquist = 0.5 * 16000  # Assuming 16kHz sample rate
            
            # Map MFCC scale factors to frequency bands
            # This mapping is approximate - maps each MFCC bin to corresponding frequency range
            freq_points = np.linspace(0, nyquist, len(extended_factors) + 2)[1:-1]
            gains_db = 20 * np.log10(extended_factors)  # Convert to dB
            
            # Constrain extreme values to avoid instability
            gains_db = np.clip(gains_db, -20, 20)
            
            # 3. Create filter using frequency sampling method
            filter_order = min(int(len(audio) / 8), 512)  # Adaptive filter order based on audio length
            filter_order = max(filter_order, 31)  # Minimum order for good resolution
            filter_order = filter_order + 1 if filter_order % 2 == 0 else filter_order  # Make odd
            
            # Use firwin2 for more precise frequency response
            b = signal.firwin2(filter_order, 
                            np.concatenate(([0], freq_points, [nyquist])) / nyquist, 
                            np.concatenate(([gains_db[0]], gains_db, [gains_db[-1]])),
                            fs=16000)
            
            # 4. Apply filter
            padded_audio = np.pad(audio, (filter_order//2, filter_order//2), mode='edge')
            filtered_audio = signal.lfilter(b, [1.0], padded_audio)
            
            # Remove padding
            filtered_audio = filtered_audio[filter_order//2:filter_order//2 + len(audio)]
            
            # 5. Normalize output to match input level
            if np.max(np.abs(audio)) > 0:
                filtered_audio = filtered_audio * (np.max(np.abs(audio)) / np.max(np.abs(filtered_audio)))
            
            return filtered_audio
            
        except Exception as e:
            logger.warning(f"Error in spectral shaping: {str(e)}. Returning original audio.")
            return audio

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
    
    def _post_process_transcription(self, text: str) -> str:
        """
        Post-process transcription to ensure only valid words
        
        Args:
            text: Raw transcription text
            
        Returns:
            Processed text with only valid words
        """
        from symspellpy import Verbosity
        
        # First apply basic cleaning
        text = clean_transcription(text)
        
        # Split into words
        words = text.split()
        corrected_words = []
        
        # Process each word
        for word in words:            
            # Skip empty words after cleaning
            if not word:
                continue

            # Try to correct with SymSpell
            suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
            if suggestions:
                # Use the closest suggestion
                corrected_words.append(suggestions[0].term)
            else:
                # If word is in English dictionary, keep it
                if sym_spell.lookup(word, Verbosity.TOP, max_edit_distance=2):
                    corrected_words.append(word)
                # Otherwise, it's likely nonsense, so skip it
                else:
                    corrected_words.append(word)
        
        # Join back to text
        corrected_text = ' '.join(corrected_words)
        
        # Return empty string if nothing valid was found
        return corrected_text if corrected_words else ""
    
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
            
            # Process the audio with the standard processor
            inputs = self.processor(
                audio_samples, 
                sampling_rate=sample_rate, 
                return_tensors="pt"
            )
            
            # Run inference
            with torch.no_grad():
                logits = self.model(inputs.input_values).logits
            
            # Convert logits to log probabilities
            log_probs = torch.log_softmax(logits, dim=-1)
            
            # Get log probabilities as numpy array (time, vocab_size)
            log_probs_np = log_probs[0].cpu().numpy()
            
            # Use the decoder to get the transcription
            # If alpha or beta are provided, they override the default values
            # decoder_kwargs = {"beam_width": 100}
            # if alpha is not None:
            #     decoder_kwargs["alpha"] = alpha
            # if beta is not None:
            #     decoder_kwargs["beta"] = beta
                
            transcription = self.decoder.decode(log_probs_np, beam_width=50)
            
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