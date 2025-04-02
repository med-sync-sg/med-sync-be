import numpy as np
import os
import torch
import logging
from typing import Optional, Dict, Any
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pyctcdecode import build_ctcdecoder

# Configure logger
logger = logging.getLogger(__name__)

class SpeechProcessor:
    """
    Class for speech-to-text transcription using Wav2Vec2 model.
    
    This class handles:
    1. Model initialization and configuration
    2. Audio preprocessing
    3. Transcription with optional language model
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
        Transcribe audio samples to text
        
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
    
    def transcribe_with_timestamps(self, audio_samples: np.ndarray, 
                                  sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Transcribe audio with word-level timestamps
        
        Args:
            audio_samples: Normalized audio samples (float32 [-1.0, 1.0])
            sample_rate: Sample rate of the audio
            
        Returns:
            Dictionary with transcription text and timestamps
        """
        try:
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
                          for w, (s, e) in zip(best_beam[0].split(), best_beam[2])]
            }
            
        except Exception as e:
            logger.error(f"Timestamp transcription error: {str(e)}")
            return {
                "text": "",
                "timestamps": [],
                "words": []
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
            "model_parameters": sum(p.numel() for p in self.model.parameters())
        }