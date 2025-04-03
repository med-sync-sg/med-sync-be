import logging
from typing import List, Dict, Any, Optional, Tuple
from app.services.audio_service import AudioService
from app.utils.speech_processor import SpeechProcessor
from app.utils.text_utils import clean_transcription, correct_spelling
from app.utils.nlp.spacy_utils import process_text, find_medical_modifiers

# Configure logger
logger = logging.getLogger(__name__)

class TranscriptionService:
    """
    Service for managing audio transcription and related processing.
    Coordinates the full pipeline from audio to structured text data.
    """
    
    def __init__(self, 
                 audio_service: AudioService = None, 
                 speech_processor: SpeechProcessor = None):
        """
        Initialize transcription service with dependencies
        
        Args:
            audio_service: Service for audio processing (uses singleton if None)
            speech_processor: Processor for speech-to-text conversion (creates if None)
        """
        self.audio_service = audio_service or AudioService()
        self.speech_processor = speech_processor or SpeechProcessor()
        
        # Transcript state
        self.full_transcript = ""
        self.transcript_segments = []
        
        logger.info("TranscriptionService initialized")

    def process_audio_segment(self, user_id: int, note_id: int) -> bool:
        """
        Process current audio buffer and attempt transcription
        
        Args:
            user_id: User ID for the transcription
            note_id: Note ID for the transcription
            
        Returns:
            True if transcription occurred, False otherwise
        """
        try:
            # Check for minimum audio duration
            if not self.audio_service.has_minimum_audio():
                return False
            
            # Check for silence (indicating end of utterance)
            if not self.audio_service.detect_silence():
                return False
                
            logger.info(f"Processing audio segment for user {user_id}, note {note_id}")
            
            # Get audio data and transcribe
            audio_samples = self.audio_service.get_wave_data()
            transcription = self.speech_processor.transcribe(audio_samples)
            
            if not transcription:
                logger.warning("Transcription produced empty result")
                return False
                
            # Process transcription text
            result = self._process_transcription_text(transcription)
            if result:
                # Reset buffer for next segment
                self.audio_service.reset_current_buffer()
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error during audio processing: {str(e)}")
            return False
            
    def _process_transcription_text(self, text: str) -> bool:
        """
        Process transcribed text: clean, correct, and update transcript
        
        Args:
            text: Raw transcription text
            
        Returns:
            True if processing succeeded, False otherwise
        """
        try:
            # Clean and correct text
            cleaned_text = clean_transcription(text)
            corrected_text = correct_spelling(cleaned_text)
            
            if not cleaned_text:
                logger.warning("Cleaning resulted in empty text")
                return False
                
            logger.debug(f"Transcription: '{text}' -> '{corrected_text}'")
            
            # Update transcript state
            self.full_transcript = (self.full_transcript + ". " + text).strip()
            self.transcript_segments.append(text)
            
            # Process with NLP pipeline for debugging
            doc = process_text(self.full_transcript)
            logger.debug(f"Entities detected: {len(doc.ents)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing transcription text: {str(e)}")
            return False
    
    def extract_keywords(self) -> List[Dict[str, Any]]:
        """
        Extract keywords from the current transcript
        
        Returns:
            List of extracted keyword dictionaries
        """
        if not self.full_transcript:
            return []
            
        try:
            # Process the full transcript with NLP
            doc = process_text(self.full_transcript)
            
            # Extract keywords using the existing pipeline
            keywords = find_medical_modifiers(doc=doc)
            logger.info(f"Extracted {len(keywords)} keyword sets from transcript")
            
            return keywords
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            return []
    
    def get_current_transcript(self) -> Dict[str, Any]:
        """
        Get the current transcription state
        
        Returns:
            Dictionary with transcript information
        """
        return {
            "text": self.full_transcript,
            "segments": self.transcript_segments,
            "segment_count": len(self.transcript_segments),
            "word_count": len(self.full_transcript.split()) if self.full_transcript else 0
        }
    
    def reset(self) -> None:
        """Reset transcription state"""
        self.full_transcript = ""
        self.transcript_segments = []
        logger.info("Transcription state reset")