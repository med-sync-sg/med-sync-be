import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

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
            speech_processor: SpeechProcessor for transcription (creates if None)
        """
        self.audio_service = audio_service or AudioService()
        self.speech_processor = speech_processor or SpeechProcessor()
        
        # Transcript state
        self.full_transcript = ""
        self.transcript_segments = []
        
        logger.info("TranscriptionService initialized")

    def process_audio_segment(self, user_id: int, note_id: int, 
                              use_adaptation: bool = False, 
                              adaptation_user_id: Optional[int] = None,
                              db_session = None) -> bool:
        """
        Process current audio buffer and attempt transcription
        
        Args:
            user_id: User ID for the transcription
            note_id: Note ID for the transcription
            use_adaptation: Whether to use speaker adaptation
            adaptation_user_id: User ID for adaptation profile
            db_session: Optional database session
            
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
                
            logger.info(f"Processing audio segment for user {user_id}, note {note_id}, adaptation={use_adaptation}")
            
            # Get audio data and transcribe
            audio_samples = self.audio_service.get_wave_data()
            
            # Perform transcription with or without adaptation
            if use_adaptation and adaptation_user_id is not None:
                transcription = self.speech_processor.transcribe_with_adaptation(
                    audio_samples, 
                    adaptation_user_id,
                    db_session
                )
                logger.info(f"Used speaker adaptation for user {adaptation_user_id}")
            else:
                transcription = self.speech_processor.transcribe(audio_samples)
                logger.info("Used standard transcription")
            
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
    
    def generate_transcript_with_timestamps(self, 
                                           user_id: Optional[int] = None,
                                           db_session = None) -> Dict[str, Any]:
        """
        Generate a transcript with word-level timestamps using the current audio buffer
        
        Args:
            user_id: Optional user ID for adaptation
            db_session: Optional database session
            
        Returns:
            Dictionary with transcript text and timestamps
        """
        try:
            # Get audio data
            audio_samples = self.audio_service.get_wave_data(use_session_buffer=True)
            
            # Generate transcript with timestamps
            if user_id is not None and db_session is not None:
                # Use adaptation if user ID provided
                result = self.speech_processor.transcribe_with_timestamps(
                    audio_samples,
                    user_id=user_id,
                    db=db_session
                )
            else:
                # Standard transcription
                result = self.speech_processor.transcribe_with_timestamps(
                    audio_samples
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating timestamp transcript: {str(e)}")
            return {
                "text": "",
                "timestamps": [],
                "words": [],
                "error": str(e)
            }
            
    def reset(self) -> None:
        """Reset transcription state"""
        self.full_transcript = ""
        self.transcript_segments = []
        logger.info("Transcription state reset")
        
    def transcribe_doctor_patient(self, audio_samples: np.ndarray, 
                                diarization_results: Dict[str, Any],
                                doctor_id: int = None) -> Dict[str, Any]:
        """
        Transcribe audio with doctor-patient diarization results
        
        Args:
            audio_samples: Audio data as numpy array
            diarization_results: Results from DiarizationService
            doctor_id: User ID of the doctor (for adaptation)
            
        Returns:
            Transcription results with doctor/patient labels
        """
        segments = diarization_results["segments"]
        speaker_mapping = diarization_results["speaker_mapping"]
        sample_rate = 16000  # Default sample rate
        
        # Store transcriptions by role
        doctor_segments = []
        patient_segments = []
        
        # Process each segment
        for i, (start_sec, end_sec) in enumerate(segments):
            # Get speaker role for this segment
            role = speaker_mapping.get(i)
            if not role:
                continue
                
            # Extract segment audio
            start_sample = int(start_sec * sample_rate)
            end_sample = int(end_sec * sample_rate)
            segment_audio = audio_samples[start_sample:end_sample]
            
            # Use adaptation for doctor segments if doctor_id is provided
            use_adaptation = (role == "doctor" and doctor_id is not None)
            
            # Transcribe segment with or without adaptation
            if use_adaptation:
                transcription = self.speech_processor.transcribe_with_adaptation(
                    segment_audio, doctor_id
                )
            else:
                transcription = self.speech_processor.transcribe(segment_audio)
            
            # Store result in appropriate list
            segment_data = {
                "start": start_sec,
                "end": end_sec,
                "text": transcription
            }
            
            if role == "doctor" or role == "speaker1":
                doctor_segments.append(segment_data)
            else:
                patient_segments.append(segment_data)
        
        # Format results
        return {
            "doctor_segments": doctor_segments,
            "patient_segments": patient_segments,
            "full_transcript": self._format_doctor_patient_transcript(doctor_segments, patient_segments)
        }
        
    def _format_doctor_patient_transcript(self, doctor_segments, patient_segments):
        """Format the transcript with doctor/patient labels"""
        # Combine all segments
        all_segments = []
        
        for segment in doctor_segments:
            all_segments.append({
                "role": "Doctor",
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"]
            })
        
        for segment in patient_segments:
            all_segments.append({
                "role": "Patient",
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"]
            })
        
        # Sort by start time
        all_segments.sort(key=lambda x: x["start"])
        
        # Format as readable transcript
        lines = []
        for segment in all_segments:
            lines.append(f"{segment['role']}: {segment['text']}")
        
        return "\n".join(lines)