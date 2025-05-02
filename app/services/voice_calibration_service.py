import os
import numpy as np
import librosa
import pickle
import logging
from typing import List, Dict, Any, Optional, BinaryIO
from app.db.local_session import DatabaseManager
from app.models.models import User, SpeakerProfile, CalibrationPhrase, CalibrationRecording
from app.schemas.calibration import CalibrationPhraseBase, CalibrationStatus
from app.utils.voice_adaptation_utils import extract_mfcc_features, estimate_warping_factor
from io import BytesIO
from sqlalchemy.orm import Session
import wave
import datetime
# Configure logger
logger = logging.getLogger(__name__)

# Router for voice calibration endpoints
get_session = DatabaseManager().get_session


class VoiceCalibrationService:
    """Service for managing voice calibration and speaker profiles using database storage"""
    
    def __init__(self, db_session: Optional[Session] = None):
        """Initialize calibration service with optional database session"""
        self.db = db_session
        self._ensure_default_phrases()
    
    def _ensure_default_phrases(self) -> None:
        """Ensure default calibration phrases exist in the database"""
        try:
            with next(get_session()) as db:
                # Check if phrases already exist
                phrase_count = db.query(CalibrationPhrase).count()
                if phrase_count > 0:
                    return  # Phrases already exist
                
                # Create default phrases
                default_phrases = [
                    {"text": "The patient presents with severe headache and nausea", 
                     "category": "general", 
                     "medical_terms": ["headache", "nausea"]},
                    {"text": "Vital signs are within normal range", 
                     "category": "examination", 
                     "medical_terms": ["vital signs"]},
                    {"text": "Prescribed amoxicillin 500mg three times daily", 
                     "category": "medication", 
                     "medical_terms": ["amoxicillin"]},
                    {"text": "Patient reports chest pain radiating to left arm", 
                     "category": "symptoms", 
                     "medical_terms": ["chest pain", "radiating"]},
                    {"text": "No known drug allergies", 
                     "category": "history", 
                     "medical_terms": ["drug allergies"]},
                    {"text": "Recommend follow-up in two weeks", 
                     "category": "plan", 
                     "medical_terms": ["follow-up"]},
                    {"text": "Laboratory results show elevated white blood cell count", 
                     "category": "results", 
                     "medical_terms": ["white blood cell", "elevated"]},
                    {"text": "Differential diagnosis includes myocardial infarction", 
                     "category": "assessment", 
                     "medical_terms": ["myocardial infarction", "differential diagnosis"]},
                    {"text": "Patient has history of hypertension and diabetes", 
                     "category": "history", 
                     "medical_terms": ["hypertension", "diabetes"]},
                    {"text": "Referred to cardiology for further evaluation", 
                     "category": "plan", 
                     "medical_terms": ["cardiology", "referred"]}
                ]
                
                # Add phrases to database
                for phrase_data in default_phrases:
                    phrase = CalibrationPhrase(
                        text=phrase_data["text"],
                        category=phrase_data["category"],
                        medical_terms=phrase_data["medical_terms"],
                        difficulty=len(phrase_data["text"].split())  # Simple difficulty metric based on word count
                    )
                    db.add(phrase)
                
                db.commit()
                logger.info(f"Added {len(default_phrases)} default calibration phrases")
                
        except Exception as e:
            logger.error(f"Error ensuring default phrases: {str(e)}")
    
    def get_calibration_phrases(self, db: Session) -> List[CalibrationPhraseBase]:
        """
        Get all calibration phrases
        
        Args:
            db: Database session
            
        Returns:
            List of calibration phrases
        """
        try:
            phrases = db.query(CalibrationPhrase).all()
            return [
                CalibrationPhraseBase(
                    id=phrase.id,
                    text=phrase.text,
                    category=phrase.category,
                    description=phrase.description,
                    difficulty=phrase.difficulty,
                    medical_terms=phrase.medical_terms
                )
                for phrase in phrases
            ]
        except Exception as e:
            logger.error(f"Error getting calibration phrases: {str(e)}")
            return []
    
    def extract_mfcc_features(self, audio_data: BinaryIO) -> np.ndarray:
        """
        Extract MFCC features from audio data
        
        Args:
            audio_data: Audio file data
            
        Returns:
            MFCC features as numpy array
        """
        try:
            # Save to temporary file
            temp_path = "temp_audio.wav"
            with open(temp_path, "wb") as f:
                f.write(audio_data.read())
            
            # Load with librosa
            y, sr = librosa.load(temp_path, sr=16000)
            
            # Extract features
            mfccs = librosa.feature.mfcc(
                y=y, 
                sr=sr, 
                n_mfcc=13,
                hop_length=int(sr * 0.01),  # 10ms hop
                n_fft=int(sr * 0.025)       # 25ms window
            )
            
            # Add delta features
            delta_mfccs = librosa.feature.delta(mfccs)
            delta2_mfccs = librosa.feature.delta(mfccs, order=2)
            
            # Combine features
            features = np.vstack([mfccs, delta_mfccs, delta2_mfccs])
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            return features
            
        except Exception as e:
            logger.error(f"Error extracting MFCC features: {str(e)}")
            if os.path.exists("temp_audio.wav"):
                os.remove("temp_audio.wav")
            raise
    
    def process_calibration_recording(self, user_id: int, phrase_id: int, audio_data: BinaryIO, db: Session) -> bool:
        """
        Process a calibration recording and store features in the database
        
        Args:
            user_id: User ID
            phrase_id: Phrase ID
            audio_data: Audio file data
            db: Database session
            
        Returns:
            True if processing successful
        """
        try:
            # Verify user and phrase exist
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                logger.error(f"User {user_id} not found")
                return False
                
            phrase = db.query(CalibrationPhrase).filter(CalibrationPhrase.id == phrase_id).first()
            if not phrase:
                logger.error(f"Phrase {phrase_id} not found")
                return False
            
            # Read the audio data
            audio_bytes = audio_data.read()
            
            # Create a BytesIO object to reuse the audio data
            audio_io = BytesIO(audio_bytes)
            
            # Extract features
            features = self.extract_mfcc_features(audio_io)
            
            # Get audio duration and sample rate
            audio_io.seek(0)  # Reset to beginning
            with wave.open(audio_io, 'rb') as wav_file:
                frames = wav_file.getnframes()
                sample_rate = wav_file.getframerate()
                duration_ms = (frames / sample_rate) * 1000
            
            
            # Serialize features
            features_binary = pickle.dumps(features)
            
            # Create recording record
            recording = CalibrationRecording(
                user_id=user_id,
                phrase_id=phrase_id,
                features=features_binary,
                audio_data=audio_bytes,
                duration_ms=duration_ms,
                sample_rate=sample_rate,
                feature_type="mfcc"
            )
            
            db.add(recording)
            db.commit()
                
            logger.info(f"Processed calibration recording for user {user_id}, phrase {phrase_id}")
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error processing calibration recording: {str(e)}")
            return False
    
    def get_calibration_status(self, user_id: int, db: Session) -> CalibrationStatus:
        """
        Get the current calibration status for a user
        
        Args:
            user_id: User ID
            db: Database session
            
        Returns:
            Calibration status
        """
        try:
            # Check for active profile
            profile = db.query(SpeakerProfile)\
                .filter(SpeakerProfile.user_id == user_id, SpeakerProfile.is_active == True)\
                .order_by(SpeakerProfile.updated_at.desc())\
                .first()
            
            profile_exists = profile is not None
            profile_id = profile.id if profile else None
            last_updated = profile.updated_at.isoformat() if profile else None
            
            # Count recorded phrases
            recording_count = db.query(CalibrationRecording)\
                .filter(CalibrationRecording.user_id == user_id)\
                .count()
            
            # Count total phrases
            total_phrases = db.query(CalibrationPhrase).count()
            
            return CalibrationStatus(
                user_id=user_id,
                calibration_complete=profile_exists,
                phrases_recorded=recording_count,
                phrases_total=total_phrases,
                profile_id=profile_id,
                last_updated=last_updated
            )
            
        except Exception as e:
            logger.error(f"Error getting calibration status: {str(e)}")
            return CalibrationStatus(
                user_id=user_id,
                calibration_complete=False,
                phrases_recorded=0,
                phrases_total=0
            )
    
    def create_speaker_profile(self, user_id: int, db: Session) -> Optional[int]:
        """
        Create a speaker profile from collected calibration recordings
        
        Args:
            user_id: User ID
            db: Database session
            
        Returns:
            Profile ID if successful, None otherwise
        """
        try:
            # Get user's recordings
            recordings = db.query(CalibrationRecording)\
                .filter(CalibrationRecording.user_id == user_id)\
                .all()
                
            if len(recordings) < 3:  # Need at least 3 recordings
                logger.error(f"Not enough calibration data for user {user_id}")
                return None
                
            # Load all features
            # Extract MFCCs from recordings
            all_mfcc_features = []
            for recording in recordings:
                # Standard MFCC extraction (no warping yet)
                mfccs = extract_mfcc_features(recording.audio_data, recording.sample_rate)
                all_mfcc_features.append(mfccs)
            
            if all_mfcc_features:
                combined_mfccs = np.concatenate(all_mfcc_features, axis=1)
            else:
                return None
            
            warp_factor = estimate_warping_factor(combined_mfccs)
            
            mean_vector = np.mean(all_mfcc_features, axis=1)
            covariance_matrix = np.cov(all_mfcc_features)
            
            # Create speaker profile dictionary
            speaker_profile_dict = {
                "user_id": user_id,
                "vtln_warp_factor": warp_factor,
                "mean_vector": mean_vector,
                "covariance_matrix": covariance_matrix,
                "feature_dimension": combined_mfccs.shape[0],
                "training_phrases": len(recordings),
                "created_at": datetime.datetime.now()
            }
            
            # Create database record
            speaker_profile = SpeakerProfile.create_from_data(user_id, speaker_profile_dict)
            
            # Set all other profiles to inactive
            db.query(SpeakerProfile)\
                .filter(SpeakerProfile.user_id == user_id, SpeakerProfile.id != speaker_profile.id)\
                .update({"is_active": False})
            
            # Add to database
            db.add(speaker_profile)
            db.flush()
            
            # Update recordings to link to this profile
            for recording in recordings:
                recording.speaker_profile_id = speaker_profile.id
            
            db.commit()
                
            logger.info(f"Created speaker profile for user {user_id}")
            return speaker_profile.id
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error creating speaker profile: {str(e)}")
            return None
    
    def get_speaker_profile(self, user_id: int, db: Session) -> Optional[Dict[str, Any]]:
        """
        Retrieve a user's speaker profile
        
        Args:
            user_id: User ID
            db: Database session
            
        Returns:
            Speaker profile dictionary or None if not found
        """
        try:
            # Get active profile
            profile = db.query(SpeakerProfile)\
                .filter(SpeakerProfile.user_id == user_id, SpeakerProfile.is_active == True)\
                .first()
            
            if not profile:
                return None
                
            return profile.get_profile_dict()
            
        except Exception as e:
            logger.error(f"Error loading speaker profile: {str(e)}")
            return None
    
    def delete_calibration_data(self, user_id: int, db: Session) -> bool:
        """
        Delete all calibration data for a user
        
        Args:
            user_id: User ID
            db: Database session
            
        Returns:
            True if successful
        """
        try:
            # Delete recordings
            db.query(CalibrationRecording)\
                .filter(CalibrationRecording.user_id == user_id)\
                .delete()
            
            # Delete profiles
            db.query(SpeakerProfile)\
                .filter(SpeakerProfile.user_id == user_id)\
                .delete()
            
            db.commit()
            
            logger.info(f"Deleted calibration data for user {user_id}")
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error deleting calibration data: {str(e)}")
            return False
