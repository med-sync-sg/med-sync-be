import webrtcvad
import numpy as np
import pandas as pd
import librosa
import logging
from typing import List, Dict, Any
from sqlalchemy.orm import Session

from app.db.local_session import DatabaseManager
from app.services.voice_calibration_service import VoiceCalibrationService
from app.services.audio_service import AudioService

logger = logging.getLogger(__name__)

class DiarizationService:
    """
    Service for doctor-patient diarization that integrates with voice calibration.
    Focuses on distinguishing the doctor (known user) from patients.
    """
    
    def __init__(self, db_session: Session = None):
        """Initialize the diarization service with dependencies"""
        self.db = db_session or next(DatabaseManager().get_session())
        self.audio_service = AudioService()
        self.calibration_service = VoiceCalibrationService(self.db)
        self.vad = webrtcvad.Vad()  # Voice Activity Detector
        
    def process_audio(self, audio_data: np.ndarray, sample_rate: int = 16000,
                      doctor_id: int = None) -> Dict[str, Any]:
        """
        Process audio data for doctor-patient diarization
        
        Args:
            audio_data: Audio samples as numpy array
            sample_rate: Audio sample rate in Hz
            doctor_id: User ID of the doctor with calibration profile
            
        Returns:
            Dictionary with diarization results (segments labeled as doctor or patient)
        """
        # Step 1: Voice Activity Detection
        speech_segments = self._detect_speech(audio_data, sample_rate)
        
        # Step 2: Speaker Segmentation (find potential change points)
        speaker_segments = self._segment_speakers(audio_data, speech_segments, sample_rate)
        
        # Step 3: Extract embeddings for each segment
        segment_embeddings = self._extract_embeddings(audio_data, speaker_segments, sample_rate)
        
        # Step 4: Binary classification - doctor or patient
        speaker_mapping = self._classify_doctor_patient(segment_embeddings, doctor_id)
        
        return {
            "segments": speaker_segments,
            "speaker_mapping": speaker_mapping
        }

    def _classify_doctor_patient(self, segment_embeddings, doctor_id):
        """
        Classify each segment as doctor or patient
        
        Args:
            segment_embeddings: List of (segment, embedding) tuples
            doctor_id: User ID of the doctor
            
        Returns:
            Dictionary mapping segment index to "doctor" or "patient"
        """
        if not doctor_id:
            # Without doctor profile, use clustering to find the two main speakers
            return self._cluster_binary(segment_embeddings)
        
        # Get doctor's profile from calibration data
        doctor_profile = self.calibration_service.get_speaker_profile(doctor_id, self.db)
        if not doctor_profile:
            logger.warning(f"No calibration profile found for doctor (User ID: {doctor_id})")
            return self._cluster_binary(segment_embeddings)
        
        # Extract doctor's mean vector
        doctor_mean = doctor_profile.get("mean_vector")
        if doctor_mean is None:
            logger.warning("Invalid doctor profile (missing mean vector)")
            return self._cluster_binary(segment_embeddings)
        
        # Calculate similarity scores
        segment_to_speaker = {}
        similarity_scores = []
        
        for i, (segment, embedding) in enumerate(segment_embeddings):
            # Compute cosine similarity with doctor's profile
            similarity = np.dot(embedding, doctor_mean) / (
                np.linalg.norm(embedding) * np.linalg.norm(doctor_mean))
            
            similarity_scores.append((i, similarity))
        
        # Sort by similarity
        similarity_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Use adaptive threshold to separate doctor from patient
        if len(similarity_scores) > 5:
            # Get similarity distribution
            similarities = [s for _, s in similarity_scores]
            
            # Find natural threshold using Otsu's method or simple statistics
            # For simplicity, let's use basic statistics
            mean_sim = np.mean(similarities)
            std_sim = np.std(similarities)
            threshold = mean_sim - 0.5 * std_sim  # Adjust factor as needed
        else:
            # With few segments, use a fixed threshold
            threshold = 0.5
        
        # Classify segments
        for i, similarity in similarity_scores:
            if similarity > threshold:
                segment_to_speaker[i] = "doctor"
            else:
                segment_to_speaker[i] = "patient"
        
        return segment_to_speaker

    def _cluster_binary(self, segment_embeddings):
        """
        Use binary clustering to separate speakers when doctor profile isn't available
        
        Args:
            segment_embeddings: List of (segment, embedding) tuples
            
        Returns:
            Dictionary mapping segment index to "speaker1" or "speaker2"
        """
        from sklearn.cluster import KMeans
        
        # Extract embeddings into a matrix
        if not segment_embeddings:
            return {}
        
        embeddings_matrix = np.array([emb for _, emb in segment_embeddings])
        
        # Apply K-means with 2 clusters
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
        labels = kmeans.fit_predict(embeddings_matrix)
        
        # Map to segment indices
        segment_to_speaker = {}
        for i, label in enumerate(labels):
            segment_to_speaker[i] = "speaker1" if label == 0 else "speaker2"
        
        return segment_to_speaker

    def _detect_speech(self, audio_data, sample_rate):
        """
        Use WebRTC VAD to detect speech segments
        
        Args:
            audio_data: Audio samples as numpy array
            sample_rate: Audio sample rate in Hz
            
        Returns:
            List of (start, end) tuples in seconds for speech segments
        """
        
        # Frame duration in milliseconds (10, 20, or 30)
        frame_duration_ms = 30
        
        # Calculate frame size
        frame_size = int(sample_rate * frame_duration_ms / 1000)
        
        # Create VAD instance (0-3, higher is more aggressive)        
        # Process frames
        speech_frames = []
        is_speech = []
        
        for i in range(0, len(audio_data) - frame_size, frame_size):
            # Extract frame
            frame = audio_data[i:i+frame_size]
            
            # Convert to 16-bit PCM
            frame_pcm = (frame * 32768).astype(np.int16).tobytes()
            
            # Check if frame is speech
            try:
                speech = self.vad.is_speech(frame_pcm, sample_rate)
                is_speech.append(speech)
            except Exception as e:
                is_speech.append(False)
        
        # Merge consecutive speech frames into segments
        segments = []
        in_speech = False
        start_frame = 0
        
        for i, speech in enumerate(is_speech):
            if speech and not in_speech:
                # Speech start
                in_speech = True
                start_frame = i
            elif not speech and in_speech:
                # Speech end
                in_speech = False
                
                # Convert to seconds
                start_time = start_frame * frame_duration_ms / 1000
                end_time = i * frame_duration_ms / 1000
                
                # Only add if segment is long enough (e.g., > 300ms)
                if end_time - start_time > 0.3:
                    segments.append((start_time, end_time))
        
        # Handle the case where speech continues until the end
        if in_speech:
            start_time = start_frame * frame_duration_ms / 1000
            end_time = len(is_speech) * frame_duration_ms / 1000
            if end_time - start_time > 0.3:
                segments.append((start_time, end_time))
        
        return segments
        
    def _segment_speakers(self, audio_data, speech_segments, sample_rate):
        """
        Find potential speaker change points within speech segments
        
        Args:
            audio_data: Audio samples as numpy array
            speech_segments: List of (start, end) tuples for speech segments
            sample_rate: Audio sample rate in Hz
            
        Returns:
            List of (start, end) tuples for speaker-homogeneous segments
        """
        from scipy.linalg import det
        
        # Parameters
        window_size_sec = 2.0
        step_size_sec = 1.0
        penalty_coefficient = 1.0  # BIC penalty parameter
        
        speaker_segments = []
        
        for speech_start, speech_end in speech_segments:
            # Convert seconds to samples
            start_sample = int(speech_start * sample_rate)
            end_sample = int(speech_end * sample_rate)
            speech_audio = audio_data[start_sample:end_sample]
            
            # Skip segments that are too short
            if len(speech_audio) < int(2 * window_size_sec * sample_rate):
                speaker_segments.append((speech_start, speech_end))
                continue
            
            # Find potential change points
            change_points = []
            
            # Window size and step in samples
            window_size = int(window_size_sec * sample_rate)
            step_size = int(step_size_sec * sample_rate)
            
            # Process the speech segment with overlapping windows
            for i in range(window_size, len(speech_audio) - window_size, step_size):
                # Extract windows
                window1 = speech_audio[i-window_size:i]
                window2 = speech_audio[i:i+window_size]
                
                # Extract MFCCs for both windows
                try:
                    mfcc1 = librosa.feature.mfcc(
                        y=window1, 
                        sr=sample_rate, 
                        n_mfcc=13
                    )
                    
                    mfcc2 = librosa.feature.mfcc(
                        y=window2, 
                        sr=sample_rate, 
                        n_mfcc=13
                    )
                    
                    # Calculate covariance matrices
                    cov1 = np.cov(mfcc1)
                    cov2 = np.cov(mfcc2)
                    
                    # Combined window
                    window_combined = speech_audio[i-window_size:i+window_size]
                    mfcc_combined = librosa.feature.mfcc(
                        y=window_combined, 
                        sr=sample_rate, 
                        n_mfcc=13
                    )
                    cov_combined = np.cov(mfcc_combined)
                    
                    # Calculate BIC value (simplification)
                    n1 = mfcc1.shape[1]
                    n2 = mfcc2.shape[1]
                    n = n1 + n2
                    d = mfcc1.shape[0]  # Feature dimension
                    
                    bic = (n * np.log(det(cov_combined) + 1e-10) - 
                        n1 * np.log(det(cov1) + 1e-10) - 
                        n2 * np.log(det(cov2) + 1e-10))
                    penalty = 0.5 * penalty_coefficient * (d + 0.5 * d * (d + 1)) * np.log(n)
                    
                    # Check if change point
                    if bic > penalty:
                        change_point_time = speech_start + (i / sample_rate)
                        change_points.append(change_point_time)
                except Exception as e:
                    # Skip on error
                    continue
            
            # Convert change points to segments
            if not change_points:
                # No change points found, keep original segment
                speaker_segments.append((speech_start, speech_end))
            else:
                # Sort change points
                change_points.sort()
                
                # Create segments from change points
                prev_time = speech_start
                for change_time in change_points:
                    # Only add if segment is long enough (e.g., > 1 second)
                    if change_time - prev_time > 1.0:
                        speaker_segments.append((prev_time, change_time))
                    prev_time = change_time
                
                # Add final segment
                if speech_end - prev_time > 1.0:
                    speaker_segments.append((prev_time, speech_end))
        
        return speaker_segments
        
    def _extract_embeddings(self, audio_data, segments, sample_rate):
        """
        Extract speaker embeddings using the same features as calibration
        
        Args:
            audio_data: Full audio samples
            segments: List of (start, end) segment tuples in seconds
            sample_rate: Audio sample rate
            
        Returns:
            List of (segment, embedding) tuples
        """
        segment_embeddings = []
        
        for start_sec, end_sec in segments:
            # Convert seconds to samples
            start_sample = int(start_sec * sample_rate)
            end_sample = int(end_sec * sample_rate)
            
            # Extract segment audio
            segment_audio = audio_data[start_sample:end_sample]
            
            # Use the same MFCC extraction as in calibration
            features = None
            try:
                # Convert to appropriate format for librosa (if needed)
                if len(segment_audio) > 0:
                    # Extract MFCCs using same method as calibration
                    mfccs = librosa.feature.mfcc(
                        y=segment_audio, 
                        sr=sample_rate, 
                        n_mfcc=13,
                        hop_length=int(sample_rate * 0.01),
                        n_fft=int(sample_rate * 0.025)
                    )
                    
                    # Add delta features as in calibration
                    delta_mfccs = librosa.feature.delta(mfccs)
                    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
                    
                    # Combine features
                    features = np.vstack([mfccs, delta_mfccs, delta2_mfccs])
                    
                    # Get mean vector as embedding
                    embedding = np.mean(features, axis=1)
                    
                    segment_embeddings.append(((start_sec, end_sec), embedding))
            except Exception as e:
                logger.error(f"Error extracting features: {str(e)}")
                continue
        
        return segment_embeddings
        
    def _match_known_speakers(self, segment_embeddings, known_speaker_ids):
        """
        Match segments to known speakers using calibration data
        
        Args:
            segment_embeddings: List of (segment, embedding) tuples
            known_speaker_ids: List of user IDs with calibration profiles
            
        Returns:
            Dictionary mapping segment index to speaker ID
        """
        # Get speaker profiles for known speakers
        speaker_profiles = {}
        for user_id in known_speaker_ids:
            profile = self.calibration_service.get_speaker_profile(user_id, self.db)
            if profile:
                # Extract mean vector from profile
                mean_vector = profile.get("mean_vector")
                if mean_vector is not None:
                    speaker_profiles[user_id] = mean_vector
        
        # Match each segment to closest speaker
        segment_to_speaker = {}
        
        for i, (segment, embedding) in enumerate(segment_embeddings):
            best_speaker = None
            best_similarity = -float('inf')
            
            for speaker_id, speaker_mean in speaker_profiles.items():
                # Compute cosine similarity
                similarity = np.dot(embedding, speaker_mean) / (
                    np.linalg.norm(embedding) * np.linalg.norm(speaker_mean))
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_speaker = speaker_id
            
            # Only assign if similarity exceeds threshold
            if best_similarity > 0.5:  # Adjust threshold as needed
                segment_to_speaker[i] = best_speaker
            else:
                # Unknown speaker
                segment_to_speaker[i] = f"unknown_{i}"
        
        return segment_to_speaker
        
    def _cluster_speakers(self, segment_embeddings):
        """Cluster segments to discover speakers"""
        # Implementation here