import webrtcvad
import numpy as np
import pandas as pd
import librosa
import logging
from typing import List, Dict, Any
from sqlalchemy.orm import Session
import torch

import time

from app.db.local_session import DatabaseManager
from app.services.voice_calibration_service import VoiceCalibrationService
from app.services.audio_service import AudioService

from speechbrain.inference.speaker import EncoderClassifier

logger = logging.getLogger(__name__)

class DiarizationService:
    """
    Service for doctor-patient diarization that integrates with voice calibration.
    Focuses on distinguishing the doctor (known user) from patients.
    """
    
    def __init__(self, db_session: Session = None):
        """Initialize the diarization service with dependencies"""
        self.db = db_session or next(DatabaseManager().get_session())
        self.audio_service = AudioService()  # Use the singleton instance
        self.calibration_service = VoiceCalibrationService(self.db)
        self.vad = webrtcvad.Vad()  # Voice Activity Detector
        self.x_vector_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", 
            run_opts={"device": "cpu"}
        )
        
        # Track when we last processed diarization
        self.last_diarization_time = 0
        self.diarization_interval_seconds = 5  # Process every 5 seconds
        
        # Store speaker profiles for consistency
        self.speaker_profiles = {
            "doctor": [],
            "patient": []
        }
        
        # Track previous diarization results for continuity
        self.previous_segments = []
        self.previous_mapping = {}
    
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
    
    def extract_xvector(self, audio_segment, sample_rate=16000):
        """
        Extract x-vector from audio segment
        
        Args:
            audio_segment: Audio samples as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            X-vector as numpy array or None if extraction failed
        """
        try:
            # Convert to appropriate format for the model
            waveform = torch.tensor(audio_segment).unsqueeze(0)
            
            # Extract x-vector
            with torch.no_grad():
                embeddings = self.x_vector_model.encode_batch(waveform)
                xvector = embeddings.squeeze().cpu().numpy()
            return xvector
        except Exception as e:
            print(f"Error extracting x-vector: {str(e)}")
            return None
    
    def diarize_buffered_audio(self, doctor_id: int = None) -> Dict[str, Any]:
        """
        Process the audio service buffer for doctor-patient diarization
        
        Args:
            doctor_id: User ID of the doctor with calibration profile
            
        Returns:
            Dictionary with diarization results (segments labeled as doctor or patient)
        """
        # Get audio data from audio service
        audio_data = self.audio_service.get_wave_data()
        sample_rate = self.audio_service.DEFAULT_SAMPLE_RATE
        
        # Skip if buffer is too small
        min_duration_seconds = 3  # Need at least 3 seconds
        min_samples = min_duration_seconds * sample_rate
        
        if len(audio_data) < min_samples:
            return {
                "segments": [],
                "speaker_mapping": {},
                "buffer_duration": len(audio_data) / sample_rate,
                "status": "insufficient_audio"
            }
        
        # Store the current buffer time bounds for segment timestamps
        buffer_duration = len(audio_data) / sample_rate
        buffer_start_time = time.time() - buffer_duration
        
        # Step 1: Voice Activity Detection on buffered audio
        speech_segments = self._detect_speech(audio_data, sample_rate)
        
        # Convert relative segment times to absolute timestamps
        timestamped_segments = []
        for start_sec, end_sec in speech_segments:
            # Convert to native Python floats
            start_sec = float(start_sec)
            end_sec = float(end_sec)
            
            # Calculate absolute timestamps
            absolute_start = buffer_start_time + start_sec
            absolute_end = buffer_start_time + end_sec
            timestamped_segments.append((start_sec, end_sec, absolute_start, absolute_end))
        
        # Step 2: Speaker Segmentation (find potential change points)
        speaker_segments = self._segment_speakers(audio_data, speech_segments, sample_rate)
        
        # Step 3: Extract embeddings for each segment
        segment_embeddings = self._extract_embeddings(audio_data, speaker_segments, sample_rate)
        
        # Step 4: Binary classification - doctor or patient
        speaker_mapping = self._classify_doctor_patient(segment_embeddings, doctor_id)
        
        # Step 5: Reconcile with previous results if available
        if self.previous_segments and self.previous_mapping:
            speaker_mapping = self._reconcile_speaker_labels(
                speaker_segments, speaker_mapping,
                self.previous_segments, self.previous_mapping
            )
        
        # Store current results for next iteration
        self.previous_segments = speaker_segments
        self.previous_mapping = speaker_mapping
        
        # Update speaker profiles with new embeddings
        self._update_speaker_profiles(segment_embeddings, speaker_mapping)
        
        # Calculate confidence scores
        confidence_scores = self._calculate_confidence_scores(segment_embeddings, speaker_mapping)
        
        # Format the results with absolute timestamps
        results = {
            "segments": [(float(start), float(end)) for start, end in speaker_segments],
            "speaker_mapping": {int(k): v for k, v in speaker_mapping.items()},
            "timestamped_segments": [(float(s), float(e), float(abs_s), float(abs_e)) 
                                    for s, e, abs_s, abs_e in timestamped_segments],
            "buffer_duration": float(buffer_duration),
            "processing_time": float(time.time()),
            "status": "success",
            "confidence_scores": {int(k): float(v) for k, v in confidence_scores.items()}
        }
        
        
        return results
    
    def _reconcile_speaker_labels(self, new_segments, new_mapping, prev_segments, prev_mapping):
        """
        Ensure consistent speaker labels between processing windows
        
        Args:
            new_segments: Current speaker segments
            new_mapping: Current segment to speaker mapping
            prev_segments: Previous speaker segments
            prev_mapping: Previous segment to speaker mapping
            
        Returns:
            Updated speaker mapping with reconciled labels
        """
        reconciled_mapping = new_mapping.copy()
        
        # Find overlapping segments between previous and current
        for i, (new_start, new_end) in enumerate(new_segments):
            # Skip if already mapped
            if i not in reconciled_mapping:
                continue
                
            # Convert to native Python types
            new_start, new_end = float(new_start), float(new_end)
                
            # Find overlapping previous segments
            overlaps = []
            for j, (prev_start, prev_end) in enumerate(prev_segments):
                if j not in prev_mapping:
                    continue
                    
                # Convert to native Python types
                prev_start, prev_end = float(prev_start), float(prev_end)
                    
                # Check for overlap
                if (new_start <= prev_end and new_end >= prev_start):
                    # Calculate overlap amount
                    overlap_start = max(new_start, prev_start)
                    overlap_end = min(new_end, prev_end)
                    overlap_amount = overlap_end - overlap_start
                    
                    if overlap_amount > 0.2:  # Minimum 0.2s overlap to consider
                        overlaps.append((j, prev_mapping[j], overlap_amount))
        
            # If we have overlaps, use majority voting weighted by overlap amount
            if overlaps:
                doctor_score = sum(overlap for _, speaker, overlap in overlaps 
                                  if speaker == "doctor")
                patient_score = sum(overlap for _, speaker, overlap in overlaps 
                                   if speaker == "patient")
                
                # Make decision based on weighted voting
                if doctor_score > patient_score:
                    reconciled_mapping[i] = "doctor"
                elif patient_score > doctor_score:
                    reconciled_mapping[i] = "patient"
                # If tied, keep current assignment
        
        return reconciled_mapping
    
    def _update_speaker_profiles(self, segment_embeddings, speaker_mapping):
        """
        Update speaker profiles based on the latest diarization results
        
        Args:
            segment_embeddings: List of (segment, embedding) tuples
            speaker_mapping: Mapping from segment index to speaker label
        """
        # Update profiles with new embeddings
        for i, (segment, embedding) in enumerate(segment_embeddings):
            speaker = speaker_mapping.get(i)
            if not speaker:
                continue
                
            # Add embedding to appropriate profile
            self.speaker_profiles[speaker].append(embedding)
            
            # Keep profiles to manageable size
            max_profile_size = 20
            if len(self.speaker_profiles[speaker]) > max_profile_size:
                self.speaker_profiles[speaker] = self.speaker_profiles[speaker][-max_profile_size:]
    
    def _calculate_confidence_scores(self, segment_embeddings, speaker_mapping):
        """
        Calculate confidence scores for speaker assignments
        
        Args:
            segment_embeddings: List of (segment, embedding) tuples
            speaker_mapping: Mapping from segment index to speaker label
            
        Returns:
            Dictionary mapping segment indices to confidence scores
        """
        confidence_scores = {}
        
        # Skip if we don't have speaker profiles yet
        if not self.speaker_profiles["doctor"] and not self.speaker_profiles["patient"]:
            return {i: 0.7 for i in speaker_mapping}  # Default medium confidence
        
        # Calculate average embeddings from profiles
        doctor_centroid = np.mean(self.speaker_profiles["doctor"], axis=0) if self.speaker_profiles["doctor"] else None
        patient_centroid = np.mean(self.speaker_profiles["patient"], axis=0) if self.speaker_profiles["patient"] else None
        
        # Calculate confidence for each segment
        for i, (segment, embedding) in enumerate(segment_embeddings):
            if i not in speaker_mapping:
                continue
                
            assigned_speaker = speaker_mapping[i]
            
            # Compare with profile centroids
            if assigned_speaker == "doctor" and doctor_centroid is not None:
                similarity = np.dot(embedding, doctor_centroid) / (
                    np.linalg.norm(embedding) * np.linalg.norm(doctor_centroid))
                confidence_scores[i] = min(1.0, max(0.5, similarity))
                
            elif assigned_speaker == "patient" and patient_centroid is not None:
                similarity = np.dot(embedding, patient_centroid) / (
                    np.linalg.norm(embedding) * np.linalg.norm(patient_centroid))
                confidence_scores[i] = min(1.0, max(0.5, similarity))
                
            else:
                confidence_scores[i] = 0.6  # Default confidence when no profile available
        
        return confidence_scores
    
    def _cluster_hierarchical(self, segment_embeddings):
        """Use hierarchical clustering for speaker segmentation"""
        from sklearn.cluster import AgglomerativeClustering

        # Extract embeddings matrix
        embeddings_matrix = np.array([emb for _, emb in segment_embeddings])
        
        # Apply hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=2,
            metric="cosine",
            linkage='complete'
        )
        labels = clustering.fit_predict(embeddings_matrix)
        
        # Map to segment indices
        segment_to_speaker = {}
        for i, label in enumerate(labels):
            segment_to_speaker[i] = "patient" if label == 0 else "doctor"
        
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
        """Extract x-vector embeddings for each segment"""
        segment_embeddings = []
        
        for start_sec, end_sec in segments:
            # Extract segment audio
            start_sample = int(start_sec * sample_rate)
            end_sample = int(end_sec * sample_rate)
            segment_audio = audio_data[start_sample:end_sample]
            
            # Extract x-vector (instead of MFCC)
            xvector = self.extract_xvector(segment_audio, sample_rate)
            if xvector is not None:
                segment_embeddings.append(((start_sec, end_sec), xvector))
        
        return segment_embeddings
    
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
            return self._cluster_hierarchical(segment_embeddings)
        
        # Get doctor's profile from calibration data
        doctor_profile = self.calibration_service.get_speaker_profile(doctor_id, self.db)
        if not doctor_profile:
            logger.warning(f"No calibration profile found for doctor (User ID: {doctor_id})")
            return self._cluster_hierarchical(segment_embeddings)
        
        # Extract doctor's mean vector
        doctor_mean = doctor_profile.get("mean_vector")
        if doctor_mean is None:
            logger.warning("Invalid doctor profile (missing mean vector)")
            return self._cluster_hierarchical(segment_embeddings)
        
        # Check for dimension mismatch
        sample_embedding_dim = segment_embeddings[0][1].shape[0] if segment_embeddings else 0
        doctor_mean_dim = doctor_mean.shape[0]
        
        if sample_embedding_dim != doctor_mean_dim:
            logger.info(f"Dimension mismatch: x-vectors ({sample_embedding_dim}) vs profile ({doctor_mean_dim})")
            
            # Handle dimension mismatch using simple dimension matching
            try:
                # Use a direct dimension matching approach instead of PCA
                if sample_embedding_dim > doctor_mean_dim:
                    # Truncate x-vectors to match doctor mean dimension
                    segment_embeddings = [(seg, emb[:doctor_mean_dim]) for seg, emb in segment_embeddings]
                    logger.info(f"Truncated x-vectors to match doctor profile dimension: {doctor_mean_dim}")
                else:
                    # Truncate doctor mean to match x-vector dimension
                    doctor_mean = doctor_mean[:sample_embedding_dim]
                    logger.info(f"Truncated doctor profile to match x-vector dimension: {sample_embedding_dim}")
            except Exception as e:
                logger.error(f"Error handling dimension mismatch: {str(e)}")
                return self._cluster_hierarchical(segment_embeddings)

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