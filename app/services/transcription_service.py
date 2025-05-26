import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor

from app.services.audio_service import AudioService
from app.utils.speech_processor import SpeechProcessor, TranscriptionConfig, TranscriptionResult, WordTiming
from app.utils.text_utils import clean_transcription, correct_spelling
from app.utils.nlp.spacy_utils import process_text, find_medical_modifiers

# Configure logger
logger = logging.getLogger(__name__)

@dataclass
class TranscriptionSegmentData:
    """Data for a transcription segment"""
    text: str
    start_time: float
    end_time: float
    speaker_id: Optional[str] = None
    confidence: float = 1.0
    words: List[WordTiming] = None

class TranscriptionService:
    """
    Refactored service for managing audio transcription with medical focus.
    Supports word-level timing and real-time streaming.
    """
    
    def __init__(self, 
                 audio_service: AudioService = None,
                 speech_processor: SpeechProcessor = None,
                 config: TranscriptionConfig = None):
        """
        Initialize transcription service
        
        Args:
            audio_service: Audio service instance
            speech_processor: Speech processor instance
            config: Transcription configuration
        """
        self.audio_service = audio_service or AudioService()
        self.config = config or TranscriptionConfig()
        self.speech_processor = speech_processor or SpeechProcessor(self.config)
        
        # Transcript state management
        self.reset()
        
        # Thread pool for async processing
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        logger.info("TranscriptionService initialized with word-level timing support")
    
    def reset(self):
        """Reset transcription state"""
        self.full_transcript = ""
        self.transcript_segments = []
        self.word_timings = []
        self.current_offset = 0.0  # Time offset for continuous streaming
        
    def process_audio_segment(self, 
                            user_id: int, 
                            note_id: int,
                            use_adaptation: bool = False,
                            adaptation_user_id: Optional[int] = None,
                            db_session = None,
                            return_timing: bool = True) -> Optional[Dict[str, Any]]:
        """
        Process current audio buffer and transcribe with timing
        
        Args:
            user_id: User ID for the transcription
            note_id: Note ID for the transcription
            use_adaptation: Whether to use speaker adaptation
            adaptation_user_id: User ID for adaptation profile
            db_session: Database session
            return_timing: Whether to return word-level timing
            
        Returns:
            Dictionary with transcription data (JSON-serializable) or None
        """
        try:
            # Check for minimum audio
            if not self.audio_service.has_minimum_audio():
                return None
            
            # Check for silence
            if not self.audio_service.detect_silence():
                return None
            
            logger.info(f"Processing segment for user {user_id}, note {note_id}")
            
            # Get audio data
            audio_samples = self.audio_service.get_wave_data()
            
            # Calculate time offset for this segment
            segment_start_time = self.current_offset
            
            # Transcribe with appropriate method
            if use_adaptation and adaptation_user_id is not None:
                result = self.speech_processor.transcribe_with_adaptation(
                    audio_samples,
                    adaptation_user_id,
                    db_session,
                    with_timing=return_timing
                )
            else:
                result = self.speech_processor.transcribe(
                    audio_samples,
                    with_timing=return_timing
                )
            
            # Handle both string and TranscriptionResult returns
            if isinstance(result, str):
                # Backward compatibility - no timing
                if not result:
                    return None
                
                segment_data = {
                    "text": result,
                    "start_time": segment_start_time,
                    "end_time": segment_start_time + (len(audio_samples) / 16000),
                    "confidence": 1.0,
                    "words": []
                }
            else:
                # New format with timing
                if not result.text:
                    return None
                
                # Convert to JSON-serializable format
                words_data = []
                for word in result.words:
                    words_data.append({
                        "word": word.word,
                        "start_time": word.start_time + segment_start_time,
                        "end_time": word.end_time + segment_start_time,
                        "confidence": word.confidence
                    })
                
                segment_data = {
                    "text": result.text,
                    "start_time": segment_start_time,
                    "end_time": segment_start_time + result.duration,
                    "confidence": result.confidence,
                    "words": words_data
                }
                
                # Store WordTiming objects internally
                for word in result.words:
                    adjusted_word = WordTiming(
                        word=word.word,
                        start_time=word.start_time + segment_start_time,
                        end_time=word.end_time + segment_start_time,
                        confidence=word.confidence,
                        speaker_id=word.speaker_id
                    )
                    self.word_timings.append(adjusted_word)
            
            # Update transcript state
            self._update_transcript_state_from_dict(segment_data)
            
            # Update time offset for next segment
            self.current_offset = segment_data["end_time"]
            
            # Reset audio buffer for next segment
            self.audio_service.reset_current_buffer()
            
            return segment_data
            
        except Exception as e:
            logger.error(f"Error processing audio segment: {str(e)}")
            return None
    
    def _update_transcript_state_from_dict(self, segment_data: Dict[str, Any]):
        """Update internal transcript state from dictionary data"""
        # Update full transcript
        if self.full_transcript:
            self.full_transcript += " " + segment_data["text"]
        else:
            self.full_transcript = segment_data["text"]
        
        # Create TranscriptionSegmentData object for internal storage
        segment = TranscriptionSegmentData(
            text=segment_data["text"],
            start_time=segment_data["start_time"],
            end_time=segment_data["end_time"],
            confidence=segment_data.get("confidence", 1.0)
        )
        self.transcript_segments.append(segment)
        
        logger.debug(f"Added segment: {segment_data['text'][:50]}...")
    
    def _update_transcript_state(self, segment: TranscriptionSegmentData):
        """Update internal transcript state with new segment"""
        # Update full transcript
        if self.full_transcript:
            self.full_transcript += " " + segment.text
        else:
            self.full_transcript = segment.text
        
        # Add to segments
        self.transcript_segments.append(segment)
        
        logger.debug(f"Added segment: {segment.text[:50]}...")
    
    def extract_keywords(self, use_word_timing: bool = True) -> List[Dict[str, Any]]:
        """
        Extract keywords from transcript with optional word timing
        
        Args:
            use_word_timing: Whether to include timing information
            
        Returns:
            List of keyword dictionaries with timing if available
        """
        if not self.full_transcript:
            return []
        
        try:
            # Process with NLP
            doc = process_text(self.full_transcript)
            
            # Extract keywords
            keywords = find_medical_modifiers(doc=doc)
            
            # Enhance with timing if available
            if use_word_timing and self.word_timings:
                keywords = self._add_timing_to_keywords(keywords)
            
            logger.info(f"Extracted {len(keywords)} keywords from transcript")
            return keywords
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            return []
    
    def _add_timing_to_keywords(self, keywords: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add timing information to extracted keywords"""
        # Create a mapping of words to timing
        word_timing_map = {}
        for word_timing in self.word_timings:
            word_lower = word_timing.word.lower()
            if word_lower not in word_timing_map:
                word_timing_map[word_lower] = []
            word_timing_map[word_lower].append(word_timing)
        
        # Enhance keywords with timing
        for keyword in keywords:
            term = keyword.get("term", "").lower()
            
            # Find timing for the term
            if term in word_timing_map:
                timings = word_timing_map[term]
                if timings:
                    # Use the first occurrence
                    keyword["start_time"] = timings[0].start_time
                    keyword["end_time"] = timings[0].end_time
                    keyword["confidence"] = timings[0].confidence
            
            # Add timing for modifiers
            for modifier_type in ["modifiers", "quantities", "temporal", "locations"]:
                if modifier_type in keyword:
                    enhanced_modifiers = []
                    for modifier in keyword[modifier_type]:
                        modifier_lower = modifier.lower()
                        if modifier_lower in word_timing_map:
                            timings = word_timing_map[modifier_lower]
                            if timings:
                                enhanced_modifiers.append({
                                    "text": modifier,
                                    "start_time": timings[0].start_time,
                                    "end_time": timings[0].end_time
                                })
                        else:
                            enhanced_modifiers.append({"text": modifier})
                    keyword[f"{modifier_type}_with_timing"] = enhanced_modifiers
        
        return keywords
    
    def get_current_transcript(self, include_timing: bool = True) -> Dict[str, Any]:
        """
        Get current transcription state with optional timing
        
        Args:
            include_timing: Whether to include word timing
            
        Returns:
            Dictionary with transcript information
        """
        response = {
            "text": self.full_transcript,
            "segments": [
                {
                    "text": seg.text,
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                    "confidence": seg.confidence,
                    "speaker_id": seg.speaker_id
                }
                for seg in self.transcript_segments
            ],
            "segment_count": len(self.transcript_segments),
            "word_count": len(self.full_transcript.split()) if self.full_transcript else 0,
            "duration": self.current_offset
        }
        
        if include_timing and self.word_timings:
            response["words"] = [
                {
                    "word": w.word,
                    "start_time": w.start_time,
                    "end_time": w.end_time,
                    "confidence": w.confidence
                }
                for w in self.word_timings
            ]
            response["word_timing_count"] = len(self.word_timings)
        
        return response
    
    def transcribe_doctor_patient(self, 
                                audio_samples: np.ndarray,
                                diarization_results: Dict[str, Any],
                                doctor_id: Optional[int] = None,
                                db_session = None) -> Dict[str, Any]:
        """
        Transcribe audio with doctor-patient diarization and word-level timing
        
        Args:
            audio_samples: Audio data as numpy array
            diarization_results: Results from DiarizationService
            doctor_id: User ID of the doctor (for adaptation)
            db_session: Database session
            
        Returns:
            Transcription results with doctor/patient labels and timing
        """
        segments = diarization_results["segments"]
        speaker_mapping = diarization_results["speaker_mapping"]
        sample_rate = 16000  # Default sample rate
        
        # Store transcriptions by role
        doctor_segments = []
        patient_segments = []
        all_words = []
        
        # Process each segment
        for i, (start_sec, end_sec) in enumerate(segments):
            # Get speaker role
            role = speaker_mapping.get(i)
            if not role:
                continue
            
            # Extract segment audio
            start_sample = int(start_sec * sample_rate)
            end_sample = int(end_sec * sample_rate)
            segment_audio = audio_samples[start_sample:end_sample]
            
            # Skip very short segments
            if len(segment_audio) < 0.5 * sample_rate:
                continue
            
            # Determine if we should use adaptation
            use_adaptation = (role == "doctor" and doctor_id is not None)
            
            # Transcribe segment with timing
            if use_adaptation:
                result = self.speech_processor.transcribe_with_adaptation(
                    segment_audio, doctor_id, db_session, with_timing=True
                )
            else:
                result = self.speech_processor.transcribe(segment_audio, with_timing=True)
            
            # Handle both string and TranscriptionResult
            if isinstance(result, str):
                # Backward compatibility
                segment_data = {
                    "start": start_sec,
                    "end": end_sec,
                    "text": result,
                    "speaker": role,
                    "words": []
                }
            else:
                # Adjust word timings to absolute position
                adjusted_words = []
                for word in result.words:
                    adjusted_word = {
                        "word": word.word,
                        "start_time": word.start_time + start_sec,
                        "end_time": word.end_time + start_sec,
                        "confidence": word.confidence,
                        "speaker": role
                    }
                    adjusted_words.append(adjusted_word)
                    all_words.append(adjusted_word)
                
                segment_data = {
                    "start": start_sec,
                    "end": end_sec,
                    "text": result.text,
                    "speaker": role,
                    "confidence": result.confidence,
                    "words": adjusted_words
                }
            
            # Store in appropriate list
            if role == "doctor" or role == "speaker1":
                doctor_segments.append(segment_data)
            else:
                patient_segments.append(segment_data)
        
        # Sort all words by time
        all_words.sort(key=lambda w: w["start_time"])
        
        # Format results
        return {
            "doctor_segments": doctor_segments,
            "patient_segments": patient_segments,
            "all_words": all_words,
            "full_transcript": self._format_doctor_patient_transcript(doctor_segments, patient_segments)
        }
    
    def _format_doctor_patient_transcript(self, doctor_segments, patient_segments):
        """Format transcript with speaker labels and timing"""
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
            timestamp = f"[{segment['start']:.1f}-{segment['end']:.1f}s]"
            lines.append(f"{timestamp} {segment['role']}: {segment['text']}")
        
        return "\n".join(lines)
    
    async def process_audio_stream_async(self,
                                       audio_chunk: bytes,
                                       user_id: int,
                                       note_id: int,
                                       use_adaptation: bool = False,
                                       adaptation_user_id: Optional[int] = None) -> Optional[TranscriptionSegmentData]:
        """
        Async wrapper for processing audio streams
        
        Args:
            audio_chunk: Audio chunk bytes
            user_id: User ID
            note_id: Note ID
            use_adaptation: Whether to use speaker adaptation
            adaptation_user_id: User ID for adaptation
            
        Returns:
            TranscriptionSegmentData or None
        """
        # Add chunk to buffer
        if not self.audio_service.add_chunk(audio_chunk):
            logger.warning("Audio buffer full")
            return None
        
        # Process in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self.process_audio_segment,
            user_id,
            note_id,
            use_adaptation,
            adaptation_user_id,
            None,  # db_session
            True   # return_timing
        )
        
        return result
    
    def get_word_at_time(self, timestamp: float) -> Optional[Dict[str, Any]]:
        """
        Get the word being spoken at a specific timestamp
        
        Args:
            timestamp: Time in seconds
            
        Returns:
            Word information or None
        """
        for word in self.word_timings:
            if word.start_time <= timestamp <= word.end_time:
                return {
                    "word": word.word,
                    "start_time": word.start_time,
                    "end_time": word.end_time,
                    "confidence": word.confidence,
                    "speaker_id": word.speaker_id
                }
        return None
    
    def get_words_in_range(self, start_time: float, end_time: float) -> List[Dict[str, Any]]:
        """
        Get all words within a time range
        
        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            List of words in the range
        """
        words_in_range = []
        for word in self.word_timings:
            if word.start_time >= start_time and word.end_time <= end_time:
                words_in_range.append({
                    "word": word.word,
                    "start_time": word.start_time,
                    "end_time": word.end_time,
                    "confidence": word.confidence,
                    "speaker_id": word.speaker_id
                })
        return words_in_range
    
    def merge_continuous_segments(self, max_gap: float = 0.5) -> List[TranscriptionSegmentData]:
        """
        Merge segments that are close in time
        
        Args:
            max_gap: Maximum gap in seconds to merge segments
            
        Returns:
            List of merged segments
        """
        if not self.transcript_segments:
            return []
        
        merged_segments = []
        current_segment = None
        
        for segment in self.transcript_segments:
            if current_segment is None:
                current_segment = TranscriptionSegmentData(
                    text=segment.text,
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    confidence=segment.confidence,
                    words=segment.words if hasattr(segment, 'words') else []
                )
            elif segment.start_time - current_segment.end_time <= max_gap:
                # Merge segments
                current_segment.text += " " + segment.text
                current_segment.end_time = segment.end_time
                if hasattr(segment, 'words') and segment.words:
                    current_segment.words.extend(segment.words)
            else:
                # Gap too large, start new segment
                merged_segments.append(current_segment)
                current_segment = TranscriptionSegmentData(
                    text=segment.text,
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    confidence=segment.confidence,
                    words=segment.words if hasattr(segment, 'words') else []
                )
        
        if current_segment:
            merged_segments.append(current_segment)
        
        return merged_segments
    
    def export_transcript(self, format: str = "json") -> Union[str, Dict[str, Any]]:
        """
        Export transcript in various formats
        
        Args:
            format: Export format ('json', 'srt', 'vtt', 'text')
            
        Returns:
            Formatted transcript
        """
        if format == "json":
            return {
                "text": self.full_transcript,
                "segments": [
                    {
                        "text": seg.text,
                        "start_time": seg.start_time,
                        "end_time": seg.end_time,
                        "confidence": seg.confidence
                    }
                    for seg in self.transcript_segments
                ],
                "words": [
                    {
                        "word": w.word,
                        "start_time": w.start_time,
                        "end_time": w.end_time,
                        "confidence": w.confidence
                    }
                    for w in self.word_timings
                ]
            }
        
        elif format == "srt":
            lines = []
            for i, segment in enumerate(self.transcript_segments, 1):
                start = self._format_timestamp_srt(segment.start_time)
                end = self._format_timestamp_srt(segment.end_time)
                lines.append(f"{i}")
                lines.append(f"{start} --> {end}")
                lines.append(segment.text)
                lines.append("")
            return "\n".join(lines)
        
        elif format == "vtt":
            lines = ["WEBVTT", ""]
            for segment in self.transcript_segments:
                start = self._format_timestamp_vtt(segment.start_time)
                end = self._format_timestamp_vtt(segment.end_time)
                lines.append(f"{start} --> {end}")
                lines.append(segment.text)
                lines.append("")
            return "\n".join(lines)
        
        else:  # text
            return self.full_transcript
    
    def _format_timestamp_srt(self, seconds: float) -> str:
        """Format timestamp for SRT format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    def _format_timestamp_vtt(self, seconds: float) -> str:
        """Format timestamp for WebVTT format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get transcription statistics"""
        total_words = len(self.word_timings)
        
        # Calculate confidence statistics
        if self.word_timings:
            confidences = [w.confidence for w in self.word_timings]
            avg_confidence = sum(confidences) / len(confidences)
            min_confidence = min(confidences)
            max_confidence = max(confidences)
        else:
            avg_confidence = min_confidence = max_confidence = 0.0
        
        # Calculate speaking rate
        if self.current_offset > 0:
            words_per_minute = (total_words / self.current_offset) * 60
        else:
            words_per_minute = 0.0
        
        return {
            "total_duration": self.current_offset,
            "total_words": total_words,
            "total_segments": len(self.transcript_segments),
            "words_per_minute": words_per_minute,
            "average_confidence": avg_confidence,
            "min_confidence": min_confidence,
            "max_confidence": max_confidence,
            "text_length": len(self.full_transcript)
        }
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)