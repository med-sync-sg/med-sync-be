import pytest
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

@pytest.mark.integration
@pytest.mark.requires_audio
class TestAudioToTranscription:
    """Integration tests for audio processing to transcription pipeline"""
    
    def test_basic_audio_loading_and_transcription(self, sample_audio_data, transcription_service):
        """Test basic audio loading and transcription"""
        # Load audio data
        audio_data = sample_audio_data["audio_data"]
        sample_rate = sample_audio_data["sample_rate"]
        
        # Transcribe
        result = transcription_service.speech_processor.transcribe(
            audio_data, sample_rate, with_timing=True
        )
        
        # Validate result
        assert result is not None
        if hasattr(result, 'text'):
            assert len(result.text.strip()) > 0
            assert result.duration > 0
        else:
            # Backward compatibility - string result
            assert len(result.strip()) > 0
    
    def test_short_audio_segment_transcription(self, short_audio_segment, transcription_service):
        """Test transcription of short audio segment"""
        audio_data = short_audio_segment["audio_data"]
        sample_rate = short_audio_segment["sample_rate"]
        expected_duration = short_audio_segment["duration"]
        
        start_time = time.time()
        result = transcription_service.speech_processor.transcribe(
            audio_data, sample_rate, with_timing=True
        )
        processing_time = time.time() - start_time
        
        # Validate timing
        assert processing_time < expected_duration * 3  # Should be reasonable processing time
        
        # Validate result structure
        if hasattr(result, 'text'):
            assert result.text is not None
            assert abs(result.duration - expected_duration) < 0.5  # Within 0.5 seconds
        
    def test_variable_length_audio_transcription(self, variable_length_audio, transcription_service):
        """Test transcription with different audio lengths"""
        audio_data = variable_length_audio["audio_data"]
        sample_rate = variable_length_audio["sample_rate"]
        expected_duration = variable_length_audio["duration"]
        label = variable_length_audio["label"]
        
        result = transcription_service.speech_processor.transcribe(
            audio_data, sample_rate, with_timing=True
        )
        
        # Validate based on length
        if label == "1sec":
            # Very short audio might not produce reliable results
            assert result is not None
        elif label in ["3sec", "5sec"]:
            # Longer audio should produce meaningful transcription
            if hasattr(result, 'text'):
                assert len(result.text.strip()) > 0
                assert result.duration > 0
            else:
                assert len(result.strip()) > 0
    
    def test_audio_service_integration(self, audio_service, pcm_byte_chunks):
        """Test audio service with real PCM byte chunks"""
        # Add chunks to audio service
        total_chunks = 0
        for chunk in pcm_byte_chunks[:5]:  # Use first 5 chunks
            success = audio_service.add_chunk(chunk)
            assert success
            total_chunks += 1
        
        # Verify audio service state
        assert audio_service.has_minimum_audio()
        
        # Get audio statistics
        stats = audio_service.get_audio_statistics()
        assert stats["sample_count"] > 0
        assert stats["duration_ms"] > 0
        assert stats["current_buffer_size"] > 0
        
        # Get wave data
        wave_data = audio_service.get_wave_data()
        assert isinstance(wave_data, np.ndarray)
        assert len(wave_data) > 0
    
    @pytest.mark.slow
    def test_streaming_audio_transcription(self, transcription_service, pcm_byte_chunks):
        """Test streaming audio transcription with chunked data"""
        transcribed_segments = []
        
        # Process chunks as they would arrive in streaming
        for i, chunk in enumerate(pcm_byte_chunks[:10]):  # Limit to 10 chunks for test speed
            # Add chunk to audio service
            transcription_service.audio_service.add_chunk(chunk)
            
            # Try to process if we have enough audio
            if transcription_service.audio_service.has_minimum_audio():
                segment_result = transcription_service.process_audio_segment(
                    user_id=1,
                    note_id=1,
                    use_adaptation=False,
                    return_timing=True
                )
                
                if segment_result:
                    transcribed_segments.append(segment_result)
        
        # Validate streaming results
        assert len(transcribed_segments) > 0
        
        # Check segment structure
        for segment in transcribed_segments:
            assert "text" in segment
            assert "start_time" in segment
            assert "end_time" in segment
            assert "confidence" in segment
            assert segment["end_time"] > segment["start_time"]
    
    def test_transcription_with_word_timing(self, short_audio_segment, transcription_service, 
                                          audio_validation_helpers):
        """Test transcription with word-level timing"""
        audio_data = short_audio_segment["audio_data"]
        sample_rate = short_audio_segment["sample_rate"]
        duration = short_audio_segment["duration"]
        
        # Transcribe with timing
        result = transcription_service.speech_processor.transcribe(
            audio_data, sample_rate, with_timing=True
        )
        
        # Validate word timing if available
        if hasattr(result, 'words') and result.words:
            # Convert to dict format for validation
            words_dict = [
                {
                    "word": w.word,
                    "start_time": w.start_time,
                    "end_time": w.end_time,
                    "confidence": w.confidence
                }
                for w in result.words
            ]
            
            # Validate timing consistency
            assert audio_validation_helpers.validate_audio_timing(words_dict, duration)
            
            # Check that words are within the audio duration
            for word in words_dict:
                assert 0 <= word["start_time"] <= duration
                assert 0 <= word["end_time"] <= duration
    
    def test_transcription_quality_metrics(self, consultation_audio_data, transcription_service,
                                         expected_transcriptions, transcription_benchmark):
        """Test transcription quality metrics"""
        audio_data = consultation_audio_data["audio_data"]
        sample_rate = consultation_audio_data["sample_rate"]
        filename = consultation_audio_data["file_path"].name
        
        # Get expected results if available
        expected = expected_transcriptions.get(filename, {})
        
        start_time = time.time()
        result = transcription_service.speech_processor.transcribe(
            audio_data, sample_rate, with_timing=True
        )
        processing_time = time.time() - start_time
        
        # Performance validation
        audio_duration = len(audio_data) / sample_rate
        max_processing_time = audio_duration * transcription_benchmark["max_processing_time_per_second"]
        assert processing_time < max_processing_time, f"Processing too slow: {processing_time:.2f}s for {audio_duration:.2f}s audio"
        
        # Quality validation
        if hasattr(result, 'text'):
            transcription_text = result.text
            confidence = result.confidence
        else:
            transcription_text = result
            confidence = 1.0  # Default for string results
        
        # Basic quality checks
        assert len(transcription_text.strip()) > 0
        
        if "min_word_count" in expected:
            word_count = len(transcription_text.split())
            assert word_count >= expected["min_word_count"], f"Too few words: {word_count} < {expected['min_word_count']}"
        
        # Confidence check
        min_confidence = transcription_benchmark["min_confidence_threshold"]
        assert confidence >= min_confidence, f"Confidence too low: {confidence} < {min_confidence}"
    
    @pytest.mark.slow
    def test_multiple_audio_files_batch(self, multiple_audio_files, transcription_service):
        """Test transcription on multiple audio files"""
        results = []
        
        for audio_data_info in multiple_audio_files:
            audio_data = audio_data_info["audio_data"]
            sample_rate = audio_data_info["sample_rate"]
            filename = audio_data_info["file_path"].name
            
            try:
                result = transcription_service.speech_processor.transcribe(
                    audio_data, sample_rate, with_timing=True
                )
                
                results.append({
                    "filename": filename,
                    "success": True,
                    "result": result,
                    "duration": audio_data_info["duration"]
                })
            except Exception as e:
                results.append({
                    "filename": filename,
                    "success": False,
                    "error": str(e),
                    "duration": audio_data_info["duration"]
                })
        
        # Validate batch results
        successful_results = [r for r in results if r["success"]]
        assert len(successful_results) > 0, "No files transcribed successfully"
        
        # Check success rate
        success_rate = len(successful_results) / len(results)
        assert success_rate >= 0.8, f"Success rate too low: {success_rate:.2f}"
    
    def test_transcription_service_state_management(self, transcription_service, audio_chunks_1sec, audio_loader):
        """Test transcription service state management across multiple calls"""
        # Reset service
        transcription_service.reset()
        assert transcription_service.full_transcript == ""
        assert len(transcription_service.transcript_segments) == 0
        
        # Process multiple chunks
        for i, chunk in enumerate(audio_chunks_1sec[:3]):
            pcm_bytes = audio_loader.audio_to_pcm_bytes(chunk)
            transcription_service.audio_service.add_chunk(pcm_bytes)
            
            if transcription_service.audio_service.has_minimum_audio():
                segment_result = transcription_service.process_audio_segment(
                    user_id=1,
                    note_id=1,
                    use_adaptation=False
                )
                
                if segment_result:
                    # Check that state is being updated
                    assert len(transcription_service.full_transcript) > 0
                    assert len(transcription_service.transcript_segments) == i + 1
        
        # Get current transcript
        current_state = transcription_service.get_current_transcript()
        assert "text" in current_state
        assert "segments" in current_state
        assert current_state["segment_count"] > 0
    
    def test_transcription_error_handling(self, transcription_service):
        """Test transcription error handling with invalid input"""
        # Test with empty audio
        empty_audio = np.array([], dtype=np.float32)
        result = transcription_service.speech_processor.transcribe(empty_audio)
        
        # Should handle gracefully
        if hasattr(result, 'text'):
            assert result.text == ""
        else:
            assert result == ""
        
        # Test with very short audio
        very_short_audio = np.random.rand(100).astype(np.float32)  # ~6ms at 16kHz
        result = transcription_service.speech_processor.transcribe(very_short_audio)
        # Should not crash
        assert result is not None
    
    def test_audio_service_buffer_limits(self, audio_service):
        """Test audio service buffer management and limits"""
        # Generate test data
        test_chunk = np.random.rand(1000).astype(np.int16).tobytes()
        
        # Test normal operation
        initial_success = audio_service.add_chunk(test_chunk)
        assert initial_success
        
        # Test buffer statistics
        stats = audio_service.get_audio_statistics()
        assert stats["current_buffer_size"] > 0
        
        # Test buffer reset
        audio_service.reset_current_buffer()
        stats_after_reset = audio_service.get_audio_statistics()
        assert stats_after_reset["current_buffer_size"] == 0
        
        # Test session buffer persistence
        audio_service.add_chunk(test_chunk)
        audio_service.reset_current_buffer()
        # Session buffer should still have data
        assert stats_after_reset["session_buffer_size"] > 0
    
    def test_silence_detection_integration(self, audio_service, transcription_service, audio_loader):
        """Test silence detection with real audio service"""
        # Add some audio data
        test_audio = np.random.rand(16000).astype(np.float32) * 0.001  # Very quiet
        pcm_bytes = audio_loader.audio_to_pcm_bytes(test_audio)
        
        audio_service.add_chunk(pcm_bytes)
        
        # Test silence detection
        is_silent = audio_service.detect_silence()
        # Should detect as silence due to low amplitude
        assert is_silent
        
        # Add louder audio
        loud_audio = np.random.rand(16000).astype(np.float32) * 0.5
        loud_pcm = audio_loader.audio_to_pcm_bytes(loud_audio)
        audio_service.add_chunk(loud_pcm)
        
        # Should not be silent now
        is_silent_after = audio_service.detect_silence()
        assert not is_silent_after