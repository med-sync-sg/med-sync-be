import pytest
import time
import json
from pathlib import Path
from typing import Dict, Any, List
import asyncio
import numpy as np

# Import your app modules
from app.utils.speech_processor import SpeechProcessor

@pytest.mark.e2e
@pytest.mark.requires_audio
class TestCompleteAudioTranscriptionWorkflow:
    """End-to-end tests for complete audio transcription workflow"""
    
    def test_complete_audio_to_text_pipeline(self, consultation_audio_data, transcription_service, 
                                           debug_dir):
        """Test complete pipeline from audio file to final transcription"""
        audio_data = consultation_audio_data["audio_data"]
        sample_rate = consultation_audio_data["sample_rate"]
        file_path = consultation_audio_data["file_path"]
        
        # Reset service state
        transcription_service.reset()
        
        # Record start time for performance tracking
        start_time = time.time()
        
        # Process the entire audio file
        result = transcription_service.speech_processor.transcribe(
            audio_data, sample_rate, with_timing=True
        )
        
        processing_time = time.time() - start_time
        audio_duration = len(audio_data) / sample_rate
        
        # Validate result exists and has content
        if hasattr(result, 'text'):
            transcription_text = result.text
            word_count = len(result.words) if hasattr(result, 'words') else 0
            confidence = result.confidence
        else:
            transcription_text = result
            word_count = len(result.split())
            confidence = 1.0
        
        # Basic validation
        assert len(transcription_text.strip()) > 0, "No transcription generated"
        
        # Performance validation
        real_time_factor = processing_time / audio_duration
        
        # Save results for debugging
        result_data = {
            "input_file": str(file_path),
            "audio_duration_seconds": audio_duration,
            "processing_time_seconds": processing_time,
            "real_time_factor": real_time_factor,
            "transcription": transcription_text,
            "word_count": word_count,
            "confidence": confidence,
            "timestamp": time.time()
        }
        
        # Save to debug directory
        result_file = debug_dir / f"e2e_transcription_{file_path.stem}.json"
        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        print(f"\nE2E Transcription Results:")
        print(f"File: {file_path.name}")
        print(f"Duration: {audio_duration:.2f}s")
        print(f"Processing time: {processing_time:.2f}s ({real_time_factor:.2f}x)")
        print(f"Words: {word_count}")
        print(f"Confidence: {confidence:.3f}")
        print(f"Transcription: {transcription_text[:100]}...")
    
    def test_streaming_workflow_simulation(self, streaming_audio_chunks, transcription_service,
                                         debug_dir, audio_loader):
        """Test streaming workflow with chunked audio processing"""
        transcription_service.reset()
        
        streaming_results = []
        total_processing_time = 0
        
        # Simulate streaming by processing chunks sequentially
        for i, chunk in enumerate(streaming_audio_chunks):  # Limit for test speed
            chunk_start_time = time.time()
            
            # Convert chunk to PCM bytes
            pcm_bytes = audio_loader.audio_to_pcm_bytes(chunk)
            
            # Add to audio service
            success = transcription_service.audio_service.add_chunk(pcm_bytes)
            
            segment_result = transcription_service.process_audio_segment(
                user_id=1,
                note_id=1,
                use_adaptation=False,
                return_timing=True
            )
            
            if segment_result is not None:
                chunk_processing_time = time.time() - chunk_start_time
                total_processing_time += chunk_processing_time
                
                streaming_results.append({
                    "chunk_index": i,
                    "text": segment_result["text"],
                    "start_time": segment_result["start_time"],
                    "end_time": segment_result["end_time"],
                    "confidence": segment_result["confidence"],
                    "processing_time": chunk_processing_time,
                    "words": segment_result.get("words", [])
                })
        
        # Validate streaming results
        assert len(streaming_results) > 0, "No streaming segments processed"
        
        # Check segment continuity
        for i in range(1, len(streaming_results)):
            prev_segment = streaming_results[i-1]
            curr_segment = streaming_results[i]
            
            # Segments should be roughly continuous
            time_gap = curr_segment["start_time"] - prev_segment["end_time"]
            assert time_gap >= -0.5, f"Segments overlap too much: {time_gap:.2f}s"
            assert time_gap <= 2.0, f"Gap too large between segments: {time_gap:.2f}s"
        
        # Get final transcript
        final_transcript = transcription_service.get_current_transcript(include_timing=True)
        
        # Save streaming results
        streaming_data = {
            "total_chunks_processed": len(streaming_results),
            "total_processing_time": total_processing_time,
            "segments": streaming_results,
            "final_transcript": final_transcript,
            "timestamp": time.time()
        }
        
        result_file = debug_dir / "e2e_streaming_workflow.json"
        with open(result_file, 'w') as f:
            json.dump(streaming_data, f, indent=2)
        
        print(f"\nStreaming Workflow Results:")
        print(f"Segments processed: {len(streaming_results)}")
        print(f"Total processing time: {total_processing_time:.2f}s")
        print(f"Final transcript length: {len(final_transcript['text'])} characters")
    
    def test_audio_quality_impact_on_transcription(self, multiple_audio_files, transcription_service,
                                                 debug_dir):
        """Test how different audio qualities affect transcription"""
        quality_results = []
        
        for audio_info in multiple_audio_files:
            file_path = audio_info["file_path"]
            audio_data = audio_info["audio_data"]
            sample_rate = audio_info["sample_rate"]
            duration = audio_info["duration"]
            
            # Calculate audio quality metrics
            rms_level = float(np.sqrt(np.mean(audio_data ** 2)))
            peak_level = float(np.max(np.abs(audio_data)))
            dynamic_range = peak_level / (rms_level + 1e-10)
            
            start_time = time.time()
            try:
                result = transcription_service.speech_processor.transcribe(
                    audio_data, sample_rate, with_timing=True
                )
                processing_time = time.time() - start_time
                
                if hasattr(result, 'text'):
                    transcription = result.text
                    confidence = result.confidence
                    word_count = len(result.words) if hasattr(result, 'words') else 0
                else:
                    transcription = result
                    confidence = 1.0
                    word_count = len(result.split())
                
                quality_results.append({
                    "filename": file_path.name,
                    "success": True,
                    "duration": duration,
                    "rms_level": rms_level,
                    "peak_level": peak_level,
                    "dynamic_range": dynamic_range,
                    "processing_time": processing_time,
                    "transcription_length": len(transcription),
                    "word_count": word_count,
                    "confidence": confidence,
                    "words_per_second": word_count / duration if duration > 0 else 0
                })
                
            except Exception as e:
                quality_results.append({
                    "filename": file_path.name,
                    "success": False,
                    "error": str(e),
                    "duration": duration,
                    "rms_level": rms_level,
                    "peak_level": peak_level,
                    "dynamic_range": dynamic_range
                })
        
        # Analyze results
        successful_results = [r for r in quality_results if r["success"]]
        assert len(successful_results) > 0, "No files processed successfully"
        
        # Save quality analysis
        quality_analysis = {
            "total_files": len(quality_results),
            "successful_files": len(successful_results),
            "success_rate": len(successful_results) / len(quality_results),
            "results": quality_results,
            "timestamp": time.time()
        }
        
        result_file = debug_dir / "e2e_quality_analysis.json"
        with open(result_file, 'w') as f:
            json.dump(quality_analysis, f, indent=2)
        
        print(f"\nAudio Quality Analysis:")
        print(f"Files processed: {len(quality_results)}")
        print(f"Success rate: {quality_analysis['success_rate']:.2%}")
        
        # Quality correlation analysis
        if len(successful_results) > 1:
            avg_confidence = sum(r["confidence"] for r in successful_results) / len(successful_results)
            print(f"Average confidence: {avg_confidence:.3f}")
    
    def test_memory_usage_during_long_transcription(self, consultation_audio_data, transcription_service):
        """Test memory usage during extended transcription"""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        audio_data = consultation_audio_data["audio_data"]
        sample_rate = consultation_audio_data["sample_rate"]
        
        # Process the audio multiple times to simulate extended usage
        memory_measurements = [initial_memory]
        
        for iteration in range(3):  # Process same audio 3 times
            transcription_service.reset()
            
            result = transcription_service.speech_processor.transcribe(
                audio_data, sample_rate, with_timing=True
            )
            
            # Measure memory after each iteration
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_measurements.append(current_memory)
        
        # Analyze memory usage
        max_memory = max(memory_measurements)
        memory_increase = max_memory - initial_memory
        
        # Memory should not grow excessively
        assert memory_increase < 500, f"Memory usage increased too much: {memory_increase:.1f}MB"
        
        print(f"\nMemory Usage Analysis:")
        print(f"Initial: {initial_memory:.1f}MB")
        print(f"Maximum: {max_memory:.1f}MB")
        print(f"Increase: {memory_increase:.1f}MB")
    
    def test_concurrent_transcription_capability(self, short_audio_segment, function_transcription_config):
        """Test ability to handle multiple transcription requests"""
        import threading
        import queue
        
        audio_data = short_audio_segment["audio_data"]
        sample_rate = short_audio_segment["sample_rate"]
        
        # Create multiple speech processors for concurrent testing
        results_queue = queue.Queue()
        errors_queue = queue.Queue()
        
        def transcribe_worker(worker_id):
            try:
                # Each worker gets its own processor to avoid conflicts
                worker_processor = SpeechProcessor(function_transcription_config)
                result = worker_processor.transcribe(audio_data, sample_rate)
                results_queue.put((worker_id, result))
            except Exception as e:
                errors_queue.put((worker_id, str(e)))
        
        # Start multiple worker threads
        num_workers = 3
        threads = []
        
        start_time = time.time()
        for i in range(num_workers):
            thread = threading.Thread(target=transcribe_worker, args=(i,))
            thread.start()
            threads.append(thread)
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=30)  # 30 second timeout
        
        total_time = time.time() - start_time
        
        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        errors = []
        while not errors_queue.empty():
            errors.append(errors_queue.get())
        
        # Validate concurrent processing
        assert len(results) == num_workers, f"Not all workers completed: {len(results)}/{num_workers}"
        assert len(errors) == 0, f"Errors in concurrent processing: {errors}"
        
        # All results should be similar (same audio)
        transcriptions = [result[1] for result in results]
        if all(hasattr(t, 'text') for t in transcriptions):
            texts = [t.text for t in transcriptions]
        else:
            texts = transcriptions
        
        # Basic consistency check - all should have some content
        for text in texts:
            assert len(text.strip()) > 0
        
        print(f"\nConcurrent Processing Results:")
        print(f"Workers: {num_workers}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average time per worker: {total_time/num_workers:.2f}s")
    
    @pytest.mark.slow
    def test_end_to_end_performance_benchmark(self, audio_file_catalog, transcription_service,
                                            transcription_benchmark, debug_dir, audio_loader):
        """Comprehensive performance benchmark across multiple files"""
        benchmark_results = {
            "benchmark_settings": transcription_benchmark,
            "test_timestamp": time.time(),
            "files_processed": [],
            "performance_summary": {}
        }
        
        total_audio_duration = 0
        total_processing_time = 0
        successful_transcriptions = 0
        
        # Process each file in catalog
        for file_info in audio_file_catalog[:5]:  # Limit to 5 files for reasonable test time
            file_path = file_info["file_path"]
            duration = file_info["duration"]
            
            try:
                # Load and process audio
                audio_data, sample_rate = audio_loader.load_audio_file(file_path)
                
                start_time = time.time()
                result = transcription_service.speech_processor.transcribe(
                    audio_data, sample_rate, with_timing=True
                )
                processing_time = time.time() - start_time
                
                # Extract result data
                if hasattr(result, 'text'):
                    transcription = result.text
                    confidence = result.confidence
                else:
                    transcription = result
                    confidence = 1.0
                
                # Calculate metrics
                real_time_factor = processing_time / duration
                words_per_minute = len(transcription.split()) / (duration / 60) if duration > 0 else 0
                
                file_result = {
                    "filename": file_path.name,
                    "success": True,
                    "duration": duration,
                    "processing_time": processing_time,
                    "real_time_factor": real_time_factor,
                    "confidence": confidence,
                    "word_count": len(transcription.split()),
                    "words_per_minute": words_per_minute,
                    "transcription_preview": transcription[:100]
                }
                
                total_audio_duration += duration
                total_processing_time += processing_time
                successful_transcriptions += 1
                
            except Exception as e:
                file_result = {
                    "filename": file_path.name,
                    "success": False,
                    "error": str(e),
                    "duration": duration
                }
            
            benchmark_results["files_processed"].append(file_result)
        
        # Calculate overall performance metrics
        if successful_transcriptions > 0:
            avg_real_time_factor = total_processing_time / total_audio_duration
            success_rate = successful_transcriptions / len(audio_file_catalog[:5])
            
            benchmark_results["performance_summary"] = {
                "total_files": len(audio_file_catalog[:5]),
                "successful_transcriptions": successful_transcriptions,
                "success_rate": success_rate,
                "total_audio_duration": total_audio_duration,
                "total_processing_time": total_processing_time,
                "average_real_time_factor": avg_real_time_factor,
                "performance_rating": "PASS" if avg_real_time_factor < 2.0 else "SLOW"
            }
            
            # Performance assertions
            assert success_rate >= 0.8, f"Success rate too low: {success_rate:.2%}"
            assert avg_real_time_factor < 3.0, f"Average processing too slow: {avg_real_time_factor:.2f}x"
        
        # Save comprehensive benchmark results
        benchmark_file = debug_dir / "e2e_performance_benchmark.json"
        with open(benchmark_file, 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        
        print(f"\nPerformance Benchmark Results:")
        if successful_transcriptions > 0:
            summary = benchmark_results["performance_summary"]
            print(f"Success rate: {summary['success_rate']:.2%}")
            print(f"Average processing speed: {summary['average_real_time_factor']:.2f}x real-time")
            print(f"Performance rating: {summary['performance_rating']}")