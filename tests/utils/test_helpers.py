import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from contextlib import contextmanager
import logging
import pytest

class TestTimer:
    """Utility for timing test operations"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    def start(self):
        """Start the timer"""
        self.start_time = time.time()
        return self
    
    def stop(self):
        """Stop the timer and return duration"""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        return self.duration
    
    @contextmanager
    def time_operation(self):
        """Context manager for timing operations"""
        self.start()
        try:
            yield self
        finally:
            self.stop()

class TranscriptionValidator:
    """Utility for validating transcription results"""
    
    @staticmethod
    def validate_basic_result(result: Any) -> Dict[str, bool]:
        """Validate basic transcription result structure"""
        validations = {
            "has_content": False,
            "has_text": False,
            "has_timing": False,
            "has_confidence": False,
            "has_words": False
        }
        
        if result is None:
            return validations
        
        # Check for string result (backward compatibility)
        if isinstance(result, str):
            validations["has_content"] = len(result.strip()) > 0
            validations["has_text"] = True
            return validations
        
        # Check for structured result
        if hasattr(result, 'text'):
            validations["has_text"] = True
            validations["has_content"] = len(result.text.strip()) > 0
        
        if hasattr(result, 'duration') or hasattr(result, 'start_time'):
            validations["has_timing"] = True
        
        if hasattr(result, 'confidence'):
            validations["has_confidence"] = True
        
        if hasattr(result, 'words') and result.words:
            validations["has_words"] = True
        
        return validations
    
    @staticmethod
    def validate_word_timing(words: List[Dict[str, Any]], total_duration: float) -> Dict[str, Any]:
        """Validate word timing consistency"""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "stats": {
                "total_words": len(words),
                "avg_word_duration": 0,
                "timing_coverage": 0
            }
        }
        
        if not words:
            validation_result["warnings"].append("No words with timing")
            return validation_result
        
        # Check individual word timing
        total_word_duration = 0
        for i, word in enumerate(words):
            # Check required fields
            required_fields = ["word", "start_time", "end_time"]
            for field in required_fields:
                if field not in word:
                    validation_result["errors"].append(f"Word {i} missing field: {field}")
                    validation_result["valid"] = False
            
            if "start_time" in word and "end_time" in word:
                start_time = word["start_time"]
                end_time = word["end_time"]
                
                # Check timing validity
                if start_time < 0:
                    validation_result["errors"].append(f"Word {i} has negative start time: {start_time}")
                    validation_result["valid"] = False
                
                if end_time <= start_time:
                    validation_result["errors"].append(f"Word {i} has invalid timing: {start_time} >= {end_time}")
                    validation_result["valid"] = False
                
                if end_time > total_duration:
                    validation_result["warnings"].append(f"Word {i} extends beyond audio duration")
                
                total_word_duration += (end_time - start_time)
                
                # Check overlap with next word
                if i < len(words) - 1:
                    next_word = words[i + 1]
                    if "start_time" in next_word:
                        if end_time > next_word["start_time"]:
                            validation_result["warnings"].append(f"Word {i} overlaps with word {i+1}")
        
        # Calculate statistics
        if len(words) > 0:
            validation_result["stats"]["avg_word_duration"] = total_word_duration / len(words)
            validation_result["stats"]["timing_coverage"] = total_word_duration / total_duration if total_duration > 0 else 0
        
        return validation_result
    
    @staticmethod
    def compare_transcriptions(transcription1: str, transcription2: str) -> Dict[str, Any]:
        """Compare two transcriptions for similarity"""
        words1 = transcription1.lower().split()
        words2 = transcription2.lower().split()
        
        # Simple word-based comparison
        common_words = set(words1) & set(words2)
        total_unique_words = set(words1) | set(words2)
        
        similarity = len(common_words) / len(total_unique_words) if total_unique_words else 0
        
        return {
            "similarity_score": similarity,
            "words_1": len(words1),
            "words_2": len(words2),
            "common_words": len(common_words),
            "length_ratio": len(words2) / len(words1) if words1 else 0
        }

class PerformanceProfiler:
    """Utility for profiling test performance"""
    
    def __init__(self):
        self.measurements = []
        self.current_operation = None
    
    @contextmanager
    def profile_operation(self, operation_name: str):
        """Profile a specific operation"""
        start_time = time.time()
        self.current_operation = operation_name
        
        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            self.measurements.append({
                "operation": operation_name,
                "duration": duration,
                "timestamp": start_time
            })
            
            self.current_operation = None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.measurements:
            return {"total_operations": 0}
        
        operations = {}
        total_time = 0
        
        for measurement in self.measurements:
            op_name = measurement["operation"]
            duration = measurement["duration"]
            
            if op_name not in operations:
                operations[op_name] = {
                    "count": 0,
                    "total_time": 0,
                    "min_time": float('inf'),
                    "max_time": 0,
                    "durations": []
                }
            
            operations[op_name]["count"] += 1
            operations[op_name]["total_time"] += duration
            operations[op_name]["min_time"] = min(operations[op_name]["min_time"], duration)
            operations[op_name]["max_time"] = max(operations[op_name]["max_time"], duration)
            operations[op_name]["durations"].append(duration)
            
            total_time += duration
        
        # Calculate averages
        for op_data in operations.values():
            op_data["avg_time"] = op_data["total_time"] / op_data["count"]
        
        return {
            "total_operations": len(self.measurements),
            "total_time": total_time,
            "operations": operations
        }

class AudioTestHelpers:
    """Helper functions for audio testing"""
    
    @staticmethod
    def generate_test_audio(duration: float = 1.0, sample_rate: int = 16000, 
                          frequency: float = 440.0) -> np.ndarray:
        """Generate test audio signal"""
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        return (0.3 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
    
    @staticmethod
    def audio_to_pcm_bytes(audio_data: np.ndarray) -> bytes:
        """Convert audio samples to PCM bytes"""
        audio_int16 = (audio_data * 32767).astype(np.int16)
        return audio_int16.tobytes()
    
    @staticmethod
    def calculate_audio_metrics(audio_data: np.ndarray) -> Dict[str, float]:
        """Calculate basic audio metrics"""
        return {
            "rms": float(np.sqrt(np.mean(audio_data ** 2))),
            "peak": float(np.max(np.abs(audio_data))),
            "mean": float(np.mean(audio_data)),
            "std": float(np.std(audio_data)),
            "duration": len(audio_data) / 16000  # Assuming 16kHz
        }

class ResultLogger:
    """Utility for logging test results"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
    
    def log_test_result(self, test_name: str, result_data: Dict[str, Any]):
        """Log a test result"""
        timestamp = time.time()
        
        log_entry = {
            "test_name": test_name,
            "timestamp": timestamp,
            "result_data": result_data
        }
        
        self.results.append(log_entry)
        
        # Save individual result file
        result_file = self.output_dir / f"{test_name}_{int(timestamp)}.json"
        with open(result_file, 'w') as f:
            json.dump(log_entry, f, indent=2, default=str)
    
    def save_summary(self, filename: str = "test_summary.json"):
        """Save summary of all test results"""
        summary_file = self.output_dir / filename
        
        summary_data = {
            "total_tests": len(self.results),
            "test_run_timestamp": time.time(),
            "results": self.results
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)

class TestAssertions:
    """Custom assertions for audio/transcription testing"""
    
    @staticmethod
    def assert_transcription_quality(result: Any, min_words: int = 1, min_confidence: float = 0.1):
        """Assert transcription meets quality thresholds"""
        # Extract text and confidence
        if isinstance(result, str):
            text = result
            confidence = 1.0  # Assume good confidence for string results
        elif hasattr(result, 'text'):
            text = result.text
            confidence = getattr(result, 'confidence', 1.0)
        else:
            raise AssertionError("Invalid transcription result format")
        
        # Check word count
        word_count = len(text.split())
        assert word_count >= min_words, f"Too few words: {word_count} < {min_words}"
        
        # Check confidence
        assert confidence >= min_confidence, f"Confidence too low: {confidence} < {min_confidence}"
        
        # Check for non-empty content
        assert len(text.strip()) > 0, "Transcription is empty"
    
    @staticmethod
    def assert_processing_speed(processing_time: float, audio_duration: float, max_ratio: float = 3.0):
        """Assert processing speed is within acceptable limits"""
        ratio = processing_time / audio_duration if audio_duration > 0 else float('inf')
        assert ratio <= max_ratio, f"Processing too slow: {ratio:.2f}x real-time (max: {max_ratio}x)"
    
    @staticmethod
    def assert_timing_consistency(words: List[Dict[str, Any]], tolerance: float = 0.1):
        """Assert word timing is consistent"""
        for i in range(len(words) - 1):
            current_word = words[i]
            next_word = words[i + 1]
            
            # Check for reasonable timing
            assert current_word["end_time"] <= next_word["start_time"] + tolerance, \
                f"Word timing inconsistent: word {i} ends at {current_word['end_time']}, word {i+1} starts at {next_word['start_time']}"
    
    @staticmethod
    def assert_audio_service_state(audio_service, expected_has_audio: bool = True):
        """Assert audio service is in expected state"""
        stats = audio_service.get_audio_statistics()
        
        if expected_has_audio:
            assert stats["current_buffer_size"] > 0, "Audio service should have audio data"
            assert stats["sample_count"] > 0, "Audio service should have samples"
        else:
            assert stats["current_buffer_size"] == 0, "Audio service should be empty"

class BenchmarkComparator:
    """Utility for comparing performance against benchmarks"""
    
    def __init__(self, benchmark_file: Optional[Path] = None):
        self.benchmarks = {}
        if benchmark_file and benchmark_file.exists():
            with open(benchmark_file, 'r') as f:
                self.benchmarks = json.load(f)
    
    def record_measurement(self, test_name: str, metric_name: str, value: float):
        """Record a performance measurement"""
        if test_name not in self.benchmarks:
            self.benchmarks[test_name] = {}
        
        if metric_name not in self.benchmarks[test_name]:
            self.benchmarks[test_name][metric_name] = []
        
        self.benchmarks[test_name][metric_name].append({
            "value": value,
            "timestamp": time.time()
        })
    
    def compare_against_baseline(self, test_name: str, metric_name: str, 
                                current_value: float, tolerance: float = 0.2) -> Dict[str, Any]:
        """Compare current measurement against historical baseline"""
        if test_name not in self.benchmarks or metric_name not in self.benchmarks[test_name]:
            return {
                "comparison": "no_baseline",
                "current_value": current_value,
                "baseline": None,
                "within_tolerance": True
            }
        
        measurements = self.benchmarks[test_name][metric_name]
        values = [m["value"] for m in measurements]
        
        baseline_avg = sum(values) / len(values)
        difference = abs(current_value - baseline_avg) / baseline_avg
        within_tolerance = difference <= tolerance
        
        return {
            "comparison": "compared",
            "current_value": current_value,
            "baseline_avg": baseline_avg,
            "difference_ratio": difference,
            "tolerance": tolerance,
            "within_tolerance": within_tolerance,
            "baseline_samples": len(values)
        }
    
    def save_benchmarks(self, output_file: Path):
        """Save benchmark data to file"""
        with open(output_file, 'w') as f:
            json.dump(self.benchmarks, f, indent=2)

@contextmanager
def suppress_logging(level=logging.WARNING):
    """Context manager to suppress logging during tests"""
    logger = logging.getLogger()
    old_level = logger.level
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(old_level)

def create_test_audio_file(output_path: Path, duration: float = 5.0, sample_rate: int = 16000):
    """Create a test audio file for testing purposes"""
    import wave
    
    # Generate test audio
    audio_data = AudioTestHelpers.generate_test_audio(duration, sample_rate)
    
    # Convert to 16-bit PCM
    audio_int16 = (audio_data * 32767).astype(np.int16)
    
    # Write WAV file
    with wave.open(str(output_path), 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())

def setup_test_environment(test_audio_dir: Path, debug_dir: Path):
    """Set up test environment with necessary directories and files"""
    # Create directories
    test_audio_dir.mkdir(exist_ok=True)
    debug_dir.mkdir(exist_ok=True)
    
    # Create test audio files if they don't exist
    test_files = [
        ("test_5sec.wav", 5.0),
        ("test_10sec.wav", 10.0),
        ("test_short.wav", 1.0)
    ]
    
    for filename, duration in test_files:
        file_path = test_audio_dir / filename
        if not file_path.exists():
            create_test_audio_file(file_path, duration)
            print(f"Created test audio file: {file_path}")

def cleanup_test_environment(debug_dir: Path, keep_results: bool = True):
    """Clean up test environment"""
    if not keep_results and debug_dir.exists():
        import shutil
        shutil.rmtree(debug_dir)
        print(f"Cleaned up debug directory: {debug_dir}")

# Pytest markers and decorators
def requires_audio_files(func):
    """Decorator to mark tests that require audio files"""
    return pytest.mark.requires_audio(func)

def slow_test(func):
    """Decorator to mark slow tests"""
    return pytest.mark.slow(func)

def integration_test(func):
    """Decorator to mark integration tests"""
    return pytest.mark.integration(func)

def e2e_test(func):
    """Decorator to mark end-to-end tests"""
    return pytest.mark.e2e(func)