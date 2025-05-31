import pytest
import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Generator, Tuple, List
import librosa
import numpy as np
import asyncio
import sys

# Import your app modules
sys.path.insert(0, '')
from app.services.audio_service import AudioService
from app.services.transcription_service import TranscriptionService
from app.utils.speech_processor import SpeechProcessor, TranscriptionConfig

# Test configuration constants
TEST_CONFIG = {
    "DEFAULT_DB_URL": "http://127.0.0.1:8002",
    "DEFAULT_APP_URL": "http://127.0.0.1:8001", 
    "DEFAULT_AUDIO_FILE": os.path.join("test_audios", "test_30sec.wav"),
    "DEBUG_PATH": "test_client_results",
    "TEST_AUDIO_FILE": os.path.join("test_audios", "day1_consultation03.wav"),
    "TEST_TRANSCRIPT_FILE": "D:\\medsync\\primock57\\output\\joined_transcripts\\day1_consultation03.txt"
}

SAMPLE_TRANSCRIPT = """
Patient: Doctor, I've had a sore throat, and it's getting worse. It feels scratchy, and swallowing is uncomfortable.
Doctor: I see. Has it been painful enough to affect eating or drinking?
"""

class AudioDataLoader:
    """Utility class for loading and processing audio files for testing"""
    
    @staticmethod
    def load_audio_file(file_path: Path, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
        """Load audio file and return samples with sample rate"""
        try:
            audio_data, sr = librosa.load(str(file_path), sr=target_sr, mono=True)
            return audio_data, sr
        except Exception as e:
            raise ValueError(f"Failed to load audio file {file_path}: {str(e)}")
    
    @staticmethod
    def load_audio_as_chunks(file_path: Path, chunk_size_ms: int = 1000, 
                           target_sr: int = 16000) -> List[np.ndarray]:
        """Load audio file and split into chunks for streaming simulation"""
        audio_data, sr = AudioDataLoader.load_audio_file(file_path, target_sr)
        
        # Calculate samples per chunk
        samples_per_chunk = int(sr * chunk_size_ms / 1000)
        
        # Split into chunks
        chunks = []
        for i in range(0, len(audio_data), samples_per_chunk):
            chunk = audio_data[i:i + samples_per_chunk]
            if len(chunk) > 0:
                chunks.append(chunk)
        
        return chunks
    
    @staticmethod
    def get_audio_duration(file_path: Path) -> float:
        """Get duration of audio file in seconds"""
        try:
            duration = librosa.get_duration(path=str(file_path))
            return duration
        except Exception as e:
            raise ValueError(f"Failed to get duration of {file_path}: {str(e)}")

# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "e2e: marks tests as end-to-end tests")
    config.addinivalue_line("markers", "requires_audio: marks tests that need audio files")
    config.addinivalue_line("markers", "slow: marks tests as slow (> 1s)")

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def test_config():
    """Test configuration constants"""
    return TEST_CONFIG

@pytest.fixture(scope="session") 
def sample_transcript():
    """Sample conversation transcript"""
    return SAMPLE_TRANSCRIPT

@pytest.fixture(scope="session")
def test_audio_dir():
    """Test audio directory path"""
    audio_dir = Path("test_audios")
    if not audio_dir.exists():
        audio_dir.mkdir()
    return audio_dir

@pytest.fixture(scope="session")
def test_transcript_dir():
    """Test transcript file"""
    transcript_file = Path("D:\\medsync\\primock57\\output\\joined_transcripts")
    return transcript_file


@pytest.fixture(scope="session")
def debug_dir():
    """Debug output directory for test results"""
    debug_path = Path(TEST_CONFIG["DEBUG_PATH"])
    if debug_path.exists():
        shutil.rmtree(debug_path)
    debug_path.mkdir()
    yield debug_path
    # Cleanup after session
    # if debug_path.exists():
    #     shutil.rmtree(debug_path)

@pytest.fixture(scope="function")
def temp_dir():
    """Temporary directory for test isolation"""
    with tempfile.TemporaryDirectory() as temp_path:
        yield Path(temp_path)

# Real service fixtures (no mocking)
@pytest.fixture(scope="session")
def audio_service():
    """Real audio service instance"""
    return AudioService()

@pytest.fixture(scope="session")
def transcription_config():
    """Transcription configuration optimized for testing"""
    return TranscriptionConfig(
        backend="whisper_onnx",
        model_size="small",
        language="en",
        enable_speaker_adaptation=False,
        enable_medical_postprocessing=False,
        enable_word_timing=True,
        use_gpu=False,  # Use CPU for consistent test environment
        confidence_threshold=0.5  # Lower threshold for test audio
    )

@pytest.fixture(scope="session")  # Session scope to avoid reloading model
def speech_processor(transcription_config):
    """Real speech processor instance (cached for session)"""
    return SpeechProcessor(transcription_config)

@pytest.fixture(scope="function")
def function_transcription_config():
    """Function-scoped transcription config for tests that need to modify it"""
    return TranscriptionConfig(
        backend="whisper_transformers",
        model_size="small",
        language="en",
        enable_speaker_adaptation=False,
        enable_medical_postprocessing=True,
        enable_word_timing=True,
        use_gpu=False,
        confidence_threshold=0.5
    )

@pytest.fixture(scope="session")
def transcription_service(audio_service, speech_processor):
    """Real transcription service instance"""
    # Use the session-scoped config for consistency
    config = TranscriptionConfig(
        backend="whisper_transformers",
        model_size="small",
        language="en",
        enable_speaker_adaptation=False,
        enable_medical_postprocessing=False,
        enable_word_timing=True,
        use_gpu=False,
        confidence_threshold=0.5
    )
    
    service = TranscriptionService(
        audio_service=audio_service,
        speech_processor=speech_processor,
        config=config
    )
    # Reset state before each test
    service.reset()
    return service

# Audio file fixtures
@pytest.fixture(scope="session")
def default_audio_file(test_config):
    """Path to default test audio file"""
    audio_path = Path(test_config["DEFAULT_AUDIO_FILE"])
    if not audio_path.exists():
        pytest.skip(f"Default audio file not found: {audio_path}")
    return audio_path

@pytest.fixture(scope="session")
def test_audio_file(test_config):
    """Path to test consultation audio file"""
    audio_path = Path(test_config["TEST_AUDIO_FILE"])
    if not audio_path.exists():
        pytest.skip(f"Test audio file not found: {audio_path}")
    return audio_path

@pytest.fixture(scope="session")
def available_audio_files(test_audio_dir):
    """List all available audio files for testing"""
    audio_extensions = ['.wav', '.mp3', '.m4a', '.flac']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(test_audio_dir.glob(f"*{ext}"))
    
    if not audio_files:
        pytest.skip("No audio files found in test_audios directory")
    
    return audio_files

# Audio fixtures
@pytest.fixture(scope="session")
def audio_loader():
    """Audio data loader utility"""
    return AudioDataLoader

@pytest.fixture(scope="session")
def sample_audio_data(default_audio_file, audio_loader):
    """Load sample audio data for testing"""
    if default_audio_file.exists():
        audio_data, sr = audio_loader.load_audio_file(default_audio_file)
        return {
            "file_path": default_audio_file,
            "audio_data": audio_data,
            "sample_rate": sr,
            "duration": len(audio_data) / sr,
        }
    else:
        pytest.skip(f"Sample audio file not found: {default_audio_file}")

@pytest.fixture(scope="session")
def consultation_audio_data(test_audio_file, audio_loader):
    """Load consultation audio data for testing"""
    if test_audio_file.exists():
        audio_data, sr = audio_loader.load_audio_file(test_audio_file)
        return {
            "file_path": test_audio_file,
            "audio_data": audio_data,
            "sample_rate": sr,
            "duration": len(audio_data) / sr,
        }
    else:
        pytest.skip(f"Consultation audio file not found: {test_audio_file}")

@pytest.fixture(scope="session")
def short_audio_segment(sample_audio_data, audio_loader):
    """Extract a short segment (first 3 seconds) for quick tests"""
    audio_data = sample_audio_data["audio_data"]
    sr = sample_audio_data["sample_rate"]
    
    # Take first 3 seconds or entire audio if shorter
    segment_samples = min(3 * sr, len(audio_data))
    short_segment = audio_data[:segment_samples]
    
    return {
        "audio_data": short_segment,
        "sample_rate": sr,
        "duration": len(short_segment) / sr,
    }

@pytest.fixture(scope="session")
def audio_chunks_1sec(sample_audio_data, audio_loader):
    """Split sample audio into 1-second chunks"""
    return audio_loader.load_audio_as_chunks(
        sample_audio_data["file_path"], 
        chunk_size_ms=1000
    )

@pytest.fixture(scope="session")
def audio_chunks_500ms(sample_audio_data, audio_loader):
    """Split sample audio into 500ms chunks for streaming tests"""
    return audio_loader.load_audio_as_chunks(
        sample_audio_data["file_path"], 
        chunk_size_ms=500
    )

@pytest.fixture(scope="session")
def streaming_audio_chunks(consultation_audio_data, audio_loader):
    """Create streaming chunks from consultation audio"""
    return audio_loader.load_audio_as_chunks(
        consultation_audio_data["file_path"],
        chunk_size_ms=1500  # 1500ms chunks for realistic streaming
    )

@pytest.fixture(scope="session", params=["1sec", "3sec", "5sec"])
def variable_length_audio(request, sample_audio_data, audio_loader):
    """Create audio segments of different lengths for testing"""
    duration_map = {"1sec": 1, "3sec": 3, "5sec": 5}
    target_duration = duration_map[request.param]
    
    audio_data = sample_audio_data["audio_data"]
    sr = sample_audio_data["sample_rate"]
    
    # Extract segment of specified duration
    segment_samples = min(target_duration * sr, len(audio_data))
    segment = audio_data[:segment_samples]
    
    return {
        "audio_data": segment,
        "sample_rate": sr,
        "duration": len(segment) / sr,
        "label": request.param
    }

@pytest.fixture(scope="session")
def multiple_audio_files(available_audio_files, audio_loader):
    """Load multiple audio files for batch testing"""
    audio_data_list = []
    
    # Limit to first 3 files to avoid long test times
    for audio_file in available_audio_files[:3]:
        try:
            audio_data, sr = audio_loader.load_audio_file(audio_file)
            audio_data_list.append({
                "file_path": audio_file,
                "audio_data": audio_data,
                "sample_rate": sr,
                "duration": len(audio_data) / sr,
            })
        except Exception as e:
            print(f"Warning: Could not load {audio_file}: {e}")
            continue
    
    if not audio_data_list:
        pytest.skip("No audio files could be loaded")
    
    return audio_data_list

@pytest.fixture(scope="session")
def audio_file_catalog(test_audio_dir, audio_loader):
    """Catalog all available audio files with metadata"""
    catalog = []
    audio_extensions = ['.wav', '.mp3', '.m4a', '.flac']
    
    for ext in audio_extensions:
        for audio_file in test_audio_dir.glob(f"*{ext}"):
            try:
                duration = audio_loader.get_audio_duration(audio_file)
                catalog.append({
                    "file_path": audio_file,
                    "filename": audio_file.name,
                    "extension": ext,
                    "duration": duration,
                    "size_bytes": audio_file.stat().st_size
                })
            except Exception as e:
                print(f"Warning: Could not process {audio_file}: {e}")
                continue
    
    return catalog


@pytest.fixture(scope="session")
def transcription_benchmark():
    """Benchmark data for transcription performance"""
    return {
        "max_processing_time_per_second": 2.0,
        "min_confidence_threshold": 0.3,
        "max_memory_usage_mb": 500,
        "expected_word_error_rate": 0.2
    }

@pytest.fixture(scope="session")
def audio_validation_helpers():
    """Helper functions for validating audio processing results"""
    
    class AudioValidationHelpers:
        @staticmethod
        def validate_transcription_result(result: Dict[str, Any], min_confidence: float = 0.1) -> bool:
            """Validate transcription result structure and content"""
            required_fields = ["text", "start_time", "end_time", "confidence"]
            
            for field in required_fields:
                if field not in result:
                    return False
            
            if not isinstance(result["text"], str):
                return False
            if result["start_time"] < 0 or result["end_time"] <= result["start_time"]:
                return False
            if not (0.0 <= result["confidence"] <= 1.0):
                return False
            if result["confidence"] < min_confidence:
                return False
            
            return True
        
        @staticmethod
        def validate_audio_timing(words: List[Dict], total_duration: float) -> bool:
            """Validate word timing consistency"""
            if not words:
                return True
            
            for i, word in enumerate(words):
                if word["start_time"] < 0 or word["end_time"] > total_duration:
                    return False
                if word["start_time"] >= word["end_time"]:
                    return False
                
                if i < len(words) - 1:
                    next_word = words[i + 1]
                    if word["end_time"] > next_word["start_time"]:
                        return False
            
            return True
    
# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_services():
    """Automatic cleanup after each test"""
    yield
    # Clear any singleton state if needed
    pass