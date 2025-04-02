import numpy as np
import logging
from typing import Optional, Tuple, List, Dict, Any
from threading import Lock

# Configure logger
logger = logging.getLogger(__name__)

class AudioService:
    """
    Service class for managing audio data and processing.
    Handles audio buffering, signal processing, and silence detection.
    """
    _instance = None
    _lock = Lock()  # Thread safety for singleton pattern
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                logger.info("Initializing AudioService singleton")
                cls._instance = super(AudioService, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        """Initialize the AudioService (only runs once due to singleton pattern)"""
        if not getattr(self, "_initialized", False):
            # Audio buffers
            self.current_buffer = bytearray()  # For real-time processing
            self.session_buffer = bytearray()  # Complete session history
            
            self._initialized = True
            logger.info("AudioService initialization complete")
    
    def add_chunk(self, chunk: bytes) -> None:
        """
        Add new PCM data to both buffers
        
        Args:
            chunk: Raw audio bytes (PCM format)
        """
        self.current_buffer.extend(chunk)
        self.session_buffer.extend(chunk)
        
    def reset_current_buffer(self) -> None:
        """Reset only the current buffer, keeping session history intact"""
        self.current_buffer = bytearray()
    
    def get_wave_data(self, use_session_buffer: bool = False) -> np.ndarray:
        """
        Convert buffer to normalized float32 samples [-1.0, 1.0]
        
        Args:
            use_session_buffer: If True, use session buffer instead of current buffer
            
        Returns:
            Numpy array of normalized audio samples
        """
        buf = self.session_buffer if use_session_buffer else self.current_buffer
        
        if len(buf) % 2 != 0:
            # Trim the last byte if the buffer length is odd
            buf = buf[:-1]
            
        if len(buf) == 0:
            return np.array([], dtype=np.float32)
            
        samples = np.frombuffer(buf, dtype=np.int16).astype(np.float32)
        samples /= 32768.0  # Normalize to [-1.0, 1.0]
        return samples
    
    def get_buffer_duration_ms(self, sample_rate: int = 16000) -> float:
        """
        Calculate the duration of the current buffer in milliseconds
        
        Args:
            sample_rate: Audio sample rate in Hz
            
        Returns:
            Duration in milliseconds
        """
        # Each sample is 2 bytes (16-bit PCM)
        num_samples = len(self.current_buffer) // 2
        return (num_samples / sample_rate) * 1000
    
    def has_minimum_audio(self, min_duration_ms: float = 1000, sample_rate: int = 16000) -> bool:
        """
        Check if buffer contains at least the minimum duration of audio
        
        Args:
            min_duration_ms: Minimum audio duration in milliseconds
            sample_rate: Audio sample rate in Hz
            
        Returns:
            True if buffer has at least the minimum duration
        """
        min_bytes = int((min_duration_ms / 1000) * sample_rate * 2)  # 2 bytes per sample
        return len(self.current_buffer) >= min_bytes
    
    def detect_silence(self, frame_duration_ms: int = 20, 
                      offset: int = 100, 
                      silence_ratio_threshold: float = 0.7,
                      sample_rate: int = 16000) -> bool:
        """
        Detect silence using adaptive threshold based on energy levels
        
        Args:
            frame_duration_ms: Frame size in milliseconds
            offset: Energy threshold offset above noise floor
            silence_ratio_threshold: Ratio of frames that must be below threshold
            sample_rate: Audio sample rate in Hz
            
        Returns:
            True if silence is detected, False otherwise
        """
        if len(self.current_buffer) == 0:
            return False
            
        # Calculate frame size in bytes
        frame_size = int(sample_rate * (frame_duration_ms / 1000.0) * 2)  # 2 bytes per sample
        
        # Split buffer into frames
        frames = []
        for i in range(0, len(self.current_buffer), frame_size):
            frame = self.current_buffer[i:i+frame_size]
            if len(frame) == frame_size:  # Only include complete frames
                frames.append(frame)
        
        if not frames:
            return False
            
        # Calculate energy for each frame
        energies = []
        for frame in frames:
            samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
            energy = np.sqrt(np.mean(samples ** 2))
            energies.append(energy)
            
        # Calculate noise floor (10th percentile of energies)
        noise_floor = np.percentile(energies, 10)
        threshold = noise_floor + offset
        
        # Count frames below threshold
        silent_count = sum(1 for energy in energies if energy < threshold)
        ratio = silent_count / len(energies)
        
        is_silent = ratio > silence_ratio_threshold
        if is_silent:
            logger.debug(f"Silence detected: {ratio:.2f} of frames below threshold")
            
        return is_silent
    
    def get_audio_statistics(self) -> Dict[str, Any]:
        """
        Calculate audio statistics for the current buffer
        
        Returns:
            Dictionary of audio statistics
        """
        if len(self.current_buffer) == 0:
            return {
                "duration_ms": 0,
                "rms": 0,
                "peak": 0,
                "sample_count": 0
            }
            
        samples = self.get_wave_data()
        return {
            "duration_ms": self.get_buffer_duration_ms(),
            "rms": float(np.sqrt(np.mean(samples ** 2))),
            "peak": float(np.max(np.abs(samples))),
            "sample_count": len(samples)
        }
    
    def clear_session(self) -> None:
        """Clear all audio data (current and session buffers)"""
        self.current_buffer = bytearray()
        self.session_buffer = bytearray()
        logger.info("Audio session cleared")