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
    
    # Constants for buffer management
    MAX_BUFFER_SIZE = 1024 * 1024 * 10  # 10MB max buffer size
    DEFAULT_SAMPLE_RATE = 16000  # 16kHz default sample rate
    MINIMUM_DURATION_MS = 1000  # 1 second minimum for processing
    
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
            
            # Silence detection adaptive parameters
            self._noise_floor = 0
            self._energy_history = []
            self._adaptive_threshold = 100
            self._history_size = 10
            
            # Buffer metrics
            self._last_buffer_size = 0
            self._peak_buffer_size = 0
            
            self._initialized = True
            logger.info("AudioService initialization complete")
    
    def add_chunk(self, chunk: bytes) -> bool:
        """
        Add new PCM data to both buffers with size checks
        
        Args:
            chunk: Raw audio bytes (PCM format)
            
        Returns:
            True if chunk was added, False if buffer would exceed max size
        """
        # Check if adding this chunk would exceed max buffer size
        if len(self.current_buffer) + len(self.session_buffer) + len(chunk) > self.MAX_BUFFER_SIZE:
            logger.warning(f"Buffer would exceed max size ({self.MAX_BUFFER_SIZE} bytes)")
            return False
            
        # Add to buffers
        self.current_buffer.extend(chunk)
        self.session_buffer.extend(chunk)
        
        # Update metrics
        current_size = len(self.current_buffer) + len(self.session_buffer)
        self._last_buffer_size = current_size
        self._peak_buffer_size = max(self._peak_buffer_size, current_size)
        
        return True
        
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
    
    def get_buffer_duration_ms(self, sample_rate: int = DEFAULT_SAMPLE_RATE) -> float:
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
    
    def has_minimum_audio(self, min_duration_ms: float = MINIMUM_DURATION_MS, 
                          sample_rate: int = DEFAULT_SAMPLE_RATE) -> bool:
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
                      silence_ratio_threshold: float = 0.7,
                      sample_rate: int = DEFAULT_SAMPLE_RATE) -> bool:
        """
        Detect silence using adaptive threshold based on energy levels
        
        Args:
            frame_duration_ms: Frame size in milliseconds
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
            
        # Update energy history for adaptive threshold
        self._energy_history.append(np.mean(energies))
        if len(self._energy_history) > self._history_size:
            self._energy_history.pop(0)
            
        # Calculate adaptive noise floor and threshold
        if len(self._energy_history) >= 3:
            self._noise_floor = np.percentile(self._energy_history, 10)
            self._adaptive_threshold = self._noise_floor * 1.5
        else:
            self._adaptive_threshold = 100  # Default threshold
            
        # Count frames below threshold
        silent_count = sum(1 for energy in energies if energy < self._adaptive_threshold)
        ratio = silent_count / len(energies)
        
        is_silent = ratio > silence_ratio_threshold
        if is_silent:
            logger.debug(f"Silence detected: {ratio:.2f} of frames below threshold {self._adaptive_threshold:.2f}")
            
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
                "sample_count": 0,
                "current_buffer_size": 0,
                "session_buffer_size": len(self.session_buffer),
                "peak_buffer_size": self._peak_buffer_size,
                "noise_floor": self._noise_floor,
                "adaptive_threshold": self._adaptive_threshold
            }
            
        samples = self.get_wave_data()
        return {
            "duration_ms": self.get_buffer_duration_ms(),
            "rms": float(np.sqrt(np.mean(samples ** 2))),
            "peak": float(np.max(np.abs(samples))),
            "sample_count": len(samples),
            "current_buffer_size": len(self.current_buffer),
            "session_buffer_size": len(self.session_buffer),
            "peak_buffer_size": self._peak_buffer_size,
            "noise_floor": self._noise_floor,
            "adaptive_threshold": self._adaptive_threshold
        }
    
    def trim_session_buffer(self, keep_duration_ms: int = 30000) -> int:
        """
        Trim the session buffer to keep only the recent audio
        
        Args:
            keep_duration_ms: Duration to keep in milliseconds
            
        Returns:
            Number of bytes removed
        """
        if len(self.session_buffer) == 0:
            return 0
            
        # Calculate bytes to keep
        bytes_to_keep = int((keep_duration_ms / 1000) * self.DEFAULT_SAMPLE_RATE * 2)
        
        # Ensure we keep at least the latest portion
        if bytes_to_keep < len(self.session_buffer):
            bytes_to_remove = len(self.session_buffer) - bytes_to_keep
            self.session_buffer = self.session_buffer[bytes_to_remove:]
            logger.info(f"Trimmed session buffer, removed {bytes_to_remove} bytes")
            return bytes_to_remove
        
        return 0
    
    def clear_session(self) -> None:
        """Clear all audio data (current and session buffers)"""
        self.current_buffer = bytearray()
        self.session_buffer = bytearray()
        self._energy_history = []
        self._noise_floor = 0
        self._adaptive_threshold = 100
        self._last_buffer_size = 0
        logger.info("Audio session cleared")