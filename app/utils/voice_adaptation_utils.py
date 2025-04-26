import numpy as np
import os
import pickle
import logging
from typing import Dict, List, Any, Optional, Tuple
import torch
import librosa
from scipy.linalg import solve
import scipy

# Configure logger
logger = logging.getLogger(__name__)

class AdaptationTransformer:
    """
    Enhanced transformer class that supports both MLLR and VTLN adaptations
    """
    
    def __init__(self, mean_vector: np.ndarray, covariance_matrix: np.ndarray):
        """
        Initialize with speaker profile statistics
        
        Args:
            mean_vector: Mean vector from speaker profile
            covariance_matrix: Covariance matrix from speaker profile
        """
        self.mean_vector = mean_vector
        self.covariance_matrix = covariance_matrix
        
        # MLLR transformation parameters
        self.transform_matrix = None
        self.bias_vector = None
        
        # VTLN parameters
        self.vtln_warp_factor = 1.0  # Default: no warping
        
        # Initialize transform (identity transformation)
        feature_dim = len(mean_vector)
        self.transform_matrix = np.eye(feature_dim)
        self.bias_vector = np.zeros(feature_dim)
        
    def estimate_transform(self, base_model_stats: Dict[str, Any]) -> None:
        """
        Estimate both MLLR and VTLN transforms
        
        Args:
            base_model_stats: Statistics of the base model
        """
        try:
            # Get base model stats
            base_mean = base_model_stats.get('mean_vector', None)
            base_cov = base_model_stats.get('covariance_matrix', None)
            
            if base_mean is None or base_cov is None:
                logger.warning("Missing base model statistics, using identity transform")
                return
            
            # Estimate MLLR transformation
            self._estimate_mllr_transform(base_mean, base_cov)
            
            # Estimate VTLN warping factor
            self._estimate_vtln_factor(base_mean)
            
        except Exception as e:
            logger.error(f"Error estimating transforms: {e}")
    
    def _estimate_mllr_transform(self, base_mean: np.ndarray, base_cov: np.ndarray) -> None:
        """
        Estimate the MLLR transform from the base model to the speaker
        """
        try:
            # Ensure dimensions match
            if len(base_mean) != len(self.mean_vector):
                logger.warning(f"Dimension mismatch: base={len(base_mean)}, speaker={len(self.mean_vector)}")
                return
                
            # Calculate transformation matrix (simplified)
            try:
                # Approach: A * base_cov = speaker_cov
                A = np.zeros_like(self.transform_matrix)
                for i in range(len(base_mean)):
                    A[i, :] = np.linalg.solve(base_cov, self.covariance_matrix[i, :])
                
                self.transform_matrix = A
                
                # Calculate bias as difference between means
                self.bias_vector = self.mean_vector - np.dot(self.transform_matrix, base_mean)
                
            except np.linalg.LinAlgError as e:
                logger.error(f"Error solving for transform matrix: {e}")
                # Fallback to simpler diagonal scaling
                diag_scale = np.sqrt(np.diag(self.covariance_matrix) / np.diag(base_cov))
                self.transform_matrix = np.diag(diag_scale)
                self.bias_vector = self.mean_vector - np.dot(self.transform_matrix, base_mean)
                
        except Exception as e:
            logger.error(f"Error estimating MLLR transform: {e}")
    
    def _estimate_vtln_factor(self, base_mean: np.ndarray) -> None:
        """
        Estimate VTLN warping factor based on mean vector comparison
        """
        try:
            # This is a simplified approach - in practice you'd use formant analysis
            # For now, we'll estimate based on spectral center of gravity 
            
            # Use lower coefficients which correspond to vocal tract characteristics
            # We only use the first few MFCC coefficients (typically 2-6) 
            # as they relate to vocal tract shape
            
            # Skip coefficient 0 (energy) and use 1-5 if available
            start_idx = 1
            end_idx = min(6, len(self.mean_vector))
            
            if end_idx <= start_idx or start_idx >= len(base_mean):
                self.vtln_warp_factor = 1.0
                return
                
            speaker_spectral_mean = np.mean(self.mean_vector[start_idx:end_idx])
            base_spectral_mean = np.mean(base_mean[start_idx:end_idx])
            
            if abs(base_spectral_mean) < 1e-6:  # Avoid division by zero
                self.vtln_warp_factor = 1.0
                return
                
            # Calculate warping factor
            # If speaker mean is lower than base mean -> warp factor > 1 (stretch)
            # If speaker mean is higher than base mean -> warp factor < 1 (compress)
            # This is a simplified heuristic
            ratio = base_spectral_mean / speaker_spectral_mean
            
            # Constrain to reasonable range
            self.vtln_warp_factor = max(0.8, min(1.2, ratio))
            
            logger.info(f"Estimated VTLN warp factor: {self.vtln_warp_factor}")
            
        except Exception as e:
            logger.error(f"Error estimating VTLN factor: {e}")
            self.vtln_warp_factor = 1.0
    
    def transform_features(self, features: np.ndarray) -> np.ndarray:
        """
        Apply the feature-space transform to input features
        
        Args:
            features: Input feature matrix [feature_dim x frames]
            
        Returns:
            Transformed features
        """
        if self.transform_matrix is None:
            return features
            
        try:
            # Apply MLLR transformation to each frame
            transformed = np.zeros_like(features)
            
            for t in range(features.shape[1]):
                frame = features[:, t]
                transformed[:, t] = np.dot(self.transform_matrix, frame) + self.bias_vector
                
            return transformed
            
        except Exception as e:
            logger.error(f"Error transforming features: {e}")
            return features
            
    def get_vtln_warp_factor(self) -> float:
        """
        Get the VTLN warping factor
        
        Returns:
            VTLN warping factor
        """
        return self.vtln_warp_factor

def extract_mfcc_features(samples: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Extract MFCC features from audio samples
    
    Args:
        samples: Audio samples
        sr: Sample rate
        
    Returns:
        MFCC features
    """
    import librosa
    
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(
        y=samples, 
        sr=sr, 
        n_mfcc=13,
        hop_length=int(sr * 0.01),  # 10ms hop
        n_fft=int(sr * 0.025)       # 25ms window
    )
    
    return mfccs

def extract_cmvn_stats(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract Cepstral Mean and Variance Normalization statistics
    
    Args:
        features: MFCC features [feature_dim x frames]
        
    Returns:
        Tuple of (mean_vector, variance_vector)
    """
    # Calculate mean and variance along time dimension
    mean_vector = np.mean(features, axis=1)
    variance_vector = np.var(features, axis=1)
    
    return mean_vector, variance_vector

def apply_cmvn(features: np.ndarray, mean_vector: np.ndarray, variance_vector: np.ndarray) -> np.ndarray:
    """
    Apply CMVN to features
    
    Args:
        features: MFCC features [feature_dim x frames]
        mean_vector: Mean vector [feature_dim]
        variance_vector: Variance vector [feature_dim]
        
    Returns:
        Normalized features
    """
    # Avoid division by zero
    std_vector = np.sqrt(variance_vector + 1e-10)
    
    # Apply normalization
    normalized = np.zeros_like(features)
    for t in range(features.shape[1]):
        normalized[:, t] = (features[:, t] - mean_vector) / std_vector
        
    return normalized

# Function to get base model statistics
def get_base_model_stats() -> Dict[str, Any]:
    """
    Get statistics of the base speech model for adaptation calculations
    
    Returns:
        Dictionary with mean_vector and covariance_matrix
    """
    # In a real system, these would be calculated from the base model
    # or loaded from a pre-computed file
    
    # For now, return placeholder values
    feature_dim = 39  # 13 MFCCs + delta + delta-delta
    
    # Reasonable defaults for speech features
    mean_vector = np.zeros(feature_dim)
    # Diagonal covariance with higher variance for lower coefficients (typical in speech)
    var_vector = np.array([1.0/(i+1) for i in range(feature_dim)])
    covariance_matrix = np.diag(var_vector)
    
    return {
        'mean_vector': mean_vector,
        'covariance_matrix': covariance_matrix
    }

def preprocess_audio_for_speaker(audio_samples: np.ndarray, speaker_profile: Dict[str, Any]) -> np.ndarray:
    """
    Preprocess audio with speaker-specific settings including VTLN
    
    Args:
        audio_samples: Raw audio samples
        speaker_profile: Speaker profile from calibration
        
    Returns:
        Processed audio samples
    """
    try:
        # Volume normalization - consistent input level is important
        if np.max(np.abs(audio_samples)) > 0:
            normalized = audio_samples / np.max(np.abs(audio_samples)) * 0.9
        else:
            normalized = audio_samples
        
        # Apply VTLN if warp factor is available
        vtln_warp_factor = speaker_profile.get('vtln_warp_factor')
        if vtln_warp_factor is not None and vtln_warp_factor != 1.0:
            # Apply frequency-domain warping
            normalized = apply_vtln_to_audio(normalized, vtln_warp_factor)
            logger.info(f"Applied VTLN with warp factor {vtln_warp_factor}")
            
        return normalized
        
    except Exception as e:
        logger.error(f"Error preprocessing audio for speaker: {e}")
        return audio_samples
    
# Functions for Vocal Tract Length Normalization (VTLN) approach
def apply_frequency_warping(spectrogram: np.ndarray, warp_factor: float) -> np.ndarray:
    """
    Apply frequency warping to a spectrogram using piecewise linear warping
    
    Args:
        spectrogram: Magnitude spectrogram
        warp_factor: Warping factor
        
    Returns:
        Warped spectrogram
    """
    # Get frequency bins
    num_freqs, num_frames = spectrogram.shape
    
    # Create warped spectrogram of same shape
    warped_spec = np.zeros_like(spectrogram)
    
    # For each frequency bin, calculate warped index
    for i in range(num_freqs):
        # Normalize frequency to [0, 1]
        norm_freq = i / (num_freqs - 1)
        
        # Apply piecewise linear warping function
        if norm_freq <= 0.5:
            warped_freq = norm_freq * warp_factor
        else:
            warped_freq = 1.0 - (1.0 - norm_freq) * warp_factor
            
        # Convert back to bin index
        warped_idx = int(warped_freq * (num_freqs - 1))
        warped_idx = max(0, min(num_freqs - 1, warped_idx))
        
        # Copy spectrogram values
        warped_spec[warped_idx, :] += spectrogram[i, :]
    
    return warped_spec

def estimate_formants(audio_samples: np.ndarray, sample_rate: int, 
                     num_formants: int = 3) -> List[float]:
    """
    Estimate formant frequencies from audio
    
    Args:
        audio_samples: Audio samples
        sample_rate: Sample rate
        num_formants: Number of formants to estimate
        
    Returns:
        List of estimated formant frequencies
    """
    try:
        # Use LPC to estimate formants
        import librosa
        
        # Preemphasis to amplify higher frequencies
        preemph = librosa.effects.preemphasis(audio_samples)
        
        # LPC analysis (this is simplified - production systems use more sophisticated methods)
        lpc_order = 2 * num_formants + 2
        lpc_coeffs = librosa.lpc(preemph, order=lpc_order)
        
        # Find roots of LPC polynomial
        roots = np.polynomial.polynomial.polyroots(np.concatenate(([1], -lpc_coeffs[1:])))
        
        # Keep only stable roots with positive imaginary part
        roots = roots[np.imag(roots) > 0]
        
        # Convert to angles and then to frequencies
        angles = np.arctan2(np.imag(roots), np.real(roots))
        formants = angles * sample_rate / (2 * np.pi)
        
        # Sort and return the first num_formants
        formants = sorted(formants)[:num_formants]
        
        return formants
        
    except Exception as e:
        logger.error(f"Error estimating formants: {e}")
        return []

def warp_frequency(freq, warp_factor, warping_point=0.5):
    """
    Apply piecewise linear warping to a frequency value
    
    Args:
        freq: Normalized frequency (0-1)
        warp_factor: Warping factor (typically 0.8-1.2)
        warping_point: Point where warping changes (typically 0.5)
        
    Returns:
        Warped frequency value
    """
    if freq <= warping_point:
        return freq * warp_factor
    else:
        return warping_point * warp_factor + (freq - warping_point) * (1 - warping_point * warp_factor) / (1 - warping_point)
    
def estimate_warping_factor(mfcc_features, reference_formants=None):
    """
    Estimate warping factor based on formant analysis
    
    Args:
        mfcc_features: MFCC features from speaker
        reference_formants: Reference formant frequencies (optional)
        
    Returns:
        Estimated warping factor
    """
    # Extract approximate formants from MFCCs
    # In practice, you'd use a more sophisticated formant extraction
    formants = estimate_formants(mfcc_features)
    
    # If no reference formants provided, use standard adult male values
    if reference_formants is None:
        reference_formants = [500, 1500, 2500]  # Hz
    
    # Calculate ratio of reference to observed formants
    ratios = []
    for ref, obs in zip(reference_formants, formants):
        if obs > 0:
            ratios.append(ref / obs)
    
    # Average the ratios to get warping factor
    if ratios:
        warp_factor = sum(ratios) / len(ratios)
        # Constrain to reasonable range
        warp_factor = max(0.8, min(1.2, warp_factor))
        return warp_factor
    
    return 1.0

def extract_vtln_mfcc(audio_samples, sample_rate, warp_factor=1.0, n_mfcc=13):
    """
    Extract MFCC features with VTLN applied
    
    Args:
        audio_samples: Audio samples
        sample_rate: Sample rate
        warp_factor: VTLN warping factor
        n_mfcc: Number of MFCC coefficients
        
    Returns:
        VTLN-warped MFCC features
    """
    # Create mel filterbank with warping
    mel_filters = create_vtln_mel_filterbank(sample_rate, warp_factor)
    
    # Compute spectrogram
    S = librosa.stft(audio_samples)
    D = np.abs(S)**2
    
    # Apply mel filterbank
    mel_spectrogram = np.dot(mel_filters, D)
    
    # Log and DCT for MFCC
    log_mel = np.log(mel_spectrogram + 1e-8)
    mfcc = scipy.fftpack.dct(log_mel, axis=0, type=2, norm='ortho')[:n_mfcc]
    
    # Add deltas and delta-deltas
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    
    # Combine features
    features = np.vstack([mfcc, delta, delta2])
    
    return features

def create_vtln_mel_filterbank(sample_rate, warp_factor, n_filters=40, n_fft=2048):
    """
    Create a mel filterbank with VTLN warping
    
    Args:
        sample_rate: Audio sample rate
        warp_factor: VTLN warping factor
        n_filters: Number of mel filters
        n_fft: FFT size
        
    Returns:
        VTLN-warped mel filterbank
    """
    # Create standard mel filterbank
    mel_filters = librosa.filters.mel(
        sr=sample_rate,
        n_fft=n_fft,
        n_mels=n_filters
    )
    
    # Apply warping to filter center frequencies
    fft_freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    n_freqs = len(fft_freqs)
    
    # Create warped filterbank
    warped_filters = np.zeros_like(mel_filters)
    
    # For each mel filter
    for i in range(n_filters):
        # Get filter weights
        filter_weights = mel_filters[i]
        
        # Find center frequency (approximate)
        center_idx = np.argmax(filter_weights)
        if center_idx > 0 and center_idx < n_freqs:
            # Normalize to 0-1 range
            norm_freq = center_idx / (n_freqs - 1)
            
            # Apply warping to center frequency
            warped_norm_freq = warp_frequency(norm_freq, warp_factor)
            
            # Convert back to frequency index
            warped_idx = int(warped_norm_freq * (n_freqs - 1))
            warped_idx = max(0, min(n_freqs - 1, warped_idx))
            
            # Calculate shift amount
            shift = warped_idx - center_idx
            
            # Shift the filter by this amount
            if shift > 0:
                warped_filters[i, shift:] = filter_weights[:-shift]
            elif shift < 0:
                warped_filters[i, :shift] = filter_weights[-shift:]
            else:
                warped_filters[i] = filter_weights
        else:
            # Keep original filter if center can't be found
            warped_filters[i] = filter_weights
    
    # Normalize filters
    for i in range(n_filters):
        if np.sum(warped_filters[i]) > 0:
            warped_filters[i] = warped_filters[i] / np.sum(warped_filters[i])
    
    return warped_filters

def apply_vtln_to_audio(audio: np.ndarray, warp_factor: float, sample_rate: int = 16000) -> np.ndarray:
    """
    Apply Vocal Tract Length Normalization to audio in the frequency domain
    
    Args:
        audio: Audio samples
        warp_factor: VTLN warping factor
        sample_rate: Sample rate of audio
    
    Returns:
        Frequency-warped audio
    """
    try:
        import librosa
        
        # Apply STFT to get complex spectrogram
        D = librosa.stft(audio)
        
        # Separate magnitude and phase
        magnitude, phase = librosa.magphase(D)
        
        # Warp the magnitude spectrogram
        warped_magnitude = np.zeros_like(magnitude)
        
        num_freqs, num_frames = magnitude.shape
        
        # Apply frequency warping to each bin
        for i in range(num_freqs):
            # Normalize frequency to [0, 1]
            norm_freq = i / (num_freqs - 1)
            
            # Apply piecewise linear warping function
            if norm_freq <= 0.5:
                warped_freq = norm_freq * warp_factor
            else:
                warped_freq = 0.5 * warp_factor + (norm_freq - 0.5) * (2 - warp_factor)
                
            # Convert back to bin index
            warped_idx = int(warped_freq * (num_freqs - 1))
            warped_idx = max(0, min(num_freqs - 1, warped_idx))
            
            # Copy magnitude
            warped_magnitude[warped_idx, :] += magnitude[i, :]
        
        # Recombine with original phase
        warped_spectrogram = warped_magnitude * phase
        
        # Inverse STFT to get time-domain signal
        warped_audio = librosa.istft(warped_spectrogram, length=len(audio))
        
        return warped_audio
    
    except Exception as e:
        logger.error(f"Error applying VTLN: {e}")
        return audio