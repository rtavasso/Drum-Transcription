"""
Audio Processing Utilities for Drum Transcription

This module handles all audio processing tasks required for the drum transcription system.

Key Components:
1. Audio loading and normalization
2. Feature extraction (mel spectrogram, STFT, etc.)
3. Data augmentation for training

Functions:
- load_audio(file_path): Load and normalize audio from file
- compute_melspectrogram(audio, sample_rate, n_fft, hop_length, n_mels): Convert audio to mel spectrogram
- compute_stft(audio, n_fft, hop_length, window): Compute Short-Time Fourier Transform
- time_stretch(audio, rate): Time stretch augmentation
- pitch_shift(audio, steps): Pitch shift augmentation
- add_noise(audio, noise_level): Add controlled noise for augmentation

Implementation Considerations:
- Use standard libraries like librosa or torchaudio for audio processing
- Ensure efficient processing for real-time applications
- Keep augmentation methods configurable via parameters
- Implement caching for frequently used audio files or computed features
- Consider multiprocessing for batch processing of audio files
- Handle different audio formats and sampling rates gracefully
- Ensure proper normalization of audio for stable model inputs
"""

import os
import numpy as np
import torch
import torchaudio
import torchaudio.functional as F
import librosa
import logging
from typing import Tuple, Optional, Union

# Configure logger
logger = logging.getLogger(__name__)

# Set default parameters
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_N_FFT = 2048
DEFAULT_HOP_LENGTH = 512
DEFAULT_N_MELS = 128


def load_audio(
    file_path: str, 
    target_sr: int = DEFAULT_SAMPLE_RATE, 
    mono: bool = True,
    normalize: bool = True
) -> Tuple[torch.Tensor, int]:
    """
    Load audio file and convert to target sample rate if needed.
    
    Args:
        file_path: Path to audio file
        target_sr: Target sample rate
        mono: Convert to mono if True
        normalize: Normalize audio to [-1, 1] if True
        
    Returns:
        Tuple of (audio_tensor, sample_rate)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    try:
        # Load audio using torchaudio
        audio, sr = torchaudio.load(file_path)
        
        # Convert to mono if needed
        if mono and audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        # Resample if needed
        if sr != target_sr:
            audio = F.resample(audio, sr, target_sr)
            sr = target_sr
            
        # Normalize if requested
        if normalize:
            if torch.max(torch.abs(audio)) > 0:
                audio = audio / torch.max(torch.abs(audio))
        
        return audio, sr
    
    except Exception as e:
        # Fall back to librosa if torchaudio fails
        logger.warning(f"torchaudio failed to load {file_path}, falling back to librosa: {str(e)}")
        audio, sr = librosa.load(file_path, sr=target_sr, mono=mono)
        audio = torch.from_numpy(audio).float()
        
        # Add channel dimension if mono
        if mono and audio.dim() == 1:
            audio = audio.unsqueeze(0)
            
        # Normalize if requested
        if normalize and torch.max(torch.abs(audio)) > 0:
            audio = audio / torch.max(torch.abs(audio))
            
        return audio, sr


def compute_melspectrogram(
    audio: torch.Tensor,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    n_mels: int = DEFAULT_N_MELS,
    power: float = 2.0,
    normalized: bool = True,
    log_mel: bool = True
) -> torch.Tensor:
    """
    Compute mel spectrogram from audio tensor.
    
    Args:
        audio: Audio tensor [channels, samples]
        sample_rate: Audio sample rate
        n_fft: FFT window size
        hop_length: Hop length between frames
        n_mels: Number of mel bands
        power: Power of the magnitude spectrogram
        normalized: Whether to normalize the mel spectrogram
        log_mel: Whether to convert to log scale
        
    Returns:
        Mel spectrogram tensor [channels, n_mels, time]
    """
    # Make sure audio is 2D [channels, samples]
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    
    # Initialize mel spectrogram transform
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=power,
        center=True,  # Match librosa's default behavior
        pad_mode='reflect'  # Match librosa's default behavior
    )
    
    # Compute mel spectrogram for each channel
    mel_spectrograms = []
    for ch in range(audio.shape[0]):
        mel_spec = mel_transform(audio[ch])
        
        # Convert to log scale if requested
        if log_mel:
            mel_spec = torch.log(torch.clamp(mel_spec, min=1e-10))
            
        # Normalize if requested
        if normalized and torch.max(mel_spec) - torch.min(mel_spec) > 0:
            mel_spec = (mel_spec - torch.min(mel_spec)) / (torch.max(mel_spec) - torch.min(mel_spec))
            
        mel_spectrograms.append(mel_spec)
    
    # Stack along channel dimension
    return torch.stack(mel_spectrograms)


def compute_stft(
    audio: torch.Tensor,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    window: Optional[torch.Tensor] = None,
    center: bool = True,
    normalized: bool = False
) -> torch.Tensor:
    """
    Compute Short-Time Fourier Transform (STFT) of audio.
    
    Args:
        audio: Audio tensor [channels, samples]
        n_fft: FFT window size
        hop_length: Hop length between frames
        window: Window function, default is hann_window
        center: Whether to pad signal on both sides
        normalized: Whether to normalize the STFT
        
    Returns:
        Complex STFT tensor [channels, n_fft//2+1, frames]
    """
    # Make sure audio is 2D [channels, samples]
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    
    # Create window if not provided
    if window is None:
        window = torch.hann_window(n_fft)
    
    # Compute STFT for each channel
    stft_features = []
    for ch in range(audio.shape[0]):
        stft = torch.stft(
            audio[ch],
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            center=center,
            normalized=normalized,
            return_complex=True,
            pad_mode='reflect'  # Match librosa's default behavior
        )
        stft_features.append(stft)
    
    # Stack along channel dimension
    return torch.stack(stft_features)


def time_stretch(
    audio: torch.Tensor,
    rate: float,
    sample_rate: int = DEFAULT_SAMPLE_RATE
) -> torch.Tensor:
    """
    Time stretch the audio by a rate factor using PyTorch.
    
    Args:
        audio: Audio tensor [channels, samples]
        rate: Stretch factor, > 1 speeds up, < 1 slows down
        sample_rate: Audio sample rate
        
    Returns:
        Time-stretched audio tensor [channels, new_samples]
    """
    # Make sure audio is 2D [channels, samples]
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    
    # Simple implementation of time stretching:
    # 1. For rate > 1 (speed up): Take every nth sample
    # 2. For rate < 1 (slow down): Repeat samples
    
    # Target number of samples after time stretching
    orig_samples = audio.shape[1]
    target_samples = int(orig_samples / rate)
    
    # Time stretch each channel
    stretched_channels = []
    for ch in range(audio.shape[0]):
        if rate > 1.0:  # Speed up - we need fewer samples
            # Use interpolation for smoother results
            indices = torch.linspace(0, orig_samples - 1, steps=target_samples)
            indices_floor = indices.long()
            frac = indices - indices_floor
            
            # Linear interpolation
            left = audio[ch, indices_floor]
            right = audio[ch, torch.minimum(indices_floor + 1, torch.tensor(orig_samples - 1))]
            stretched = left + frac * (right - left)
        else:  # Slow down - we need more samples
            # Use simple linear interpolation
            indices = torch.linspace(0, orig_samples - 1, steps=target_samples)
            indices_floor = indices.long()
            frac = indices - indices_floor
            
            # Linear interpolation
            left = audio[ch, indices_floor]
            right = audio[ch, torch.minimum(indices_floor + 1, torch.tensor(orig_samples - 1))]
            stretched = left + frac * (right - left)
        
        stretched_channels.append(stretched)
    
    # Stack channels
    result = torch.stack(stretched_channels)
    
    return result


def pitch_shift(
    audio: torch.Tensor,
    steps: float,
    sample_rate: int = DEFAULT_SAMPLE_RATE
) -> torch.Tensor:
    """
    Shift the pitch of audio by steps semitones using PyTorch.
    
    Args:
        audio: Audio tensor [channels, samples]
        steps: Number of semitones to shift (can be fractional)
        sample_rate: Audio sample rate
        
    Returns:
        Pitch-shifted audio tensor [channels, samples]
    """
    # Make sure audio is 2D [channels, samples]
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    
    # Calculate the frequency shift factor
    # Each semitone is a factor of 2^(1/12)
    factor = 2.0 ** (steps / 12.0)
    
    # Simplify the approach: use integer ratios for resampling
    # Convert factor to a rational approximation
    factor_mult = 1000
    ratio_num = int(factor * factor_mult)
    ratio_denom = factor_mult
    
    # Pitch shift each channel
    shifted_channels = []
    for ch in range(audio.shape[0]):
        # First resample to change the pitch (keeping the same number of samples)
        # This is intentionally "wrong" - we're changing the sample rate which changes the pitch
        # but also changes the duration, which we'll fix in the next step
        audio_tensor = audio[ch].unsqueeze(0)  # Add batch dimension
        
        # Resample with integer sample rates to avoid errors
        stretched = F.resample(
            audio_tensor, 
            orig_freq=ratio_denom, 
            new_freq=ratio_num
        ).squeeze(0)
        
        # Now resample back to the original number of samples to fix the duration
        orig_len = audio.shape[1]
        target_samples = int(stretched.shape[0])
        
        # Simple resampling to original length
        if target_samples != orig_len:
            indices = torch.linspace(0, target_samples - 1, steps=orig_len)
            indices_floor = indices.long()
            frac = indices - indices_floor
            
            # Linear interpolation
            left = stretched[indices_floor]
            right = stretched[torch.minimum(indices_floor + 1, torch.tensor(target_samples - 1))]
            shifted = left + frac * (right - left)
        else:
            shifted = stretched
            
        shifted_channels.append(shifted)
    
    # Stack channels
    result = torch.stack(shifted_channels)
    
    return result


def add_noise(
    audio: torch.Tensor,
    noise_level: float = 0.005,
    noise_type: str = 'gaussian'
) -> torch.Tensor:
    """
    Add noise to audio tensor.
    
    Args:
        audio: Audio tensor [channels, samples]
        noise_level: Level of noise to add
        noise_type: Type of noise ('gaussian', 'uniform')
        
    Returns:
        Noisy audio tensor [channels, samples]
    """
    # Make sure audio is 2D [channels, samples]
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    
    # Generate noise based on specified type
    if noise_type == 'gaussian':
        noise = torch.randn_like(audio) * noise_level
    elif noise_type == 'uniform':
        noise = (torch.rand_like(audio) * 2 - 1) * noise_level
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")
    
    # Add noise to audio
    noisy_audio = audio + noise
    
    # Clip to [-1, 1] to avoid distortion
    noisy_audio = torch.clamp(noisy_audio, -1.0, 1.0)
    
    return noisy_audio


def augment_audio(
    audio: torch.Tensor,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    time_stretch_rate: Optional[float] = None,
    pitch_shift_steps: Optional[float] = None,
    noise_level: Optional[float] = None
) -> torch.Tensor:
    """
    Apply multiple augmentations to audio tensor.
    
    Args:
        audio: Audio tensor [channels, samples]
        sample_rate: Audio sample rate
        time_stretch_rate: Rate for time stretching, if None, no stretching is applied
        pitch_shift_steps: Steps for pitch shifting, if None, no shifting is applied
        noise_level: Level of noise to add, if None, no noise is added
        
    Returns:
        Augmented audio tensor [channels, new_samples]
    """
    # Make sure audio is 2D [channels, samples]
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    
    result = audio
    
    # Apply time stretching if requested
    if time_stretch_rate is not None:
        result = time_stretch(result, time_stretch_rate, sample_rate)
    
    # Apply pitch shifting if requested
    if pitch_shift_steps is not None:
        result = pitch_shift(result, pitch_shift_steps, sample_rate)
    
    # Add noise if requested
    if noise_level is not None:
        result = add_noise(result, noise_level)
    
    return result 