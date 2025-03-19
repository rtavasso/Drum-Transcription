"""
General Utilities for Drum Transcription

This module provides general utility functions used throughout the drum transcription system.

Key Components:
1. Logging utilities
2. Visualization helpers
3. MIDI processing utilities
4. File handling utilities
5. Data conversion utilities

Functions:
- setup_logger(log_dir): Set up logging to file and console
- plot_spectrogram(spec, title): Plot spectrogram for visualization
- plot_predictions(audio, onset_pred, onset_target, velocity_pred, velocity_target): Plot predictions vs targets
- plot_training_curves(train_losses, val_losses, metrics, output_path): Plot training progress curves
- read_midi(midi_path): Read MIDI file and return note events
- write_midi(notes, output_path): Write note events to MIDI file
- ensure_dir(directory): Ensure directory exists, create if not
- get_timestamp(): Get current timestamp for file naming
- set_seed(seed): Set random seeds for reproducibility
- get_device(): Get appropriate device (CPU/GPU) for computation

Implementation Considerations:
- Keep utilities modular and focused on specific tasks
- Implement visualization helpers for debugging and result analysis
- Add profiling utilities for performance optimization
- Ensure proper error handling and logging
- Use consistent formats for data conversion
- Keep utilities compatible with both training and inference pipelines
- Make utilities configurable through parameters
- Add type hints for better code documentation
- Ensure thread safety for utilities used in parallel processing
"""

import os
import sys
import time
import logging
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import mido
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any
import datetime
import json
import librosa
import librosa.display


def setup_logger(log_dir: Optional[str] = None, name: str = "drum_transcription") -> logging.Logger:
    """
    Set up logging to file and console.
    
    Args:
        log_dir: Directory to save log files (None for no file logging)
        name: Logger name
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_dir is provided
    if log_dir:
        # Ensure log directory exists
        ensure_dir(log_dir)
        
        # Create timestamped log file
        timestamp = get_timestamp()
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
        
        # Add file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def plot_spectrogram(
    spec: np.ndarray,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    hop_length: int = 512,
    sample_rate: int = 44100,
    y_axis: str = "log"
) -> None:
    """
    Plot spectrogram for visualization.
    
    Args:
        spec: Spectrogram array (can be magnitude, mel, or dB scale)
        title: Plot title (optional)
        save_path: Path to save the visualization (optional)
        hop_length: Hop length used for STFT
        sample_rate: Audio sample rate
        y_axis: Y-axis scale ('log', 'linear', 'mel', 'hz')
    """
    plt.figure(figsize=(10, 4))
    
    # Plot spectrogram
    librosa.display.specshow(
        spec,
        hop_length=hop_length,
        sr=sample_rate,
        x_axis="time",
        y_axis=y_axis
    )
    
    # Add colorbar
    plt.colorbar(format="%+2.0f dB")
    
    # Set title if provided
    if title:
        plt.title(title)
    
    plt.tight_layout()
    
    # Save or display
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_waveform(
    audio: np.ndarray,
    sample_rate: int = 44100,
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Plot audio waveform.
    
    Args:
        audio: Audio waveform
        sample_rate: Audio sample rate
        title: Plot title (optional)
        save_path: Path to save the visualization (optional)
    """
    plt.figure(figsize=(10, 3))
    
    # Create time axis
    time = np.arange(0, len(audio)) / sample_rate
    
    # Plot waveform
    plt.plot(time, audio)
    
    # Set labels
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    
    # Set title if provided
    if title:
        plt.title(title)
    
    plt.tight_layout()
    
    # Save or display
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    metrics: Dict[str, List[float]],
    output_path: Optional[str] = None
) -> None:
    """
    Plot training progress curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        metrics: Dictionary of metrics to plot
        output_path: Path to save the visualization (optional)
    """
    n_metrics = len(metrics) + 1  # +1 for loss
    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 3 * n_metrics), sharex=True)
    
    # If only one subplot, wrap it in a list for consistent indexing
    if n_metrics == 1:
        axes = [axes]
    
    # Plot loss
    axes[0].plot(train_losses, label="Train")
    axes[0].plot(val_losses, label="Validation")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot other metrics
    for i, (name, values) in enumerate(metrics.items(), 1):
        axes[i].plot(values)
        axes[i].set_ylabel(name)
        axes[i].set_title(f"Validation {name}")
        axes[i].grid(True)
    
    # Set common x-axis label
    axes[-1].set_xlabel("Epoch")
    
    plt.tight_layout()
    
    # Save or display
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


def read_midi(midi_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Read MIDI file and return note events.
    
    Args:
        midi_path: Path to MIDI file
        
    Returns:
        List of note events, each containing onset time, note number, and velocity
    """
    # Load MIDI file
    midi_data = mido.MidiFile(midi_path)
    
    # Convert ticks to seconds for each note
    notes = []
    current_time = 0
    
    for track in midi_data.tracks:
        track_time = 0
        
        for msg in track:
            track_time += msg.time
            
            # Only process note_on events with velocity > 0 (actual note hits)
            if msg.type == 'note_on' and msg.velocity > 0:
                note_time = mido.tick2second(track_time, midi_data.ticks_per_beat, mido.bpm2tempo(120))
                
                notes.append({
                    'onset_time': note_time,
                    'note': msg.note,
                    'velocity': msg.velocity / 127.0  # Normalize to [0, 1]
                })
    
    # Sort notes by onset time
    notes.sort(key=lambda x: x['onset_time'])
    
    return notes


def write_midi(
    notes: List[Dict[str, Any]],
    output_path: Union[str, Path],
    ticks_per_beat: int = 480,
    tempo: int = 120
) -> None:
    """
    Write note events to MIDI file.
    
    Args:
        notes: List of note events (each with onset_time, note, velocity)
        output_path: Path to save MIDI file
        ticks_per_beat: MIDI ticks per beat
        tempo: Tempo in BPM
    """
    # Create MIDI file
    midi_file = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    midi_file.tracks.append(track)
    
    # Add tempo message
    tempo_value = mido.bpm2tempo(tempo)
    track.append(mido.MetaMessage('set_tempo', tempo=tempo_value, time=0))
    
    # Sort notes by onset time
    sorted_notes = sorted(notes, key=lambda x: x['onset_time'])
    
    # Add note events
    last_time = 0
    
    for note_data in sorted_notes:
        # Get note data
        onset_time = note_data['onset_time']
        note = note_data.get('note', 36)  # Default to bass drum (36) if not specified
        velocity = int(note_data['velocity'] * 127)  # Scale from [0, 1] to [0, 127]
        
        # Calculate delta time in ticks
        current_time = mido.second2tick(onset_time, ticks_per_beat, tempo_value)
        delta_time = current_time - last_time
        last_time = current_time
        
        # Add note_on message
        track.append(mido.Message('note_on', note=note, velocity=velocity, time=int(delta_time)))
        
        # Add note_off message (immediately after note_on)
        track.append(mido.Message('note_off', note=note, velocity=0, time=10))
    
    # Save MIDI file
    midi_file.save(str(output_path))


def ensure_dir(directory: Union[str, Path]) -> str:
    """
    Ensure directory exists, create if not.
    
    Args:
        directory: Directory path
        
    Returns:
        Absolute path to directory
    """
    directory = os.path.abspath(directory)
    os.makedirs(directory, exist_ok=True)
    return directory


def get_timestamp() -> str:
    """
    Get current timestamp for file naming.
    
    Returns:
        Timestamp string
    """
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get appropriate device (CPU/GPU) for computation.
    
    Args:
        device: Device string ('cpu', 'cuda', or None for auto-detection)
        
    Returns:
        PyTorch device
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return torch.device(device)


def save_json(data: Any, path: Union[str, Path]) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        path: Path to save file
    """
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(path: Union[str, Path]) -> Any:
    """
    Load data from JSON file.
    
    Args:
        path: Path to JSON file
        
    Returns:
        Loaded data
    """
    with open(path, 'r') as f:
        return json.load(f)


def time_function(func):
    """
    Decorator to measure function execution time.
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds to run")
        return result
    return wrapper


def load_audio(
    file_path: Union[str, Path],
    sample_rate: int = 44100,
    mono: bool = True,
    normalize: bool = True
) -> np.ndarray:
    """
    Load audio file with normalization.
    
    Args:
        file_path: Path to audio file
        sample_rate: Target sample rate
        mono: Whether to convert to mono
        normalize: Whether to normalize audio
        
    Returns:
        Audio array
    """
    # Load audio file
    audio, sr = librosa.load(file_path, sr=sample_rate, mono=mono)
    
    # Normalize if requested
    if normalize:
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
    
    return audio


def compute_melspectrogram(
    audio: np.ndarray,
    sample_rate: int = 44100,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    fmin: int = 20,
    fmax: int = 20000,
    power: float = 2.0
) -> np.ndarray:
    """
    Compute mel spectrogram from audio.
    
    Args:
        audio: Audio waveform
        sample_rate: Audio sample rate
        n_fft: FFT window size
        hop_length: Hop length
        n_mels: Number of mel bands
        fmin: Minimum frequency
        fmax: Maximum frequency
        power: Exponent for the magnitude spectrogram
        
    Returns:
        Mel spectrogram
    """
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=power
    )
    
    # Convert to dB scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db


def augment_audio(
    audio: np.ndarray,
    sample_rate: int = 44100,
    time_stretch_factor: Optional[float] = None,
    pitch_shift_steps: Optional[int] = None,
    noise_factor: Optional[float] = None
) -> np.ndarray:
    """
    Apply audio augmentation.
    
    Args:
        audio: Audio waveform
        sample_rate: Audio sample rate
        time_stretch_factor: Time stretch factor (None for no stretching)
        pitch_shift_steps: Pitch shift in semitones (None for no shifting)
        noise_factor: Noise level factor (None for no noise)
        
    Returns:
        Augmented audio
    """
    # Apply time stretching
    if time_stretch_factor is not None:
        audio = librosa.effects.time_stretch(audio, rate=time_stretch_factor)
    
    # Apply pitch shifting
    if pitch_shift_steps is not None:
        audio = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=pitch_shift_steps)
    
    # Add noise
    if noise_factor is not None:
        noise = np.random.randn(*audio.shape) * noise_factor
        audio = audio + noise
        
        # Clip to valid range
        audio = np.clip(audio, -1.0, 1.0)
    
    return audio


def find_audio_files(directory: Union[str, Path], extensions: List[str] = ['.wav', '.mp3', '.flac']) -> List[Path]:
    """
    Find all audio files in a directory (recursive).
    
    Args:
        directory: Directory to search
        extensions: List of audio file extensions to include
        
    Returns:
        List of audio file paths
    """
    directory = Path(directory)
    audio_files = []
    
    for ext in extensions:
        audio_files.extend(directory.glob(f"**/*{ext}"))
    
    return sorted(audio_files) 