"""
Inference Pipeline for Drum Transcription

This module handles using the trained model for drum transcription.

Key Components:
1. Model loading utilities
2. Audio processing for inference
3. Prediction generation
4. Post-processing of predictions
5. MIDI conversion utilities

Functions:
- load_model(checkpoint_path, model_config, device): Load a trained model from checkpoint
- transcribe_file(audio_path, model, config, output_midi): Transcribe audio file to MIDI
- predictions_to_midi(onset_pred, velocity_pred, threshold, output_path): Convert model predictions to MIDI file
- batch_transcribe(audio_dir, model, config, output_dir): Transcribe multiple audio files
- post_process_predictions(onset_pred, velocity_pred): Apply post-processing to clean up predictions
- adjust_onsets(onsets, velocities, threshold): Adjust onset predictions based on threshold

Implementation Considerations:
- Support batch processing for multiple files
- Implement efficient processing for real-time use cases
- Add post-processing to clean up predictions
- Provide adjustable thresholds for onset detection
- Support different output formats (MIDI, JSON, etc.)
- Ensure proper error handling for missing files or models
- Optimize for inference speed with techniques like model quantization
- Consider memory usage for large audio files
- Support both CPU and GPU inference
- Add progress reporting for batch processing
"""

import os
import json
import torch
import numpy as np
import librosa
import mido
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Union, Any
from tqdm import tqdm

from src.model import get_model, DrumTranscriptionModel
from src.audio import compute_melspectrogram


def load_model(
    checkpoint_path: Union[str, Path],
    model_config: Optional[Dict[str, Any]] = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> DrumTranscriptionModel:
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        model_config: Model configuration dictionary (optional, loaded from checkpoint if not provided)
        device: Device to load the model on ('cpu' or 'cuda')
        
    Returns:
        Loaded model
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    # Get config from checkpoint if not provided
    if model_config is None:
        model_config = checkpoint.get('config', {})
    
    # Get model name from config
    model_name = model_config.get('model_name', 'cnn_lstm')
    
    # Create model instance
    model = get_model(model_name, model_config)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move model to device and set to evaluation mode
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"Model type: {model_name}")
    
    return model


def transcribe_file(
    audio_path: Union[str, Path],
    model: DrumTranscriptionModel,
    config: Dict[str, Any],
    output_midi: Optional[Union[str, Path]] = None,
    output_json: Optional[Union[str, Path]] = None,
    onset_threshold: float = 0.5,
    min_gap_ms: int = 10
) -> Dict[str, np.ndarray]:
    """
    Transcribe audio file to MIDI.
    
    Args:
        audio_path: Path to audio file
        model: Trained drum transcription model
        config: Configuration dictionary
        output_midi: Path to save MIDI output (optional)
        output_json: Path to save JSON output (optional)
        onset_threshold: Threshold for onset detection
        min_gap_ms: Minimum gap between onsets in milliseconds
        
    Returns:
        Dictionary containing onset and velocity predictions
    """
    # Extract parameters from config
    sample_rate = config.get('sample_rate', 44100)
    hop_length = config.get('hop_length', 512)  # Controls time resolution
    midi_sample_rate = config.get('midi_sample_rate', 100)  # Frames per second for MIDI output
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load audio file
    print(f"Loading audio: {audio_path}")
    try:
        audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return {'onset_probs': np.array([]), 'velocities': np.array([])}
    
    # Compute features
    features = process_audio_for_inference(audio, sr, config)
    
    # Move features to device
    features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(device)
    
    # Get predictions
    with torch.no_grad():
        predictions = model(features_tensor)
    
    # Move predictions to CPU and convert to numpy
    onset_probs = predictions['onset_probs'].squeeze().cpu().numpy()
    velocities = predictions['velocities'].squeeze().cpu().numpy()
    
    # Post-process predictions
    onset_probs, velocities = post_process_predictions(
        onset_probs, 
        velocities,
        onset_threshold=onset_threshold,
        min_gap_ms=min_gap_ms,
        sample_rate=midi_sample_rate
    )
    
    # Save to MIDI if specified
    if output_midi is not None:
        predictions_to_midi(
            onset_probs,
            velocities,
            threshold=onset_threshold,
            output_path=output_midi,
            sample_rate=midi_sample_rate
        )
        print(f"MIDI saved to: {output_midi}")
    
    # Save to JSON if specified
    if output_json is not None:
        save_predictions_to_json(
            onset_probs,
            velocities,
            output_path=output_json,
            sample_rate=midi_sample_rate,
            audio_path=audio_path
        )
        print(f"JSON saved to: {output_json}")
    
    return {
        'onset_probs': onset_probs,
        'velocities': velocities
    }


def process_audio_for_inference(
    audio: np.ndarray,
    sample_rate: int,
    config: Dict[str, Any]
) -> np.ndarray:
    """
    Process audio for inference.
    
    Args:
        audio: Audio array
        sample_rate: Audio sample rate
        config: Configuration dictionary
        
    Returns:
        Processed audio features
    """
    # Extract parameters from config
    n_fft = config.get('n_fft', 2048)
    hop_length = config.get('hop_length', 512)
    n_mels = config.get('n_mels', 128)
    
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio, 
        sr=sample_rate, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        n_mels=n_mels
    )
    
    # Convert to dB scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize to [0, 1] range
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
    
    return mel_spec_norm


def post_process_predictions(
    onset_probs: np.ndarray,
    velocities: np.ndarray,
    onset_threshold: float = 0.5,
    min_gap_ms: int = 10,
    sample_rate: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply post-processing to clean up predictions.
    
    Args:
        onset_probs: Onset probability predictions
        velocities: Velocity predictions
        onset_threshold: Threshold for onset detection
        min_gap_ms: Minimum gap between onsets in milliseconds
        sample_rate: Sample rate in Hz (frames per second)
        
    Returns:
        Tuple of (cleaned_onsets, cleaned_velocities)
    """
    # Convert min_gap_ms to frames
    min_gap_frames = int(min_gap_ms * sample_rate / 1000)
    
    # Threshold onsets
    onset_binary = (onset_probs > onset_threshold).astype(np.float32)
    
    # Find onset positions (where the signal changes from 0 to 1)
    onset_positions = np.where(np.diff(np.concatenate([[0], onset_binary])) > 0)[0]
    
    # Initialize cleaned onsets and velocities
    cleaned_onsets = np.zeros_like(onset_probs)
    cleaned_velocities = np.zeros_like(velocities)
    
    # Process each onset
    last_onset_pos = -min_gap_frames
    for pos in onset_positions:
        # Skip if too close to previous onset
        if pos - last_onset_pos < min_gap_frames:
            continue
        
        # Add onset
        cleaned_onsets[pos] = 1.0
        cleaned_velocities[pos] = velocities[pos]
        last_onset_pos = pos
    
    return cleaned_onsets, cleaned_velocities


def adjust_onsets(
    onsets: np.ndarray,
    velocities: np.ndarray,
    threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adjust onset predictions based on threshold.
    
    Args:
        onsets: Onset predictions
        velocities: Velocity predictions
        threshold: Threshold for onset detection
        
    Returns:
        Tuple of (thresholded_onsets, thresholded_velocities)
    """
    # Apply threshold to onsets
    thresholded_onsets = (onsets > threshold).astype(np.float32)
    
    # Apply onset mask to velocities 
    thresholded_velocities = velocities * thresholded_onsets
    
    return thresholded_onsets, thresholded_velocities


def predictions_to_midi(
    onset_pred: np.ndarray,
    velocity_pred: np.ndarray,
    threshold: float = 0.5,
    output_path: Union[str, Path] = 'output.mid',
    sample_rate: int = 100,
    ticks_per_beat: int = 480,
    tempo: int = 120
) -> None:
    """
    Convert model predictions to MIDI file.
    
    Args:
        onset_pred: Onset predictions
        velocity_pred: Velocity predictions
        threshold: Threshold for onset detection
        output_path: Path to save MIDI file
        sample_rate: Sample rate in Hz (frames per second)
        ticks_per_beat: MIDI ticks per beat
        tempo: Tempo in BPM
    """
    # Create MIDI file
    midi_file = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    midi_file.tracks.append(track)
    
    # Set tempo
    tempo_us = mido.bpm2tempo(tempo)
    track.append(mido.MetaMessage('set_tempo', tempo=tempo_us, time=0))
    
    # Define drum mappings (General MIDI drum map)
    drum_map = {
        'kick': 36,
        'snare': 38,
        'hihat': 42,
        'tom': 45,
        'cymbal': 49
    }
    
    # For a real system, we would have a drum classifier to assign each onset to a drum type
    # For this implementation, we'll use a simplified approach: assign all onsets to kick drum
    # In a full implementation, this would be replaced with a proper drum classification
    drum_type = 'kick'
    note_number = drum_map[drum_type]
    
    # Apply threshold
    onset_binary = (onset_pred > threshold).astype(np.float32)
    
    # Find onset positions (where the signal changes from 0 to 1)
    onset_positions = np.where(np.diff(np.concatenate([[0], onset_binary])) > 0)[0]
    
    # Add note events to track
    prev_time = 0
    for pos in onset_positions:
        # Calculate time in seconds
        time_sec = pos / sample_rate
        
        # Convert to MIDI ticks
        time_ticks = int(time_sec * tempo / 60 * ticks_per_beat)
        
        # Calculate delta time
        delta_time = time_ticks - prev_time
        
        # Calculate velocity (scale to MIDI range 1-127)
        velocity = int(velocity_pred[pos] * 127)
        if velocity < 1:
            velocity = 1  # Ensure minimum velocity
        
        # Add note_on event
        track.append(mido.Message('note_on', note=note_number, velocity=velocity, time=delta_time))
        
        # Add note_off event (10 ticks later)
        track.append(mido.Message('note_off', note=note_number, velocity=0, time=10))
        
        # Update previous time
        prev_time = time_ticks + 10
    
    # Save MIDI file
    midi_file.save(output_path)


def save_predictions_to_json(
    onset_pred: np.ndarray,
    velocity_pred: np.ndarray,
    output_path: Union[str, Path],
    sample_rate: int = 100,
    audio_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Save predictions to JSON file.
    
    Args:
        onset_pred: Onset predictions
        velocity_pred: Velocity predictions
        output_path: Path to save JSON file
        sample_rate: Sample rate in Hz (frames per second)
        audio_path: Path to source audio file (optional)
    """
    # Find onset positions (where predictions exceed 0.5)
    onset_positions = np.where(onset_pred > 0.5)[0]
    
    # Create list of events
    events = []
    for pos in onset_positions:
        events.append({
            'time': float(pos / sample_rate),
            'velocity': float(velocity_pred[pos])
        })
    
    # Create output data
    data = {
        'sample_rate': sample_rate,
        'events': events
    }
    
    # Add source audio path if provided
    if audio_path is not None:
        data['source_audio'] = str(audio_path)
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def batch_transcribe(
    audio_dir: Union[str, Path],
    model: DrumTranscriptionModel,
    config: Dict[str, Any],
    output_dir: Union[str, Path],
    file_pattern: str = '*.wav',
    onset_threshold: float = 0.5,
    save_midi: bool = True,
    save_json: bool = True
) -> Dict[str, Any]:
    """
    Transcribe multiple audio files.
    
    Args:
        audio_dir: Directory containing audio files
        model: Trained drum transcription model
        config: Configuration dictionary
        output_dir: Directory to save outputs
        file_pattern: Pattern to match audio files
        onset_threshold: Threshold for onset detection
        save_midi: Whether to save MIDI files
        save_json: Whether to save JSON files
        
    Returns:
        Dictionary containing results for each file
    """
    # Convert to Path objects
    audio_dir = Path(audio_dir)
    output_dir = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of audio files
    audio_files = list(audio_dir.glob(file_pattern))
    
    # Check if any files were found
    if not audio_files:
        print(f"No audio files found in {audio_dir} matching {file_pattern}")
        return {}
    
    # Initialize results dictionary
    results = {}
    
    # Process each file
    for audio_file in tqdm(audio_files, desc="Transcribing files"):
        # Determine output paths
        file_stem = audio_file.stem
        midi_path = output_dir / f"{file_stem}.mid" if save_midi else None
        json_path = output_dir / f"{file_stem}.json" if save_json else None
        
        # Transcribe file
        try:
            predictions = transcribe_file(
                audio_path=audio_file,
                model=model,
                config=config,
                output_midi=midi_path,
                output_json=json_path,
                onset_threshold=onset_threshold
            )
            
            # Store results
            results[file_stem] = {
                'audio_path': str(audio_file),
                'midi_path': str(midi_path) if midi_path else None,
                'json_path': str(json_path) if json_path else None,
                'onset_count': int(np.sum(predictions['onset_probs'] > onset_threshold)),
                'status': 'success'
            }
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            results[file_stem] = {
                'audio_path': str(audio_file),
                'status': 'error',
                'error': str(e)
            }
    
    # Save summary
    summary_path = output_dir / 'transcription_summary.json'
    with open(summary_path, 'w') as f:
        json.dump({
            'config': {
                'onset_threshold': onset_threshold,
                'file_pattern': file_pattern,
                'save_midi': save_midi,
                'save_json': save_json
            },
            'results': results,
            'total_files': len(audio_files),
            'successful': sum(1 for r in results.values() if r['status'] == 'success'),
            'failed': sum(1 for r in results.values() if r['status'] == 'error')
        }, f, indent=2)
    
    print(f"Transcription complete. Results saved to {summary_path}")
    
    return results


def real_time_inference_setup(
    model: DrumTranscriptionModel,
    config: Dict[str, Any],
    buffer_size: int = 4096,
    hop_size: int = 512
) -> Dict[str, Any]:
    """
    Set up model for real-time inference.
    
    Args:
        model: Trained drum transcription model
        config: Configuration dictionary
        buffer_size: Audio buffer size for processing
        hop_size: Hop size for processing
        
    Returns:
        Dictionary containing setup information
    """
    # Extract parameters
    sample_rate = config.get('sample_rate', 44100)
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    # Calculate frames per second
    frames_per_second = sample_rate / hop_size
    
    # Set model to evaluation mode
    model.eval()
    
    # Create processing buffers
    audio_buffer = np.zeros(buffer_size, dtype=np.float32)
    feature_buffer = np.zeros((config.get('n_mels', 128), buffer_size // hop_size), dtype=np.float32)
    
    # Return setup information
    return {
        'model': model,
        'config': config,
        'buffer_size': buffer_size,
        'hop_size': hop_size,
        'sample_rate': sample_rate,
        'frames_per_second': frames_per_second,
        'audio_buffer': audio_buffer,
        'feature_buffer': feature_buffer,
        'device': device
    }


def process_audio_chunk(
    audio_chunk: np.ndarray,
    inference_state: Dict[str, Any]
) -> Dict[str, np.ndarray]:
    """
    Process a chunk of audio for real-time inference.
    
    Args:
        audio_chunk: Audio chunk to process
        inference_state: Inference state from real_time_inference_setup
        
    Returns:
        Dictionary containing onset and velocity predictions
    """
    # Extract components from inference state
    model = inference_state['model']
    config = inference_state['config']
    buffer_size = inference_state['buffer_size']
    hop_size = inference_state['hop_size']
    audio_buffer = inference_state['audio_buffer']
    device = inference_state['device']
    
    # Update audio buffer (shift left and add new chunk)
    shift_size = min(len(audio_chunk), buffer_size)
    audio_buffer = np.roll(audio_buffer, -shift_size)
    audio_buffer[-shift_size:] = audio_chunk[-shift_size:]
    
    # Process audio
    features = process_audio_for_inference(audio_buffer, inference_state['sample_rate'], config)
    
    # Move features to device
    features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(device)
    
    # Get predictions
    with torch.no_grad():
        predictions = model(features_tensor)
    
    # Move predictions to CPU and convert to numpy
    onset_probs = predictions['onset_probs'].squeeze().cpu().numpy()
    velocities = predictions['velocities'].squeeze().cpu().numpy()
    
    # Post-process predictions (only return predictions for the new chunk)
    frames_per_chunk = shift_size // hop_size
    onset_probs = onset_probs[-frames_per_chunk:] if frames_per_chunk > 0 else np.array([])
    velocities = velocities[-frames_per_chunk:] if frames_per_chunk > 0 else np.array([])
    
    return {
        'onset_probs': onset_probs,
        'velocities': velocities
    } 