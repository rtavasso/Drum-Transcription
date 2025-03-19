"""
Dataset Loading and Processing for Drum Transcription

This module handles dataset loading, processing, and augmentation for the drum transcription system.

Key Components:
1. Dataset classes for different data formats
2. Data loading and preprocessing utilities
3. Training/validation/test splitting
4. Synthetic dataset generation

Classes:
- DrumAudioDataset: Dataset for paired audio and MIDI drum files
    - Loads and processes audio files
    - Loads and processes corresponding MIDI files
    - Applies augmentation during training
    - Returns (audio_features, onsets, velocities) tuples

Functions:
- create_synthetic_dataset(output_dir, num_samples): Create synthetic drum data
- load_dataset(dataset_path, split): Load dataset with specified split
- midi_to_onsets(midi_file, sample_rate): Convert MIDI file to onset and velocity arrays
- align_audio_midi(audio, midi, sample_rate): Ensure audio and MIDI are aligned

Implementation Considerations:
- Support multiple dataset formats (paired audio/MIDI, audio with annotations)
- Implement efficient loading with caching when possible
- Use on-the-fly augmentation to increase training variety
- Create train/validation/test splits deterministically
- Support generating synthetic data for testing
- Ensure proper synchronization between audio and MIDI data
- Handle variable-length inputs efficiently
- Use lazy loading for large datasets to reduce memory usage
- Support batch processing for efficient training
"""

import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import mido
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union

class DrumAudioDataset(Dataset):
    """Dataset for paired audio and MIDI drum files"""
    
    def __init__(self, 
                directory: str,
                sample_length: float = 5.0, 
                sample_rate: int = 44100, 
                midi_sample_rate: int = 100,
                split: str = 'train', 
                augmentation: bool = True,
                cache_size: int = 100,
                test_split: float = 0.1,
                val_split: float = 0.1,
                seed: int = 42):
        """
        Initialize the drum audio dataset.
        
        Args:
            directory: Path to dataset directory containing audio and MIDI files
            sample_length: Length of audio samples in seconds
            sample_rate: Audio sample rate
            midi_sample_rate: MIDI sampling rate in Hz (frames per second)
            split: Dataset split ('train', 'val', 'test')
            augmentation: Whether to apply augmentation during training
            cache_size: Number of samples to cache in memory
            test_split: Fraction of data to use for testing
            val_split: Fraction of data to use for validation
            seed: Random seed for reproducibility
        """
        self.directory = Path(directory)
        self.sample_length = sample_length
        self.sample_rate = sample_rate
        self.midi_sample_rate = midi_sample_rate
        self.split = split
        self.augmentation = augmentation and split == 'train'
        self.cache_size = cache_size
        self.cache = {}
        
        # Set random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        
        # Get list of audio files and their corresponding MIDI files
        self.audio_midi_pairs = self._get_audio_midi_pairs()
        
        # Split dataset
        self._create_splits(test_split, val_split)
        
        # Select the appropriate split
        if split == 'train':
            self.file_pairs = self.train_pairs
        elif split == 'val':
            self.file_pairs = self.val_pairs
        elif split == 'test':
            self.file_pairs = self.test_pairs
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")
        
        print(f"Loaded {len(self.file_pairs)} samples for {split} split")
    
    def _get_audio_midi_pairs(self) -> List[Tuple[Path, Path]]:
        """
        Get pairs of audio and corresponding MIDI files.
        
        Returns:
            List of (audio_path, midi_path) tuples
        """
        audio_files = list(self.directory.glob("**/*.wav"))
        pairs = []
        
        for audio_file in audio_files:
            # Look for matching MIDI file (same name, different extension)
            midi_file = audio_file.with_suffix('.mid')
            if midi_file.exists():
                pairs.append((audio_file, midi_file))
        
        return pairs
    
    def _create_splits(self, test_split: float, val_split: float) -> None:
        """
        Create train/validation/test splits.
        
        Args:
            test_split: Fraction of data to use for testing
            val_split: Fraction of data to use for validation
        """
        # Shuffle the pairs for random splitting
        all_pairs = self.audio_midi_pairs.copy()
        random.shuffle(all_pairs)
        
        # Calculate split indices
        n_test = max(1, int(len(all_pairs) * test_split))
        n_val = max(1, int(len(all_pairs) * val_split))
        
        # Create splits
        self.test_pairs = all_pairs[:n_test]
        self.val_pairs = all_pairs[n_test:n_test + n_val]
        self.train_pairs = all_pairs[n_test + n_val:]
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.file_pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (audio_features, onsets, velocities)
        """
        # Check if sample is in cache
        if idx in self.cache:
            return self.cache[idx]
        
        # Load audio and MIDI files
        audio_path, midi_path = self.file_pairs[idx]
        
        # Load and process audio
        audio = self._load_and_process_audio(audio_path)
        
        # Load and process MIDI
        onsets, velocities = midi_to_onsets(midi_path, self.sample_rate, self.midi_sample_rate)
        
        # Align audio and MIDI data
        audio, onsets, velocities = align_audio_midi(audio, onsets, velocities, self.sample_rate, self.midi_sample_rate)
        
        # Convert to tensors
        audio_tensor = torch.from_numpy(audio).float()
        onsets_tensor = torch.from_numpy(onsets).float()
        velocities_tensor = torch.from_numpy(velocities).float()
        
        # Apply augmentation if enabled
        if self.augmentation:
            audio_tensor, onsets_tensor, velocities_tensor = self._apply_augmentation(
                audio_tensor, onsets_tensor, velocities_tensor
            )
        
        # Add to cache if not full
        if len(self.cache) < self.cache_size:
            self.cache[idx] = (audio_tensor, onsets_tensor, velocities_tensor)
        
        return audio_tensor, onsets_tensor, velocities_tensor
    
    def _load_and_process_audio(self, audio_path: Path) -> np.ndarray:
        """
        Load and process audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Processed audio array
        """
        # Load audio file
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        
        # Ensure correct length
        target_length = int(self.sample_length * self.sample_rate)
        
        if len(audio) < target_length:
            # Pad if too short
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        elif len(audio) > target_length:
            # Take a random segment if in training mode, otherwise take the beginning
            if self.split == 'train':
                start = random.randint(0, len(audio) - target_length)
                audio = audio[start:start + target_length]
            else:
                audio = audio[:target_length]
        
        return audio
    
    def _apply_augmentation(
        self, 
        audio: torch.Tensor, 
        onsets: torch.Tensor, 
        velocities: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply augmentation to audio and labels.
        
        Args:
            audio: Audio tensor
            onsets: Onset tensor
            velocities: Velocity tensor
            
        Returns:
            Augmented audio, onsets, and velocities
        """
        # Apply simple time shift augmentation
        # This is a basic example - more complex augmentations would be implemented here
        if random.random() < 0.5:
            shift = random.randint(0, int(0.1 * self.sample_rate))  # Shift up to 100ms
            audio = torch.roll(audio, shifts=shift, dims=0)
            onsets = torch.roll(onsets, shifts=shift // (self.sample_rate // self.midi_sample_rate), dims=0)
            velocities = torch.roll(velocities, shifts=shift // (self.sample_rate // self.midi_sample_rate), dims=0)
        
        # Volume augmentation
        if random.random() < 0.5:
            volume_factor = random.uniform(0.8, 1.2)
            audio = audio * volume_factor
            # Clamp to valid range
            audio = torch.clamp(audio, -1.0, 1.0)
        
        return audio, onsets, velocities


def midi_to_onsets(
    midi_file: Union[str, Path],
    sample_rate: int,
    midi_sample_rate: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert MIDI file to onset and velocity arrays.
    
    Args:
        midi_file: Path to MIDI file
        sample_rate: Audio sample rate
        midi_sample_rate: MIDI sampling rate in Hz
        
    Returns:
        Tuple of (onsets, velocities) arrays
    """
    # Calculate total number of frames based on MIDI file duration
    midi_data = mido.MidiFile(midi_file)
    duration_seconds = midi_data.length
    n_frames = int(duration_seconds * midi_sample_rate)
    
    # Initialize empty arrays
    onsets = np.zeros(n_frames, dtype=np.float32)
    velocities = np.zeros(n_frames, dtype=np.float32)
    
    # Convert ticks to seconds for each note
    current_time = 0
    
    for msg in midi_data:
        current_time += msg.time
        
        # Only process note_on events with velocity > 0 (actual drum hits)
        if msg.type == 'note_on' and msg.velocity > 0:
            # Convert time to frame index
            frame_idx = int(current_time * midi_sample_rate)
            
            # Ensure we don't exceed array bounds
            if 0 <= frame_idx < n_frames:
                onsets[frame_idx] = 1.0
                velocities[frame_idx] = msg.velocity / 127.0  # Normalize to [0, 1]
    
    return onsets, velocities


def align_audio_midi(
    audio: np.ndarray,
    onsets: np.ndarray,
    velocities: np.ndarray,
    sample_rate: int,
    midi_sample_rate: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Ensure audio and MIDI data are aligned and have matching lengths.
    
    Args:
        audio: Audio array
        onsets: Onset array
        velocities: Velocity array
        sample_rate: Audio sample rate
        midi_sample_rate: MIDI sampling rate
        
    Returns:
        Aligned (audio, onsets, velocities)
    """
    # Calculate expected frame counts
    audio_frames = len(audio)
    expected_midi_frames = int(audio_frames / sample_rate * midi_sample_rate)
    
    # Trim or pad MIDI data as needed
    if len(onsets) > expected_midi_frames:
        onsets = onsets[:expected_midi_frames]
        velocities = velocities[:expected_midi_frames]
    elif len(onsets) < expected_midi_frames:
        onsets = np.pad(onsets, (0, expected_midi_frames - len(onsets)), mode='constant')
        velocities = np.pad(velocities, (0, expected_midi_frames - len(velocities)), mode='constant')
    
    return audio, onsets, velocities


def create_synthetic_dataset(
    output_dir: Union[str, Path],
    num_samples: int = 100,
    sample_rate: int = 44100,
    duration: float = 10.0,
    instruments: List[str] = ["kick", "snare", "hihat"],
    complexity: int = 5,
    seed: int = 42
) -> Dict:
    """
    Create synthetic drum data for testing and development.
    
    Args:
        output_dir: Directory to save synthetic data
        num_samples: Number of synthetic samples to generate
        sample_rate: Sample rate for audio files
        duration: Duration of each sample in seconds
        instruments: List of drum instruments to include
        complexity: Complexity of patterns from 1-10
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing metadata about the generated dataset
    """
    from scipy.io import wavfile
    import pretty_midi
    import numpy as np
    import random
    import json
    from pathlib import Path
    
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Define drum samples (would normally load from files)
    # Here we're creating synthetic drum sounds for demonstration
    def create_drum_sample(name, sample_rate):
        """Create a synthetic drum sound"""
        length = int(sample_rate * 0.5)  # 500ms samples
        sample = np.zeros(length)
        
        if name == "kick":
            # Synthesize a kick drum (simple sine sweep with decay)
            t = np.linspace(0, 0.5, length)
            freq = np.exp(np.linspace(np.log(150), np.log(50), length))
            sample = np.sin(2 * np.pi * freq * t / sample_rate)
            sample *= np.exp(-t * 10)  # Decay envelope
            
        elif name == "snare":
            # Synthesize a snare drum (noise with decay)
            t = np.linspace(0, 0.5, length)
            sample = np.random.randn(length) * 0.5
            sample += np.sin(2 * np.pi * 200 * t)  # Add some tonality
            sample *= np.exp(-t * 15)  # Decay envelope
            
        elif name == "hihat":
            # Synthesize a hi-hat (filtered noise with fast decay)
            t = np.linspace(0, 0.5, length)
            sample = np.random.randn(length)
            # Crude high-pass filter (just to make it sound higher)
            sample = np.diff(sample, prepend=0)
            sample *= np.exp(-t * 30)  # Fast decay envelope
            
        elif name == "tom":
            # Synthesize a tom (mid-range sine with decay)
            t = np.linspace(0, 0.5, length)
            freq = np.exp(np.linspace(np.log(200), np.log(100), length))
            sample = np.sin(2 * np.pi * freq * t / sample_rate)
            sample *= np.exp(-t * 8)  # Decay envelope
            
        elif name == "cymbal":
            # Synthesize a cymbal (filtered noise with slow decay)
            t = np.linspace(0, 0.5, length)
            sample = np.random.randn(length)
            # Crude band-pass filter
            sample = np.diff(np.diff(sample, prepend=0), prepend=0)
            sample *= np.exp(-t * 5)  # Slow decay envelope
        
        # Normalize
        if np.max(np.abs(sample)) > 0:
            sample = sample / np.max(np.abs(sample)) * 0.9
            
        return sample
    
    # Create drum samples
    drum_samples = {name: create_drum_sample(name, sample_rate) for name in instruments}
    
    # Define MIDI note numbers for drum instruments (GM standard)
    midi_notes = {
        "kick": 36,   # Bass Drum 1
        "snare": 38,  # Acoustic Snare
        "hihat": 42,  # Closed Hi-Hat
        "tom": 45,    # Low Tom
        "cymbal": 49  # Crash Cymbal 1
    }
    
    # Create metadata dictionary
    metadata = {
        "dataset_info": {
            "num_samples": num_samples,
            "sample_rate": sample_rate,
            "duration": duration,
            "instruments": instruments,
            "complexity": complexity,
            "seed": seed
        },
        "samples": []
    }
    
    # Generate samples
    for i in range(num_samples):
        # Sample ID
        sample_id = f"sample_{i:04d}"
        
        # Define file paths - now in the same main directory
        audio_path = output_dir / f"{sample_id}.wav"
        midi_path = output_dir / f"{sample_id}.mid"
        
        # Generate a drum pattern
        # The complexity parameter affects:
        # - Number of hits
        # - Rhythmic variation
        # - Velocity variation
        
        # Create an empty audio array
        audio_length = int(duration * sample_rate)
        audio = np.zeros(audio_length)
        
        # Create a MIDI file
        midi = pretty_midi.PrettyMIDI()
        drum_track = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
        
        # Sample info for metadata
        sample_info = {
            "id": sample_id,
            "audio_file": str(audio_path.relative_to(output_dir)),
            "midi_file": str(midi_path.relative_to(output_dir)),
            "pattern": {}
        }
        
        # For each instrument, create a pattern
        for instrument in instruments:
            # Determine the base pattern density based on complexity
            # Higher complexity means more notes
            base_density = 0.1 + (complexity / 10) * 0.4  # Maps 1-10 to 0.14-0.5
            
            # Add some randomness to the density
            density = base_density * random.uniform(0.8, 1.2)
            
            # Create rhythmic patterns based on musical measures
            # We'll use a 4/4 time signature with 16th note resolution
            beats_per_measure = 4
            notes_per_beat = 4  # 16th notes
            measures = int(duration / (60/120 * beats_per_measure))
            total_steps = measures * beats_per_measure * notes_per_beat
            
            # Generate hits based on density and some musical rules
            pattern = []
            
            # For kick and snare, prefer musically common positions
            if instrument == "kick":
                # Kicks often on beats 1 and 3 in common patterns
                for measure in range(measures):
                    for beat in range(beats_per_measure):
                        for note in range(notes_per_beat):
                            step = measure * beats_per_measure * notes_per_beat + beat * notes_per_beat + note
                            
                            # Higher probability on beats 1 and 3
                            if beat in [0, 2] and note == 0:
                                if random.random() < 0.7 + (complexity / 20):  # Ensure high likelihood on main beats
                                    velocity = random.uniform(0.8, 1.0)
                                    pattern.append((step, velocity))
                            # Sometimes add kick on off-beats for complexity
                            elif complexity > 5 and random.random() < density * 0.3:
                                velocity = random.uniform(0.6, 0.9)
                                pattern.append((step, velocity))
                
            elif instrument == "snare":
                # Snares often on beats 2 and 4
                for measure in range(measures):
                    for beat in range(beats_per_measure):
                        for note in range(notes_per_beat):
                            step = measure * beats_per_measure * notes_per_beat + beat * notes_per_beat + note
                            
                            # Higher probability on beats 2 and 4
                            if beat in [1, 3] and note == 0:
                                if random.random() < 0.7 + (complexity / 20):
                                    velocity = random.uniform(0.8, 1.0)
                                    pattern.append((step, velocity))
                            # Sometimes add ghost notes for complexity
                            elif complexity > 6 and random.random() < density * 0.2:
                                velocity = random.uniform(0.3, 0.5)  # Lower velocity for ghost notes
                                pattern.append((step, velocity))
            
            elif instrument == "hihat":
                # Hi-hats often on every 8th or 16th note
                note_division = 2 if complexity < 5 else 1  # 8th or 16th notes based on complexity
                for measure in range(measures):
                    for beat in range(beats_per_measure):
                        for note in range(notes_per_beat):
                            if note % note_division == 0:  # Every 8th or 16th
                                step = measure * beats_per_measure * notes_per_beat + beat * notes_per_beat + note
                                if random.random() < 0.9:  # High probability for consistent hi-hat
                                    # Vary velocity to simulate accent patterns
                                    # Typically accent on the beat
                                    if note == 0:
                                        velocity = random.uniform(0.7, 0.9)
                                    else:
                                        velocity = random.uniform(0.5, 0.7)
                                    pattern.append((step, velocity))
            
            else:
                # For other instruments, use uniform probability based on density
                for step in range(total_steps):
                    if random.random() < density:
                        velocity = random.uniform(0.5, 1.0)
                        pattern.append((step, velocity))
            
            # Convert pattern to note events and add to audio/MIDI
            note_events = []
            for step, velocity in pattern:
                # Calculate timing
                seconds_per_step = duration / total_steps
                time = step * seconds_per_step
                
                # Add to audio
                sample = drum_samples[instrument]
                sample_scaled = sample * velocity
                
                # Calculate sample position
                pos = int(time * sample_rate)
                end_pos = min(pos + len(sample_scaled), audio_length)
                
                # Mix in the sample
                audio[pos:end_pos] += sample_scaled[:end_pos-pos]
                
                # Add to MIDI
                note_number = midi_notes[instrument]
                velocity_midi = int(velocity * 127)
                note = pretty_midi.Note(
                    velocity=velocity_midi,
                    pitch=note_number,
                    start=time,
                    end=time + 0.1  # Short duration for drum hits
                )
                drum_track.notes.append(note)
                
                # Add to note events for metadata
                note_events.append({
                    "time": time,
                    "velocity": velocity
                })
            
            # Add pattern to sample info
            sample_info["pattern"][instrument] = {
                "num_hits": len(pattern),
                "note_events": note_events
            }
        
        # Normalize audio to prevent clipping
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.9
        
        # Add some background noise if complexity is higher (simulate real recordings)
        if complexity > 3:
            noise_level = 0.01 * (complexity / 10)
            noise = np.random.randn(audio_length) * noise_level
            audio += noise
        
        # Ensure audio doesn't clip
        audio = np.clip(audio, -1.0, 1.0)
        
        # Convert to int16 format for WAV file
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Save audio file
        wavfile.write(audio_path, sample_rate, audio_int16)
        
        # Add instrument track to MIDI file
        midi.instruments.append(drum_track)
        
        # Save MIDI file
        midi.write(str(midi_path))
        
        # Add sample info to metadata
        metadata["samples"].append(sample_info)
        
        # Print progress
        if (i+1) % 10 == 0:
            print(f"Generated {i+1}/{num_samples} samples")
    
    # Save metadata to JSON file
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Generated {num_samples} synthetic drum samples")
    print(f"Audio saved to: {output_dir}")
    print(f"MIDI saved to: {output_dir}")
    print(f"Metadata saved to: {metadata_path}")
    
    return metadata


def load_dataset(
    dataset_path: str,
    split: str = 'train',
    batch_size: int = 32,
    sample_length: float = 5.0,
    sample_rate: int = 44100,
    midi_sample_rate: int = 100,
    num_workers: int = 4,
    augmentation: bool = True
) -> DataLoader:
    """
    Load dataset with specified split.
    
    Args:
        dataset_path: Path to dataset directory
        split: Dataset split ('train', 'val', 'test')
        batch_size: Batch size for DataLoader
        sample_length: Sample length in seconds
        sample_rate: Audio sample rate
        midi_sample_rate: MIDI sampling rate
        num_workers: Number of workers for DataLoader
        augmentation: Whether to apply augmentation (only in training)
        
    Returns:
        DataLoader for the specified dataset split
    """
    dataset = DrumAudioDataset(
        directory=dataset_path,
        sample_length=sample_length,
        sample_rate=sample_rate,
        midi_sample_rate=midi_sample_rate,
        split=split,
        augmentation=augmentation
    )
    
    shuffle = split == 'train'
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=split == 'train'  # Drop incomplete batches during training
    ) 