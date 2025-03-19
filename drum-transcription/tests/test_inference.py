"""
Tests for Inference Pipeline

This module contains tests for the inference pipeline in the drum transcription system.

Test Cases:
1. Model loading
   - Test loading models from checkpoints
   - Test configuration handling
   - Test device placement
   - Test handling of missing checkpoints

2. Audio transcription
   - Test single file transcription
   - Test batch transcription
   - Test different audio formats
   - Test error handling for corrupted files

3. Prediction processing
   - Test onset detection with different thresholds
   - Test velocity prediction
   - Test post-processing
   - Test MIDI conversion

4. Performance
   - Test transcription speed
   - Test batch processing efficiency
"""

import unittest
import os
import numpy as np
import pytest
import torch
import tempfile
import json
import shutil
from unittest.mock import patch, MagicMock
import time
from pathlib import Path

# Add the parent directory to the path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import (
    load_model, transcribe_file, batch_transcribe, predictions_to_midi, 
    post_process_predictions, adjust_onsets, process_audio_for_inference,
    save_predictions_to_json, real_time_inference_setup, process_audio_chunk
)
from src.config import get_default_config
from src.model import CnnLstmTranscriber, SmallCnnTranscriber

# Create a minimal test configuration
def get_test_config():
    """Get a minimal configuration for testing inference."""
    config = {
        'model_name': 'small_cnn',  # Using smaller model for testing
        'encoder': {
            'input_channels': 1,
            'base_channels': 8,  # Smaller channels for faster tests
            'kernel_sizes': [3, 3],
            'strides': [1, 1],
            'paddings': [1, 1],
            'dropout': 0.1
        },
        'sample_rate': 22050,  # Lower sample rate for faster tests
        'hop_length': 512,
        'n_fft': 1024,
        'n_mels': 64,
        'midi_sample_rate': 100,
        'device': 'cpu'  # Use CPU for testing
    }
    return config

def create_test_audio(path, duration=1.0, sample_rate=22050):
    """Create a test audio file with a simple sine wave."""
    # Generate a simple sine wave
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    # Add some "drum hits" (short impulses)
    for i in range(4):
        hit_pos = int((i + 0.5) * sample_rate / 4)
        audio[hit_pos:hit_pos+50] += np.linspace(1.0, 0.0, 50)
    
    # Normalize
    audio = audio / np.max(np.abs(audio))
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    
    # Save as WAV file
    import soundfile as sf
    sf.write(path, audio, sample_rate, 'PCM_16')
    
    return audio

def create_test_checkpoint(model, path):
    """Create a test checkpoint file."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    
    # Save checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': model.config
    }
    torch.save(checkpoint, path)


class TestModelLoading(unittest.TestCase):
    """Test cases for model loading functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a test configuration
        self.config = get_test_config()
        
        # Create a test model
        torch.manual_seed(42)  # For reproducibility
        self.test_model = SmallCnnTranscriber(self.config)
        
        # Create a test checkpoint
        self.checkpoint_path = os.path.join(self.temp_dir, "test_model.pt")
        create_test_checkpoint(self.test_model, self.checkpoint_path)
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_load_model(self):
        """Test loading model from checkpoint."""
        # Load the model from checkpoint
        loaded_model = load_model(self.checkpoint_path, self.config)
        
        # Check that the model is loaded correctly
        self.assertIsInstance(loaded_model, SmallCnnTranscriber)
        
        # Check that the model is in evaluation mode
        self.assertFalse(loaded_model.training)
        
        # Verify all parameters are equal
        for p1, p2 in zip(self.test_model.parameters(), loaded_model.parameters()):
            self.assertTrue(torch.allclose(p1, p2))
    
    def test_load_model_without_config(self):
        """Test loading model without providing a config."""
        # Load the model from checkpoint without providing a config
        loaded_model = load_model(self.checkpoint_path)
        
        # Check that the model is loaded correctly
        self.assertIsInstance(loaded_model, SmallCnnTranscriber)
        
        # Verify all parameters are equal
        for p1, p2 in zip(self.test_model.parameters(), loaded_model.parameters()):
            self.assertTrue(torch.allclose(p1, p2))
    
    def test_missing_checkpoint(self):
        """Test handling of missing checkpoint files."""
        # Try to load a model from a non-existent checkpoint
        with self.assertRaises(FileNotFoundError):
            load_model(os.path.join(self.temp_dir, "nonexistent.pt"))
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_device_placement(self):
        """Test loading model to a specific device."""
        # Load the model to CUDA
        loaded_model = load_model(self.checkpoint_path, self.config, device='cuda')
        
        # Check that the model is on CUDA
        self.assertTrue(next(loaded_model.parameters()).is_cuda)


class TestAudioTranscription(unittest.TestCase):
    """Test cases for audio transcription functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a test configuration
        self.config = get_test_config()
        
        # Create a test model
        torch.manual_seed(42)  # For reproducibility
        self.test_model = SmallCnnTranscriber(self.config)
        
        # Create test audio files
        self.audio_path = os.path.join(self.temp_dir, "test_audio.wav")
        self.audio_data = create_test_audio(self.audio_path)
        
        # Create a directory for batch transcription
        self.audio_dir = os.path.join(self.temp_dir, "audio")
        os.makedirs(self.audio_dir, exist_ok=True)
        for i in range(3):
            audio_path = os.path.join(self.audio_dir, f"test_audio_{i}.wav")
            create_test_audio(audio_path)
        
        # Create output directories
        self.output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_transcribe_file(self):
        """Test single file transcription."""
        # Transcribe the test audio file
        midi_path = os.path.join(self.temp_dir, "output.mid")
        json_path = os.path.join(self.temp_dir, "output.json")
        
        results = transcribe_file(
            self.audio_path,
            self.test_model,
            self.config,
            output_midi=midi_path,
            output_json=json_path
        )
        
        # Check that results are returned
        self.assertIn('onset_probs', results)
        self.assertIn('velocities', results)
        
        # Check that output files are created
        self.assertTrue(os.path.exists(midi_path))
        self.assertTrue(os.path.exists(json_path))
        
        # Check JSON output format
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        
        self.assertIn('onsets', json_data)
        self.assertIn('velocities', json_data)
        self.assertIn('audio_path', json_data)
        self.assertIn('sample_rate', json_data)
    
    def test_batch_transcribe(self):
        """Test batch transcription."""
        # Transcribe all files in the audio directory
        results = batch_transcribe(
            self.audio_dir,
            self.test_model,
            self.config,
            self.output_dir,
            file_pattern="*.wav"
        )
        
        # Check that results contain entries for each file
        self.assertEqual(len(results['files']), 3)
        
        # Check that output files are created
        for i in range(3):
            midi_path = os.path.join(self.output_dir, f"test_audio_{i}.mid")
            json_path = os.path.join(self.output_dir, f"test_audio_{i}.json")
            self.assertTrue(os.path.exists(midi_path))
            self.assertTrue(os.path.exists(json_path))
    
    def test_error_handling(self):
        """Test error handling for corrupted files."""
        # Create an empty file that's not a valid audio file
        bad_audio_path = os.path.join(self.temp_dir, "bad_audio.wav")
        with open(bad_audio_path, 'w') as f:
            f.write("Not a valid WAV file")
        
        # Mock librosa.load to simulate an error
        with patch('librosa.load', side_effect=Exception("Audio file cannot be loaded")):
            # Transcribe should handle the error gracefully
            results = transcribe_file(
                bad_audio_path,
                self.test_model,
                self.config
            )
            
            # Should return empty arrays
            self.assertEqual(len(results['onset_probs']), 0)
            self.assertEqual(len(results['velocities']), 0)
    
    def test_process_audio_for_inference(self):
        """Test audio processing for inference."""
        # Process audio for inference
        features = process_audio_for_inference(
            self.audio_data,
            self.config['sample_rate'],
            self.config
        )
        
        # Check that features have the right shape and range
        self.assertEqual(features.shape[0], self.config['n_mels'])
        self.assertTrue(np.all(features >= 0))
        self.assertTrue(np.all(features <= 1))


class TestPredictionProcessing(unittest.TestCase):
    """Test cases for prediction processing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample predictions for testing
        self.time_steps = 100
        
        # Create fake onset probabilities with some clear onsets
        self.onset_probs = np.zeros(self.time_steps)
        for i in range(4):
            pos = 10 + i * 20
            self.onset_probs[pos:pos+5] = 0.9
        
        # Add some noise
        np.random.seed(42)
        self.onset_probs += np.random.uniform(0, 0.2, self.time_steps)
        self.onset_probs = np.clip(self.onset_probs, 0, 1)
        
        # Create fake velocities
        self.velocities = np.zeros(self.time_steps)
        for i in range(4):
            pos = 10 + i * 20
            self.velocities[pos:pos+5] = 0.8 - i * 0.1  # Decreasing velocities
        
        # Create a temporary directory for outputs
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_post_process_predictions(self):
        """Test post-processing of predictions."""
        # Post-process the predictions
        processed_onsets, processed_velocities = post_process_predictions(
            self.onset_probs,
            self.velocities,
            onset_threshold=0.5,
            min_gap_ms=50,
            sample_rate=100
        )
        
        # Check that we detected the expected number of onsets
        onset_count = np.sum(np.diff(np.concatenate([[0], processed_onsets])) > 0)
        self.assertEqual(onset_count, 4)
        
        # Check that velocities are preserved at onset positions
        for i in range(4):
            pos = 10 + i * 20
            self.assertTrue(processed_onsets[pos] > 0.5)
            self.assertAlmostEqual(processed_velocities[pos], 0.8 - i * 0.1, delta=0.05)
    
    def test_adjust_onsets(self):
        """Test adjustment of onsets based on threshold."""
        # Apply onset adjustment
        binary_onsets = self.onset_probs > 0.5
        adjusted_onsets, adjusted_velocities = adjust_onsets(
            binary_onsets,
            self.velocities,
            threshold=0.5
        )
        
        # Check that we detected the expected number of onsets
        onset_count = np.sum(adjusted_onsets)
        self.assertEqual(onset_count, 4)
    
    def test_predictions_to_midi(self):
        """Test conversion of predictions to MIDI."""
        # Convert predictions to MIDI
        midi_path = os.path.join(self.temp_dir, "test.mid")
        predictions_to_midi(
            self.onset_probs,
            self.velocities,
            threshold=0.5,
            output_path=midi_path,
            sample_rate=100
        )
        
        # Check that MIDI file is created
        self.assertTrue(os.path.exists(midi_path))
        
        # Check file size is non-zero
        self.assertGreater(os.path.getsize(midi_path), 0)
    
    def test_save_predictions_to_json(self):
        """Test saving predictions to JSON."""
        # Save predictions to JSON
        json_path = os.path.join(self.temp_dir, "test.json")
        save_predictions_to_json(
            self.onset_probs,
            self.velocities,
            output_path=json_path,
            sample_rate=100,
            audio_path="test_audio.wav"
        )
        
        # Check that JSON file is created
        self.assertTrue(os.path.exists(json_path))
        
        # Check JSON contents
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        self.assertIn('onsets', data)
        self.assertIn('velocities', data)
        self.assertEqual(data['audio_path'], "test_audio.wav")
    
    def test_threshold_variation(self):
        """Test effect of different thresholds on onset detection."""
        thresholds = [0.3, 0.5, 0.7, 0.9]
        onset_counts = []
        
        for threshold in thresholds:
            # Post-process with different thresholds
            processed_onsets, _ = post_process_predictions(
                self.onset_probs,
                self.velocities,
                onset_threshold=threshold
            )
            
            # Count onsets
            onset_count = np.sum(np.diff(np.concatenate([[0], processed_onsets])) > 0)
            onset_counts.append(onset_count)
        
        # Higher thresholds should detect fewer onsets
        self.assertTrue(onset_counts[0] >= onset_counts[1] >= onset_counts[2] >= onset_counts[3])


class TestPerformance(unittest.TestCase):
    """Test cases for performance of the inference pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a test configuration
        self.config = get_test_config()
        
        # Create a test model
        torch.manual_seed(42)  # For reproducibility
        self.test_model = SmallCnnTranscriber(self.config)
        
        # Create test audio files for batch processing
        self.audio_dir = os.path.join(self.temp_dir, "audio")
        os.makedirs(self.audio_dir, exist_ok=True)
        
        # Create 5 audio files of varying lengths
        self.audio_files = []
        for i, duration in enumerate([0.5, 1.0, 1.5, 2.0, 2.5]):
            audio_path = os.path.join(self.audio_dir, f"test_audio_{i}.wav")
            create_test_audio(audio_path, duration=duration)
            self.audio_files.append(audio_path)
        
        # Create output directory
        self.output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_transcription_speed(self):
        """Test transcription speed."""
        # Measure time to transcribe a file
        start_time = time.time()
        
        transcribe_file(
            self.audio_files[1],  # 1 second file
            self.test_model,
            self.config
        )
        
        duration = time.time() - start_time
        
        # Transcription should be reasonably fast (adjust threshold as needed)
        # This is a basic check to catch significant performance regressions
        self.assertLess(duration, 5.0)  # Should be faster than 5 seconds
    
    def test_batch_transcription_efficiency(self):
        """Test batch transcription efficiency."""
        # Measure time to transcribe multiple files
        start_time = time.time()
        
        batch_transcribe(
            self.audio_dir,
            self.test_model,
            self.config,
            self.output_dir
        )
        
        batch_duration = time.time() - start_time
        
        # Now measure time to transcribe each file individually
        individual_start_time = time.time()
        
        for audio_path in self.audio_files:
            transcribe_file(
                audio_path,
                self.test_model,
                self.config
            )
        
        individual_duration = time.time() - individual_start_time
        
        # Batch processing should not be significantly slower than individual processing
        # It might be faster due to shared setup costs
        self.assertLessEqual(batch_duration / individual_duration, 1.5)
    
    def test_real_time_inference_setup(self):
        """Test real-time inference setup."""
        # Set up real-time inference
        inference_state = real_time_inference_setup(
            self.test_model,
            self.config,
            buffer_size=4096,
            hop_size=512
        )
        
        # Check that the state contains expected keys
        self.assertIn('model', inference_state)
        self.assertIn('buffer', inference_state)
        self.assertIn('config', inference_state)
        
        # Test processing an audio chunk
        audio_chunk = np.random.uniform(-0.5, 0.5, 4096)
        result = process_audio_chunk(audio_chunk, inference_state)
        
        # Check that result contains expected keys
        self.assertIn('onset_probs', result)
        self.assertIn('velocities', result)


if __name__ == '__main__':
    unittest.main() 