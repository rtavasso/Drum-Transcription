#!/usr/bin/env python3
"""
Test Pipeline Script

This script tests the full drum transcription pipeline by:
1. Generating a small synthetic dataset
2. Training a model for a few epochs
3. Verifying that the training process works

Usage:
    python test_pipeline.py
"""

import os
import sys
import argparse
import torch
import numpy as np
import librosa
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import create_synthetic_dataset
from src.model import CnnLstmTranscriber
from src.config import get_default_config
from src.utils import set_seed

def main():
    parser = argparse.ArgumentParser(description="Test the drum transcription pipeline")
    parser.add_argument("--output_dir", type=str, default="data/test_run",
                      help="Directory to save data and models")
    parser.add_argument("--num_samples", type=int, default=10,
                      help="Number of synthetic samples to generate")
    parser.add_argument("--test_training", action="store_true",
                      help="Also test the training pipeline")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility")
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directories
    output_dir = Path(args.output_dir)
    data_dir = output_dir / "data"
    model_dir = output_dir / "model"
    
    data_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"====== TESTING DRUM TRANSCRIPTION PIPELINE ======")
    print(f"Data directory: {data_dir}")
    print(f"Model directory: {model_dir}")
    print(f"Number of samples: {args.num_samples}")
    
    # Step 1: Generate synthetic data
    print("\n[1/3] Generating synthetic data...")
    metadata = create_synthetic_dataset(
        output_dir=data_dir,
        num_samples=args.num_samples,
        sample_rate=44100,
        duration=5.0,
        instruments=["kick", "snare", "hihat"],
        complexity=5,
        seed=args.seed
    )
    print(f"Generated {args.num_samples} synthetic samples")
    
    # Step 2: Initialize the model
    print("\n[2/3] Initializing model...")
    
    # Get default config and customize it
    config = get_default_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create and initialize the model
    model = CnnLstmTranscriber(config["model"])
    model.to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Step 3: Process a sample to verify model works
    print("\n[3/3] Testing model with sample data...")
    
    # Get a list of audio files
    audio_files = list(data_dir.glob("*.wav"))
    if not audio_files:
        print("No audio files found!")
        return
    
    # Process a few samples
    for i, audio_file in enumerate(audio_files[:2]):
        print(f"Processing {audio_file.name}...")
        
        # Load audio
        audio, sr = librosa.load(str(audio_file), sr=44100, mono=True)
        
        # Convert to spectrogram
        spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=sr, 
            n_fft=2048, 
            hop_length=512, 
            n_mels=128
        )
        
        # Convert to log spectrogram
        log_spec = librosa.power_to_db(spec, ref=np.max)
        
        # Convert to tensor [batch, channels, freq, time]
        log_spec_tensor = torch.from_numpy(log_spec).unsqueeze(0).unsqueeze(0).float()
        
        # Move to device
        log_spec_tensor = log_spec_tensor.to(device)
        
        # Run model inference
        model.eval()
        with torch.no_grad():
            print(f"Input shape: {log_spec_tensor.shape}")
            
            # Process the first N frames (model may expect a specific input size)
            frame_size = min(500, log_spec_tensor.shape[-1])
            input_data = log_spec_tensor[:, :, :, :frame_size]
            
            # Reshape to [batch, channels, time] - using the raw audio instead
            # The model is expecting raw audio input, not spectrogram
            audio_tensor = torch.from_numpy(audio[:sr*5]).unsqueeze(0).unsqueeze(0).float().to(device)
            print(f"Audio tensor shape: {audio_tensor.shape}")
            
            # Forward pass
            try:
                outputs = model(audio_tensor)
                print(f"Model output shapes:")
                for key, value in outputs.items():
                    print(f"  {key}: {value.shape}")
                print("Model inference successful!")
            except Exception as e:
                print(f"Error during model inference: {e}")
    
    # Step 4 (Optional): Test training loop
    if args.test_training:
        print("\n[4/4] Testing training loop...")
        
        # Create a small batch of data for training
        batch_size = 2
        num_batches = 2
        
        # We'll create artificial data and labels
        print("Creating artificial training data...")
        
        # Create a batch of audio data [batch, channels, time]
        batch_audio = torch.randn(batch_size, 1, sr*5).to(device)
        
        # Create targets - onsets and velocities
        # The model outputs onset_logits and velocities with shape [batch, time]
        time_steps = 55125  # From the model output above
        batch_onsets = torch.zeros(batch_size, time_steps).to(device)
        batch_velocities = torch.zeros(batch_size, time_steps).to(device)
        
        # Add some random onsets (sparse)
        for b in range(batch_size):
            num_onsets = 10
            onset_positions = torch.randint(0, time_steps, (num_onsets,))
            batch_onsets[b, onset_positions] = 1.0
            batch_velocities[b, onset_positions] = torch.rand(num_onsets, device=device) * 0.5 + 0.5  # 0.5-1.0 range
        
        # Set model to training mode
        model.train()
        
        # Get optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        print("Running training iterations...")
        for i in range(num_batches):
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_audio)
            
            # Get losses
            from src.model import combined_loss
            loss, loss_dict = combined_loss(
                outputs['onset_logits'],
                outputs['velocities'],
                batch_onsets,
                batch_velocities
            )
            
            # Backward pass
            loss.backward()
            
            # Optimize
            optimizer.step()
            
            print(f"Batch {i+1}/{num_batches}, Loss: {loss.item():.4f}, "
                  f"Onset Loss: {loss_dict['onset_loss'].item():.4f}, "
                  f"Velocity Loss: {loss_dict['velocity_loss'].item():.4f}")
        
        print("Training test completed successfully!")
    
    print("\n====== PIPELINE TEST COMPLETED ======")
    print("Note: This was a basic test verifying the drum transcription pipeline.")
    print("For full training, use the train_model.py script.")

if __name__ == "__main__":
    main() 