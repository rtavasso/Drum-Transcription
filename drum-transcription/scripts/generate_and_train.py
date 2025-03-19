#!/usr/bin/env python3
"""
Generate Dataset and Train Model Script

This script:
1. Generates a large synthetic drum dataset
2. Trains a drum transcription model on this dataset

Usage:
    python generate_and_train.py [--num_samples NUM_SAMPLES] [--epochs EPOCHS] [--output_dir OUTPUT_DIR]
"""

import os
import sys
import time
import argparse
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import create_synthetic_dataset, DrumAudioDataset
from src.model import CnnLstmTranscriber
from src.train import train_model
from src.config import get_default_config, save_config
from src.utils import set_seed, ensure_dir

def collate_batch(batch):
    """
    Custom collate function that ensures model input has the correct shape.
    
    Args:
        batch: List of (audio, onsets, velocities) tuples
        
    Returns:
        Formatted batch tensors
    """
    # Unzip the batch
    audio, onsets, velocities = zip(*batch)
    
    # Process audio tensors - ensure they're [batch, channels, time]
    processed_audio = []
    for audio_tensor in audio:
        # Ensure audio is 1D first (mono)
        if len(audio_tensor.shape) > 1:
            audio_tensor = audio_tensor.mean(dim=0)
            
        # Add channel dimension if missing
        if len(audio_tensor.shape) == 1:
            audio_tensor = audio_tensor.unsqueeze(0)  # Add channel dimension [channels, time]
        
        processed_audio.append(audio_tensor)
    
    # Stack along batch dimension
    audio_batch = torch.stack(processed_audio)
    
    # Process target tensors - resample to expected model output size
    expected_length = 55125  # From model output
    processed_onsets = []
    processed_velocities = []
    
    for onset_tensor, velocity_tensor in zip(onsets, velocities):
        # Resample onsets and velocities to match expected length
        # Use simple linear interpolation
        if onset_tensor.shape[0] != expected_length:
            # Create interpolated indices
            src_indices = torch.linspace(0, onset_tensor.shape[0] - 1, onset_tensor.shape[0])
            dst_indices = torch.linspace(0, onset_tensor.shape[0] - 1, expected_length)
            
            # Create new tensors
            resampled_onsets = torch.zeros(expected_length, device=onset_tensor.device)
            resampled_velocities = torch.zeros(expected_length, device=velocity_tensor.device)
            
            # Copy values directly for onset (binary values should not be interpolated)
            # We'll use nearest neighbor approach for onsets
            for i, idx in enumerate(dst_indices):
                nearest_idx = int(round(idx.item()))
                nearest_idx = min(nearest_idx, onset_tensor.shape[0] - 1)  # Ensure within bounds
                resampled_onsets[i] = onset_tensor[nearest_idx]
                resampled_velocities[i] = velocity_tensor[nearest_idx]
            
            processed_onsets.append(resampled_onsets)
            processed_velocities.append(resampled_velocities)
        else:
            processed_onsets.append(onset_tensor)
            processed_velocities.append(velocity_tensor)
    
    # Stack along batch dimension
    onsets_batch = torch.stack(processed_onsets)
    velocities_batch = torch.stack(processed_velocities)
    
    return audio_batch, onsets_batch, velocities_batch

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic data and train model")
    parser.add_argument("--output_dir", type=str, default="models/synthetic_run",
                      help="Directory to save data and models")
    parser.add_argument("--data_dir", type=str, default=None,
                      help="Directory to save/load data (default: OUTPUT_DIR/data)")
    parser.add_argument("--num_samples", type=int, default=3000,
                      help="Number of synthetic samples to generate")
    parser.add_argument("--epochs", type=int, default=20,
                      help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=32,
                      help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                      help="Learning rate")
    parser.add_argument("--generate_only", action="store_true",
                      help="Only generate data, don't train")
    parser.add_argument("--train_only", action="store_true",
                      help="Only train, don't generate data")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility")
    parser.add_argument("--num_workers", type=int, default=4,
                      help="Number of workers for data loading")
    parser.add_argument("--use_wandb", action="store_true",
                      help="Use Weights & Biases for logging")
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directories
    output_dir = Path(args.output_dir)
    data_dir = Path(args.data_dir) if args.data_dir else output_dir / "data"
    model_dir = output_dir / "model"
    log_dir = output_dir / "logs"
    
    # Create directories
    ensure_dir(output_dir)
    ensure_dir(data_dir)
    ensure_dir(model_dir)
    ensure_dir(log_dir)
    
    print(f"====== DRUM TRANSCRIPTION SYSTEM ======")
    print(f"Output directory: {output_dir}")
    print(f"Data directory: {data_dir}")
    print(f"Model directory: {model_dir}")
    
    # Set up config
    config = get_default_config()
    
    # Configure training
    config["training"]["num_epochs"] = args.epochs
    config["training"]["batch_size"] = args.batch_size
    config["training"]["learning_rate"] = args.learning_rate
    config["training"]["checkpoint_dir"] = str(model_dir)
    config["training"]["log_dir"] = str(log_dir)
    config["training"]["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    config["training"]["use_wandb"] = args.use_wandb
    config["training"]["wandb_project"] = "drum-transcription"
    config["training"]["wandb_run_name"] = f"synthetic-{args.num_samples}-{time.strftime('%Y%m%d-%H%M%S')}"
    config["training"]["num_workers"] = args.num_workers
    
    # Save config
    save_config(config, output_dir / "config.yaml")
    
    # Step 1: Generate synthetic data (if not train_only)
    if not args.train_only:
        print(f"\n[1/2] Generating {args.num_samples} synthetic samples...")
        start_time = time.time()
        
        metadata = create_synthetic_dataset(
            output_dir=data_dir,
            num_samples=args.num_samples,
            sample_rate=44100,
            duration=5.0,
            instruments=["kick", "snare", "hihat", "tom", "cymbal"],
            complexity=7,  # Higher complexity for more realistic patterns
            seed=args.seed
        )
        
        elapsed = time.time() - start_time
        print(f"Generated {args.num_samples} synthetic samples in {elapsed:.1f} seconds")
        print(f"Data saved to: {data_dir}")
    
    # Exit if generate_only
    if args.generate_only:
        print("Data generation complete. Exiting as requested (--generate_only)")
        return
    
    # Step 2: Train model
    print(f"\n[2/2] Training model...")
    
    # Create datasets
    train_dataset = DrumAudioDataset(
        directory=data_dir,
        sample_length=5.0,
        sample_rate=44100,
        midi_sample_rate=100,
        split='train',
        augmentation=True,
        test_split=0.1,
        val_split=0.1,
        seed=args.seed
    )
    
    val_dataset = DrumAudioDataset(
        directory=data_dir,
        sample_length=5.0,
        sample_rate=44100,
        midi_sample_rate=100,
        split='val',
        augmentation=False,
        test_split=0.1,
        val_split=0.1,
        seed=args.seed
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch
    )
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Initialize model
    model = CnnLstmTranscriber(config["model"])
    model.to(config["training"]["device"])
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Training on {config['training']['device']} for {args.epochs} epochs")
    
    # Train model
    start_time = time.time()
    
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config["training"]
    )
    
    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    
    # Save final model
    final_model_path = model_dir / "final_model.pt"
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'config': config,
        'history': history
    }, final_model_path)
    
    print(f"\n====== TRAINING COMPLETED ======")
    print(f"Final model saved to: {final_model_path}")
    
    # Report final metrics
    if history['val_onset_f1']:
        final_f1 = history['val_onset_f1'][-1]
        final_mae = history['val_velocity_mae'][-1]
        best_f1 = max(history['val_onset_f1'])
        best_f1_epoch = history['val_onset_f1'].index(best_f1) + 1
        
        print(f"Final validation metrics:")
        print(f"  Onset F1: {final_f1:.4f}")
        print(f"  Velocity MAE: {final_mae:.4f}")
        print(f"Best Onset F1: {best_f1:.4f} (epoch {best_f1_epoch})")

if __name__ == "__main__":
    main() 