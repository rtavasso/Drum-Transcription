"""
Script to Train Drum Transcription Model

This script handles the training of the drum transcription model, including
dataset loading, model initialization, training loop, and model saving.

Usage:
    python train_model.py --config configs/train_config.yaml --output_dir models/my_model

Arguments:
    --config: Path to the configuration file (YAML)
    --output_dir: Directory to save trained model and logs
    --data_dir: Directory containing the dataset (overrides config)
    --resume: Path to checkpoint to resume training from
    --seed: Random seed for reproducibility (default: 42)
    --gpu: GPU index to use, -1 for CPU (default: 0)
    --num_workers: Number of data loading workers (default: 4)

Configuration:
    The config file should contain settings for:
    - audio_config: Audio processing parameters
    - model_config: Model architecture parameters
    - training_config: Training hyperparameters

Implementation:
    The script performs the following steps:
    1. Load and parse the configuration
    2. Set up the dataset and data loaders
    3. Initialize the model
    4. Set up optimizer, scheduler, and loss functions
    5. Run the training loop
    6. Save the trained model and configuration
    7. Generate validation metrics and visualizations

This script integrates the components from the src directory to create a complete training pipeline.
"""

import argparse
import sys
import os
import torch
import yaml
from pathlib import Path
from datetime import datetime

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import DrumAudioDataset
from src.model import CnnLstmTranscriber
from src.train import train_model
from src.utils import setup_logger, ensure_dir, set_seed, get_device
from src.config import load_config, save_config, merge_configs, get_default_config

def main():
    parser = argparse.ArgumentParser(description="Train drum transcription model")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save model and logs")
    parser.add_argument("--data_dir", type=str, help="Directory containing dataset (overrides config)")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume training from")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to use, -1 for CPU")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    
    args = parser.parse_args()
    
    # 1. Set up environment
    set_seed(args.seed)
    device = get_device(args.gpu)
    
    # 2. Create output directory and set up logging
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    log_dir = output_dir / "logs"
    checkpoint_dir = output_dir / "checkpoints"
    ensure_dir(log_dir)
    ensure_dir(checkpoint_dir)
    
    logger = setup_logger(log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger.info(f"Starting training with arguments: {args}")
    logger.info(f"Using device: {device}")
    
    # 3. Load and merge configurations
    # Load default config
    config = get_default_config()
    
    # Load user config from file
    user_config = load_config(args.config)
    config = merge_configs(config, user_config)
    
    # Override config with command line arguments
    if args.data_dir:
        config['data_dir'] = args.data_dir
    config['training_config']['device'] = device
    config['training_config']['num_workers'] = args.num_workers
    config['training_config']['checkpoint_dir'] = str(checkpoint_dir)
    
    # Save the merged config for reference
    save_config(config, output_dir / "config.yaml")
    logger.info(f"Configuration saved to {output_dir / 'config.yaml'}")
    
    # 4. Set up datasets and data loaders
    logger.info("Creating datasets and data loaders")
    
    # Extract dataset parameters from config
    data_dir = config['data_dir']
    sample_rate = config['audio_config']['sample_rate']
    sample_length = config['audio_config'].get('sample_length', 5.0)
    midi_sample_rate = config['audio_config'].get('midi_sample_rate', 100)
    batch_size = config['training_config']['batch_size']
    
    # Create datasets
    train_dataset = DrumAudioDataset(
        directory=data_dir,
        sample_length=sample_length,
        sample_rate=sample_rate,
        midi_sample_rate=midi_sample_rate,
        split='train',
        augmentation=True
    )
    
    val_dataset = DrumAudioDataset(
        directory=data_dir,
        sample_length=sample_length,
        sample_rate=sample_rate,
        midi_sample_rate=midi_sample_rate,
        split='validation',
        augmentation=False
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device != 'cpu' else False
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device != 'cpu' else False
    )
    
    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    # 5. Initialize model
    logger.info("Initializing model")
    model = CnnLstmTranscriber(config['model_config'])
    
    # Print model summary
    model_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {model_params:,}")
    
    # 6. Run training loop
    logger.info("Starting training")
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config['training_config'],
        resume_from=args.resume
    )
    
    # 7. Save final model and training history
    final_model_path = output_dir / "final_model.pt"
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'config': config,
        'history': history
    }, final_model_path)
    
    logger.info(f"Final model saved to {final_model_path}")
    logger.info("Training completed successfully")
    
    # 8. Print final validation metrics
    if history['val_onset_f1']:
        final_f1 = history['val_onset_f1'][-1]
        final_mae = history['val_velocity_mae'][-1]
        logger.info(f"Final validation metrics - Onset F1: {final_f1:.4f}, Velocity MAE: {final_mae:.4f}")
    
    return trained_model, history

if __name__ == "__main__":
    main() 