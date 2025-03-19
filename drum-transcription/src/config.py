"""
Configuration Handling for Drum Transcription

This module handles configuration management for the drum transcription system.

Key Components:
1. Default configuration definitions
2. Configuration loading and saving
3. Command-line argument parsing
4. Configuration validation

Functions:
- load_config(config_path): Load configuration from YAML file
- save_config(config, config_path): Save configuration to YAML file
- parse_args(): Parse command-line arguments
- validate_config(config): Validate configuration values
- merge_configs(base_config, override_config): Merge two configurations
- get_default_config(): Get default configuration

Constants:
- DEFAULT_AUDIO_CONFIG: Default configuration for audio processing
- DEFAULT_MODEL_CONFIG: Default configuration for model architecture
- DEFAULT_TRAINING_CONFIG: Default configuration for training
- DEFAULT_INFERENCE_CONFIG: Default configuration for inference

Implementation Considerations:
- Use a hierarchical configuration structure for clarity
- Support command-line overrides of configuration values
- Validate configuration values to catch errors early
- Save configurations with model checkpoints for reproducibility
- Use typed configurations for better IDE support
- Support environment variable overrides
- Include documentation for each configuration option
- Implement sensible defaults for all parameters
- Support different configurations for different environments (dev, prod)
"""

import os
import yaml
import argparse
import json
from typing import Dict, Any, Optional
from pathlib import Path


# Default configuration for audio processing
DEFAULT_AUDIO_CONFIG = {
    # Audio parameters
    'sample_rate': 44100,           # Audio sample rate in Hz
    'n_fft': 2048,                  # FFT window size
    'hop_length': 512,              # Hop length for STFT
    'n_mels': 128,                  # Number of mel bands
    'f_min': 20,                    # Minimum frequency for mel bands
    'f_max': 20000,                 # Maximum frequency for mel bands
    
    # Feature extraction parameters
    'window_size_ms': 25,           # Window size in milliseconds
    'hop_size_ms': 10,              # Hop size in milliseconds
    'midi_sample_rate': 100,        # MIDI sampling rate in Hz (frames per second)
    
    # Data augmentation parameters
    'time_stretch_range': [0.95, 1.05],  # Range for time stretching
    'pitch_shift_range': [-2, 2],        # Range for pitch shifting in semitones
    'noise_level_range': [0.0, 0.01],    # Range for noise level
    'use_augmentation': True,            # Whether to use data augmentation
}


# Default configuration for model architecture
DEFAULT_MODEL_CONFIG = {
    # Model type
    'model_name': 'cnn_lstm',       # Model architecture to use
    
    # Encoder parameters
    'encoder': {
        'input_channels': 1,        # Number of input channels (mono audio)
        'base_channels': 16,        # Base number of channels in CNN
        'kernel_sizes': [7, 5, 3, 3],  # Kernel sizes for CNN layers
        'strides': [2, 2, 1, 1],    # Strides for CNN layers
        'paddings': [3, 2, 1, 1],   # Paddings for CNN layers
        'dropout': 0.2              # Dropout rate in encoder
    },
    
    # Temporal model parameters
    'temporal': {
        'hidden_dim': 256,          # Hidden dimension for LSTM
        'num_layers': 2,            # Number of LSTM layers
        'dropout': 0.2              # Dropout rate in temporal model
    },
    
    # Loss weights
    'onset_loss_weight': 1.0,       # Weight for onset loss
    'velocity_loss_weight': 0.5,    # Weight for velocity loss
    'onset_positive_weight': 5.0,   # Positive class weight for onset loss
    
    # Other parameters
    'learning_rate': 0.001,         # Initial learning rate
    'weight_decay': 1e-5,           # Weight decay (L2 regularization)
}


# Default configuration for training
DEFAULT_TRAINING_CONFIG = {
    # Dataset parameters
    'dataset_path': './data',       # Path to dataset
    'sample_length': 5.0,           # Sample length in seconds
    
    # Training parameters
    'batch_size': 32,               # Batch size for training
    'num_epochs': 100,              # Maximum number of epochs
    'early_stopping_patience': 10,  # Patience for early stopping
    'grad_clip_value': 1.0,         # Gradient clipping value
    'mixed_precision': True,        # Whether to use mixed precision training
    
    # Optimizer parameters
    'learning_rate': 0.001,         # Initial learning rate
    'weight_decay': 1e-5,           # Weight decay (L2 regularization)
    
    # Logging and checkpoints
    'checkpoint_dir': './checkpoints',  # Directory to save checkpoints
    'log_dir': './logs',                # Directory to save logs
    'use_wandb': False,                 # Whether to use Weights & Biases
    'wandb_project': 'drum-transcription',  # W&B project name
    
    # Data loading
    'num_workers': 4,               # Number of workers for data loading
    
    # Data splitting
    'test_split': 0.1,              # Fraction of data to use for testing
    'val_split': 0.1,               # Fraction of data to use for validation
    
    # Random seed
    'seed': 42,                     # Random seed for reproducibility
    
    # Device
    'device': 'cuda',               # Device to use ('cpu' or 'cuda')
}


# Default configuration for inference
DEFAULT_INFERENCE_CONFIG = {
    # Inference parameters
    'checkpoint_path': 'best_model.pt',  # Path to model checkpoint
    'onset_threshold': 0.5,              # Threshold for onset detection
    'min_gap_ms': 10,                    # Minimum gap between notes in milliseconds
    
    # Output parameters
    'save_midi': True,                   # Whether to save MIDI output
    'save_predictions': True,            # Whether to save raw predictions
    'save_visualizations': False,        # Whether to save visualizations
    
    # Evaluation parameters
    'tolerance_ms': 50.0,                # Time tolerance for note matching in milliseconds
    'evaluate': True,                    # Whether to evaluate if ground truth is available
    
    # Device
    'device': 'cuda',                    # Device to use ('cpu' or 'cuda')
    
    # Batch processing
    'batch_size': 1,                     # Batch size for inference
}


def get_default_config() -> Dict[str, Any]:
    """
    Get the default configuration.
    
    Returns:
        Default configuration dictionary
    """
    return {
        'audio': DEFAULT_AUDIO_CONFIG.copy(),
        'model': DEFAULT_MODEL_CONFIG.copy(),
        'training': DEFAULT_TRAINING_CONFIG.copy(),
        'inference': DEFAULT_INFERENCE_CONFIG.copy()
    }


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Merge with default config to ensure all keys are present
    config = merge_configs(get_default_config(), config)
    
    # Validate config
    validate_config(config)
    
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configurations, with values in override_config taking precedence.
    
    Args:
        base_config: Base configuration dictionary
        override_config: Override configuration dictionary
        
    Returns:
        Merged configuration dictionary
    """
    merged_config = base_config.copy()
    
    for key, value in override_config.items():
        if (
            key in merged_config and 
            isinstance(merged_config[key], dict) and 
            isinstance(value, dict)
        ):
            # Recursively merge nested dictionaries
            merged_config[key] = merge_configs(merged_config[key], value)
        else:
            # Override or add the value
            merged_config[key] = value
    
    return merged_config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration values.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Validate audio config
    audio_config = config.get('audio', {})
    if audio_config.get('sample_rate') <= 0:
        raise ValueError("Sample rate must be positive")
    if audio_config.get('n_fft') <= 0:
        raise ValueError("FFT window size must be positive")
    if audio_config.get('hop_length') <= 0:
        raise ValueError("Hop length must be positive")
    if audio_config.get('n_mels') <= 0:
        raise ValueError("Number of mel bands must be positive")
        
    # Validate model config
    model_config = config.get('model', {})
    if model_config.get('model_name') not in ['cnn_lstm', 'small_cnn']:
        raise ValueError(f"Unknown model name: {model_config.get('model_name')}")
    
    # Validate training config
    training_config = config.get('training', {})
    if training_config.get('batch_size') <= 0:
        raise ValueError("Batch size must be positive")
    if training_config.get('num_epochs') <= 0:
        raise ValueError("Number of epochs must be positive")
    
    # Validate inference config
    inference_config = config.get('inference', {})
    if inference_config.get('onset_threshold') < 0 or inference_config.get('onset_threshold') > 1:
        raise ValueError("Onset threshold must be between 0 and 1")
    if inference_config.get('min_gap_ms') < 0:
        raise ValueError("Minimum gap must be non-negative")
    

def parse_args() -> Dict[str, Any]:
    """
    Parse command-line arguments.
    
    Returns:
        Dictionary of parsed arguments
    """
    parser = argparse.ArgumentParser(description="Drum Transcription System")
    
    # Common arguments
    parser.add_argument('--config', type=str, help="Path to configuration file")
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'transcribe'], 
                        default='train', help="Mode of operation")
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], 
                        help="Device to use")
    
    # Training arguments
    parser.add_argument('--dataset_path', type=str, help="Path to dataset")
    parser.add_argument('--batch_size', type=int, help="Batch size for training")
    parser.add_argument('--num_epochs', type=int, help="Maximum number of epochs")
    parser.add_argument('--learning_rate', type=float, help="Initial learning rate")
    parser.add_argument('--checkpoint_dir', type=str, help="Directory to save checkpoints")
    parser.add_argument('--use_wandb', action='store_true', help="Whether to use Weights & Biases")
    
    # Model arguments
    parser.add_argument('--model_name', type=str, choices=['cnn_lstm', 'small_cnn'], 
                        help="Model architecture to use")
    
    # Inference arguments
    parser.add_argument('--checkpoint_path', type=str, help="Path to model checkpoint")
    parser.add_argument('--input_path', type=str, help="Path to input audio file or directory")
    parser.add_argument('--output_path', type=str, help="Path to output directory")
    parser.add_argument('--onset_threshold', type=float, help="Threshold for onset detection")
    
    args = parser.parse_args()
    
    # Convert args to dictionary and remove None values
    args_dict = {k: v for k, v in vars(args).items() if v is not None}
    
    return args_dict


def create_config_from_args(args_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create configuration from command-line arguments.
    
    Args:
        args_dict: Dictionary of parsed arguments
        
    Returns:
        Configuration dictionary
    """
    # Start with default config
    config = get_default_config()
    
    # If config file provided, load it
    if 'config' in args_dict:
        file_config = load_config(args_dict['config'])
        config = merge_configs(config, file_config)
    
    # Map command-line arguments to config sections
    arg_to_section = {
        'device': ['training', 'inference'],
        'batch_size': ['training', 'inference'],
        'learning_rate': ['model', 'training'],
        'model_name': ['model'],
        'dataset_path': ['training'],
        'num_epochs': ['training'],
        'checkpoint_dir': ['training'],
        'use_wandb': ['training'],
        'checkpoint_path': ['inference'],
        'onset_threshold': ['inference'],
    }
    
    # Apply arguments to config
    for arg, value in args_dict.items():
        if arg in arg_to_section:
            for section in arg_to_section[arg]:
                config[section][arg] = value
    
    # Special handling for mode-specific arguments
    if args_dict.get('mode') == 'transcribe':
        config['inference']['input_path'] = args_dict.get('input_path')
        config['inference']['output_path'] = args_dict.get('output_path')
    
    return config


def get_config(args_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get configuration based on command-line arguments and/or defaults.
    
    Args:
        args_dict: Dictionary of parsed arguments (optional)
        
    Returns:
        Configuration dictionary
    """
    if args_dict is None:
        args_dict = parse_args()
    
    return create_config_from_args(args_dict)


def save_config_with_model(config: Dict[str, Any], model_path: str) -> None:
    """
    Save configuration alongside model checkpoint.
    
    Args:
        config: Configuration dictionary
        model_path: Path to model checkpoint
    """
    # Create config path from model path
    config_path = Path(model_path).with_suffix('.yaml')
    
    # Save config
    save_config(config, str(config_path))
    
    # Also save as JSON for easier loading in other contexts
    json_path = Path(model_path).with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump(config, f, indent=2) 