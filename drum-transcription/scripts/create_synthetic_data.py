"""
Script to Generate Synthetic Drum Data

This script generates synthetic drum data for testing and development purposes.
It creates paired audio and MIDI files with known patterns, which can be used
for testing the drum transcription system.

Usage:
    python create_synthetic_data.py --output_dir data/synthetic --num_samples 100

Arguments:
    --output_dir: Directory where synthetic data will be saved
    --num_samples: Number of synthetic samples to generate
    --sample_rate: Sample rate for audio files (default: 44100)
    --duration: Duration of each sample in seconds (default: 10.0)
    --instruments: List of drum instruments to include (default: kick,snare,hihat)
    --complexity: Complexity of patterns from 1-10 (default: 5)
    --seed: Random seed for reproducibility (default: 42)

Implementation:
    The script uses the synthetic dataset generation function from the dataset module
    to create drum patterns with known characteristics. For each sample, it:
    1. Generates a random drum pattern
    2. Creates an audio file using drum samples
    3. Creates a corresponding MIDI file
    4. Saves metadata about the generated sample

This allows creation of a controlled test set with ground truth annotations.
"""

import argparse
import sys
import os
import logging
from pathlib import Path

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import create_synthetic_dataset
from src.utils import ensure_dir, set_seed

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic drum data for testing")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save synthetic data")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of synthetic samples to generate")
    parser.add_argument("--sample_rate", type=int, default=44100, help="Sample rate for audio files")
    parser.add_argument("--duration", type=float, default=10.0, help="Duration of each sample in seconds")
    parser.add_argument("--instruments", type=str, default="kick,snare,hihat", help="Comma-separated list of drum instruments")
    parser.add_argument("--complexity", type=int, default=5, help="Complexity of patterns from 1-10")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    
    # Parse instruments list
    instruments = args.instruments.split(",")
    
    logging.info(f"Generating {args.num_samples} synthetic drum samples")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Instruments: {instruments}")
    
    # Generate the synthetic dataset
    metadata = create_synthetic_dataset(
        output_dir=output_dir,
        num_samples=args.num_samples,
        sample_rate=args.sample_rate,
        duration=args.duration,
        instruments=instruments,
        complexity=args.complexity
    )
    
    # Save metadata about the generated dataset
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        import json
        json.dump(metadata, f, indent=4)
    
    logging.info(f"Synthetic dataset generation complete. Metadata saved to {metadata_path}")
    logging.info(f"Generated {args.num_samples} audio-MIDI pairs in {output_dir}")

if __name__ == "__main__":
    main() 