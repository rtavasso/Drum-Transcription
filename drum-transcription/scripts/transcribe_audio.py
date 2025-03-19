"""
Script for Drum Transcription Inference

This script transcribes drum audio into MIDI using a trained model. It supports
processing a single file or batch processing a directory of files.

Usage:
    python transcribe_audio.py --model models/my_model/best_model.pt --input audio/my_drum_loop.wav --output output/my_drum_loop.mid
    python transcribe_audio.py --model models/my_model/best_model.pt --input_dir audio/drum_loops/ --output_dir output/midi/

Arguments:
    --model: Path to the trained model checkpoint
    --config: Path to the configuration file (optional, if not stored with model)
    --input: Path to input audio file (for single file processing)
    --input_dir: Path to directory containing audio files (for batch processing)
    --output: Path to output MIDI file (for single file processing)
    --output_dir: Path to directory for output MIDI files (for batch processing)
    --threshold: Onset detection threshold (default: 0.5)
    --gpu: GPU index to use, -1 for CPU (default: 0)
    --preprocess: Whether to use HTDemucs for drum separation (default: False)
    --formats: Output formats, comma-separated (midi,json) (default: midi)
    --visualize: Generate visualizations of transcriptions (default: False)

Implementation:
    The script performs the following steps:
    1. Load the trained model and configuration
    2. Process input audio (optionally using HTDemucs for drum separation)
    3. Run inference to detect onsets and velocities
    4. Post-process predictions
    5. Convert predictions to MIDI and/or other formats
    6. Save output files
    7. Optionally generate visualizations

This script combines the components from the src directory to create a complete inference pipeline.
"""

import argparse
import sys
import os
import torch
from pathlib import Path
import logging
import time

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import load_model, transcribe_file, batch_transcribe
from src.utils import ensure_dir, get_device, setup_logger
from src.config import load_config
from src.demucs_adapter import get_htdemucs_model, process_audio_file

def main():
    parser = argparse.ArgumentParser(description="Transcribe drum audio to MIDI")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--config", type=str, help="Path to configuration file (optional)")
    parser.add_argument("--input", type=str, help="Path to input audio file")
    parser.add_argument("--input_dir", type=str, help="Path to directory containing audio files")
    parser.add_argument("--output", type=str, help="Path to output MIDI file")
    parser.add_argument("--output_dir", type=str, help="Path to directory for output MIDI files")
    parser.add_argument("--threshold", type=float, default=0.5, help="Onset detection threshold")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to use, -1 for CPU")
    parser.add_argument("--preprocess", action="store_true", help="Use HTDemucs for drum separation")
    parser.add_argument("--formats", type=str, default="midi", help="Output formats, comma-separated (midi,json)")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations of transcriptions")
    
    args = parser.parse_args()
    
    # Ensure either input or input_dir is provided
    if args.input is None and args.input_dir is None:
        parser.error("Either --input or --input_dir must be provided")
        
    # Ensure output is provided for single file, output_dir for batch
    if args.input and not args.output and not args.output_dir:
        parser.error("--output must be provided for single file processing")
    if args.input_dir and not args.output_dir:
        parser.error("--output_dir must be provided for batch processing")
    
    # 1. Set up environment
    device = get_device(args.gpu)
    
    # Create logger
    log_dir = Path("logs")
    ensure_dir(log_dir)
    logger = setup_logger(log_dir / f"transcription_{time.strftime('%Y%m%d_%H%M%S')}.log")
    logger.info(f"Starting transcription with arguments: {args}")
    logger.info(f"Using device: {device}")
    
    # 2. Load model and configuration
    logger.info(f"Loading model from {args.model}")
    model, config = load_model(args.model, args.config, device)
    
    # 3. Initialize HTDemucs if preprocessing is enabled
    demucs_model = None
    if args.preprocess:
        logger.info("Loading HTDemucs model for drum separation")
        demucs_model = get_htdemucs_model(device)
        if demucs_model is None:
            logger.error("Failed to load HTDemucs model. Make sure it's installed.")
            return
    
    # 4. Parse output formats
    formats = args.formats.split(',')
    for fmt in formats:
        if fmt not in ['midi', 'json', 'csv']:
            logger.warning(f"Unsupported output format: {fmt}. Will be ignored.")
    
    # 5. Process audio file(s)
    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
        ensure_dir(output_dir)
        if args.visualize:
            ensure_dir(output_dir / "visualizations")
    
    # 6. Process single file or batch
    try:
        if args.input:
            # Single file processing
            input_path = Path(args.input)
            output_path = Path(args.output) if args.output else (output_dir / f"{input_path.stem}.mid")
            
            logger.info(f"Transcribing file: {input_path}")
            
            # Transcribe the file
            result = transcribe_file(
                audio_path=input_path,
                model=model,
                config=config,
                threshold=args.threshold,
                device=device,
                demucs_model=demucs_model if args.preprocess else None,
                output_path=output_path,
                formats=formats,
                visualize=args.visualize,
                visualization_path=output_dir / "visualizations" if args.visualize and output_dir else None
            )
            
            logger.info(f"Transcription complete: {output_path}")
            
        else:
            # Batch processing
            input_dir = Path(args.input_dir)
            
            logger.info(f"Batch transcribing files from: {input_dir}")
            
            # Get all audio files in the directory
            audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
            audio_files = [f for f in input_dir.glob('**/*') if f.suffix.lower() in audio_extensions]
            
            logger.info(f"Found {len(audio_files)} audio files to process")
            
            # Process all files
            results = batch_transcribe(
                audio_paths=audio_files,
                model=model,
                config=config,
                threshold=args.threshold,
                device=device,
                demucs_model=demucs_model if args.preprocess else None,
                output_dir=output_dir,
                formats=formats,
                visualize=args.visualize,
                visualization_dir=output_dir / "visualizations" if args.visualize else None
            )
            
            # Log summary
            successful = sum(1 for r in results if r['success'])
            logger.info(f"Batch transcription complete. {successful}/{len(audio_files)} files processed successfully.")
            
            # Log failures if any
            failures = [r['file'] for r in results if not r['success']]
            if failures:
                logger.warning(f"Failed to transcribe {len(failures)} files:")
                for f in failures:
                    logger.warning(f"  - {f}")
    
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}", exc_info=True)
        return
    
    logger.info("Transcription process completed successfully")

if __name__ == "__main__":
    main() 