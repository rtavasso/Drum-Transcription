"""
HTDemucs Adapter for Drum Transcription

This module provides integration with the publicly available HTDemucs library for source separation.
It serves as an adapter that interfaces with HTDemucs to extract drums from mixed audio.

Key Components:
1. Model loading utilities for HTDemucs
2. Source separation functionality
3. Audio processing for HTDemucs

Functions:
- get_htdemucs_model(device): Load the HTDemucs model from the publicly available library
- separate_drums(audio_tensor, model, sample_rate): Separate drums from audio using HTDemucs
- process_audio_file(audio_path, model, output_path): Process an audio file and save the drum stem
- batch_separate(audio_dir, output_dir, device): Batch process multiple audio files

Implementation Considerations:
- Use the public API of HTDemucs as a dependency
- Clearly document the use of this external library
- Handle errors gracefully if the library is not installed
- Provide clear installation instructions
- Optimize for memory usage when processing large files
- Support both CPU and GPU processing
- Implement proper device management
- Add progress reporting for batch processing
- Ensure output formats are compatible with the rest of the system
- Cache model to avoid reloading for multiple files
"""

# Implementation will be added here 