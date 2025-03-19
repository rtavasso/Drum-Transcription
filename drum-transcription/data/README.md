# Data Preparation Instructions

This directory should contain the datasets used for training and evaluating the drum transcription models.

## Datasets

The implementation can use these public datasets:

1. **IDMT-SMT-Drums**: A dataset of drum recordings with annotations
   - Available at: https://www.idmt.fraunhofer.de/en/business_units/m2d/smt/drums.html

2. **Groove MIDI Dataset**: A MIDI dataset of drum performances
   - Available at: https://magenta.tensorflow.org/datasets/groove

3. **ENST Drums**: Audio recordings of drum performances with annotations
   - Available at: http://www.tsi.telecom-paristech.fr/aao/en/2010/02/19/enst-drums-an-extensive-audio-visual-database-for-drum-signals-processing/

## Data Structure

Each dataset should be organized in its own subdirectory with the following structure:

```
data/
├── dataset_name/
│   ├── audio/          # Contains audio files (.wav)
│   ├── midi/           # Contains MIDI files (.mid)
│   ├── metadata/       # Contains metadata files
│   └── splits/         # Contains train/validation/test splits
```

## Synthetic Data

For development and testing purposes, a synthetic dataset can be generated using the provided script:

```
python scripts/create_synthetic_data.py --output_dir data/synthetic --num_samples 100
```

## Data Preprocessing

Before using the datasets, ensure that:

1. Audio files are in WAV format, mono or stereo, with sample rate of 44.1kHz
2. MIDI files are properly aligned with audio
3. Train/validation/test splits are properly defined

The dataset.py module provides utilities for loading and preprocessing these datasets. 