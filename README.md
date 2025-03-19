# Automatic Drum Transcription

## Project Overview
**NOTE: This project is a recreation of the true work I did for Fadr. Due to confidentiality and IP clauses in my employment contract, I cannot share the actual code for the SOTA system, but this repo acts as a comparable open-source reference without any IP infringement issues.

Developed a SOTA drum transcription model to accurately identify different percussion elements from audio recordings using a novel multi-modal transformer architecture.

## My Role and Project Timeline
I independently developed this project over the course of 3 months, including lit review, ideation, experimentation, and finetuning.

## Results and Impact
- +7% hit onset detection accuracy wrt Onsets and Frames (previous SOTA)
- Achieved SOTA quality in listening tests over previous best models in literature
- Model effectively generalizes to varied recording conditions, drum kits, styles, and demixing bleed/artifacts
- Enabled new product features for pro users on the platform

## Challenges Overcome
- The model had problems with distribution shift from training set to real world recordings.
    - Created a custom dataset to address this by minimizing distribution shift from training to deployment
- The initial loss function would overly penalize temporal errors in onset detection leading to poor optimization.
    - Implemented a new loss function that weighted temporal error efficiently leading to better training dynamics and performance

## Technical Approach

### Architecture Overview
The model uses a modified transformer architecture with a bespoke attention mechanism that processes multi-modal data simultaneously. This allows the model to leverage complementary information from different audio representations for better onset detection and hit classification. This repo does not reimplement this architecture and opts for a basic CNN & LSTM model due to IP constraints.

### Key Technical Innovations

#### Multi-Modal Processing
- **Complementary Feature Extraction**: Each modality captures different aspects of percussion events

#### Novel Attention Mechanism
Designed and implemented a custom attention mechanism that:
- Processes information from different audio representations simultaneously
- Combines representations through latent space alignment
- Enhances robustness to distribution shifts in audio data

### Note Event Representation
- Discrete representation of onset, pitch, and velocity

### Training Innovations
- **Improved Loss Function**: Designed a custom loss function to weight temporal error in prediction
- **Data Augmentation**: Implemented sample mixing technique to expand training variety

## Technical Skills Used
- Deep learning architecture design
- Multi-modal neural network implementation
- Custom loss function development
- Audio signal processing
- Transformer model optimization
- Sequence modeling

## Technologies Used
- PyTorch
- HuggingFace Accelerate (multi-GPU training)
- Weights & Biases (experiment tracking)

## Development Process
- Initial pretraining on clean solo drum recordings
- Fine-tuning on a custom dataset of isolated drum stems
- Hyperparameter optimization through structured sweeps
- Extensive evaluation on diverse drum recording styles with PR curves and detailed metrics
