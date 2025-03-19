# AI-Based Drum Transcription System

## Project Overview
Developed a SOTA drum transcription model to accurately identify different percussion elements from audio recordings using a novel multi-modal neural network architecture.

## Technical Approach

### Architecture Overview
The model uses a modified transformer architecture with a novel attention mechanism that processes multi-modal data simultaneously. This unique approach allows the model to leverage complementary information from different audio representations for better onset detection and hit classification.

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

## Results and Impact
- Achieved SOTA quality in listening tests over previous best models in literature
- Model effectively generalizes to varied recording conditions, drum kits, styles, and demixing bleed/artifacts

## Technologies Used
- PyTorch
- HuggingFace Accelerate (multi-GPU training)
- Weights & Biases (experiment tracking)

## Development Process
- Initial pretraining on clean solo drum recordings
- Fine-tuning on a custom dataset of isolated drum stems
- Hyperparameter optimization through structured sweeps
- Extensive evaluation on diverse drum recording styles with PR curves and detailed metrics
