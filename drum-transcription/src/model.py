"""
Model Architecture Definitions for Drum Transcription

This module defines the neural network architectures used for drum transcription.

Key Components:
1. Base model class with common functionality
2. Feature encoder implementations (processes audio features)
3. Temporal model implementations (captures timing relationships)
4. Output heads (onset and velocity prediction)

Classes:
- DrumTranscriptionModel: Base class for all drum transcription models
- CnnLstmTranscriber: CNN-LSTM model for drum transcription
    - Uses CNN layers for feature extraction
    - Uses LSTM layers for temporal modeling
    - Has separate heads for onset and velocity prediction
- Alternative model architectures as needed

Implementation Considerations:
- Use a modular design that allows swapping components
- Support different model sizes (small, medium, large)
- Allow for different input features (mel specs, CQT, raw audio)
- Use bidirectional LSTM for better temporal modeling
- Consider attention mechanisms for complex patterns
- Ensure models can be initialized with random or pretrained weights
- Include proper weight initialization methods
- Support both training and inference modes with appropriate optimizations
- Consider model quantization for inference efficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List, Union


class DrumTranscriptionModel(nn.Module):
    """Base class for drum transcription models"""
    
    def __init__(self, config: Dict):
        """
        Initialize the drum transcription model.
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__()
        self.config = config
    
    def forward(self, x: torch.Tensor, conditioning: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor (audio features)
            conditioning: Optional conditioning tensor
            
        Returns:
            Dictionary of model outputs (onsets, velocities)
        """
        raise NotImplementedError("Subclasses must implement forward method")
    
    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        """
        Configure optimizer and learning rate scheduler.
        
        Returns:
            Tuple of (optimizer, scheduler)
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.get('learning_rate', 0.001),
            weight_decay=self.config.get('weight_decay', 1e-5)
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        return optimizer, scheduler
    
    def compute_loss(
        self, 
        predictions: Dict[str, torch.Tensor], 
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss for model predictions.
        
        Args:
            predictions: Model predictions dictionary
            targets: Target values dictionary
            
        Returns:
            Dictionary of losses
        """
        # Onset loss with positive weighting to handle class imbalance
        onset_loss = F.binary_cross_entropy_with_logits(
            predictions['onset_logits'],
            targets['onsets'],
            pos_weight=torch.tensor(self.config.get('onset_positive_weight', 5.0), 
                                    device=predictions['onset_logits'].device)
        )
        
        # Only compute velocity loss where onsets exist
        onset_mask = targets['onsets'] > 0.5
        if onset_mask.sum() > 0:
            velocity_loss = F.mse_loss(
                predictions['velocities'][onset_mask],
                targets['velocities'][onset_mask]
            )
        else:
            velocity_loss = torch.tensor(0.0, device=predictions['onset_logits'].device)
        
        # Total loss is weighted sum
        onset_weight = self.config.get('onset_loss_weight', 1.0)
        velocity_weight = self.config.get('velocity_loss_weight', 0.5)
        total_loss = onset_weight * onset_loss + velocity_weight * velocity_loss
        
        return {
            'onset_loss': onset_loss,
            'velocity_loss': velocity_loss,
            'total_loss': total_loss
        }


class FeatureEncoder(nn.Module):
    """Feature encoder based on CNN architecture"""
    
    def __init__(self, config: Dict):
        """
        Initialize the feature encoder.
        
        Args:
            config: Encoder configuration dictionary
        """
        super().__init__()
        
        input_channels = config.get('input_channels', 1)  # Default to mono audio
        base_channels = config.get('base_channels', 16)
        kernel_sizes = config.get('kernel_sizes', [7, 5, 3, 3])
        strides = config.get('strides', [2, 2, 1, 1])
        paddings = config.get('paddings', [3, 2, 1, 1])
        
        # Use a simple CNN architecture with increasing channel depth
        layers = []
        in_channels = input_channels
        
        for i, (kernel, stride, padding) in enumerate(zip(kernel_sizes, strides, paddings)):
            # For the tests to pass, we need to use the formula: base_channels * (2 ** i)
            out_channels = base_channels * (2 ** i)
            
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel, stride=stride, padding=padding),
                nn.BatchNorm1d(out_channels),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout(config.get('dropout', 0.2))
            ])
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*layers)
        
        # For test_feature_encoder_init
        # Expected output dimension is base_channels * (2 ** (num_layers - 1))
        num_layers = len(kernel_sizes)
        expected_output_dim = base_channels * (2 ** (num_layers - 1))
        self.output_dim = expected_output_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the feature encoder.
        
        Args:
            x: Input tensor [batch_size, channels, time]
            
        Returns:
            Encoded features [batch_size, features, time]
        """
        return self.cnn(x)


class TemporalModel(nn.Module):
    """Temporal model based on bidirectional LSTM"""
    
    def __init__(self, config: Dict, input_dim: int):
        """
        Initialize the temporal model.
        
        Args:
            config: Temporal model configuration dictionary
            input_dim: Input feature dimension from encoder
        """
        super().__init__()
        
        hidden_dim = config.get('hidden_dim', 256)
        num_layers = config.get('num_layers', 2)
        dropout = config.get('dropout', 0.2)
        
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=False  # LSTM expects [time, batch, features]
        )
        
        self.output_dim = hidden_dim * 2  # Bidirectional, so 2x hidden_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the temporal model.
        
        Args:
            x: Input tensor [batch_size, features, time]
            
        Returns:
            Processed features [batch_size, output_dim, time]
        """
        # Permute to [time, batch, features] for LSTM
        x = x.permute(2, 0, 1)
        
        outputs, _ = self.lstm(x)
        
        # Permute back to [batch, features, time]
        outputs = outputs.permute(1, 2, 0)
        
        return outputs


class CnnLstmTranscriber(DrumTranscriptionModel):
    """
    CNN-LSTM model for drum transcription.
    
    Architecture:
    1. Feature encoder (CNN) processes input features
    2. Temporal model (LSTM) captures timing patterns
    3. Separate heads for onset and velocity prediction
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the CNN-LSTM transcriber.
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__(config)
        
        # Feature encoder
        self.encoder = FeatureEncoder(config.get('encoder', {}))
        
        # Temporal model
        self.temporal = TemporalModel(
            config.get('temporal', {}),
            self.encoder.output_dim
        )
        
        # Prediction heads
        self.onset_head = nn.Conv1d(
            self.temporal.output_dim,
            1,  # Single channel for onsets
            kernel_size=1
        )
        
        self.velocity_head = nn.Sequential(
            nn.Conv1d(self.temporal.output_dim, 8, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(8, 1, kernel_size=1),
            nn.Sigmoid()  # Velocities are normalized to [0, 1]
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights for better convergence"""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    
    def forward(self, x: torch.Tensor, conditioning: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor [batch_size, input_channels, time]
            conditioning: Optional conditioning tensor (unused in this implementation)
            
        Returns:
            Dictionary of model outputs:
                - onset_logits: Raw logits for onset prediction
                - onset_probs: Probabilities of onsets (after sigmoid)
                - velocities: Predicted velocities
        """
        # Feature encoding
        features = self.encoder(x)
        
        # Temporal processing
        temporal_features = self.temporal(features)
        
        # Prediction heads
        onset_logits = self.onset_head(temporal_features).squeeze(1)
        onset_probs = torch.sigmoid(onset_logits)
        velocities = self.velocity_head(temporal_features).squeeze(1)
        
        return {
            'onset_logits': onset_logits,
            'onset_probs': onset_probs,
            'velocities': velocities
        }


class SmallCnnTranscriber(DrumTranscriptionModel):
    """
    Smaller CNN-only model for drum transcription when computational resources are limited.
    
    Architecture:
    1. Feature encoder (CNN) processes input features
    2. 1D convolutions with dilation for temporal context
    3. Separate heads for onset and velocity prediction
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the small CNN transcriber.
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__(config)
        
        # Feature encoder (simpler than the CnnLstm version)
        encoder_config = config.get('encoder', {})
        encoder_config['base_channels'] = encoder_config.get('base_channels', 8)  # Smaller channel sizes
        self.encoder = FeatureEncoder(encoder_config)
        
        # Temporal context with dilated convolutions instead of LSTM
        channels = self.encoder.output_dim
        dilation_factors = [1, 2, 4, 8]
        temporal_layers = []
        
        for dilation in dilation_factors:
            temporal_layers.extend([
                nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
                nn.BatchNorm1d(channels),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout(config.get('dropout', 0.1))
            ])
        
        self.temporal = nn.Sequential(*temporal_layers)
        
        # Prediction heads (simpler than CnnLstm version)
        self.onset_head = nn.Conv1d(channels, 1, kernel_size=1)
        
        self.velocity_head = nn.Sequential(
            nn.Conv1d(channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights for better convergence"""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor, conditioning: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor [batch_size, input_channels, time]
            conditioning: Optional conditioning tensor (unused in this implementation)
            
        Returns:
            Dictionary of model outputs
        """
        # Feature encoding
        features = self.encoder(x)
        
        # Temporal processing with dilated convolutions
        temporal_features = self.temporal(features)
        
        # Prediction heads
        onset_logits = self.onset_head(temporal_features).squeeze(1)
        onset_probs = torch.sigmoid(onset_logits)
        velocities = self.velocity_head(temporal_features).squeeze(1)
        
        return {
            'onset_logits': onset_logits,
            'onset_probs': onset_probs,
            'velocities': velocities
        }


# Loss functions that can be used outside the model class

def onset_loss(predictions: torch.Tensor, targets: torch.Tensor, pos_weight: Optional[float] = None) -> torch.Tensor:
    """
    Calculate onset detection loss with optional positive weighting.
    
    Args:
        predictions: Model predictions (logits)
        targets: Target labels (0/1)
        pos_weight: Positive class weight for handling class imbalance
        
    Returns:
        Loss value
    """
    if pos_weight is not None:
        pos_weight_tensor = torch.tensor(pos_weight, device=predictions.device)
        return F.binary_cross_entropy_with_logits(predictions, targets, pos_weight=pos_weight_tensor)
    else:
        return F.binary_cross_entropy_with_logits(predictions, targets)


def velocity_loss(
    predictions: torch.Tensor, 
    targets: torch.Tensor, 
    onset_mask: torch.Tensor
) -> torch.Tensor:
    """
    Calculate velocity prediction loss where onsets exist.
    
    Args:
        predictions: Model velocity predictions (normalized to [0, 1])
        targets: Target velocities (normalized to [0, 1])
        onset_mask: Boolean mask indicating where onsets exist
        
    Returns:
        Loss value
    """
    if onset_mask.sum() > 0:
        return F.mse_loss(predictions[onset_mask], targets[onset_mask])
    else:
        return torch.tensor(0.0, device=predictions.device)


def combined_loss(
    onset_pred: torch.Tensor,
    velocity_pred: torch.Tensor,
    onset_targets: torch.Tensor,
    velocity_targets: torch.Tensor,
    onset_weight: float = 1.0,
    velocity_weight: float = 0.5,
    pos_weight: Optional[float] = 5.0
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Combined loss function for joint optimization.
    
    Args:
        onset_pred: Onset predictions (logits)
        velocity_pred: Velocity predictions [0, 1]
        onset_targets: Onset targets (0/1)
        velocity_targets: Velocity targets [0, 1]
        onset_weight: Weight for onset loss
        velocity_weight: Weight for velocity loss
        pos_weight: Positive class weight for onset loss
        
    Returns:
        Tuple of (total_loss, loss_dict)
    """
    onset_targets_bool = onset_targets > 0.5
    
    # Calculate individual losses
    o_loss = onset_loss(onset_pred, onset_targets, pos_weight)
    v_loss = velocity_loss(velocity_pred, velocity_targets, onset_targets_bool)
    
    # Calculate weighted total loss
    total_loss = onset_weight * o_loss + velocity_weight * v_loss
    
    return total_loss, {
        'onset_loss': o_loss,
        'velocity_loss': v_loss,
        'total_loss': total_loss
    }


def get_model(model_name: str, config: Dict) -> DrumTranscriptionModel:
    """
    Factory function to create model instances.
    
    Args:
        model_name: Name of the model to create
        config: Model configuration dictionary
        
    Returns:
        Model instance
    """
    if model_name == 'cnn_lstm':
        return CnnLstmTranscriber(config)
    elif model_name == 'small_cnn':
        return SmallCnnTranscriber(config)
    else:
        raise ValueError(f"Unknown model name: {model_name}") 