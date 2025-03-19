"""
Training Loop and Utilities for Drum Transcription

This module handles model training for the drum transcription system.

Key Components:
1. Training loop implementation
2. Validation loop implementation
3. Checkpointing and logging
4. Hyperparameter management
5. Loss function implementations

Functions:
- train_model(model, train_loader, val_loader, config): Main training function
- train_epoch(model, dataloader, optimizer, device): Run a single training epoch
- validate(model, dataloader, device): Run validation
- onset_loss(predictions, targets, pos_weight): Calculate onset detection loss
- velocity_loss(predictions, targets, onset_mask): Calculate velocity prediction loss
- combined_loss(onset_pred, velocity_pred, onset_targets, velocity_targets): Combined loss function
- save_checkpoint(model, optimizer, epoch, metrics, path): Save model checkpoint
- load_checkpoint(path, model, optimizer): Load model checkpoint

Implementation Considerations:
- Use mixed precision training for efficiency
- Implement early stopping and learning rate scheduling
- Log training progress (tensorboard, wandb, etc.)
- Save best models based on validation metrics
- Support distributed training if needed
- Ensure proper gradient clipping to prevent exploding gradients
- Implement proper device management (CPU/GPU/multi-GPU)
- Add training resumption capability
- Track and visualize metrics during training
- Include proper exception handling and training status reporting
"""

import os
import time
import json
import torch
import numpy as np
import wandb
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Union, Any
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from src.model import DrumTranscriptionModel, combined_loss


def train_model(
    model: DrumTranscriptionModel,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    config: Dict[str, Any],
    resume_from: Optional[str] = None
) -> Tuple[DrumTranscriptionModel, Dict[str, List[float]]]:
    """
    Main training function.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        config: Training configuration dictionary
        resume_from: Path to checkpoint to resume from (optional)
        
    Returns:
        Tuple of (trained model, training history)
    """
    # Extract training parameters from config
    num_epochs = config.get('num_epochs', 100)
    early_stopping_patience = config.get('early_stopping_patience', 10)
    learning_rate = config.get('learning_rate', 0.001)
    weight_decay = config.get('weight_decay', 1e-5)
    grad_clip_value = config.get('grad_clip_value', 1.0)
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    use_mixed_precision = config.get('mixed_precision', True) and device != 'cpu'
    checkpoint_dir = Path(config.get('checkpoint_dir', './checkpoints'))
    
    # Initialize wandb if enabled
    use_wandb = config.get('use_wandb', True)
    if use_wandb:
        wandb.init(
            project=config.get('wandb_project', 'drum-transcription'),
            name=config.get('wandb_run_name', f"{config.get('model_name', 'model')}_{time.strftime('%Y%m%d_%H%M%S')}"),
            config=config
        )
        wandb.watch(model, log_freq=100)
    
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Move model to device
    model = model.to(device)
    
    # Initialize optimizer and scheduler
    optimizer, scheduler = model.configure_optimizers()
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler() if use_mixed_precision else None
    
    # Initialize training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_onset_f1': [],
        'val_velocity_mae': [],
        'learning_rate': []
    }
    
    # Initialize early stopping variables
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if resume_from:
        start_epoch, history = load_checkpoint(resume_from, model, optimizer)
        print(f"Resuming training from epoch {start_epoch}")
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        # Train for one epoch
        train_metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip_value=grad_clip_value,
            use_mixed_precision=use_mixed_precision,
            scaler=scaler
        )
        
        # Validate
        val_metrics = validate(model, val_loader, device)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_onset_f1'].append(val_metrics['onset_f1'])
        history['val_velocity_mae'].append(val_metrics['velocity_mae'])
        history['learning_rate'].append(current_lr)
        
        # Update learning rate scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_metrics['loss']:.4f} - "
              f"Val Loss: {val_metrics['loss']:.4f} - "
              f"Val Onset F1: {val_metrics['onset_f1']:.4f} - "
              f"Val Velocity MAE: {val_metrics['velocity_mae']:.4f} - "
              f"LR: {current_lr:.6f}")
        
        # Log to wandb if enabled
        if use_wandb:
            wandb.log({
                'epoch': epoch,
                'train/loss': train_metrics['loss'],
                'train/onset_loss': train_metrics['onset_loss'],
                'train/velocity_loss': train_metrics['velocity_loss'],
                'val/loss': val_metrics['loss'],
                'val/onset_loss': val_metrics['onset_loss'],
                'val/velocity_loss': val_metrics['velocity_loss'],
                'val/onset_f1': val_metrics['onset_f1'],
                'val/onset_precision': val_metrics['onset_precision'],
                'val/onset_recall': val_metrics['onset_recall'],
                'val/velocity_mae': val_metrics['velocity_mae'],
                'learning_rate': current_lr
            })
        
        # Save checkpoint
        is_best = val_metrics['loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['loss']
            best_epoch = epoch
            epochs_without_improvement = 0
            
            # Save best model
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                path=checkpoint_dir / 'best_model.pt',
                config=config
            )
        else:
            epochs_without_improvement += 1
        
        # Save latest model
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=val_metrics,
            path=checkpoint_dir / 'latest_model.pt',
            config=config
        )
        
        # Early stopping
        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            print(f"Best model was at epoch {best_epoch+1} with validation loss {best_val_loss:.4f}")
            break
    
    # Log final best model metrics
    if use_wandb:
        wandb.run.summary['best_epoch'] = best_epoch
        wandb.run.summary['best_val_loss'] = best_val_loss
        wandb.finish()
    
    # Load best model before returning
    load_checkpoint(checkpoint_dir / 'best_model.pt', model, None)
    
    return model, history


def train_epoch(
    model: DrumTranscriptionModel,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    grad_clip_value: float = 1.0,
    use_mixed_precision: bool = False,
    scaler: Optional[GradScaler] = None
) -> Dict[str, float]:
    """
    Run a single training epoch.
    
    Args:
        model: The model to train
        dataloader: DataLoader for training data
        optimizer: Optimizer for training
        device: Device to train on ('cpu' or 'cuda')
        grad_clip_value: Value for gradient clipping
        use_mixed_precision: Whether to use mixed precision training
        scaler: Gradient scaler for mixed precision training
        
    Returns:
        Dictionary of training metrics
    """
    model.train()
    
    running_loss = 0.0
    running_onset_loss = 0.0
    running_velocity_loss = 0.0
    
    # Initialize progress bar
    pbar = tqdm(dataloader, desc="Training")
    
    for audio, onsets, velocities in pbar:
        # Move data to device
        audio = audio.to(device)
        onsets = onsets.to(device)
        velocities = velocities.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with or without mixed precision
        if use_mixed_precision:
            with autocast():
                # Forward pass
                predictions = model(audio)
                
                # Compute loss
                loss, loss_dict = combined_loss(
                    predictions['onset_logits'],
                    predictions['velocities'],
                    onsets,
                    velocities,
                    onset_weight=model.config.get('onset_loss_weight', 1.0),
                    velocity_weight=model.config.get('velocity_loss_weight', 0.5),
                    pos_weight=model.config.get('onset_positive_weight', 5.0)
                )
                
            # Backward pass
            scaler.scale(loss).backward()
            
            # Clip gradients
            if grad_clip_value > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
            
            # Update weights
            scaler.step(optimizer)
            scaler.update()
        else:
            # Forward pass
            predictions = model(audio)
            
            # Compute loss
            loss, loss_dict = combined_loss(
                predictions['onset_logits'],
                predictions['velocities'],
                onsets,
                velocities,
                onset_weight=model.config.get('onset_loss_weight', 1.0),
                velocity_weight=model.config.get('velocity_loss_weight', 0.5),
                pos_weight=model.config.get('onset_positive_weight', 5.0)
            )
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            if grad_clip_value > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
            
            # Update weights
            optimizer.step()
        
        # Update running loss values
        running_loss += loss.item()
        running_onset_loss += loss_dict['onset_loss'].item()
        running_velocity_loss += loss_dict['velocity_loss'].item() if 'velocity_loss' in loss_dict else 0.0
        
        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item(),
            'onset_loss': loss_dict['onset_loss'].item(),
            'velocity_loss': loss_dict['velocity_loss'].item() if 'velocity_loss' in loss_dict else 0.0
        })
    
    # Calculate average losses
    avg_loss = running_loss / len(dataloader)
    avg_onset_loss = running_onset_loss / len(dataloader)
    avg_velocity_loss = running_velocity_loss / len(dataloader)
    
    return {
        'loss': avg_loss,
        'onset_loss': avg_onset_loss,
        'velocity_loss': avg_velocity_loss
    }


def validate(
    model: DrumTranscriptionModel,
    dataloader: torch.utils.data.DataLoader,
    device: str
) -> Dict[str, float]:
    """
    Run validation.
    
    Args:
        model: The model to validate
        dataloader: DataLoader for validation data
        device: Device to validate on ('cpu' or 'cuda')
        
    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    
    running_loss = 0.0
    running_onset_loss = 0.0
    running_velocity_loss = 0.0
    
    # For F1 calculation
    all_onset_preds = []
    all_onset_targets = []
    
    # For velocity MAE calculation
    all_velocity_errors = []
    
    # Disable gradient calculation for validation
    with torch.no_grad():
        for audio, onsets, velocities in tqdm(dataloader, desc="Validating"):
            # Move data to device
            audio = audio.to(device)
            onsets = onsets.to(device)
            velocities = velocities.to(device)
            
            # Forward pass
            predictions = model(audio)
            
            # Compute loss
            loss, loss_dict = combined_loss(
                predictions['onset_logits'],
                predictions['velocities'],
                onsets,
                velocities,
                onset_weight=model.config.get('onset_loss_weight', 1.0),
                velocity_weight=model.config.get('velocity_loss_weight', 0.5),
                pos_weight=model.config.get('onset_positive_weight', 5.0)
            )
            
            # Update running loss values
            running_loss += loss.item()
            running_onset_loss += loss_dict['onset_loss'].item()
            running_velocity_loss += loss_dict['velocity_loss'].item() if 'velocity_loss' in loss_dict else 0.0
            
            # Store predictions and targets for metric calculation
            onset_preds = (torch.sigmoid(predictions['onset_logits']) > 0.5).float()
            all_onset_preds.append(onset_preds.cpu().numpy())
            all_onset_targets.append(onsets.cpu().numpy())
            
            # Calculate velocity error only where onsets exist
            onset_mask = onsets > 0.5
            if onset_mask.sum() > 0:
                velocity_errors = torch.abs(predictions['velocities'][onset_mask] - velocities[onset_mask])
                all_velocity_errors.append(velocity_errors.cpu().numpy())
    
    # Calculate average losses
    avg_loss = running_loss / len(dataloader)
    avg_onset_loss = running_onset_loss / len(dataloader)
    avg_velocity_loss = running_velocity_loss / len(dataloader)
    
    # Calculate onset detection metrics (precision, recall, F1)
    onset_metrics = calculate_onset_metrics(all_onset_preds, all_onset_targets)
    
    # Calculate velocity MAE
    velocity_mae = np.mean(np.concatenate(all_velocity_errors)) if all_velocity_errors else 0.0
    
    return {
        'loss': avg_loss,
        'onset_loss': avg_onset_loss,
        'velocity_loss': avg_velocity_loss,
        'onset_precision': onset_metrics['precision'],
        'onset_recall': onset_metrics['recall'],
        'onset_f1': onset_metrics['f1'],
        'velocity_mae': velocity_mae
    }


def calculate_onset_metrics(predictions: List[np.ndarray], targets: List[np.ndarray]) -> Dict[str, float]:
    """
    Calculate onset detection metrics (precision, recall, F1).
    
    Args:
        predictions: List of prediction arrays
        targets: List of target arrays
        
    Returns:
        Dictionary of metrics
    """
    # Concatenate all batches
    all_preds = np.concatenate([p.flatten() for p in predictions])
    all_targets = np.concatenate([t.flatten() for t in targets])
    
    # Calculate true positives, false positives, false negatives
    tp = np.sum((all_preds == 1) & (all_targets == 1))
    fp = np.sum((all_preds == 1) & (all_targets == 0))
    fn = np.sum((all_preds == 0) & (all_targets == 1))
    
    # Calculate precision, recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def save_checkpoint(
    model: DrumTranscriptionModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    path: Union[str, Path],
    config: Dict[str, Any]
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: The model to save
        optimizer: The optimizer to save
        epoch: Current epoch number
        metrics: Dictionary of metrics
        path: Path to save checkpoint to
        config: Configuration dictionary
    """
    # Create checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config
    }
    
    # Save checkpoint
    torch.save(checkpoint, path)
    
    # Convert metrics for JSON serialization
    json_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, np.float32) or isinstance(v, np.float64):
            json_metrics[k] = float(v)
        elif isinstance(v, np.int32) or isinstance(v, np.int64):
            json_metrics[k] = int(v)
        else:
            json_metrics[k] = v
    
    # Save metrics to JSON for easy review
    metrics_path = Path(path).with_suffix('.json')
    with open(metrics_path, 'w') as f:
        json.dump({
            'epoch': epoch,
            'metrics': json_metrics,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }, f, indent=2)


def load_checkpoint(
    path: Union[str, Path],
    model: DrumTranscriptionModel,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> Tuple[int, Dict[str, List[float]]]:
    """
    Load model checkpoint.
    
    Args:
        path: Path to checkpoint
        model: Model to load checkpoint into
        optimizer: Optimizer to load checkpoint into (optional)
        
    Returns:
        Tuple of (epoch, history)
    """
    # Load checkpoint
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Reconstruct history (may not be present in older checkpoints)
    history = {
        'train_loss': checkpoint.get('train_loss', []),
        'val_loss': checkpoint.get('val_loss', []),
        'val_onset_f1': checkpoint.get('val_onset_f1', []),
        'val_velocity_mae': checkpoint.get('val_velocity_mae', []),
        'learning_rate': checkpoint.get('learning_rate', [])
    }
    
    return checkpoint['epoch'], history 