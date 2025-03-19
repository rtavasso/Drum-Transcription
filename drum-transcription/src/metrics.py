"""
Evaluation Metrics for Drum Transcription

This module implements metrics for evaluating drum transcription models.

Key Components:
1. Note-level metrics (precision, recall, F1)
2. Frame-level metrics
3. Velocity accuracy metrics
4. Note extraction from frames

Functions:
- frame_metrics(onset_pred, onset_target, threshold=0.5): Frame-level metrics
- note_metrics(onset_pred, onset_target, velocity_pred, velocity_target, threshold=0.5): Note-level metrics
- extract_notes_from_frames(onset_frames, velocity_frames, threshold=0.5): Convert frames to notes
- velocity_metrics(pred_velocities, target_velocities, onset_mask): Calculate velocity accuracy
- visualize_metrics(metrics_dict, save_path=None): Visualize metrics
- evaluate_model(model, dataloader, device, threshold): Comprehensive model evaluation
- threshold_sweep(onset_pred, onset_target): Find optimal threshold
- precision_recall_curve(onset_pred, onset_target): Calculate precision-recall curve

Implementation Considerations:
- Support both frame-level and note-level evaluation
- Consider velocity in evaluation when appropriate
- Support different thresholds for onset detection
- Provide detailed breakdown by drum type if possible
- Implement visualizations for easier interpretation
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional, Union, Any
from sklearn.metrics import precision_recall_fscore_support, mean_absolute_error, mean_squared_error, precision_recall_curve as sk_precision_recall_curve, auc


def frame_metrics(
    onset_pred: np.ndarray,
    onset_target: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate frame-level metrics (precision, recall, F1).
    
    Args:
        onset_pred: Predicted onset probabilities or logits [batch_size, time] or [time]
        onset_target: Target onset values (0/1) [batch_size, time] or [time]
        threshold: Threshold for binary prediction
        
    Returns:
        Dictionary of metrics
    """
    # Ensure inputs are numpy arrays
    if isinstance(onset_pred, torch.Tensor):
        onset_pred = onset_pred.detach().cpu().numpy()
    if isinstance(onset_target, torch.Tensor):
        onset_target = onset_target.detach().cpu().numpy()
    
    # Flatten if batched
    if onset_pred.ndim > 1:
        onset_pred = onset_pred.reshape(-1)
    if onset_target.ndim > 1:
        onset_target = onset_target.reshape(-1)
    
    # Convert predictions to binary
    onset_pred_binary = (onset_pred > threshold).astype(np.int32)
    
    # Calculate true positives, false positives, false negatives
    tp = np.sum((onset_pred_binary == 1) & (onset_target == 1))
    fp = np.sum((onset_pred_binary == 1) & (onset_target == 0))
    fn = np.sum((onset_pred_binary == 0) & (onset_target == 1))
    
    # Calculate precision, recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Calculate accuracy
    accuracy = np.mean(onset_pred_binary == onset_target)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


def extract_notes_from_frames(
    onset_frames: np.ndarray,
    velocity_frames: np.ndarray,
    threshold: float = 0.5,
    min_gap_ms: int = 10,
    sample_rate: int = 100
) -> List[Dict[str, Any]]:
    """
    Convert frame-level predictions to note events.
    
    Args:
        onset_frames: Onset frame predictions [time]
        velocity_frames: Velocity frame predictions [time]
        threshold: Threshold for onset detection
        min_gap_ms: Minimum gap between notes in milliseconds
        sample_rate: Frame sample rate in Hz
        
    Returns:
        List of note events, each with onset time and velocity
    """
    # Ensure inputs are numpy arrays
    if isinstance(onset_frames, torch.Tensor):
        onset_frames = onset_frames.detach().cpu().numpy()
    if isinstance(velocity_frames, torch.Tensor):
        velocity_frames = velocity_frames.detach().cpu().numpy()
    
    # Apply threshold to get binary onsets
    binary_onsets = onset_frames > threshold
    
    # Convert min_gap from ms to frames
    min_gap_frames = max(1, int((min_gap_ms / 1000) * sample_rate))
    
    # Initialize list to store note events
    notes = []
    
    # Find onset positions
    onset_positions = np.where(binary_onsets)[0]
    
    if len(onset_positions) > 0:
        # Initialize with the first onset
        current_onset = onset_positions[0]
        notes.append({
            'onset_frame': current_onset,
            'onset_time': current_onset / sample_rate,
            'velocity': velocity_frames[current_onset]
        })
        
        # Process remaining onsets
        for pos in onset_positions[1:]:
            # Check if this onset is separated enough from the previous one
            if pos - current_onset >= min_gap_frames:
                # Add a new note
                notes.append({
                    'onset_frame': pos,
                    'onset_time': pos / sample_rate,
                    'velocity': velocity_frames[pos]
                })
                current_onset = pos
    
    return notes


def note_metrics(
    pred_notes: List[Dict[str, Any]],
    target_notes: List[Dict[str, Any]],
    tolerance_ms: float = 50.0
) -> Dict[str, float]:
    """
    Calculate note-level metrics with velocity considerations.
    
    Args:
        pred_notes: List of predicted note events
        target_notes: List of target note events
        tolerance_ms: Time tolerance in milliseconds for matching notes
        
    Returns:
        Dictionary of note-level metrics
    """
    if not target_notes:
        if not pred_notes:
            return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'velocity_error': 0.0}
        else:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'velocity_error': 0.0}
    
    if not pred_notes:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'velocity_error': 0.0}
    
    # Convert tolerance from ms to seconds
    tolerance_sec = tolerance_ms / 1000.0
    
    # Initialize counters
    true_positives = 0
    velocity_errors = []
    
    # For each target note, find the closest predicted note within tolerance
    matched_preds = set()
    
    for target in target_notes:
        target_time = target['onset_time']
        target_velocity = target['velocity']
        
        # Find closest predicted note
        closest_pred = None
        min_distance = float('inf')
        
        for i, pred in enumerate(pred_notes):
            if i in matched_preds:
                continue
                
            pred_time = pred['onset_time']
            time_diff = abs(pred_time - target_time)
            
            if time_diff < tolerance_sec and time_diff < min_distance:
                closest_pred = pred
                min_distance = time_diff
                closest_idx = i
        
        # If a match was found
        if closest_pred is not None:
            true_positives += 1
            matched_preds.add(closest_idx)
            
            # Calculate velocity error
            pred_velocity = closest_pred['velocity']
            velocity_error = abs(pred_velocity - target_velocity)
            velocity_errors.append(velocity_error)
    
    # Calculate metrics
    false_positives = len(pred_notes) - true_positives
    false_negatives = len(target_notes) - true_positives
    
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    
    # Calculate average velocity error for matched notes
    avg_velocity_error = np.mean(velocity_errors) if velocity_errors else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'velocity_error': avg_velocity_error,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }


def velocity_metrics(
    pred_velocities: np.ndarray,
    target_velocities: np.ndarray,
    onset_mask: np.ndarray
) -> Dict[str, float]:
    """
    Calculate velocity prediction accuracy metrics.
    
    Args:
        pred_velocities: Predicted velocities [batch_size, time] or [time]
        target_velocities: Target velocities [batch_size, time] or [time]
        onset_mask: Boolean mask indicating where onsets exist
        
    Returns:
        Dictionary of velocity metrics
    """
    # Ensure inputs are numpy arrays
    if isinstance(pred_velocities, torch.Tensor):
        pred_velocities = pred_velocities.detach().cpu().numpy()
    if isinstance(target_velocities, torch.Tensor):
        target_velocities = target_velocities.detach().cpu().numpy()
    if isinstance(onset_mask, torch.Tensor):
        onset_mask = onset_mask.detach().cpu().numpy()
    
    # Flatten if batched
    if pred_velocities.ndim > 1:
        pred_velocities = pred_velocities.reshape(-1)
    if target_velocities.ndim > 1:
        target_velocities = target_velocities.reshape(-1)
    if onset_mask.ndim > 1:
        onset_mask = onset_mask.reshape(-1)
    
    # Only evaluate where onsets exist
    if np.sum(onset_mask) == 0:
        return {
            'mae': 0.0,
            'mse': 0.0,
            'rmse': 0.0,
            'r2': 0.0,
            'count': 0
        }
    
    pred_velocities_masked = pred_velocities[onset_mask]
    target_velocities_masked = target_velocities[onset_mask]
    
    # Calculate metrics
    mae = mean_absolute_error(target_velocities_masked, pred_velocities_masked)
    mse = mean_squared_error(target_velocities_masked, pred_velocities_masked)
    rmse = np.sqrt(mse)
    
    # Calculate R-squared
    ss_total = np.sum((target_velocities_masked - np.mean(target_velocities_masked)) ** 2)
    ss_residual = np.sum((target_velocities_masked - pred_velocities_masked) ** 2)
    r2 = 1 - (ss_residual / ss_total) if ss_total > 0 else 0.0
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'count': np.sum(onset_mask)
    }


def visualize_metrics(
    metrics_dict: Dict[str, float],
    save_path: Optional[str] = None,
    title: str = "Evaluation Metrics"
) -> None:
    """
    Visualize metrics as bar charts.
    
    Args:
        metrics_dict: Dictionary of metrics to visualize
        save_path: Path to save the visualization (optional)
        title: Title for the visualization
    """
    # Filter metrics for visualization
    viz_metrics = {}
    for k, v in metrics_dict.items():
        if isinstance(v, (int, float)) and not k.startswith('_'):
            # Skip count fields and other non-metric values
            if not any(k.endswith(suffix) for suffix in ['count', 'tp', 'fp', 'fn']):
                viz_metrics[k] = v
    
    plt.figure(figsize=(10, 6))
    
    # Create bar chart
    bars = plt.bar(viz_metrics.keys(), viz_metrics.values())
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{height:.3f}', ha='center', va='bottom')
    
    # Set title and labels
    plt.title(title)
    plt.ylabel('Value')
    plt.ylim(0, min(1.1, max(list(viz_metrics.values()) + [1.0]) * 1.2))
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save or display
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def visualize_predictions(
    audio: np.ndarray,
    onset_pred: np.ndarray,
    onset_target: np.ndarray,
    velocity_pred: Optional[np.ndarray] = None,
    velocity_target: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    sample_rate: int = 100,
    audio_sample_rate: int = 44100,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize predictions vs targets for analysis.
    
    Args:
        audio: Audio waveform
        onset_pred: Predicted onset probabilities
        onset_target: Target onset values (0/1)
        velocity_pred: Predicted velocities (optional)
        velocity_target: Target velocities (optional)
        threshold: Threshold for binary onset prediction
        sample_rate: Frame sample rate in Hz
        audio_sample_rate: Audio sample rate in Hz
        save_path: Path to save the visualization (optional)
    """
    # Ensure inputs are numpy arrays
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    if isinstance(onset_pred, torch.Tensor):
        onset_pred = onset_pred.detach().cpu().numpy()
    if isinstance(onset_target, torch.Tensor):
        onset_target = onset_target.detach().cpu().numpy()
    
    if velocity_pred is not None and isinstance(velocity_pred, torch.Tensor):
        velocity_pred = velocity_pred.detach().cpu().numpy()
    if velocity_target is not None and isinstance(velocity_target, torch.Tensor):
        velocity_target = velocity_target.detach().cpu().numpy()
    
    # Create time axis for frames and audio
    frame_times = np.arange(len(onset_pred)) / sample_rate
    audio_times = np.arange(len(audio)) / audio_sample_rate
    
    # Determine number of subplots
    n_plots = 2 + (velocity_pred is not None) + (velocity_target is not None)
    
    # Create plot
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 8), sharex=True)
    
    # Plot audio waveform
    axes[0].plot(audio_times, audio)
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Audio Waveform')
    
    # Plot onset predictions and targets
    axes[1].plot(frame_times, onset_pred, label='Predicted')
    axes[1].plot(frame_times, onset_target, 'r--', label='Target')
    axes[1].axhline(y=threshold, color='g', linestyle=':', label=f'Threshold ({threshold})')
    axes[1].set_ylabel('Probability')
    axes[1].set_title('Onset Detection')
    axes[1].legend()
    
    # Plot velocities if provided
    plot_idx = 2
    if velocity_pred is not None:
        axes[plot_idx].plot(frame_times, velocity_pred, label='Predicted')
        if velocity_target is not None:
            axes[plot_idx].plot(frame_times, velocity_target, 'r--', label='Target')
        axes[plot_idx].set_ylabel('Velocity')
        axes[plot_idx].set_title('Velocity Prediction')
        axes[plot_idx].legend()
        plot_idx += 1
    
    # Set common x-axis label
    axes[-1].set_xlabel('Time (s)')
    
    plt.tight_layout()
    
    # Save or display
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    threshold: float = 0.5,
    tolerance_ms: float = 50.0,
    sample_rate: int = 100
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation data
        device: Device to evaluate on ('cpu' or 'cuda')
        threshold: Threshold for onset detection
        tolerance_ms: Time tolerance for note matching in milliseconds
        sample_rate: Frame sample rate in Hz
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    # Initialize lists to store results
    all_frame_metrics = []
    all_note_metrics = []
    all_velocity_metrics = []
    
    # Process each batch
    with torch.no_grad():
        for audio, onsets, velocities in dataloader:
            # Move data to device
            audio = audio.to(device)
            onsets = onsets.to(device)
            velocities = velocities.to(device)
            
            # Forward pass
            predictions = model(audio)
            
            # Process each example in batch
            for i in range(audio.size(0)):
                # Get predictions and targets for this example
                onset_pred = predictions['onset_probs'][i].cpu().numpy()
                onset_target = onsets[i].cpu().numpy()
                velocity_pred = predictions['velocities'][i].cpu().numpy()
                velocity_target = velocities[i].cpu().numpy()
                
                # Calculate frame-level metrics
                frame_result = frame_metrics(onset_pred, onset_target, threshold)
                all_frame_metrics.append(frame_result)
                
                # Calculate velocity metrics
                onset_mask = onset_target > 0.5
                velocity_result = velocity_metrics(velocity_pred, velocity_target, onset_mask)
                all_velocity_metrics.append(velocity_result)
                
                # Extract notes and calculate note-level metrics
                pred_notes = extract_notes_from_frames(onset_pred, velocity_pred, threshold, sample_rate=sample_rate)
                target_notes = extract_notes_from_frames(onset_target, velocity_target, 0.5, sample_rate=sample_rate)
                note_result = note_metrics(pred_notes, target_notes, tolerance_ms)
                all_note_metrics.append(note_result)
    
    # Aggregate metrics
    aggregated_results = {
        'frame': _aggregate_metrics(all_frame_metrics),
        'note': _aggregate_metrics(all_note_metrics),
        'velocity': _aggregate_metrics(all_velocity_metrics)
    }
    
    return aggregated_results


def _aggregate_metrics(metric_list: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Aggregate metrics from multiple examples.
    
    Args:
        metric_list: List of metric dictionaries
        
    Returns:
        Aggregated metrics
    """
    if not metric_list:
        return {}
    
    # Initialize with first entry
    keys = metric_list[0].keys()
    aggregated = {k: [] for k in keys}
    
    # Collect values
    for metrics in metric_list:
        for k, v in metrics.items():
            if k in aggregated:
                aggregated[k].append(v)
    
    # Calculate means
    result = {k: np.mean(v) for k, v in aggregated.items()}
    
    return result


def threshold_sweep(
    onset_pred: np.ndarray,
    onset_target: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    metric: str = 'f1'
) -> Tuple[float, Dict[str, np.ndarray]]:
    """
    Find optimal threshold by sweeping through threshold values.
    
    Args:
        onset_pred: Predicted onset probabilities
        onset_target: Target onset values (0/1)
        thresholds: Array of threshold values to try (default: np.linspace(0.01, 0.99, 50))
        metric: Metric to optimize ('f1', 'precision', 'recall', 'accuracy')
        
    Returns:
        Tuple of (optimal_threshold, metrics_dict)
    """
    # Ensure inputs are numpy arrays
    if isinstance(onset_pred, torch.Tensor):
        onset_pred = onset_pred.detach().cpu().numpy()
    if isinstance(onset_target, torch.Tensor):
        onset_target = onset_target.detach().cpu().numpy()
    
    # Default thresholds
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 50)
    
    # Initialize results arrays
    precision_values = np.zeros_like(thresholds)
    recall_values = np.zeros_like(thresholds)
    f1_values = np.zeros_like(thresholds)
    accuracy_values = np.zeros_like(thresholds)
    
    # Compute metrics for each threshold
    for i, threshold in enumerate(thresholds):
        metrics = frame_metrics(onset_pred, onset_target, threshold)
        precision_values[i] = metrics['precision']
        recall_values[i] = metrics['recall']
        f1_values[i] = metrics['f1']
        accuracy_values[i] = metrics['accuracy']
    
    # Determine optimal threshold based on specified metric
    if metric == 'f1':
        optimal_idx = np.argmax(f1_values)
    elif metric == 'precision':
        optimal_idx = np.argmax(precision_values)
    elif metric == 'recall':
        optimal_idx = np.argmax(recall_values)
    elif metric == 'accuracy':
        optimal_idx = np.argmax(accuracy_values)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    optimal_threshold = thresholds[optimal_idx]
    
    # Collect results
    results = {
        'thresholds': thresholds,
        'precision': precision_values,
        'recall': recall_values,
        'f1': f1_values,
        'accuracy': accuracy_values,
        'optimal_threshold': optimal_threshold,
        'optimal_'+metric: np.max(locals()[metric+'_values'])
    }
    
    return optimal_threshold, results


def precision_recall_curve(
    onset_pred: np.ndarray,
    onset_target: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Calculate precision-recall curve for onset detection.
    
    Args:
        onset_pred: Predicted onset probabilities
        onset_target: Target onset values (0/1)
        
    Returns:
        Dictionary with precision, recall, thresholds, and AUC
    """
    # Ensure inputs are numpy arrays
    if isinstance(onset_pred, torch.Tensor):
        onset_pred = onset_pred.detach().cpu().numpy()
    if isinstance(onset_target, torch.Tensor):
        onset_target = onset_target.detach().cpu().numpy()
    
    # Flatten if batched
    if onset_pred.ndim > 1:
        onset_pred = onset_pred.reshape(-1)
    if onset_target.ndim > 1:
        onset_target = onset_target.reshape(-1)
    
    # Calculate precision-recall curve
    precision, recall, thresholds = sk_precision_recall_curve(onset_target, onset_pred)
    
    # Calculate AUC
    auc_score = auc(recall, precision)
    
    return {
        'precision': precision,
        'recall': recall,
        'thresholds': thresholds,
        'auc': auc_score
    }


def plot_precision_recall_curve(
    onset_pred: np.ndarray,
    onset_target: np.ndarray,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Plot precision-recall curve for onset detection.
    
    Args:
        onset_pred: Predicted onset probabilities
        onset_target: Target onset values (0/1)
        save_path: Path to save the visualization (optional)
        
    Returns:
        Dictionary with precision, recall, thresholds, and AUC
    """
    # Calculate precision-recall curve
    pr_curve = precision_recall_curve(onset_pred, onset_target)
    
    # Create plot
    plt.figure(figsize=(8, 6))
    plt.plot(pr_curve['recall'], pr_curve['precision'], lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (AUC = {pr_curve["auc"]:.3f})')
    plt.grid(True)
    
    # Add iso-f1 curves
    f1_levels = [0.2, 0.4, 0.6, 0.8]
    for f1 in f1_levels:
        x = np.linspace(0.01, 1.0, 100)
        y = (f1 * x) / (2 * x - f1)
        mask = (y <= 1.0) & (y >= 0.0)
        plt.plot(x[mask], y[mask], linestyle=':', color='gray', alpha=0.5)
        plt.annotate(f'F1={f1}', xy=(x[mask][-1], y[mask][-1]), 
                    xytext=(5, 0), textcoords='offset points',
                    fontsize=8, color='gray')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    
    # Save or display
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
    
    return pr_curve
