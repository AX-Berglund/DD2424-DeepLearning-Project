#!/usr/bin/env python
"""
Evaluation metrics for model assessment.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)


def compute_accuracy(outputs, targets, binary=False):
    """
    Compute classification accuracy.
    
    Args:
        outputs (torch.Tensor): Model outputs.
        targets (torch.Tensor): Target labels.
        binary (bool): Whether this is a binary classification task.
        
    Returns:
        float: Accuracy score.
    """
    if binary:
        # For binary classification, outputs are logits
        probs = torch.sigmoid(outputs)
        preds = (probs >= 0.5).float()
        return (preds == targets).float().mean().item()
    else:
        # For multi-class classification
        _, preds = torch.max(outputs, 1)
        return (preds == targets).float().mean().item()


def compute_loss(outputs, targets, weights=None):
    """
    Compute cross-entropy loss.
    
    Args:
        outputs (torch.Tensor): Model outputs.
        targets (torch.Tensor): Target labels.
        weights (torch.Tensor, optional): Class weights.
        
    Returns:
        torch.Tensor: Loss value.
    """
    return F.cross_entropy(outputs, targets, weight=weights)


def compute_metrics(outputs, targets, class_names=None, binary=False):
    """
    Compute comprehensive classification metrics.
    
    Args:
        outputs (torch.Tensor): Model outputs.
        targets (torch.Tensor): Target labels.
        class_names (list, optional): List of class names.
        binary (bool): Whether this is a binary classification task.
        
    Returns:
        dict: Dictionary of metrics.
    """
    # Convert to numpy arrays
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    # Get predictions
    if binary:
        # For binary classification, outputs are already logits
        probs = torch.sigmoid(torch.tensor(outputs)).numpy()
        preds = (probs >= 0.5).astype(int)
        # Remove extra dimension if present
        if preds.ndim > 1:
            preds = preds.squeeze()
        if targets.ndim > 1:
            targets = targets.squeeze()
    else:
        # For multi-class classification
        if outputs.ndim > 1:
            probs = F.softmax(torch.tensor(outputs), dim=1).numpy()
            preds = np.argmax(outputs, axis=1)
        else:
            preds = outputs
            probs = None
    
    # Create metrics dictionary
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(targets, preds)
    
    # For binary classification
    if binary or (class_names is not None and len(class_names) == 2):
        metrics['precision'] = precision_score(targets, preds, average='binary', zero_division=0)
        metrics['recall'] = recall_score(targets, preds, average='binary', zero_division=0)
        metrics['f1'] = f1_score(targets, preds, average='binary', zero_division=0)
        
        # ROC AUC (if probabilities are available)
        if probs is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(targets, probs)
            except Exception:
                metrics['roc_auc'] = float('nan')
    else:
        # For multi-class classification
        metrics['precision_macro'] = precision_score(targets, preds, average='macro')
        metrics['recall_macro'] = recall_score(targets, preds, average='macro')
        metrics['f1_macro'] = f1_score(targets, preds, average='macro')
        
        metrics['precision_weighted'] = precision_score(targets, preds, average='weighted')
        metrics['recall_weighted'] = recall_score(targets, preds, average='weighted')
        metrics['f1_weighted'] = f1_score(targets, preds, average='weighted')
    
    # Confusion matrix
    cm = confusion_matrix(targets, preds)
    metrics['confusion_matrix'] = cm
    
    # Store predictions and targets for further analysis
    metrics['y_pred'] = preds
    metrics['y_true'] = targets
    
    # Per-class metrics
    if class_names is not None:
        metrics['class_names'] = class_names
        
        # Compute per-class accuracy
        per_class_acc = {}
        for i, class_name in enumerate(class_names):
            class_indices = (targets == i)
            if np.sum(class_indices) > 0:
                class_acc = np.mean(preds[class_indices] == targets[class_indices])
                per_class_acc[class_name] = class_acc
        
        metrics['per_class_accuracy'] = per_class_acc
        
        # Classification report
        report = classification_report(targets, preds, target_names=class_names, output_dict=True)
        metrics['classification_report'] = report
    
    return metrics


def confusion_matrix_to_figure(cm, class_names):
    """
    Convert confusion matrix to a matplotlib figure.
    
    Args:
        cm (numpy.ndarray): Confusion matrix.
        class_names (list): List of class names.
        
    Returns:
        matplotlib.figure.Figure: Figure with confusion matrix plot.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot confusion matrix
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Set labels
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel='True label',
        xlabel='Predicted label'
    )
    
    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data and create text annotations
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )
    
    fig.tight_layout()
    return fig


def compute_mixup_loss(outputs, targets_a, targets_b, lam):
    """
    Compute cross-entropy loss for mixup training.
    
    Args:
        outputs (torch.Tensor): Model outputs.
        targets_a (torch.Tensor): First target labels.
        targets_b (torch.Tensor): Second target labels.
        lam (float): Lambda value for mixing.
        
    Returns:
        torch.Tensor: Mixed loss value.
    """
    return lam * F.cross_entropy(outputs, targets_a) + \
           (1 - lam) * F.cross_entropy(outputs, targets_b)


def compute_cutmix_loss(outputs, targets_a, targets_b, lam):
    """
    Compute cross-entropy loss for CutMix training (same as mixup loss).
    
    Args:
        outputs (torch.Tensor): Model outputs.
        targets_a (torch.Tensor): First target labels.
        targets_b (torch.Tensor): Second target labels.
        lam (float): Lambda value for mixing.
        
    Returns:
        torch.Tensor: Mixed loss value.
    """
    return compute_mixup_loss(outputs, targets_a, targets_b, lam)


def compute_weighted_loss(outputs, targets, class_counts):
    """
    Compute weighted cross-entropy loss based on class frequencies.
    
    Args:
        outputs (torch.Tensor): Model outputs.
        targets (torch.Tensor): Target labels.
        class_counts (list or torch.Tensor): Count of samples for each class.
        
    Returns:
        torch.Tensor: Weighted loss value.
    """
    # Convert class_counts to tensor if it's a list
    if isinstance(class_counts, list):
        class_counts = torch.tensor(class_counts, dtype=torch.float32, device=outputs.device)
    
    # Calculate weights as inverse of class frequencies
    weights = 1.0 / class_counts
    
    # Normalize weights
    weights = weights / weights.sum() * len(weights)
    
    return F.cross_entropy(outputs, targets, weight=weights)


def get_class_counts(dataloader):
    """
    Get count of samples for each class in a dataset.
    
    Args:
        dataloader (torch.utils.data.DataLoader): Data loader.
        
    Returns:
        list: List of class counts.
    """
    class_counts = {}
    
    for _, targets in dataloader:
        for target in targets:
            target_item = target.item()
            class_counts[target_item] = class_counts.get(target_item, 0) + 1
    
    # Convert dictionary to list, ensuring proper order
    num_classes = max(class_counts.keys()) + 1
    counts_list = [0] * num_classes
    
    for class_idx, count in class_counts.items():
        counts_list[class_idx] = count
    
    return counts_list


def visualize_model_predictions(model, dataloader, class_names, device, num_images=8):
    """
    Visualize model predictions on a batch of images.
    
    Args:
        model (nn.Module): Trained model.
        dataloader (torch.utils.data.DataLoader): Data loader.
        class_names (list): List of class names.
        device (torch.device): Device to run model on.
        num_images (int): Number of images to visualize.
        
    Returns:
        matplotlib.figure.Figure: Figure with prediction visualizations.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torchvision
    
    # Set model to evaluation mode
    model.eval()
    
    # Get a batch of images
    images, labels = next(iter(dataloader))
    images = images[:num_images].to(device)
    labels = labels[:num_images].cpu().numpy()
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
    
    # Convert predictions to numpy
    preds = preds.cpu().numpy()
    
    # Create figure
    fig, axes = plt.subplots(2, num_images // 2, figsize=(15, 8))
    axes = axes.flatten()
    
    # Compute mean and std for unnormalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Plot images with predictions
    for i, ax in enumerate(axes):
        # Unnormalize image
        img = images[i].cpu().numpy().transpose((1, 2, 0))
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        # Display image
        ax.imshow(img)
        
        # Set title with prediction result
        title_color = "green" if preds[i] == labels[i] else "red"
        ax.set_title(
            f"True: {class_names[labels[i]]}\nPred: {class_names[preds[i]]}",
            color=title_color
        )
        ax.axis('off')
    
    fig.tight_layout()
    return fig