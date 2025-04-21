#!/usr/bin/env python
"""
Visualization utilities for model analysis and debugging.
"""

import io
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def plot_to_image(figure):
    """
    Convert a matplotlib figure to an image tensor.
    
    Args:
        figure (matplotlib.figure.Figure): Figure to convert.
        
    Returns:
        torch.Tensor: Image tensor.
    """
    # Save the plot to a buffer
    buf = io.BytesIO()
    figure.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    
    # Load the image and convert to tensor
    image = Image.open(buf)
    image = torchvision.transforms.ToTensor()(image)
    
    return image


def plot_training_metrics(metrics, save_path=None):
    """
    Plot training and validation metrics.
    
    Args:
        metrics (dict): Dictionary of metrics.
        save_path (str, optional): Path to save the figure.
        
    Returns:
        matplotlib.figure.Figure: Figure with plots.
    """
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot training and validation loss
    epochs = range(1, len(metrics['train']['loss']) + 1)
    ax1.plot(epochs, metrics['train']['loss'], 'b-', label='Training loss')
    ax1.plot(epochs, metrics['val']['loss'], 'r-', label='Validation loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot training and validation accuracy
    ax2.plot(epochs, metrics['train']['acc'], 'b-', label='Training accuracy')
    ax2.plot(epochs, metrics['val']['acc'], 'r-', label='Validation accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    fig.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        fig.savefig(save_path)
    
    return fig


def visualize_model_activations(model, images, device, layer_name=None):
    """
    Visualize activations of a specific layer in the model.
    
    Args:
        model (nn.Module): Model to analyze.
        images (torch.Tensor): Input images.
        device (torch.device): Device to run model on.
        layer_name (str, optional): Name of the layer to visualize.
        
    Returns:
        matplotlib.figure.Figure: Figure with activation visualizations.
    """
    # Set model to evaluation mode
    model.eval()
    
    # Register hook to capture activations
    activations = {}
    
    def hook_fn(module, input, output):
        activations['features'] = output.detach()
    
    # If layer_name is provided, attach hook to that layer
    if layer_name:
        # Find the layer by name
        for name, module in model.named_modules():
            if name == layer_name:
                module.register_forward_hook(hook_fn)
                break
    else:
        # Otherwise, attach hook to the layer before the final classifier
        if hasattr(model, 'model'):
            # For wrapped models like TransferModel
            if hasattr(model.model, 'fc'):
                # ResNet
                model.model.layer4.register_forward_hook(hook_fn)
            elif hasattr(model.model, 'classifier'):
                # DenseNet or VGG
                if hasattr(model.model, 'features'):
                    model.model.features.register_forward_hook(hook_fn)
        else:
            # Direct model
            if hasattr(model, 'fc'):
                # ResNet
                model.layer4.register_forward_hook(hook_fn)
            elif hasattr(model, 'classifier'):
                # DenseNet or VGG
                if hasattr(model, 'features'):
                    model.features.register_forward_hook(hook_fn)
    
    # Forward pass to get activations
    with torch.no_grad():
        _ = model(images.to(device))
    
    # Check if activations were captured
    if 'features' not in activations:
        print("No activations captured. Check layer_name or model architecture.")
        return None
    
    # Get activations
    features = activations['features']
    
    # Convert tensor to numpy (move to CPU first if needed)
    if features.is_cuda:
        features = features.cpu()
    
    # Reshape features for visualization
    if len(features.shape) == 4:  # Conv layer output
        # Take first image's activations
        features = features[0]
        # Reorder dimensions for plotting
        features = features.permute(1, 2, 0)
        # If there are too many channels, only show a subset
        if features.shape[2] > 64:
            features = features[:, :, :64]
    
    # Create figure
    if len(features.shape) == 3:  # Conv layer features
        # Number of filters to visualize
        num_filters = min(64, features.shape[2])
        # Create grid dimensions
        grid_size = int(np.ceil(np.sqrt(num_filters)))
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        axes = axes.flatten()
        
        for i in range(num_filters):
            if i < len(axes):
                # Plot feature map
                im = axes[i].imshow(features[:, :, i], cmap='viridis')
                axes[i].set_title(f'Filter {i}')
                axes[i].axis('off')
        
        # Turn off any unused axes
        for i in range(num_filters, len(axes)):
            axes[i].axis('off')
        
        fig.colorbar(im, ax=axes.tolist(), shrink=0.8)
    else:
        # For FC layer features, create a heatmap or bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(features.unsqueeze(0), cmap='viridis', aspect='auto')
        ax.set_title(f'Activations ({features.shape[0]} units)')
        ax.set_xlabel('Neuron index')
        fig.colorbar(im, ax=ax)
    
    fig.tight_layout()
    return fig


def visualize_feature_space(model, dataloader, device, class_names=None, num_samples=1000, method='tsne'):
    """
    Visualize the feature space of the model using dimensionality reduction.
    
    Args:
        model (nn.Module): Model to analyze.
        dataloader (torch.utils.data.DataLoader): Data loader.
        device (torch.device): Device to run model on.
        class_names (list, optional): List of class names.
        num_samples (int): Maximum number of samples to visualize.
        method (str): Dimensionality reduction method ('tsne' or 'pca').
        
    Returns:
        matplotlib.figure.Figure: Figure with feature space visualization.
    """
    # Set model to evaluation mode
    model.eval()
    
    # Register hook to capture features before the classifier
    features = []
    labels = []
    
    def hook_fn(module, input, output):
        # For ResNet, input[0] contains the features before the final linear layer
        features.append(input[0].detach().cpu())
    
    # Attach hook to the final layer
    if hasattr(model, 'model'):
        # For wrapped models like TransferModel
        if hasattr(model.model, 'fc'):
            # ResNet
            model.model.fc.register_forward_hook(hook_fn)
        elif hasattr(model.model, 'classifier'):
            # DenseNet or VGG
            if isinstance(model.model.classifier, torch.nn.Sequential):
                model.model.classifier[0].register_forward_hook(hook_fn)
            else:
                model.model.classifier.register_forward_hook(hook_fn)
    else:
        # Direct model
        if hasattr(model, 'fc'):
            # ResNet
            model.fc.register_forward_hook(hook_fn)
        elif hasattr(model, 'classifier'):
            # DenseNet or VGG
            if isinstance(model.classifier, torch.nn.Sequential):
                model.classifier[0].register_forward_hook(hook_fn)
            else:
                model.classifier.register_forward_hook(hook_fn)
    
    # Collect features and labels
    num_collected = 0
    with torch.no_grad():
        for images, targets in dataloader:
            # Forward pass
            _ = model(images.to(device))
            # Collect labels
            labels.append(targets.cpu())
            
            # Count collected samples
            num_collected += len(targets)
            if num_collected >= num_samples:
                break
    
    # Concatenate features and labels
    features = torch.cat(features, dim=0)[:num_samples]
    labels = torch.cat(labels, dim=0)[:num_samples]
    
    # Convert to numpy
    features = features.numpy()
    labels = labels.numpy()
    
    # Apply dimensionality reduction
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:  # 'pca'
        reducer = PCA(n_components=2, random_state=42)
    
    # Reduce dimensionality
    features_2d = reducer.fit_transform(features)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get unique labels
    unique_labels = np.unique(labels)
    
    # Create scatter plot for each class
    for i in unique_labels:
        indices = labels == i
        ax.scatter(
            features_2d[indices, 0],
            features_2d[indices, 1],
            label=class_names[i] if class_names else f'Class {i}',
            alpha=0.6
        )
    
    ax.set_title(f'Feature Space Visualization using {method.upper()}')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.legend()
    
    fig.tight_layout()
    return fig


def visualize_weight_distribution(model, layer_name=None):
    """
    Visualize the distribution of weights in the model or a specific layer.
    
    Args:
        model (nn.Module): Model to analyze.
        layer_name (str, optional): Name of the layer to visualize. If None, visualize all layers.
        
    Returns:
        matplotlib.figure.Figure: Figure with weight distribution visualization.
    """
    # Get all parameters or specific layer parameters
    params = {}
    
    if layer_name:
        # Get parameters of specific layer
        for name, param in model.named_parameters():
            if layer_name in name:
                params[name] = param.detach().cpu().numpy().flatten()
    else:
        # Get all parameters
        for name, param in model.named_parameters():
            if 'weight' in name:  # Only include weights, not biases
                params[name] = param.detach().cpu().numpy().flatten()
    
    # Create figure
    n_params = len(params)
    fig, axes = plt.subplots(n_params, 1, figsize=(10, 3 * n_params))
    
    # Handle single parameter case
    if n_params == 1:
        axes = [axes]
    
    # Plot histogram for each parameter
    for (name, weights), ax in zip(params.items(), axes):
        ax.hist(weights, bins=50, alpha=0.7)
        ax.set_title(f'Weight Distribution: {name}')
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Frequency')
        
        # Add statistics
        mean = np.mean(weights)
        std = np.std(weights)
        ax.axvline(mean, color='r', linestyle='dashed', linewidth=1, label=f'Mean: {mean:.4f}')
        ax.axvline(mean + std, color='g', linestyle='dashed', linewidth=1, label=f'Mean + Std: {mean + std:.4f}')
        ax.axvline(mean - std, color='g', linestyle='dashed', linewidth=1, label=f'Mean - Std: {mean - std:.4f}')
        ax.legend()
    
    fig.tight_layout()
    return fig


def visualize_learning_rates(optimizer):
    """
    Visualize the learning rates for different parameter groups.
    
    Args:
        optimizer (torch.optim.Optimizer): Optimizer to visualize.
        
    Returns:
        matplotlib.figure.Figure: Figure with learning rate visualization.
    """
    # Get learning rates from parameter groups
    groups = optimizer.param_groups
    group_names = [f"Group {i}" for i in range(len(groups))]
    learning_rates = [group['lr'] for group in groups]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Create bar plot
    ax.bar(group_names, learning_rates)
    ax.set_title('Learning Rates per Parameter Group')
    ax.set_xlabel('Parameter Group')
    ax.set_ylabel('Learning Rate')
    
    # Add learning rate values on top of bars
    for i, lr in enumerate(learning_rates):
        ax.text(i, lr + 0.0001, f'{lr:.6f}', ha='center')
    
    fig.tight_layout()
    return fig


def visualize_class_distribution(dataloader, class_names=None):
    """
    Visualize the distribution of classes in a dataset.
    
    Args:
        dataloader (torch.utils.data.DataLoader): Data loader.
        class_names (list, optional): List of class names.
        
    Returns:
        matplotlib.figure.Figure: Figure with class distribution visualization.
    """
    # Count classes
    class_counts = {}
    for _, targets in dataloader:
        for target in targets:
            target_item = target.item()
            class_counts[target_item] = class_counts.get(target_item, 0) + 1
    
    # Sort by class index
    classes = sorted(class_counts.keys())
    counts = [class_counts[cls] for cls in classes]
    
    # Use class names if provided
    if class_names:
        labels = [class_names[cls] for cls in classes]
    else:
        labels = [f'Class {cls}' for cls in classes]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create bar plot
    ax.bar(labels, counts)
    ax.set_title('Class Distribution')
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    
    # Rotate x-axis labels if there are many classes
    if len(labels) > 10:
        plt.xticks(rotation=45, ha='right')
    
    # Add count values on top of bars
    for i, count in enumerate(counts):
        ax.text(i, count + 0.5, str(count), ha='center')
    
    fig.tight_layout()
    return fig


def visualize_batch(images, targets, class_names=None, normalize=True):
    """
    Visualize a batch of images with their class labels.
    
    Args:
        images (torch.Tensor): Batch of images.
        targets (torch.Tensor): Batch of targets.
        class_names (list, optional): List of class names.
        normalize (bool): Whether images are normalized.
        
    Returns:
        matplotlib.figure.Figure: Figure with batch visualization.
    """
    # Convert targets to numpy
    targets = targets.cpu().numpy()
    
    # Get batch size
    batch_size = images.size(0)
    
    # Create grid of images
    grid = torchvision.utils.make_grid(images, nrow=int(np.sqrt(batch_size)))
    
    # Convert grid to numpy
    grid = grid.cpu().numpy().transpose((1, 2, 0))
    
    # Unnormalize if needed
    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        grid = std * grid + mean
        grid = np.clip(grid, 0, 1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Display grid
    ax.imshow(grid)
    ax.axis('off')
    
    # Set title
    if class_names:
        class_labels = [class_names[t] for t in targets]
        ax.set_title(f'Batch of {batch_size} images\nClasses: {", ".join(class_labels[:10])}' + 
                     ('...' if len(class_labels) > 10 else ''))
    else:
        ax.set_title(f'Batch of {batch_size} images')
    
    fig.tight_layout()
    return fig


def visualize_grad_flow(model):
    """
    Visualize the gradient flow through the layers of the model.
    
    Args:
        model (nn.Module): Model after backward pass.
        
    Returns:
        matplotlib.figure.Figure: Figure with gradient flow visualization.
    """
    # Get average gradient for each layer
    named_parameters = list(filter(
        lambda p: p[1].grad is not None and p[0].find('.bias') == -1,
        model.named_parameters()
    ))
    
    layers = [n.replace('module.', '') for n, p in named_parameters]
    ave_grads = [p.grad.abs().mean().item() for n, p in named_parameters]
    max_grads = [p.grad.abs().max().item() for n, p in named_parameters]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot average gradients
    ax.bar(np.arange(len(ave_grads)), ave_grads, alpha=0.3, lw=1, color="b")
    # Plot max gradients as a second series with transparency
    ax.bar(np.arange(len(max_grads)), max_grads, alpha=0.3, lw=1, color="r")
    
    # Set labels
    ax.set_title("Gradient Flow")
    ax.set_xlabel("Layers")
    ax.set_ylabel("Average Gradient")
    
    # Configure tick labels
    ax.set_xticks(np.arange(len(layers)))
    ax.set_xticklabels(layers, rotation=90)
    
    # Add legend
    ax.legend(['Mean Gradient', 'Max Gradient'])
    
    fig.tight_layout()
    return fig


def visualize_parameter_changes(model, original_params):
    """
    Visualize how much parameters have changed during training.
    
    Args:
        model (nn.Module): Model after training.
        original_params (dict): Original parameters before training.
        
    Returns:
        matplotlib.figure.Figure: Figure with parameter changes visualization.
    """
    # Calculate parameter changes
    changes = {}
    
    for name, param in model.named_parameters():
        if name in original_params:
            original = original_params[name].to(param.device)
            # Calculate relative change
            relative_change = torch.norm(param - original) / torch.norm(original)
            changes[name] = relative_change.item()
    
    # Sort layers by change magnitude
    sorted_changes = sorted(changes.items(), key=lambda x: x[1], reverse=True)
    layers = [name for name, _ in sorted_changes]
    magnitudes = [change for _, change in sorted_changes]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot parameter changes
    ax.bar(np.arange(len(magnitudes)), magnitudes, alpha=0.7)
    
    # Set labels
    ax.set_title("Parameter Changes During Training")
    ax.set_xlabel("Layers")
    ax.set_ylabel("Relative Change (norm)")
    
    # Configure tick labels
    ax.set_xticks(np.arange(len(layers)))
    ax.set_xticklabels(layers, rotation=90)
    
    fig.tight_layout()
    return fig