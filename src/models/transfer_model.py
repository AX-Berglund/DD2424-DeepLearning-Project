#!/usr/bin/env python
"""
Transfer learning model wrapper.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet50_Weights


class TransferModel(nn.Module):
    """Transfer learning model using a pre-trained backbone."""
    
    def __init__(self, config):
        """
        Initialize the model.
        
        Args:
            config (dict): Model configuration.
        """
        super().__init__()
        
        # Store config
        self.config = config
        
        # Get model configuration
        self.architecture = config['model']['architecture']
        self.pretrained = config['model']['pretrained']
        self.num_classes = config['model']['num_classes']
        self.dropout_rate = config['model']['dropout_rate']
        
        # Load pre-trained model
        if self.architecture == 'resnet18':
            self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if self.pretrained else None)
            self.feature_dim = self.model.fc.in_features
        elif self.architecture == 'resnet50':
            self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if self.pretrained else None)
            self.feature_dim = self.model.fc.in_features
        else:
            raise ValueError(f"Unsupported architecture: {self.architecture}")
        
        # Replace the final layer with a proper classification head
        if self.num_classes == 2:  # Binary classification
            self.model.fc = nn.Sequential(
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.feature_dim, 1)  # Single output for binary classification
            )
        else:  # Multi-class classification
            self.model.fc = nn.Sequential(
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.feature_dim, self.num_classes)
            )
        
        # Initialize weights for the new layers
        self._initialize_weights()
        
        # Set all parameters to require gradients
        for param in self.parameters():
            param.requires_grad = True
    
    def _initialize_weights(self):
        """Initialize the weights of the new layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x)
    
    def get_optimizer_param_groups(self):
        """
        Get parameter groups for the optimizer.
        
        Returns:
            list: List of parameter groups.
        """
        # Get the number of layers to unfreeze
        unfreeze_layers = self.config['training'].get('unfreeze_layers', 1)
        
        # Get all layers
        layers = list(self.model.children())
        
        # Create parameter groups
        param_groups = []
        
        # Add parameters for unfrozen layers
        for i in range(unfreeze_layers):
            layer = layers[-(i + 1)]
            param_groups.append({
                'params': layer.parameters(),
                'lr': self.config['training']['learning_rate']
            })
        
        # Add parameters for the final layer
        param_groups.append({
            'params': self.model.fc.parameters(),
            'lr': self.config['training']['learning_rate']
        })
        
        return param_groups
    
    def _initialize_model(self):
        """Initialize the pre-trained model and adjust the final layer."""
        # Get the model function based on architecture name
        model_fn = getattr(models, self.architecture)
        
        # Load pre-trained model
        self.model = model_fn(pretrained=self.pretrained)
        
        # Save original parameters for logging/visualization
        self.original_params = {name: param.clone().detach() 
                              for name, param in self.model.named_parameters()}
        
        # Modify the final layer based on model architecture
        if self.architecture.startswith('resnet'):
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, self.num_classes)
        elif self.architecture.startswith('densenet'):
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(in_features, self.num_classes)
        elif self.architecture.startswith('vgg'):
            in_features = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(in_features, self.num_classes)
        else:
            raise ValueError(f"Unsupported architecture: {self.architecture}")
        
        # Freeze all layers initially
        self._freeze_all_layers()
    
    def _freeze_all_layers(self):
        """Freeze all layers of the model."""
        for param in self.model.parameters():
            param.requires_grad = False
    
    def unfreeze_last_n_layers(self, n):
        """
        Unfreeze the last n layers of the model.
        
        Args:
            n (int): Number of layers to unfreeze.
        """
        # Helper function to get layers for different architectures
        def get_layers():
            if self.architecture.startswith('resnet'):
                # For ResNet, layer groups are layer1, layer2, layer3, layer4, and fc
                return [
                    self.model.layer1,
                    self.model.layer2,
                    self.model.layer3,
                    self.model.layer4,
                    self.model.fc
                ]
            elif self.architecture.startswith('densenet'):
                # For DenseNet, layer groups are denseblock1-4, transition1-3, and classifier
                return [
                    self.model.features.denseblock1, 
                    self.model.features.transition1,
                    self.model.features.denseblock2,
                    self.model.features.transition2,
                    self.model.features.denseblock3,
                    self.model.features.transition3,
                    self.model.features.denseblock4,
                    self.model.classifier
                ]
            elif self.architecture.startswith('vgg'):
                # For VGG, features (convolutional blocks) and classifier (FC layers)
                # Extract the convolutional blocks (typically 5 blocks)
                feature_layers = []
                current_block = []
                
                for layer in self.model.features:
                    current_block.append(layer)
                    if isinstance(layer, nn.MaxPool2d):
                        feature_layers.append(nn.Sequential(*current_block))
                        current_block = []
                
                # Add classifier components
                classifier_layers = []
                for i in range(0, len(self.model.classifier), 3):
                    if i+2 < len(self.model.classifier):
                        block = nn.Sequential(
                            self.model.classifier[i],
                            self.model.classifier[i+1],
                            self.model.classifier[i+2]
                        )
                        classifier_layers.append(block)
                
                # Final layer
                if len(self.model.classifier) % 3 != 0:
                    classifier_layers.append(
                        nn.Sequential(*[self.model.classifier[i] 
                                      for i in range(len(classifier_layers)*3, len(self.model.classifier))])
                    )
                
                return feature_layers + classifier_layers
        
        # Get model layers
        layers = get_layers()
        
        # Default to unfreeze at least the final layer (FC or classifier)
        n = min(max(1, n), len(layers))
        
        # Unfreeze the last n layers
        for layer in layers[-n:]:
            for param in layer.parameters():
                param.requires_grad = True
    
    def unfreeze_batch_norm(self):
        """Unfreeze all BatchNorm layers while keeping other layers frozen."""
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                for param in module.parameters():
                    param.requires_grad = True
    
    def apply_strategy(self, strategy, epoch=None):
        """
        Apply a fine-tuning strategy.
        
        Args:
            strategy (str): Fine-tuning strategy name.
            epoch (int, optional): Current epoch (for gradual unfreezing).
        """
        if strategy == 'last_layer':
            # Freeze all layers except the final classification layer
            self._freeze_all_layers()
            
            if self.architecture.startswith('resnet'):
                for param in self.model.fc.parameters():
                    param.requires_grad = True
            elif self.architecture.startswith('densenet'):
                for param in self.model.classifier.parameters():
                    param.requires_grad = True
            elif self.architecture.startswith('vgg'):
                for param in self.model.classifier[6].parameters():
                    param.requires_grad = True
        
        elif strategy == 'multi_layer':
            # Unfreeze the last n layers
            self._freeze_all_layers()
            self.unfreeze_last_n_layers(self.unfreeze_layers)
        
        elif strategy == 'gradual_unfreeze' and epoch is not None:
            # Gradually unfreeze layers as training progresses
            gradual_config = self.config['training'].get('gradual_unfreeze', {})
            initial_layers = gradual_config.get('initial_layers', 1)
            unfreeze_every = gradual_config.get('unfreeze_every', 1)
            
            # Calculate how many layers to unfreeze based on epoch
            if epoch == 0:
                layers_to_unfreeze = initial_layers
            else:
                additional_unfreezes = epoch // unfreeze_every
                layers_to_unfreeze = initial_layers + additional_unfreezes
            
            # Apply unfreezing
            self._freeze_all_layers()
            self.unfreeze_last_n_layers(layers_to_unfreeze)
        
        elif strategy == 'batch_norm_only':
            # Only unfreeze batch norm layers and final layer
            self._freeze_all_layers()
            self.unfreeze_batch_norm()
            
            # Also unfreeze the final classification layer
            if self.architecture.startswith('resnet'):
                for param in self.model.fc.parameters():
                    param.requires_grad = True
            elif self.architecture.startswith('densenet'):
                for param in self.model.classifier.parameters():
                    param.requires_grad = True
            elif self.architecture.startswith('vgg'):
                for param in self.model.classifier[6].parameters():
                    param.requires_grad = True
    
    def get_optimizer_param_groups(self):
        """
        Get parameter groups for the optimizer with different learning rates.
        
        Returns:
            list: List of parameter dictionaries with 'params' and 'lr' keys.
        """
        layer_specific_lr = self.config['training'].get('layer_specific_lr', {})
        if not layer_specific_lr.get('enabled', False):
            return self.parameters()
        
        base_lr = layer_specific_lr.get('base_lr', 1e-4)
        new_layers_lr = layer_specific_lr.get('new_layers_lr', 1e-3)
        
        # Group parameters
        pretrained_params = []
        new_params = []
        
        if self.architecture.startswith('resnet'):
            for name, param in self.model.named_parameters():
                if 'fc' in name:
                    new_params.append(param)
                else:
                    pretrained_params.append(param)
        elif self.architecture.startswith('densenet'):
            for name, param in self.model.named_parameters():
                if 'classifier' in name:
                    new_params.append(param)
                else:
                    pretrained_params.append(param)
        elif self.architecture.startswith('vgg'):
            for name, param in self.model.named_parameters():
                if 'classifier.6' in name:
                    new_params.append(param)
                else:
                    pretrained_params.append(param)
        
        return [
            {'params': pretrained_params, 'lr': base_lr},
            {'params': new_params, 'lr': new_layers_lr}
        ]
    
    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)



class MaskedFineTuning(nn.Module):
    """
    Masked fine-tuning implementation that only updates a subset of parameters.
    """
    
    def __init__(self, model, mask_ratio=0.1, seed=42):
        """
        Initialize masked fine-tuning.
        
        Args:
            model (nn.Module): Base model.
            mask_ratio (float): Fraction of parameters to tune.
            seed (int): Random seed.
        """
        super(MaskedFineTuning, self).__init__()
        
        self.model = model
        self.mask_ratio = mask_ratio
        
        # Set seed for reproducibility
        torch.manual_seed(seed)
        
        # Create masks for parameters
        self._create_masks()
    
    def _create_masks(self):
        """Create binary masks for each parameter tensor."""
        self.masks = {}
        
        for name, param in self.model.named_parameters():
            # Exclude parameters that shouldn't be tuned
            if 'bn' in name or 'bias' in name or 'fc' in name or 'classifier' in name:
                # For batch norm, bias, and final layer parameters: allow tuning
                self.masks[name] = torch.ones_like(param, dtype=torch.bool)
            else:
                # For other parameters: randomly select subset to tune
                mask = torch.zeros_like(param, dtype=torch.bool)
                num_params = param.numel()
                num_to_tune = int(num_params * self.mask_ratio)
                
                # Randomly select indices to tune
                indices = torch.randperm(num_params)[:num_to_tune]
                flat_mask = mask.flatten()
                flat_mask[indices] = True
                
                # Reshape back to original shape
                self.masks[name] = flat_mask.view_as(param)
                
                # Freeze parameters not selected for tuning
                param.requires_grad = False
    
    def apply_masks(self):
        """Apply the masks to gradients during training."""
        for name, param in self.model.named_parameters():
            if name in self.masks and param.grad is not None:
                # Zero-out gradients for parameters not selected for tuning
                param.grad.mul_(self.masks[name])
    
    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)


def create_model(config):
    """
    Create a transfer learning model based on the configuration.
    
    Args:
        config (dict): Configuration dictionary.
        
    Returns:
        TransferModel: Transfer learning model.
    """
    base_model = TransferModel(config)
    
    # Check if using LoRA
    if 'lora' in config['model'] and config['model']['lora'].get('enabled', False):
        return LoRAModel(base_model, config['model']['lora'])
    
    # Check if using masked fine-tuning
    if 'masked_finetuning' in config['model'] and config['model']['masked_finetuning'].get('enabled', False):
        masked_ratio = config['model']['masked_finetuning'].get('mask_ratio', 0.1)
        return MaskedFineTuning(base_model, mask_ratio=masked_ratio)
    
    return base_model