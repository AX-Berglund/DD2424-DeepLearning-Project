#!/usr/bin/env python
"""
Implementation of Low-Rank Adaptation (LoRA) for efficient fine-tuning.

LoRA: Low-Rank Adaptation of Large Language Models
Paper: https://arxiv.org/abs/2106.09685

This implementation is adapted for convolutional neural networks.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation layer.
    
    This layer adds low-rank decomposition matrices to adapt the weights
    of pre-trained models with a small number of trainable parameters.
    """
    
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        """
        Initialize LoRA layer.
        
        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            rank (int): Rank of the low-rank matrices (r << min(in_features, out_features)).
            alpha (float): Scaling factor for the LoRA contribution.
        """
        super(LoRALayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Initialize weights
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize the parameters of the LoRA matrices."""
        # Initialize A with Kaiming uniform
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # Initialize B with zeros
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
        """
        Forward pass of the LoRA layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_features].
            
        Returns:
            torch.Tensor: LoRA adaptation of shape [batch_size, out_features].
        """
        # (rank, in_features) @ (batch_size, in_features).T -> (rank, batch_size)
        # (out_features, rank) @ (rank, batch_size) -> (out_features, batch_size)
        # (out_features, batch_size).T -> (batch_size, out_features)
        return (self.lora_B @ self.lora_A @ x.T).T * self.scaling


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation.
    
    This module combines a regular linear layer with a LoRA adaptation.
    """
    
    def __init__(self, original_layer, rank=4, alpha=1.0, trainable_original=False):
        """
        Initialize the LoRA-adapted linear layer.
        
        Args:
            original_layer (nn.Linear): Original linear layer to adapt.
            rank (int): Rank of the low-rank matrices.
            alpha (float): Scaling factor for the LoRA contribution.
            trainable_original (bool): Whether to train the original weights as well.
        """
        super(LoRALinear, self).__init__()
        
        self.original_layer = original_layer
        
        # Freeze the original weights if not trainable
        if not trainable_original:
            self.original_layer.weight.requires_grad = False
            if self.original_layer.bias is not None:
                self.original_layer.bias.requires_grad = False
        
        # Add LoRA adaptation
        self.lora = LoRALayer(
            original_layer.in_features,
            original_layer.out_features,
            rank=rank,
            alpha=alpha
        )
    
    def forward(self, x):
        """
        Forward pass with LoRA adaptation.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output with LoRA adaptation.
        """
        # Regular forward pass
        original_output = self.original_layer(x)
        # LoRA adaptation
        lora_output = self.lora(x)
        # Combine outputs
        return original_output + lora_output


class LoRAConv2d(nn.Module):
    """
    Convolutional layer with LoRA adaptation.
    
    This module adapts a 2D convolutional layer with LoRA.
    """
    
    def __init__(self, original_layer, rank=4, alpha=1.0, trainable_original=False):
        """
        Initialize the LoRA-adapted convolutional layer.
        
        Args:
            original_layer (nn.Conv2d): Original convolutional layer to adapt.
            rank (int): Rank of the low-rank matrices.
            alpha (float): Scaling factor for the LoRA contribution.
            trainable_original (bool): Whether to train the original weights as well.
        """
        super(LoRAConv2d, self).__init__()
        
        self.original_layer = original_layer
        
        # Freeze the original weights if not trainable
        if not trainable_original:
            self.original_layer.weight.requires_grad = False
            if self.original_layer.bias is not None:
                self.original_layer.bias.requires_grad = False
        
        # Calculate in_features and out_features
        in_features = original_layer.in_channels * original_layer.kernel_size[0] * original_layer.kernel_size[1]
        out_features = original_layer.out_channels
        
        # Add LoRA adaptation
        self.lora = LoRALayer(
            in_features,
            out_features,
            rank=rank,
            alpha=alpha
        )
        
        # Store original layer parameters
        self.in_channels = original_layer.in_channels
        self.out_channels = original_layer.out_channels
        self.kernel_size = original_layer.kernel_size
        self.stride = original_layer.stride
        self.padding = original_layer.padding
        self.dilation = original_layer.dilation
        self.groups = original_layer.groups
    
    def forward(self, x):
        """
        Forward pass with LoRA adaptation.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_channels, height, width].
            
        Returns:
            torch.Tensor: Output with LoRA adaptation.
        """
        # Regular forward pass
        original_output = self.original_layer(x)
        
        # LoRA adaptation
        batch_size, in_channels, height, width = x.shape
        
        # Reshape input for LoRA
        # [batch_size, in_channels, height, width] -> [batch_size * height_out * width_out, in_channels * kernel_h * kernel_w]
        # where height_out, width_out are the spatial dimensions after applying convolution
        unfold = nn.Unfold(
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride
        )
        x_unfolded = unfold(x)
        
        # Calculate output spatial dimensions
        height_out = (height + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        width_out = (width + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
        
        # Reshape unfolded input
        x_unfolded = x_unfolded.transpose(1, 2).reshape(-1, in_channels * self.kernel_size[0] * self.kernel_size[1])
        
        # Apply LoRA
        lora_output = self.lora(x_unfolded)
        
        # Reshape LoRA output
        lora_output = lora_output.reshape(batch_size, height_out, width_out, self.out_channels).permute(0, 3, 1, 2)
        
        # Combine outputs
        return original_output + lora_output


def apply_lora_to_model(model, rank=4, alpha=1.0, target_modules=None):
    """
    Apply LoRA to specific layers of a model.
    
    Args:
        model (nn.Module): Model to adapt.
        rank (int): Rank of the low-rank matrices.
        alpha (float): Scaling factor for the LoRA contribution.
        target_modules (list): List of module names to adapt. If None, adapt all linear and conv2d layers.
        
    Returns:
        nn.Module: Model with LoRA adaptation.
    """
    for name, module in list(model.named_children()):
        # Skip if this module shouldn't be adapted
        if target_modules is not None and name not in target_modules:
            continue
        
        # Adapt the module based on its type
        if isinstance(module, nn.Linear):
            setattr(model, name, LoRALinear(module, rank=rank, alpha=alpha))
        elif isinstance(module, nn.Conv2d):
            setattr(model, name, LoRAConv2d(module, rank=rank, alpha=alpha))
        else:
            # Recursively apply LoRA to child modules
            apply_lora_to_model(module, rank=rank, alpha=alpha, target_modules=target_modules)
    
    return model


def count_lora_parameters(model):
    """
    Count the number of trainable parameters in LoRA layers.
    
    Args:
        model (nn.Module): Model with LoRA adaptation.
        
    Returns:
        tuple: (lora_params, total_params) - Number of LoRA parameters and total parameters.
    """
    lora_params = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'lora_A' in name or 'lora_B' in name:
                lora_params += param.numel()
            total_params += param.numel()
    
    return lora_params, total_params


class LoRAWrapper(nn.Module):
    """
    Wrapper for a model with LoRA adaptation.
    
    This class wraps a model and applies LoRA adaptation to specific layers.
    """
    
    def __init__(self, base_model, config):
        """
        Initialize the LoRA-adapted model.
        
        Args:
            base_model (nn.Module): Base model to adapt.
            config (dict): LoRA configuration parameters.
        """
        super(LoRAWrapper, self).__init__()
        
        self.base_model = base_model
        self.rank = config.get('rank', 4)
        self.alpha = config.get('alpha', 1.0)
        self.target_modules = config.get('target_modules', None)
        
        # Apply LoRA to the model
        apply_lora_to_model(
            self.base_model,
            rank=self.rank,
            alpha=self.alpha,
            target_modules=self.target_modules
        )
        
        # Count parameters
        lora_params, total_params = count_lora_parameters(self.base_model)
        print(f"LoRA parameters: {lora_params:,} / {total_params:,} ({100.0 * lora_params / total_params:.2f}%)")
    
    def forward(self, x):
        """
        Forward pass through the LoRA-adapted model.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Model output.
        """
        return self.base_model(x)


def create_lora_model(base_model, config):
    """
    Create a LoRA-adapted model.
    
    Args:
        base_model (nn.Module): Base model to adapt.
        config (dict): LoRA configuration parameters.
        
    Returns:
        LoRAWrapper: LoRA-adapted model.
    """
    return LoRAWrapper(base_model, config)