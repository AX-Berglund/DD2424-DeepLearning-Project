#!/usr/bin/env python
"""
Trainer for binary semi-supervised learning with pseudo-labeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from src.trainers.binary_trainer import BinaryTrainer
from src.utils.logger import MetricTracker
from src.utils.metrics import compute_accuracy, compute_metrics


class BinaryPseudoLabelingTrainer(BinaryTrainer):
    """Trainer for binary semi-supervised learning with pseudo-labeling."""
    
    def __init__(self, model, dataloaders, config, logger):
        """
        Initialize the trainer.
        
        Args:
            model (nn.Module): Model to train.
            dataloaders (dict): Dictionary of data loaders for each split.
            config (dict): Configuration dictionary.
            logger (Logger): Logger instance.
        """
        super().__init__(model, dataloaders, config, logger)
        
        # Get pseudo-labeling settings
        self.confidence_threshold = config['training']['pseudo_labeling']['confidence_threshold']
        self.rampup_epochs = config['training']['pseudo_labeling']['rampup_epochs']
        self.alpha = config['training']['pseudo_labeling']['alpha']
        
        # Get unlabeled data loader if it exists
        self.unlabeled_loader = dataloaders.get('unlabeled')
        self.has_unlabeled_data = self.unlabeled_loader is not None
        
        if not self.has_unlabeled_data:
            logger.info("No unlabeled data found. Running in supervised mode.")
        
        # Initialize metrics for pseudo-labeling
        self.pseudo_label_metrics = {
            'pseudo_label_accuracy': [],
            'pseudo_label_confidence': [],
            'pseudo_label_count': []
        }
        
        # Apply initial fine-tuning strategy
        if hasattr(self.model, 'apply_strategy'):
            strategy = self.config['training']['strategy']
            self.model.apply_strategy(strategy, epoch=0)
    
    def get_pseudo_label_weight(self, epoch):
        """
        Get the weight for pseudo-labeled loss based on the current epoch.
        
        Args:
            epoch (int): Current epoch.
            
        Returns:
            float: Weight for pseudo-labeled loss.
        """
        if not self.has_unlabeled_data:
            return 0.0
            
        if epoch < self.rampup_epochs:
            return self.alpha * (epoch / self.rampup_epochs)
        return self.alpha
    
    def generate_pseudo_labels(self, unlabeled_data):
        """
        Generate pseudo-labels for unlabeled data.
        
        Args:
            unlabeled_data (torch.Tensor): Unlabeled data batch.
            
        Returns:
            tuple: (pseudo_labels, mask) where mask indicates which samples to use.
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(unlabeled_data)
            probs = torch.sigmoid(outputs)
            pseudo_labels = (probs >= 0.5).float()
            
            # Create mask for confident predictions
            confidence = torch.abs(probs - 0.5) * 2  # Scale to [0, 1]
            mask = confidence >= self.confidence_threshold
            
        self.model.train()  # Set back to training mode
        return pseudo_labels, mask
    
    def train_epoch(self, epoch):
        """
        Train for one epoch.
        
        Args:
            epoch (int): Current epoch.
            
        Returns:
            tuple: Tuple of (epoch_loss, epoch_accuracy).
        """
        # Set model to training mode
        self.model.train()
        
        # Initialize metrics tracker
        tracker = MetricTracker()
        
        # Get data loaders
        labeled_loader = self.dataloaders['train']
        unlabeled_loader = self.dataloaders.get('unlabeled')
        
        # Initialize unlabeled data iterator if we have unlabeled data
        if unlabeled_loader is not None:
            unlabeled_iter = iter(unlabeled_loader)
        
        # Training loop
        for batch_idx, (labeled_data, labeled_targets) in enumerate(labeled_loader):
            # Move data to device
            labeled_data = labeled_data.to(self.device)
            labeled_targets = labeled_targets.to(self.device).float().unsqueeze(1)  # Add channel dimension
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass for labeled data
            labeled_outputs = self.model(labeled_data)
            
            # Compute loss for labeled data
            labeled_loss = F.binary_cross_entropy_with_logits(
                labeled_outputs, labeled_targets
            )
            
            total_loss = labeled_loss
            
            # If we have unlabeled data and past warmup, use pseudo-labeling
            if unlabeled_loader is not None and epoch >= self.config['training']['pseudo_labeling']['warmup_epochs']:
                try:
                    unlabeled_data, _ = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(unlabeled_loader)
                    unlabeled_data, _ = next(unlabeled_iter)
                
                unlabeled_data = unlabeled_data.to(self.device)
                
                # Get pseudo-labels
                unlabeled_outputs = self.model(unlabeled_data)
                pseudo_probs = torch.sigmoid(unlabeled_outputs)
                
                # Create confidence mask
                confidence = torch.abs(pseudo_probs - 0.5) * 2  # Scale to [0, 1]
                confidence_mask = confidence >= self.confidence_threshold
                
                if confidence_mask.any():
                    # Get pseudo-labels for confident predictions
                    pseudo_labels = (pseudo_probs > 0.5).float()
                    
                    # Compute loss for pseudo-labeled data
                    pseudo_loss = F.binary_cross_entropy_with_logits(
                        unlabeled_outputs[confidence_mask],
                        pseudo_labels[confidence_mask]
                    )
                    
                    # Get weight for pseudo-labeled loss
                    alpha = self.get_pseudo_label_weight(epoch)
                    
                    # Combine losses
                    total_loss = labeled_loss + alpha * pseudo_loss
                    
                    # Log pseudo-labeling metrics
                    self.pseudo_label_metrics['pseudo_label_count'].append(confidence_mask.sum().item())
                    self.pseudo_label_metrics['pseudo_label_confidence'].append(confidence[confidence_mask].mean().item())
            
            # Backward pass
            total_loss.backward()
            
            # Update weights
            self.optimizer.step()
            
            # Update metrics with total loss
            tracker.update(total_loss.item(), labeled_outputs, labeled_targets)
            
            # Log batch progress
            self.logger.log_batch(
                epoch=epoch,
                batch_idx=batch_idx,
                n_batches=len(labeled_loader),
                loss=total_loss.item(),
                acc=compute_accuracy(labeled_outputs, labeled_targets, binary=True)
            )
        
        # Get epoch metrics
        metrics = tracker.get_metrics()
        
        # Add pseudo-labeling metrics if we have them
        if self.pseudo_label_metrics['pseudo_label_count']:
            metrics.update({
                'pseudo_label_count': np.mean(self.pseudo_label_metrics['pseudo_label_count']),
                'pseudo_label_confidence': np.mean(self.pseudo_label_metrics['pseudo_label_confidence'])
            })
        
        return metrics['loss'], metrics['accuracy']

    def validate_epoch(self, epoch):
        """
        Validate for one epoch.
        
        Args:
            epoch (int): Current epoch.
            
        Returns:
            tuple: Tuple of (epoch_loss, epoch_accuracy).
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize metrics tracker
        tracker = MetricTracker()
        
        # Get data loader
        val_loader = self.dataloaders['val']
        
        # Storage for detailed metrics
        all_outputs = []
        all_targets = []
        
        # Validation loop
        with torch.no_grad():
            for data, target in val_loader:
                # Move data to device
                data = data.to(self.device)
                target = target.to(self.device).float().unsqueeze(1)  # Add channel dimension
                
                # Forward pass
                output = self.model(data)
                
                # Compute loss
                loss = F.binary_cross_entropy_with_logits(output, target)
                
                # Update metrics
                tracker.update(loss.item(), output, target)
                
                # Collect outputs and targets for detailed metrics
                all_outputs.append(output)
                all_targets.append(target)
        
        # Concatenate all outputs and targets
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Get epoch metrics
        metrics = tracker.get_metrics()
        
        # Log detailed metrics
        detailed_metrics = compute_metrics(
            outputs=all_outputs,
            targets=all_targets,
            class_names=self.class_names,
            binary=True
        )
        
        self.logger.log_metrics(epoch, detailed_metrics, split="val")
        
        return metrics['loss'], metrics['accuracy'] 