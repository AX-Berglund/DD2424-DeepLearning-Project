#!/usr/bin/env python
"""
Trainer for multiclass semi-supervised learning with pseudo-labeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from src.trainers.multiclass_trainer import MultiClassTrainer
from src.utils.logger import MetricTracker
from src.utils.metrics import compute_accuracy


class MultiClassPseudoLabelingTrainer(MultiClassTrainer):
    """Trainer for multiclass semi-supervised learning with pseudo-labeling."""
    
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
        
        # Initialize loss function
        self.criterion = nn.CrossEntropyLoss()
        
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
            probs = F.softmax(outputs, dim=1)
            max_probs, pseudo_labels = torch.max(probs, dim=1)
            
            # Create mask for confident predictions
            mask = max_probs >= self.confidence_threshold
            
        self.model.train()  # Set back to training mode
        return pseudo_labels, mask
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        # Get labeled and unlabeled data loaders
        labeled_loader = self.dataloaders['train']
        unlabeled_loader = self.dataloaders['unlabeled']
        
        # Create progress bar
        pbar = tqdm(zip(labeled_loader, unlabeled_loader), total=len(labeled_loader))
        
        for (labeled_data, labeled_targets), (unlabeled_data, _) in pbar:
            # Move data to device
            labeled_data = labeled_data.to(self.device)
            labeled_targets = labeled_targets.to(self.device)
            unlabeled_data = unlabeled_data.to(self.device)
            
            # Forward pass for labeled data
            labeled_outputs = self.model(labeled_data)
            labeled_loss = self.criterion(labeled_outputs, labeled_targets)
            
            # Forward pass for unlabeled data
            with torch.no_grad():
                unlabeled_outputs = self.model(unlabeled_data)
                pseudo_labels = torch.argmax(unlabeled_outputs, dim=1)
                confidence = torch.max(torch.softmax(unlabeled_outputs, dim=1), dim=1)[0]
            
            # Create mask for confident predictions
            mask = confidence > self.confidence_threshold
            
            # Compute pseudo-label loss if there are confident predictions
            if mask.any():
                unlabeled_loss = self.criterion(
                    unlabeled_outputs[mask],
                    pseudo_labels[mask]
                )
                
                # Compute weight for pseudo-label loss
                weight = self.get_pseudo_label_weight(epoch)
                
                # Combine losses
                loss = labeled_loss + weight * unlabeled_loss
            else:
                loss = labeled_loss
            
            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update learning rate scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            total_correct += (labeled_outputs.argmax(dim=1) == labeled_targets).sum().item()
            total_samples += labeled_targets.size(0)
            
            # Update progress bar
            pbar.set_description(
                f'Epoch {epoch} | Loss: {loss.item():.4f} | Acc: {total_correct/total_samples:.4f}'
            )
        
        # Compute average metrics
        avg_loss = total_loss / len(labeled_loader)
        avg_acc = total_correct / total_samples
        
        return avg_loss, avg_acc 