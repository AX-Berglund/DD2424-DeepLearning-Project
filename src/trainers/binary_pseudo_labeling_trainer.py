#!/usr/bin/env python
"""
Trainer for binary semi-supervised learning with pseudo-labeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Dict

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
        
        # Get data loaders
        self.train_loader = dataloaders['train']
        self.val_loader = dataloaders['val']
        self.test_loader = dataloaders['test']
        
        # Get training settings
        self.batch_size = config['data']['batch_size']
        self.num_epochs = config['training']['num_epochs']
        self.learning_rate = config['training']['learning_rate']
        self.early_stopping_patience = config['training']['early_stopping']['patience']
        self.log_interval = config['logging']['log_interval']
        
        # Get pseudo-labeling settings
        self.confidence_threshold = config['training']['pseudo_labeling']['confidence_threshold']
        self.rampup_epochs = config['training']['pseudo_labeling']['rampup_epochs']
        self.warmup_epochs = config['training']['pseudo_labeling']['warmup_epochs']
        self.alpha = config['training']['pseudo_labeling']['alpha']
        
        # Get unlabeled data loader if it exists
        self.unlabeled_loader = dataloaders.get('unlabeled')
        self.has_unlabeled_data = self.unlabeled_loader is not None
        
        # Get fast mode setting
        self.fast_mode = config.get('training', {}).get('fast_mode', False)
        
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
        Calculate the weight applied to the pseudo-labeled loss term during training.
        This implements a linear ramp-up schedule for the pseudo-label weight.
        
        During the ramp-up period (first rampup_epochs), the weight increases linearly 
        from 0 to alpha. After ramp-up, it stays constant at alpha. If there is no 
        unlabeled data, the weight is always 0.
        
        Args:
            epoch (int): Current training epoch.
            
        Returns:
            float: Weight for pseudo-labeled loss:
                  - 0.0 if no unlabeled data exists
                  - Linear ramp from 0 to alpha during first rampup_epochs 
                  - Constant alpha after rampup_epochs
        """
        # If no unlabeled data available, pseudo-label loss weight is 0
        if not self.has_unlabeled_data:
            return 0.0
            
        # During ramp-up period, linearly increase weight from 0 to alpha
        if epoch < self.rampup_epochs:
            return self.alpha * (epoch / self.rampup_epochs)
            
        # After ramp-up, use constant alpha weight
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
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        metrics = MetricTracker()
        
        # Reset unlabeled data iterator at start of epoch
        if self.unlabeled_loader is not None:
            self.unlabeled_iter = iter(self.unlabeled_loader)
        
        # Pre-allocate tensors for pseudo-labeling metrics if unlabeled data is available
        if self.unlabeled_loader is not None:
            total_samples = len(self.unlabeled_loader.dataset)
            num_confident = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # Move data to device
            data, target = data.to(self.device), target.to(self.device)
            
            # Add channel dimension to target for binary cross entropy
            target = target.float().unsqueeze(1)
            
            # Zero gradients efficiently
            self.optimizer.zero_grad(set_to_none=True)
            
            # Forward pass for labeled data
            output = self.model(data)
            loss = self.criterion(output, target)
            self.logger.detailed(f"Batch {batch_idx}: Labeled loss: {loss.item():.6f}")
            # Handle unlabeled data if available
            if self.unlabeled_loader is not None:
                try:
                    unlabeled_data, _ = next(self.unlabeled_iter)
                except StopIteration:
                    self.unlabeled_iter = iter(self.unlabeled_iter)
                    unlabeled_data, _ = next(self.unlabeled_iter)
                
                unlabeled_data = unlabeled_data.to(self.device)
                
                # Forward pass for unlabeled data
                with torch.no_grad():
                    unlabeled_output = self.model(unlabeled_data)
                    pseudo_labels, mask = self.generate_pseudo_labels(unlabeled_data)
                
                # Update pseudo-labeling metrics
                num_confident += mask.sum().item()
                
                # Compute loss for pseudo-labeled data
                if mask.any():
                    pseudo_loss = self.criterion(unlabeled_output, pseudo_labels)
                    loss = loss + self.alpha * pseudo_loss
                    
                    self.logger.detailed(f"Batch {batch_idx}: Pseudo-labeling stats - "
                                      f"Confident samples: {mask.sum().item()}/{len(mask)} "
                                      f"(Confidence threshold: {self.confidence_threshold})")
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            metrics.update(loss.item(), output, target)
            
            # Log batch progress
            if batch_idx % self.log_interval == 0:
                self.logger.detailed(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(self.train_loader.dataset)} "
                                  f"({100. * batch_idx / len(self.train_loader):.0f}%)]\t"
                                  f"Loss: {loss.item():.6f}")
        
        # Log pseudo-labeling statistics for the epoch
        if self.unlabeled_loader is not None:
            self.logger.detailed(f"Epoch {epoch} Pseudo-labeling Summary:")
            self.logger.detailed(f"Found {num_confident}/{total_samples} confident predictions "
                              f"({100. * num_confident / total_samples:.2f}%)")
        
        return metrics.get_metrics()

    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        metrics = MetricTracker()
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.val_loader):
                data, target = data.to(self.device), target.to(self.device)
                target = target.float().unsqueeze(1)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                metrics.update(loss.item(), output, target)
                
                if batch_idx % self.log_interval == 0:
                    self.logger.detailed(f"Validation Epoch: {epoch} "
                                      f"[{batch_idx * len(data)}/{len(self.val_loader.dataset)} "
                                      f"({100. * batch_idx / len(self.val_loader):.0f}%)]\t"
                                      f"Loss: {loss.item():.6f}")
        
        # Skip detailed metrics in fast mode
        if not self.fast_mode:
            # Compute additional metrics
            all_outputs = []
            all_targets = []
            
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                target = target.float().unsqueeze(1)
                output = self.model(data)
                all_outputs.append(output)
                all_targets.append(target)
            
            all_outputs = torch.cat(all_outputs)
            all_targets = torch.cat(all_targets)
            
            # Compute and log detailed metrics
            detailed_metrics = compute_metrics(all_outputs, all_targets)
            for metric_name, metric_value in detailed_metrics.items():
                metrics.update(metric_name, metric_value)
                self.logger.detailed(f"Validation {metric_name}: {metric_value:.4f}")
        
        return metrics.get_metrics()

    def train(self) -> Dict[str, float]:
        """Train the model."""
        best_val_loss = float('inf')
        best_val_acc = 0.0
        patience = self.config['training']['early_stopping']['patience']
        patience_counter = 0
        
        self.logger.detailed("Starting training...")
        self.logger.detailed(f"Training configuration:")
        self.logger.detailed(f"- Device: {self.device}")
        self.logger.detailed(f"- Batch size: {self.batch_size}")
        self.logger.detailed(f"- Learning rate: {self.learning_rate}")
        self.logger.detailed(f"- Number of epochs: {self.num_epochs}")
        self.logger.detailed(f"- Early stopping patience: {self.early_stopping_patience}")
        
        if self.unlabeled_loader is not None:
            self.logger.detailed(f"Pseudo-labeling configuration:")
            self.logger.detailed(f"- Confidence threshold: {self.confidence_threshold}")
            self.logger.detailed(f"- Alpha (pseudo-label weight): {self.alpha}")
            self.logger.detailed(f"- Warmup epochs: {self.warmup_epochs}")
            self.logger.detailed(f"- Rampup epochs: {self.rampup_epochs}")
        
        final_metrics = None
        for epoch in range(self.num_epochs):
            self.logger.detailed(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            # Train for one epoch
            train_metrics = self.train_epoch(epoch)
            self.logger.detailed(f"Training metrics:")
            for metric_name, metric_value in train_metrics.items():
                self.logger.detailed(f"- {metric_name}: {metric_value:.4f}")
            
            # Validate
            val_metrics = self.validate_epoch(epoch)
            self.logger.detailed(f"Validation metrics:")
            for metric_name, metric_value in val_metrics.items():
                self.logger.detailed(f"- {metric_name}: {metric_value:.4f}")
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                self.logger.detailed(f"Learning rate: {current_lr:.6f}")
            
            # Log metrics
            self.logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}"
            )
            
            # Save checkpoint
            is_best = val_metrics['loss'] < best_val_loss
            if is_best:
                best_val_loss = val_metrics['loss']
                best_val_acc = val_metrics['accuracy']
                patience_counter = 0
                final_metrics = val_metrics
            else:
                patience_counter += 1
            
            self.logger.save_model(
                self.model,
                self.optimizer,
                epoch,
                val_metrics['loss'],
                val_metrics['accuracy'],
                is_best
            )
            
            # Early stopping
            if patience_counter >= patience:
                self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Clean up logger resources
        self.logger.finish()
        
        self.logger.detailed("Training completed!")
        if not self.fast_mode:
            self.logger.detailed("Computing final evaluation metrics...")
            final_metrics = self.validate_epoch(self.num_epochs)
            self.logger.detailed("Final evaluation metrics:")
            for metric_name, metric_value in final_metrics.items():
                self.logger.detailed(f"- {metric_name}: {metric_value:.4f}")
        
        return final_metrics 