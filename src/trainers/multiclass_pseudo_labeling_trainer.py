#!/usr/bin/env python
"""
Trainer for multiclass semi-supervised learning with pseudo-labeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from typing import Dict

from src.trainers.multiclass_trainer import MultiClassTrainer
from src.utils.logger import MetricTracker
from src.utils.metrics import compute_accuracy, compute_metrics


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
            
            # Zero gradients efficiently
            self.optimizer.zero_grad(set_to_none=True)
            
            # Forward pass for labeled data
            output = self.model(data)
            loss = self.criterion(output, target)
            
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
                    loss = loss + self.get_pseudo_label_weight(epoch) * pseudo_loss
                    
                    self.logger.detailed(f"Batch {batch_idx}: Pseudo-labeling stats - "
                                      f"Confident samples: {mask.sum().item()}/{len(mask)} "
                                      f"(Confidence threshold: {self.confidence_threshold})")
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            metrics.update('loss', loss.item())
            metrics.update('accuracy', self._compute_accuracy(output, target))
            
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
        
        return metrics.result()

    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        metrics = MetricTracker()
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.val_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                metrics.update('loss', loss.item())
                metrics.update('accuracy', self._compute_accuracy(output, target))
                
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
        
        return metrics.result()

    def train(self) -> None:
        """Train the model."""
        best_val_loss = float('inf')
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
            self.logger.detailed(f"- Warmup epochs: {self.rampup_epochs}")
            self.logger.detailed(f"- Rampup epochs: {self.rampup_epochs}")
        
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
            
            # Early stopping
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                self.save_checkpoint(epoch, val_metrics)
                self.logger.detailed(f"New best model saved! Validation loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                self.logger.detailed(f"No improvement for {patience_counter} epochs")
            
            if patience_counter >= self.early_stopping_patience:
                self.logger.detailed(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        self.logger.detailed("Training completed!")
        if not self.fast_mode:
            self.logger.detailed("Computing final evaluation metrics...")
            final_metrics = self.validate_epoch(self.num_epochs)
            self.logger.detailed("Final evaluation metrics:")
            for metric_name, metric_value in final_metrics.items():
                self.logger.detailed(f"- {metric_name}: {metric_value:.4f}") 