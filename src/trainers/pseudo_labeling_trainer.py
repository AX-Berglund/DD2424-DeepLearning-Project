#!/usr/bin/env python
"""
Trainer for semi-supervised learning with pseudo-labeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from src.trainers.multiclass_trainer import MultiClassTrainer
from src.utils.logger import MetricTracker
from src.utils.metrics import compute_accuracy


class PseudoLabelingTrainer(MultiClassTrainer):
    """Trainer for semi-supervised learning with pseudo-labeling."""
    
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
            
            # Log model summary after applying strategy
            self.logger.log_model_summary(self.model)
    
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
        """
        Train for one epoch with pseudo-labeling.
        
        Args:
            epoch (int): Current epoch.
            
        Returns:
            tuple: Tuple of (epoch_loss, epoch_accuracy).
        """
        # Set model to training mode
        self.model.train()
        
        # Apply fine-tuning strategy if model supports it
        if hasattr(self.model, 'apply_strategy'):
            strategy = self.config['training']['strategy']
            self.model.apply_strategy(strategy, epoch)
        
        # Initialize metrics tracker
        tracker = MetricTracker()
        batch_idx = 0
        
        # Get data loaders
        labeled_loader = self.dataloaders['train']
        
        # Calculate pseudo-label weight
        pseudo_weight = self.get_pseudo_label_weight(epoch)
        
        # Training loop
        if self.has_unlabeled_data:
            # Train with both labeled and unlabeled data
            for (labeled_data, labeled_targets), (unlabeled_data, _) in zip(
                labeled_loader, self.unlabeled_loader
            ):
                # Move data to device
                labeled_data = labeled_data.to(self.device)
                labeled_targets = labeled_targets.to(self.device)
                unlabeled_data = unlabeled_data.to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass for labeled data
                labeled_outputs = self.model(labeled_data)
                labeled_loss = F.cross_entropy(labeled_outputs, labeled_targets)
                
                # Generate pseudo-labels for unlabeled data
                pseudo_labels, mask = self.generate_pseudo_labels(unlabeled_data)
                
                # Forward pass for unlabeled data
                unlabeled_outputs = self.model(unlabeled_data)
                unlabeled_loss = F.cross_entropy(
                    unlabeled_outputs[mask], 
                    pseudo_labels[mask]
                ) if mask.any() else torch.tensor(0.0, device=self.device)
                
                # Combine losses
                total_loss = labeled_loss + pseudo_weight * unlabeled_loss
                
                # Backward pass
                total_loss.backward()
                
                # Apply gradient masks if using masked fine-tuning
                if hasattr(self.model, 'apply_masks'):
                    self.model.apply_masks()
                
                # Update weights
                self.optimizer.step()
                
                # Update metrics (only for labeled data)
                tracker.update(labeled_loss.item(), labeled_outputs, labeled_targets)
                
                # Log batch progress
                self.logger.log_batch(
                    epoch=epoch,
                    batch_idx=batch_idx,
                    n_batches=len(labeled_loader),
                    loss=total_loss.item(),
                    acc=compute_accuracy(labeled_outputs, labeled_targets)
                )
                batch_idx += 1
                
                # Update pseudo-label metrics
                if mask.any():
                    pseudo_acc = compute_accuracy(
                        unlabeled_outputs[mask], 
                        pseudo_labels[mask]
                    )
                    self.pseudo_label_metrics['pseudo_label_accuracy'].append(pseudo_acc)
                    self.pseudo_label_metrics['pseudo_label_confidence'].append(
                        F.softmax(unlabeled_outputs[mask], dim=1).max(dim=1)[0].mean().item()
                    )
                    self.pseudo_label_metrics['pseudo_label_count'].append(mask.sum().item())
        else:
            # Train with only labeled data
            for labeled_data, labeled_targets in labeled_loader:
                # Move data to device
                labeled_data = labeled_data.to(self.device)
                labeled_targets = labeled_targets.to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                labeled_outputs = self.model(labeled_data)
                total_loss = F.cross_entropy(labeled_outputs, labeled_targets)
                
                # Backward pass
                total_loss.backward()
                
                # Apply gradient masks if using masked fine-tuning
                if hasattr(self.model, 'apply_masks'):
                    self.model.apply_masks()
                
                # Update weights
                self.optimizer.step()
                
                # Update metrics
                tracker.update(total_loss.item(), labeled_outputs, labeled_targets)
                
                # Log batch progress
                self.logger.log_batch(
                    epoch=epoch,
                    batch_idx=batch_idx,
                    n_batches=len(labeled_loader),
                    loss=total_loss.item(),
                    acc=compute_accuracy(labeled_outputs, labeled_targets)
                )
                batch_idx += 1
        
        # Get epoch metrics
        metrics = tracker.get_metrics()
        
        # Add pseudo-label metrics if we have unlabeled data
        if self.has_unlabeled_data and self.pseudo_label_metrics['pseudo_label_accuracy']:
            metrics.update({
                'pseudo_label_accuracy': np.mean(self.pseudo_label_metrics['pseudo_label_accuracy']),
                'pseudo_label_confidence': np.mean(self.pseudo_label_metrics['pseudo_label_confidence']),
                'pseudo_label_count': np.mean(self.pseudo_label_metrics['pseudo_label_count'])
            })
        
        return metrics['loss'], metrics['accuracy']
    
    def train(self):
        """
        Train the model with pseudo-labeling.
        
        Returns:
            dict: Dictionary of training and validation metrics.
        """
        # Get number of epochs
        num_epochs = self.config['training']['num_epochs']
        
        # Initialize best metrics
        best_val_acc = 0.0
        
        # Training loop
        for epoch in range(num_epochs):
            # Log epoch start
            self.logger.start_epoch(epoch)
            
            # Train for one epoch
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate for one epoch
            val_loss, val_acc = self.validate_epoch(epoch)
            
            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log epoch end
            self.logger.end_epoch(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                lr=current_lr
            )
            
            # Check if this is the best model
            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
            
            # Save model checkpoint
            self.logger.save_model(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                val_loss=val_loss,
                val_acc=val_acc,
                is_best=is_best
            )
            
            # Early stopping check
            if self.early_stopping_counter >= self.early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        return {
            'best_val_acc': best_val_acc,
            'final_train_loss': train_loss,
            'final_train_acc': train_acc,
            'final_val_loss': val_loss,
            'final_val_acc': val_acc
        } 