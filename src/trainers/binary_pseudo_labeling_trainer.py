#!/usr/bin/env python
"""
Trainer for binary semi-supervised learning with pseudo-labeling.

This trainer implements a two-stage approach for each batch:
1. First train on labeled data
2. Then use the model to predict labels for unlabeled data
3. If predictions have high confidence, use them as additional training signals
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Dict, Optional, Tuple, Iterator

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
        
        # Initialize unlabeled data loader
        self.unlabeled_loader = dataloaders.get('unlabeled')
        self.has_unlabeled_data = self.unlabeled_loader is not None
        self.unlabeled_iter = None
        
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
    
    def get_pseudo_label_weight(self, epoch: int) -> float:
        """
        Calculate the weight applied to the pseudo-labeled loss term during training.
        
        Args:
            epoch (int): Current training epoch.
            
        Returns:
            float: Weight for pseudo-labeled loss.
        """
        # During warmup period, don't use pseudo-labels at all
        if epoch < self.warmup_epochs:
            return 0.0
            
        # If no unlabeled data available, pseudo-label loss weight is 0
        if not self.has_unlabeled_data:
            return 0.0
            
        # During ramp-up period after warmup, linearly increase weight from 0 to alpha
        if epoch < self.warmup_epochs + self.rampup_epochs:
            return self.alpha * ((epoch - self.warmup_epochs) / self.rampup_epochs)
            
        # After ramp-up, use constant alpha weight
        return self.alpha
    
    def get_next_unlabeled_batch(self) -> Optional[torch.Tensor]:
        """
        Get the next batch of unlabeled data, handling iterator reset if needed.
        
        Returns:
            torch.Tensor or None: Unlabeled data batch, or None if no unlabeled data available.
        """
        if not self.has_unlabeled_data:
            return None
            
        if self.unlabeled_iter is None:
            self.unlabeled_iter = iter(self.unlabeled_loader)
            
        try:
            unlabeled_data, _ = next(self.unlabeled_iter)
            return unlabeled_data.to(self.device)
        except StopIteration:
            # Reset iterator and try again
            self.unlabeled_iter = iter(self.unlabeled_loader)
            try:
                unlabeled_data, _ = next(self.unlabeled_iter)
                return unlabeled_data.to(self.device)
            except StopIteration:
                # This should not happen unless unlabeled dataset is empty
                self.logger.detailed("Warning: Empty unlabeled dataset detected")
                return None
    
    def generate_pseudo_labels(self, unlabeled_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate pseudo-labels for unlabeled data.
        
        Args:
            unlabeled_data (torch.Tensor): Unlabeled data batch.
            
        Returns:
            tuple: (pseudo_labels, mask) where mask indicates which samples to use.
        """
        self.model.eval()  # Set model to evaluation mode for prediction
        with torch.no_grad():
            outputs = self.model(unlabeled_data)
            probs = torch.sigmoid(outputs)
            
            # Generate binary labels (0 or 1) based on probability threshold
            pseudo_labels = (probs >= 0.5).float()
            
            # Create mask for confident predictions
            # Scale to [0, 1] range where 1.0 is highest confidence
            confidence = torch.abs(probs - 0.5) * 2  
            mask = confidence >= self.confidence_threshold
            
        self.model.train()  # Set back to training mode
        return pseudo_labels, mask
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train the model for one epoch using both labeled and unlabeled data.
        
        Args:
            epoch (int): Current epoch number.
            
        Returns:
            Dict[str, float]: Dictionary of training metrics.
        """
        self.model.train()
        metrics = MetricTracker()
        
        # Get the weight for pseudo-labeled loss for current epoch
        pseudo_label_weight = self.get_pseudo_label_weight(epoch)
        
        # Track the number of confident pseudo-labels
        total_unlabeled_samples = 0
        total_confident_samples = 0
        
        # First, process all labeled data
        for batch_idx, (labeled_data, labels) in enumerate(self.train_loader):
            # Move labeled data to device
            labeled_data = labeled_data.to(self.device)
            labels = labels.to(self.device).float().unsqueeze(1)  # Add channel dimension
            
            # Zero gradients efficiently
            self.optimizer.zero_grad(set_to_none=True)
            
            # Forward pass for labeled data
            labeled_outputs = self.model(labeled_data)
            labeled_loss = self.criterion(labeled_outputs, labels)
            
            # Log labeled loss
            self.logger.detailed(f"Batch {batch_idx}: Labeled loss: {labeled_loss.item():.6f}")
            
            # Initialize combined loss to labeled loss
            total_loss = labeled_loss
            
            # Backward pass and optimization for labeled data
            total_loss.backward()
            self.optimizer.step()
            
            # Update metrics
            metrics.update(total_loss.item(), labeled_outputs, labels)
            
            # Log batch progress
            if batch_idx % self.log_interval == 0:
                self.logger.detailed(
                    f"Train Epoch: {epoch} [{batch_idx * len(labeled_data)}/{len(self.train_loader.dataset)} "
                    f"({100. * batch_idx / len(self.train_loader):.0f}%)]\t"
                    f"Loss: {total_loss.item():.6f}"
                )
        
        # Then, process all unlabeled data if we're using pseudo-labels
        if self.has_unlabeled_data and pseudo_label_weight > 0:
            # Reset unlabeled data iterator
            self.unlabeled_iter = iter(self.unlabeled_loader)
            unlabeled_batch_idx = 0
            
            # Process all unlabeled data
            while True:
                try:
                    unlabeled_data = self.get_next_unlabeled_batch()
                    if unlabeled_data is None:
                        self.logger.detailed("No more unlabeled data available")
                        break
                        
                    # Generate pseudo-labels
                    pseudo_labels, confidence_mask = self.generate_pseudo_labels(unlabeled_data)
                    
                    # Update statistics
                    batch_confident = confidence_mask.sum().item()
                    total_confident_samples += batch_confident
                    total_unlabeled_samples += len(unlabeled_data)
                    
                    # Log pseudo-labeling stats with batch index
                    self.logger.detailed(
                        f"Unlabeled Batch {unlabeled_batch_idx}: Pseudo-labeling stats - "
                        f"Confident samples: {batch_confident}/{len(unlabeled_data)} "
                        f"(Confidence threshold: {self.confidence_threshold})"
                    )
                    
                    # Compute loss for confident pseudo-labeled samples
                    if confidence_mask.any():
                        # Zero gradients efficiently
                        self.optimizer.zero_grad(set_to_none=True)
                        
                        # Forward pass for unlabeled data
                        unlabeled_outputs = self.model(unlabeled_data)
                        
                        # Apply confidence mask to both outputs and pseudo-labels
                        masked_outputs = unlabeled_outputs[confidence_mask]
                        masked_pseudo_labels = pseudo_labels[confidence_mask]
                        
                        # Compute loss for confident pseudo-labeled data
                        if masked_outputs.numel() > 0:  # Check there are confident samples
                            pseudo_loss = self.criterion(masked_outputs, masked_pseudo_labels)
                            # Add weighted pseudo-loss to total loss
                            total_loss = pseudo_label_weight * pseudo_loss
                            
                            # Backward pass and optimization for pseudo-labeled data
                            total_loss.backward()
                            self.optimizer.step()
                            
                            # Update metrics - ensure outputs and labels are properly shaped
                            metrics.update(
                                total_loss.item(),
                                masked_outputs.unsqueeze(1) if masked_outputs.dim() == 1 else masked_outputs,
                                masked_pseudo_labels.unsqueeze(1) if masked_pseudo_labels.dim() == 1 else masked_pseudo_labels
                            )
                        
                    # Increment batch counter
                    unlabeled_batch_idx += 1
                    
                    # Add a safety check to prevent infinite loops
                    if unlabeled_batch_idx >= len(self.unlabeled_loader):
                        self.logger.detailed(f"Processed all {unlabeled_batch_idx} unlabeled batches")
                        break
                    
                except StopIteration:
                    self.logger.detailed(f"Finished processing {unlabeled_batch_idx} unlabeled batches")
                    break
                except Exception as e:
                    self.logger.warning(f"Error processing unlabeled batch {unlabeled_batch_idx}: {str(e)}")
                    break
        
        # Log pseudo-labeling statistics for the epoch
        if self.has_unlabeled_data:
            confidence_percentage = 0.0
            if total_unlabeled_samples > 0:
                confidence_percentage = 100.0 * total_confident_samples / total_unlabeled_samples
                
            self.logger.detailed(f"Epoch {epoch} Pseudo-labeling Summary:")
            self.logger.detailed(f"Found {total_confident_samples}/{total_unlabeled_samples} confident predictions "
                              f"({confidence_percentage:.2f}%)")
            
            # Log to wandb
            if self.logger.wandb is not None:
                self.logger.wandb.log({
                    "pseudo_labeling/confident_samples": total_confident_samples,
                    "pseudo_labeling/total_samples": total_unlabeled_samples,
                    "pseudo_labeling/confidence_ratio": total_confident_samples / total_unlabeled_samples if total_unlabeled_samples > 0 else 0
                }, step=epoch)
        
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
                # Skip accuracy as it's already calculated in the main validation loop
                if metric_name != 'accuracy':
                    metrics.update_metric(metric_name, metric_value)
                if isinstance(metric_value, np.ndarray):
                    self.logger.detailed(f"Validation {metric_name}:\n{metric_value}")
                else:
                    self.logger.detailed(f"Validation {metric_name}: {metric_value:.4f}")
        
        # Log to wandb
        if self.logger.wandb is not None:
            self.logger.wandb.log({
                "val/loss": metrics.get_metrics()['loss'],
                "val/accuracy": metrics.get_metrics()['accuracy']
            }, step=epoch)
        
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
                if isinstance(metric_value, np.ndarray):
                    self.logger.detailed(f"- {metric_name}:\n{metric_value}")
                else:
                    self.logger.detailed(f"- {metric_name}: {metric_value:.4f}")
            
            # Log training metrics to wandb
            if self.logger.wandb is not None:
                self.logger.wandb.log({
                    "train/loss": train_metrics['loss'],
                    "train/accuracy": train_metrics['accuracy']
                }, step=epoch)
            
            # Validate
            val_metrics = self.validate_epoch(epoch)
            self.logger.detailed(f"Validation metrics:")
            for metric_name, metric_value in val_metrics.items():
                if isinstance(metric_value, np.ndarray):
                    self.logger.detailed(f"- {metric_name}:\n{metric_value}")
                else:
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

            # In your train method, replace multiple wandb.log calls with a single one at end of each epoch:
            if self.logger.wandb is not None:
                wandb_log_dict = {
                    "train/loss": train_metrics['loss'],
                    "train/accuracy": train_metrics['accuracy'],
                    "val/loss": val_metrics['loss'],
                    "val/accuracy": val_metrics['accuracy'],
                    "learning_rate": self.optimizer.param_groups[0]['lr'],
                    "epoch": epoch,
                    "_step": epoch
                }
                
                # Add pseudo-labeling metrics if available
                if hasattr(self, 'total_confident_samples') and hasattr(self, 'total_unlabeled_samples'):
                    wandb_log_dict.update({
                        "pseudo_labeling/confident_samples": self.total_confident_samples,
                        "pseudo_labeling/total_samples": self.total_unlabeled_samples,
                        "pseudo_labeling/confidence_ratio": self.total_confident_samples / self.total_unlabeled_samples if self.total_unlabeled_samples > 0 else 0,
                    })
                
                self.logger.wandb.log(wandb_log_dict)
            
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
        
        # Evaluate on test set
        self.logger.info("Evaluating model on test set...")
        test_metrics = self.evaluate(split='test')
        self.logger.info(f"Test metrics:")
        for metric_name, metric_value in test_metrics.items():
            if isinstance(metric_value, np.ndarray):
                self.logger.info(f"- {metric_name}:\n{metric_value}")
            else:
                self.logger.info(f"- {metric_name}: {metric_value:.4f}")
        
        # Log test metrics to wandb at the current epoch step
        if self.logger.wandb is not None:
            wandb_log_dict = {}
            for metric_name, metric_value in test_metrics.items():
                if not isinstance(metric_value, np.ndarray):
                    wandb_log_dict[f"test/{metric_name}"] = metric_value
            # Add epoch step to ensure metrics are logged at the correct step
            wandb_log_dict["epoch"] = epoch
            self.logger.wandb.log(wandb_log_dict)
        
        # Clean up logger resources
        self.logger.finish()
        return final_metrics