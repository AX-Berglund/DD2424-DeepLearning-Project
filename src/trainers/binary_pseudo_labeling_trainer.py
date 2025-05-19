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
            
        # Log info about confident predictions
        num_confident = mask.sum().item()
        total_samples = mask.numel()
        self.logger.info(
            f"Found {num_confident}/{total_samples} confident predictions "
            f"({(num_confident/total_samples)*100:.1f}%)"
        )
            
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
        
        # Reset unlabeled data iterator at the start of each epoch
        unlabeled_iter = None
        if unlabeled_loader is not None and len(unlabeled_loader) > 0:
            unlabeled_iter = iter(unlabeled_loader)
        
        # Pre-allocate tensors for pseudo-labeling metrics
        if self.has_unlabeled_data:
            self.pseudo_label_metrics['pseudo_label_count'] = []
            self.pseudo_label_metrics['pseudo_label_confidence'] = []
        
        # Enable automatic mixed precision if configured
        scaler = torch.cuda.amp.GradScaler() if self.config['training'].get('mixed_precision', False) else None
        
        # Training loop
        for batch_idx, (labeled_data, labeled_targets) in enumerate(labeled_loader):
            # Move data to device (do this once for the whole batch)
            labeled_data = labeled_data.to(self.device, non_blocking=True)
            labeled_targets = labeled_targets.to(self.device, non_blocking=True).float().unsqueeze(1)
            
            # Zero gradients
            self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            # Forward pass for labeled data with mixed precision
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                labeled_outputs = self.model(labeled_data)
                labeled_loss = F.binary_cross_entropy_with_logits(labeled_outputs, labeled_targets)
                total_loss = labeled_loss
            
            # If we have unlabeled data and past warmup, use pseudo-labeling
            if unlabeled_iter is not None and epoch >= self.config['training']['pseudo_labeling']['warmup_epochs']:
                try:
                    # Get next batch of unlabeled data
                    unlabeled_data, _ = next(unlabeled_iter)
                    unlabeled_data = unlabeled_data.to(self.device, non_blocking=True)
                    
                    # Get pseudo-labels with mixed precision
                    with torch.cuda.amp.autocast(enabled=scaler is not None):
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
                            
                            # Log pseudo-labeling metrics (do this on CPU to avoid GPU memory fragmentation)
                            with torch.no_grad():
                                self.pseudo_label_metrics['pseudo_label_count'].append(
                                    confidence_mask.sum().item()
                                )
                                self.pseudo_label_metrics['pseudo_label_confidence'].append(
                                    confidence[confidence_mask].mean().item()
                                )
                    
                except StopIteration:
                    # Reset iterator if we've exhausted it
                    unlabeled_iter = iter(unlabeled_loader)
            
            # Backward pass with mixed precision
            if scaler is not None:
                scaler.scale(total_loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                total_loss.backward()
                self.optimizer.step()
            
            # Update metrics with total loss (do this on CPU)
            with torch.no_grad():
                tracker.update(total_loss.item(), labeled_outputs, labeled_targets)
            
            # Log batch progress (only every N batches to reduce overhead)
            if batch_idx % self.config['logging']['log_interval'] == 0:
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
        
        # Pre-allocate lists for outputs and targets
        all_outputs = []
        all_targets = []
        
        # Validation loop
        with torch.no_grad(), torch.amp.autocast(device_type=self.device.type):  # Use automatic mixed precision
            for data, target in val_loader:
                # Move data to device (do this once for the whole batch)
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True).float().unsqueeze(1)
                
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
        
        # Convert to float32 before computing metrics
        all_outputs = all_outputs.float()
        all_targets = all_targets.float()
        
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

    def train(self):
        """
        Train the model for the specified number of epochs.
        
        Returns:
            dict: Dictionary of training and validation metrics.
        """
        # Get number of epochs
        num_epochs = self.config['training']['num_epochs']
        
        # Initialize lists to store metrics
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        learning_rates = []
        pseudo_label_counts = []
        pseudo_label_confidences = []
        
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
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Store metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            learning_rates.append(current_lr)
            
            # Store pseudo-labeling metrics if available
            if self.pseudo_label_metrics['pseudo_label_count']:
                pseudo_label_counts.append(np.mean(self.pseudo_label_metrics['pseudo_label_count']))
                pseudo_label_confidences.append(np.mean(self.pseudo_label_metrics['pseudo_label_confidence']))
            
            # Log epoch end
            self.logger.end_epoch(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                lr=current_lr
            )
            
            # Log pseudo-label metrics if available
            if self.pseudo_label_metrics['pseudo_label_count']:
                avg_count = np.mean(self.pseudo_label_metrics['pseudo_label_count'])
                avg_confidence = np.mean(self.pseudo_label_metrics['pseudo_label_confidence'])
                self.logger.info(
                    f"Pseudo-label metrics - Count: {avg_count:.1f}, "
                    f"Confidence: {avg_confidence:.4f}"
                )
            
            # Check if this is the best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict()
            
            # Save model checkpoint
            self.logger.save_model(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                val_loss=val_loss,
                val_acc=val_acc,
                is_best=is_best
            )
            
            # Early stopping
            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                self.logger.info(f"Early stopping triggered after epoch {epoch}")
                break
        
        # Final evaluation on test set
        self.logger.info("Evaluating model on test set...")
        test_metrics = self.evaluate(split='test')
        self.logger.info(f"Test accuracy: {test_metrics['accuracy']:.4f}")
        
        # Create and save training history plots
        import os
        from datetime import datetime
        import matplotlib.pyplot as plt
        
        # Create timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directory if it doesn't exist
        save_dir = os.path.join("results", "graphs", "training_history")
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Training vs Validation Loss
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training vs Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f"loss_history_{timestamp}.png"))
        plt.show()
        plt.close()
        
        # 2. Training vs Validation Accuracy
        plt.figure(figsize=(10, 6))
        plt.plot(train_accs, label='Training Accuracy')
        plt.plot(val_accs, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training vs Validation Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f"accuracy_history_{timestamp}.png"))
        plt.show()
        plt.close()
        
        # 3. Learning Rate History
        plt.figure(figsize=(10, 6))
        plt.plot(learning_rates)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate History')
        plt.yscale('log')  # Use log scale for better visualization
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f"learning_rate_history_{timestamp}.png"))
        plt.show()
        plt.close()
        
        # 4. Pseudo-labeling Metrics (if available)
        if pseudo_label_counts:
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
            
            # Plot pseudo-label count
            ax1.plot(pseudo_label_counts, 'b-')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Number of Pseudo-labels')
            ax1.set_title('Number of Pseudo-labels per Epoch')
            ax1.grid(True)
            
            # Plot pseudo-label confidence
            ax2.plot(pseudo_label_confidences, 'r-')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Average Confidence')
            ax2.set_title('Average Pseudo-label Confidence per Epoch')
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"pseudo_labeling_metrics_{timestamp}.png"))
            plt.show()
            plt.close()
        
        # Return training metrics
        return {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'test_acc': test_metrics['accuracy']
        } 