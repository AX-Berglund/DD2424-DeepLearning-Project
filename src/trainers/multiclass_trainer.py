#!/usr/bin/env python
"""
Trainer for multi-class classification (37 breeds).
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
from tqdm import tqdm

from src.utils.logger import MetricTracker
from src.utils.metrics import (
    compute_accuracy, compute_metrics, compute_mixup_loss, 
    compute_cutmix_loss, compute_weighted_loss, get_class_counts
)
from src.utils.visualization import (
    visualize_model_predictions, plot_to_image, 
    visualize_class_distribution, visualize_parameter_changes
)


class MultiClassTrainer:
    """Trainer for multi-class classification task."""
    
    def __init__(self, model, dataloaders, config, logger):
        """
        Initialize the trainer.
        
        Args:
            model (nn.Module): Model to train.
            dataloaders (dict): Dictionary of data loaders for each split.
            config (dict): Configuration dictionary.
            logger (Logger): Logger instance.
        """
        self.model = model
        self.dataloaders = dataloaders
        self.config = config
        self.logger = logger
        
        # Set device
        self.device = torch.device(config['device'])
        self.model = self.model.to(self.device)
        
        # Get class names (breed names)
        self.class_names = self._get_class_names()
        
        # Handle class imbalance if enabled
        self.class_weights = None
        if (config['training'].get('class_imbalance', {}).get('enabled', False) and 
            config['training']['class_imbalance']['strategy'] == 'weighted_loss'):
            self.class_weights = self._compute_class_weights()
        
        # Initialize optimizer
        self.optimizer = self._initialize_optimizer()
        
        # Initialize learning rate scheduler
        self.scheduler = self._initialize_lr_scheduler()
        
        # Initialize early stopping
        self.early_stopping_patience = config['training']['early_stopping_patience']
        self.early_stopping_counter = 0
        self.best_val_acc = 0.0
        
        # Store original model parameters for comparison
        if hasattr(self.model, 'original_params'):
            self.original_params = self.model.original_params
        else:
            self.original_params = {name: param.clone().detach() 
                                  for name, param in self.model.named_parameters()}
        
        # Log model summary
        self.logger.log_model_summary(self.model)
        
        # Visualize class distribution
        self._visualize_class_distribution()
    
    def _get_class_names(self):
        """
        Get class names from data loader.
        
        Returns:
            list: List of class names.
        """
        # Define cat and dog breeds

        all_breeds = ['Abyssinian',
            'american_bulldog',
            'american_pit_bull_terrier',
            'basset_hound',
            'beagle',
            'Bengal',
            'Birman',
            'Bombay',
            'boxer',
            'British_Shorthair',
            'chihuahua',
            'Egyptian_Mau',
            'english_cocker_spaniel',
            'english_setter',
            'german_shorthaired',
            'great_pyrenees',
            'havanese',
            'japanese_chin',
            'keeshond',
            'leonberger',
            'Maine_Coon',
            'miniature_pinscher',
            'newfoundland',
            'Persian',
            'pomeranian',
            'pug',
            'Ragdoll',
            'Russian_Blue',
            'saint_bernard',
            'samoyed',
            'scottish_terrier',
            'shiba_inu',
            'Siamese',
            'Sphynx',
            'staffordshire_bull_terrier',
            'wheaten_terrier',
            'yorkshire_terrier']
        
        return all_breeds
    
    def _compute_class_weights(self):
        """
        Compute class weights for weighted loss function.
        
        Returns:
            torch.Tensor: Tensor of class weights.
        """
        # Get class counts from training data
        class_counts = get_class_counts(self.dataloaders['train'])
        self.logger.info(f"Class counts: {class_counts}")
        
        # Convert to tensor
        class_counts = torch.tensor(class_counts, dtype=torch.float32)
        
        # Compute weights as inverse of frequency
        weights = 1.0 / class_counts
        
        # Normalize weights
        weights = weights / weights.sum() * len(weights)
        
        # Move to device
        weights = weights.to(self.device)
        
        self.logger.info(f"Class weights: {weights}")
        
        return weights
    
    def _initialize_optimizer(self):
        """
        Initialize the optimizer based on configuration.
        
        Returns:
            torch.optim.Optimizer: Initialized optimizer.
        """
        # Get optimizer parameters
        optimizer_name = self.config['training']['optimizer']
        lr = self.config['training']['learning_rate']
        weight_decay = self.config['training']['weight_decay']
        
        # Get parameter groups (for different learning rates per layer)
        if hasattr(self.model, 'get_optimizer_param_groups'):
            parameters = self.model.get_optimizer_param_groups()
        else:
            parameters = self.model.parameters()
        
        # Create optimizer
        if optimizer_name.lower() == 'adam':
            return optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'adamw':
            return optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'sgd':
            momentum = self.config['training'].get('momentum', 0.9)
            return optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    def _initialize_lr_scheduler(self):
        """
        Initialize the learning rate scheduler based on configuration.
        
        Returns:
            torch.optim.lr_scheduler._LRScheduler: Initialized scheduler or None.
        """
        scheduler_config = self.config['training']['lr_scheduler']
        
        if not scheduler_config.get('use_scheduler', False):
            return None
        
        scheduler_type = scheduler_config.get('type', 'step').lower()
        
        if scheduler_type == 'step':
            return StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 10),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        elif scheduler_type == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['num_epochs'],
                eta_min=scheduler_config.get('eta_min', 0)
            )
        elif scheduler_type == 'reduce_on_plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_config.get('gamma', 0.1),
                patience=scheduler_config.get('patience', 5),
                verbose=True
            )
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_type}")
    
    def _visualize_class_distribution(self):
        """Visualize class distribution in the training set."""
        # Create visualization
        fig = visualize_class_distribution(
            dataloader=self.dataloaders['train'],
            class_names=self.class_names
        )
        
        # Log to TensorBoard
        self.logger.log_image("class_distribution", plot_to_image(fig))
    
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
        
        # Apply fine-tuning strategy if model supports it
        if hasattr(self.model, 'apply_strategy'):
            strategy = self.config['training']['strategy']
            self.model.apply_strategy(strategy, epoch)
        
        # Initialize metrics tracker
        tracker = MetricTracker()
        
        # Get data loader
        train_loader = self.dataloaders['train']
        num_batches = len(train_loader)
        
        # Check for mixup/cutmix augmentation
        aug_config = self.config['data']['augmentation']
        use_mixup = aug_config.get('mixup', False)
        use_cutmix = aug_config.get('cutmix', False)
        
        # Train loop
        for batch_idx, data in enumerate(train_loader):
            # Handle regular or augmented data
            if use_mixup or use_cutmix:
                # For MixUp/CutMix datasets, the data includes mixed targets
                if len(data) == 4:  # MixUp/CutMix
                    inputs, targets_a, targets_b, lam = data
                    inputs = inputs.to(self.device)
                    targets_a = targets_a.to(self.device)
                    targets_b = targets_b.to(self.device)
                else:
                    # Regular batch format
                    inputs, targets = data
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
            else:
                # Regular batch format
                inputs, targets = data
                inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Compute loss
            if use_mixup and len(data) == 4:
                loss = compute_mixup_loss(outputs, targets_a, targets_b, lam)
                # For accuracy tracking during training, just use the main target
                preds = torch.argmax(outputs, dim=1)
                accuracy = (preds == targets_a).float().mean().item()
            elif use_cutmix and len(data) == 4:
                loss = compute_cutmix_loss(outputs, targets_a, targets_b, lam)
                # For accuracy tracking during training, just use the main target
                preds = torch.argmax(outputs, dim=1)
                accuracy = (preds == targets_a).float().mean().item()
            else:
                # Regular loss computation
                if self.class_weights is not None:
                    loss = F.cross_entropy(outputs, targets, weight=self.class_weights)
                else:
                    loss = F.cross_entropy(outputs, targets)
                
                # Compute accuracy
                accuracy = compute_accuracy(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Apply gradient masks if using masked fine-tuning
            if hasattr(self.model, 'apply_masks'):
                self.model.apply_masks()
            
            # Update weights
            self.optimizer.step()
            
            # Update metrics (using regular data format for tracking)
            if use_mixup or use_cutmix:
                if len(data) == 4:
                    tracker.update(loss.item(), outputs, targets_a)
                else:
                    tracker.update(loss.item(), outputs, targets)
            else:
                tracker.update(loss.item(), outputs, targets)
            
            # Log batch progress
            if batch_idx % self.config['logging']['log_interval'] == 0:
                self.logger.info(
                    f"Epoch {epoch} [{batch_idx}/{num_batches} "
                    f"({100. * batch_idx / num_batches:.0f}%)] - "
                    f"Loss: {loss.item():.6f}, Acc: {accuracy:.4f}"
                )
        
        # Get epoch metrics
        metrics = tracker.get_metrics()
        
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
            for inputs, targets in val_loader:
                # Move data to device
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Compute loss (same as training, but without mixup/cutmix)
                if self.class_weights is not None:
                    loss = F.cross_entropy(outputs, targets, weight=self.class_weights)
                else:
                    loss = F.cross_entropy(outputs, targets)
                
                # Update metrics
                tracker.update(loss.item(), outputs, targets)
                
                # Collect outputs and targets for detailed metrics
                all_outputs.append(outputs)
                all_targets.append(targets)
        
        # Concatenate all outputs and targets
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Get epoch metrics
        metrics = tracker.get_metrics()
        
        # Log detailed metrics
        detailed_metrics = compute_metrics(
            outputs=all_outputs,
            targets=all_targets,
            class_names=self.class_names
        )
        self.logger.log_metrics(epoch, detailed_metrics, split="val")
        
        # Visualize model predictions if it's a multiple of 5 epochs
        if epoch % 5 == 0 or epoch == self.config['training']['num_epochs'] - 1:
            fig = visualize_model_predictions(
                model=self.model,
                dataloader=val_loader,
                class_names=self.class_names,
                device=self.device
            )
            self.logger.log_image(f"predictions_epoch_{epoch}", plot_to_image(fig), global_step=epoch)
            
            # Visualize parameter changes
            if hasattr(self, 'original_params'):
                fig = visualize_parameter_changes(self.model, self.original_params)
                self.logger.log_image(f"parameter_changes_epoch_{epoch}", plot_to_image(fig), global_step=epoch)
        
        return metrics['loss'], metrics['accuracy']
    
    def evaluate(self, split='test'):
        """
        Evaluate the model on the specified split.
        
        Args:
            split (str): Data split ('train', 'val', or 'test').
            
        Returns:
            dict: Dictionary of evaluation metrics.
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Get data loader
        data_loader = self.dataloaders[split]
        
        # Initialize lists for outputs and targets
        all_outputs = []
        all_targets = []
        
        # Evaluation loop
        with torch.no_grad():
            for inputs, targets in tqdm(data_loader, desc=f"Evaluating on {split}"):
                # Move data to device
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Collect outputs and targets
                all_outputs.append(outputs)
                all_targets.append(targets)
        
        # Concatenate all outputs and targets
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Compute metrics
        metrics = compute_metrics(
            outputs=all_outputs,
            targets=all_targets,
            class_names=self.class_names
        )
        
        # Log metrics
        self.logger.log_metrics(0, metrics, split=split)
        
        # Create and log confusion matrix visualization
        from src.utils.visualization import confusion_matrix_to_figure
        fig = confusion_matrix_to_figure(metrics['confusion_matrix'], self.class_names)
        self.logger.log_image(f"confusion_matrix_{split}", plot_to_image(fig))
        
        return metrics
    
    def train(self):
        """
        Train the model for the specified number of epochs.
        
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
                if isinstance(self.scheduler, ReduceLROnPlateau):
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
            
            # Early stopping
            if self.early_stopping_counter >= self.early_stopping_patience:
                self.logger.info(f"Early stopping triggered after epoch {epoch}")
                break
        
        # Final evaluation on test set
        self.logger.info("Evaluating model on test set...")
        test_metrics = self.evaluate(split='test')
        self.logger.info(f"Test accuracy: {test_metrics['accuracy']:.4f}")
        
        # Return training metrics
        return {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'test_acc': test_metrics['accuracy']
        }