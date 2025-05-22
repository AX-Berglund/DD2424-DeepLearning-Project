#!/usr/bin/env python
"""
Trainer for binary semi-supervised learning with pseudo-labeling.

This trainer implements a two-stage approach for each epoch:
1. First train on labeled data
2. Then generate pseudo-labels for unlabeled data
3. Save confident predictions to labeled_added directory
4. Use both original labeled data and accumulated pseudo-labeled data in next epoch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import os
import shutil
from pathlib import Path
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
        
        # Create directories for pseudo-labeled data and unlabeled_rm
        self.pseudo_labeled_dir = Path('data/processed/semisupervised/binary/labeled_added')
        self.pseudo_labeled_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up unlabeled_rm directory
        self.unlabeled_rm_dir = Path('data/processed/semisupervised/binary/1_percent_labeled/unlabeled_rm')
        self.unlabeled_rm_images_dir = self.unlabeled_rm_dir / 'images'
        self.unlabeled_rm_dir.mkdir(parents=True, exist_ok=True)
        self.unlabeled_rm_images_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy unlabeled data to unlabeled_rm if it doesn't exist
        self.setup_unlabeled_rm()
        
        # Always use unlabeled_rm/images for the unlabeled loader
        from src.data.dataset import BinaryPetDataset
        from torch.utils.data import DataLoader
        
        # Create a new dataset with the correct path structure
        unlabeled_dataset = BinaryPetDataset(
            root_dir=str(self.unlabeled_rm_images_dir),
            split='unlabeled',
            transform=dataloaders['unlabeled'].dataset.transform if dataloaders.get('unlabeled') else None
        )
        
        # Verify that all files exist
        missing_files = []
        for img_path, _ in unlabeled_dataset.samples:
            if not os.path.exists(img_path):
                missing_files.append(img_path)
        
        if missing_files:
            self.logger.warning(f"Found {len(missing_files)} missing files in unlabeled dataset. Attempting to fix...")
            # Try to copy missing files from source
            source_dir = Path('data/processed/semisupervised/binary/1_percent_labeled/unlabeled/images')
            for missing_file in missing_files:
                src_path = source_dir / Path(missing_file).name
                if src_path.exists():
                    shutil.copy2(src_path, missing_file)
                    self.logger.info(f"Copied {src_path} to {missing_file}")
                else:
                    self.logger.error(f"Source file {src_path} does not exist")
        
        # Create the data loader
        self.unlabeled_loader = DataLoader(
            unlabeled_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=config['data']['num_workers'],
            pin_memory=True
        )
        self.has_unlabeled_data = self.unlabeled_loader is not None
        
        # Track pseudo-labeled data
        self.pseudo_labeled_count = 0
        self.pseudo_labeled_data = []
        
        # Track original labeled count
        self.original_labeled_count = len(self.train_loader.dataset)
        
        # Apply initial fine-tuning strategy
        if hasattr(self.model, 'apply_strategy'):
            strategy = self.config['training']['strategy']
            self.model.apply_strategy(strategy, epoch=0)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=config['training']['weight_decay']
        )
        
        # Initialize learning rate scheduler
        if config['training']['lr_scheduler']['use_scheduler']:
            scheduler_config = config['training']['lr_scheduler']
            if scheduler_config['type'] == 'step':
                from torch.optim.lr_scheduler import MultiStepLR
                self.scheduler = MultiStepLR(
                    self.optimizer,
                    milestones=scheduler_config['milestones'],
                    gamma=scheduler_config['gamma']
                )
            elif scheduler_config['type'] == 'cosine':
                from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
                self.scheduler = CosineAnnealingWarmRestarts(
                    self.optimizer,
                    T_0=scheduler_config['T_0'],
                    T_mult=scheduler_config['T_mult'],
                    eta_min=scheduler_config['min_lr']
                )
            elif scheduler_config['type'] == 'reduce_on_plateau':
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='min',
                    factor=0.1,
                    patience=5,
                    verbose=True
                )
        else:
            self.scheduler = None
    
    def setup_unlabeled_rm(self):
        """Set up the unlabeled_rm directory by copying unlabeled data if needed."""
        unlabeled_dir = Path('data/processed/semisupervised/binary/1_percent_labeled/unlabeled/images')
        unlabeled_rm_images_dir = self.unlabeled_rm_images_dir
        
        # Get list of files in source and destination
        source_files = set(f.name for f in unlabeled_dir.glob('*') if f.is_file())
        dest_files = set(f.name for f in unlabeled_rm_images_dir.glob('*') if f.is_file())
        
        # Find files that need to be copied
        files_to_copy = source_files - dest_files
        
        if files_to_copy:
            self.logger.info(f"Setting up unlabeled_rm directory... Copying {len(files_to_copy)} files")
            for file_name in files_to_copy:
                src_path = unlabeled_dir / file_name
                dst_path = unlabeled_rm_images_dir / file_name
                shutil.copy2(src_path, dst_path)
            self.logger.info(f"Copied {len(files_to_copy)} files to unlabeled_rm")
        else:
            self.logger.info("All files already present in unlabeled_rm directory")
    
    def move_confident_sample(self, img_path: Path, pseudo_label: int, confidence: float, epoch: int):
        """
        Move a confident sample from unlabeled_rm to labeled_added.
        
        Args:
            img_path (Path): Path to the image file
            pseudo_label (int): Predicted label
            confidence (float): Confidence score
            epoch (int): Current epoch number
        """
        # Create epoch-specific directory in labeled_added
        epoch_dir = self.pseudo_labeled_dir / f'epoch_{epoch}'
        epoch_dir.mkdir(exist_ok=True)
        
        # Create class-specific directory
        class_name = 'cat' if pseudo_label == 0 else 'dog'
        class_dir = epoch_dir / class_name
        class_dir.mkdir(exist_ok=True)
        
        # Move the file
        new_path = class_dir / img_path.name
        shutil.move(str(img_path), str(new_path))
        
        # Remove from unlabeled_rm
        if img_path.exists():
            img_path.unlink()
    
    def save_pseudo_labeled_data(self, images, pseudo_labels, confidence_scores, image_paths, epoch):
        """
        Save confident pseudo-labeled data to the labeled_added directory.
        
        Args:
            images (torch.Tensor): Batch of images
            pseudo_labels (torch.Tensor): Generated pseudo-labels
            confidence_scores (torch.Tensor): Confidence scores for predictions
            image_paths (list): List of paths to the original images
            epoch (int): Current epoch number
        """
        # Convert tensors to numpy
        pseudo_labels = pseudo_labels.cpu().numpy()
        confidence_scores = confidence_scores.cpu().numpy()
        
        # Process each confident prediction
        for idx, (img_path, label, conf) in enumerate(zip(image_paths, pseudo_labels, confidence_scores)):
            if conf >= self.confidence_threshold:
                # Move the file to labeled_added
                self.move_confident_sample(
                    Path(img_path),
                    int(label),
                    float(conf),
                    epoch
                )
                
                self.pseudo_labeled_count += 1
                self.pseudo_labeled_data.append({
                    'image_path': img_path,
                    'label': label,
                    'confidence': conf,
                    'epoch': epoch
                })
    
    def generate_pseudo_labels(self, unlabeled_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate pseudo-labels for unlabeled data.
        
        Args:
            unlabeled_data (torch.Tensor): Unlabeled data batch.
            
        Returns:
            tuple: (pseudo_labels, confidence_mask, confidence_scores)
        """
        self.model.eval()  # Set model to evaluation mode for prediction
        with torch.no_grad():
            outputs = self.model(unlabeled_data)
            probs = torch.sigmoid(outputs)
            
            # Generate binary labels (0 or 1) based on probability threshold
            pseudo_labels = (probs >= 0.5).float()
            
            # Calculate confidence scores
            confidence_scores = torch.abs(probs - 0.5) * 2  # Scale to [0, 1] range
            
            # Create mask for confident predictions
            confidence_mask = confidence_scores >= self.confidence_threshold
            
        self.model.train()  # Set back to training mode
        return pseudo_labels, confidence_mask, confidence_scores
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train the model for one epoch using labeled data.
        
        Args:
            epoch (int): Current epoch number.
            
        Returns:
            Dict[str, float]: Dictionary of training metrics.
        """
        self.model.train()
        metrics = MetricTracker()
        
        # Update original labeled count
        self.original_labeled_count = len(self.train_loader.dataset)
        
        # Log the number of samples in the training set
        self.logger.detailed(f"\nTraining on {self.original_labeled_count} original labeled samples")
        
        # Train on labeled data
        for batch_idx, (labeled_data, labels) in enumerate(self.train_loader):
            # Move labeled data to device
            labeled_data = labeled_data.to(self.device)
            labels = labels.to(self.device).float().unsqueeze(1)  # Add channel dimension
            
            # Zero gradients efficiently
            self.optimizer.zero_grad(set_to_none=True)
            
            # Forward pass for labeled data
            labeled_outputs = self.model(labeled_data)
            labeled_loss = self.criterion(labeled_outputs, labels)
            
            # Backward pass and optimization
            labeled_loss.backward()
            self.optimizer.step()
            
            # Update metrics
            metrics.update(labeled_loss.item(), labeled_outputs, labels)
            
            # Log batch progress
            if batch_idx % self.log_interval == 0:
                self.logger.detailed(
                    f"Train Epoch: {epoch} [{batch_idx * len(labeled_data)}/{len(self.train_loader.dataset)} "
                    f"({100. * batch_idx / len(self.train_loader):.0f}%)]\t"
                    f"Loss: {labeled_loss.item():.6f}"
                )
        
        return metrics.get_metrics()
    
    def generate_and_save_pseudo_labels(self, epoch: int) -> Dict[str, float]:
        """
        Generate pseudo-labels for all unlabeled data and save confident predictions.
        
        Args:
            epoch (int): Current epoch number.
            
        Returns:
            Dict[str, float]: Dictionary of pseudo-labeling metrics.
        """
        self.model.eval()
        metrics = MetricTracker()
        total_unlabeled_samples = 0
        total_confident_samples = 0
        skipped_files = 0
        
        # Log the start of pseudo-labeling
        self.logger.detailed(f"\nStarting pseudo-labeling for epoch {epoch}")
        
        # Create a list to store valid samples
        valid_samples = []
        missing_files = []
        
        # First, verify all files exist and collect valid samples
        for img_path, _ in self.unlabeled_loader.dataset.samples:
            if os.path.exists(img_path):
                valid_samples.append((img_path, -1))
            else:
                missing_files.append(img_path)
        
        if missing_files:
            self.logger.warning(f"Found {len(missing_files)} missing files. Attempting to recover...")
            # Try to copy missing files from source
            source_dir = Path('data/processed/semisupervised/binary/1_percent_labeled/unlabeled/images')
            for missing_file in missing_files:
                src_path = source_dir / Path(missing_file).name
                if src_path.exists():
                    shutil.copy2(src_path, missing_file)
                    valid_samples.append((missing_file, -1))
                    self.logger.info(f"Recovered {missing_file}")
                else:
                    self.logger.error(f"Could not recover {missing_file} - source file {src_path} does not exist")
        
        # Create a new dataset with only valid samples
        from src.data.dataset import BinaryPetDataset
        from torch.utils.data import DataLoader
        
        # Create a new dataset with only valid files
        valid_dataset = BinaryPetDataset(
            root_dir=str(self.unlabeled_rm_images_dir),
            split='unlabeled',
            transform=self.unlabeled_loader.dataset.transform
        )
        valid_dataset.samples = valid_samples
        
        # Create a new DataLoader with the valid dataset
        self.unlabeled_loader = DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True
        )
        
        # Now process the valid samples
        with torch.no_grad():
            for batch_idx, (unlabeled_data, _) in enumerate(tqdm(self.unlabeled_loader, desc="Generating pseudo-labels")):
                # Get image paths for this batch
                start = batch_idx * self.batch_size
                end = start + len(unlabeled_data)
                image_paths = [self.unlabeled_loader.dataset.samples[i][0] for i in range(start, end)]
                
                # Verify all files in this batch still exist
                valid_indices = []
                valid_paths = []
                for idx, path in enumerate(image_paths):
                    if os.path.exists(path):
                        valid_indices.append(idx)
                        valid_paths.append(path)
                    else:
                        skipped_files += 1
                        self.logger.warning(f"File {path} was moved/deleted during processing, skipping...")
                
                if not valid_indices:
                    self.logger.warning(f"All files in batch {batch_idx} were moved/deleted, skipping batch...")
                    continue
                
                # Only process valid files
                valid_data = unlabeled_data[valid_indices]
                valid_data = valid_data.to(self.device)
                
                pseudo_labels, confidence_mask, confidence_scores = self.generate_pseudo_labels(valid_data)
                batch_confident = confidence_mask.sum().item()
                total_confident_samples += batch_confident
                total_unlabeled_samples += len(valid_data)
                
                # Save confident predictions
                self.save_pseudo_labeled_data(
                    valid_data,
                    pseudo_labels,
                    confidence_scores,
                    valid_paths,
                    epoch
                )
                
                self.logger.detailed(
                    f"Pseudo-labeling stats - "
                    f"Confident samples: {batch_confident}/{len(valid_data)} "
                    f"(Confidence threshold: {self.confidence_threshold})"
                )
        
        confidence_percentage = 0.0
        if total_unlabeled_samples > 0:
            confidence_percentage = 100.0 * total_confident_samples / total_unlabeled_samples
        
        self.logger.detailed(f"Epoch {epoch} Pseudo-labeling Summary:")
        self.logger.detailed(f"Found {total_confident_samples}/{total_unlabeled_samples} confident predictions "
                          f"({confidence_percentage:.2f}%)")
        if skipped_files > 0:
            self.logger.detailed(f"Skipped {skipped_files} files that were moved/deleted during processing")
        
        if total_confident_samples > 0:
            self.logger.detailed(f"\nMoving {total_confident_samples} confident samples to labeled_added directory")
            self.update_unlabeled_loader()
            # Log the updated training set size
            new_labeled_count = len(self.train_loader.dataset)
            self.logger.detailed(f"Training set now contains {new_labeled_count} samples "
                              f"({new_labeled_count - self.original_labeled_count} newly added)")
        
        if self.logger.wandb is not None:
            self.logger.wandb.log({
                "pseudo_labeling/confident_samples": total_confident_samples,
                "pseudo_labeling/total_samples": total_unlabeled_samples,
                "pseudo_labeling/skipped_files": skipped_files,
                "pseudo_labeling/confidence_ratio": total_confident_samples / total_unlabeled_samples if total_unlabeled_samples > 0 else 0,
                "pseudo_labeling/training_set_size": len(self.train_loader.dataset)
            }, step=epoch)
        
        return metrics.get_metrics()
    
    def update_unlabeled_loader(self):
        """Update the unlabeled data loader to use only remaining samples."""
        from torch.utils.data import DataLoader
        from src.data.dataset import BinaryPetDataset
        
        # Create a new dataset with remaining samples
        remaining_dataset = BinaryPetDataset(
            root_dir=str(self.unlabeled_rm_images_dir),
            split='unlabeled',
            transform=self.unlabeled_loader.dataset.transform
        )
        
        # Update the samples list to only include files that exist
        remaining_dataset.update_samples()
        
        # Check if we have any samples left
        if len(remaining_dataset) == 0:
            self.logger.info("No unlabeled samples remaining. Pseudo-labeling complete.")
            self.has_unlabeled_data = False
            self.unlabeled_loader = None
            return
        
        # Create a new DataLoader with the remaining dataset
        self.unlabeled_loader = DataLoader(
            remaining_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True
        )
        self.logger.info(f"Updated unlabeled loader with {len(remaining_dataset)} remaining samples")
    
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
            
            # Train on labeled data
            train_metrics = self.train_epoch(epoch)
            self.logger.detailed(f"Training metrics:")
            for metric_name, metric_value in train_metrics.items():
                if isinstance(metric_value, np.ndarray):
                    self.logger.detailed(f"- {metric_name}:\n{metric_value}")
                else:
                    self.logger.detailed(f"- {metric_name}: {metric_value:.4f}")
            
            # Generate and save pseudo-labels if we're past warmup
            if self.has_unlabeled_data and epoch >= self.warmup_epochs:
                pseudo_metrics = self.generate_and_save_pseudo_labels(epoch)
                self.logger.detailed(f"Pseudo-labeling metrics:")
                for metric_name, metric_value in pseudo_metrics.items():
                    if isinstance(metric_value, np.ndarray):
                        self.logger.detailed(f"- {metric_name}:\n{metric_value}")
                    else:
                        self.logger.detailed(f"- {metric_name}: {metric_value:.4f}")
            
            # Validate
            val_loss, val_acc = self.validate_epoch(epoch)
            self.logger.detailed(f"Validation metrics:")
            self.logger.detailed(f"- loss: {val_loss:.4f}")
            self.logger.detailed(f"- accuracy: {val_acc:.4f}")
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                self.logger.detailed(f"Learning rate: {current_lr:.6f}")
            
            # Log metrics
            self.logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Acc: {val_acc:.4f}"
            )
            
            # Log to wandb
            if self.logger.wandb is not None:
                wandb_log_dict = {
                    "train/loss": train_metrics['loss'],
                    "train/accuracy": train_metrics['accuracy'],
                    "val/loss": val_loss,
                    "val/accuracy": val_acc,
                    "learning_rate": self.optimizer.param_groups[0]['lr'],
                    "epoch": epoch,
                    "_step": epoch
                }
                
                # Add pseudo-labeling metrics if available
                if hasattr(self, 'pseudo_labeled_count'):
                    wandb_log_dict.update({
                        "pseudo_labeling/total_pseudo_labeled": self.pseudo_labeled_count,
                    })
                
                self.logger.wandb.log(wandb_log_dict)
            
            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                best_val_acc = val_acc
                patience_counter = 0
                final_metrics = {'loss': val_loss, 'accuracy': val_acc}
            else:
                patience_counter += 1
            
            self.logger.save_model(
                self.model,
                self.optimizer,
                epoch,
                val_loss,
                val_acc,
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
            if isinstance(metric_value, (list, np.ndarray)):
                self.logger.info(f"- {metric_name}:\n{metric_value}")
            elif isinstance(metric_value, (int, float)):
                self.logger.info(f"- {metric_name}: {metric_value:.4f}")
            else:
                self.logger.info(f"- {metric_name}: {metric_value}")
        
        # Log test metrics to wandb
        if self.logger.wandb is not None:
            wandb_log_dict = {}
            for metric_name, metric_value in test_metrics.items():
                if isinstance(metric_value, (int, float)):
                    wandb_log_dict[f"test/{metric_name}"] = metric_value
            wandb_log_dict["epoch"] = epoch
            self.logger.wandb.log(wandb_log_dict)
        
        # Clean up logger resources
        self.logger.finish()
        return final_metrics