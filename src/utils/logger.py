#!/usr/bin/env python
"""
Logging utilities for training and evaluation.
"""

import os
import json
import logging
import warnings
from pathlib import Path
import time
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import wandb
import numpy as np
import matplotlib.pyplot as plt

# Filter out PyTorch DataLoader warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*")

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Filter out PyTorch DataLoader warnings
logging.getLogger('torch.utils.data.dataloader').setLevel(logging.ERROR)


class Logger:
    """Logger class for training and evaluation metrics."""
    
    def __init__(self, config):
        """
        Initialize logger.
        
        Args:
            config (dict): Configuration dictionary.
        """
        self.config = config
        self.log_dir = config['logging']['log_dir']
        self.experiment_name = config['logging']['experiment_name']
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_level = config['logging'].get('log_level', 'info')  # Default to 'info'
        
        # Create log directory
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Set up file logging
        self.log_file = os.path.join(self.log_dir, f"{self.experiment_name}_{self.timestamp}.log")
        self.setup_file_logging()
        
        # Set up TensorBoard
        if config['logging']['tensorboard']:
            self.tensorboard_dir = os.path.join(self.log_dir, 'tensorboard', f"{self.experiment_name}_{self.timestamp}")
            self.writer = SummaryWriter(self.tensorboard_dir)
        else:
            self.writer = None
        
        # Set up Weights & Biases
        self.wandb = None
        if config['logging'].get('wandb', False):
            self.wandb = wandb.init(
                project=config['logging'].get('wandb_project', 'deep-learning'),
                config=config
            )
        
        # Log experiment start
        self.info(f"Starting experiment: {self.experiment_name}_{self.timestamp}")
        self.info(f"Config: {config}")
    
    def setup_file_logging(self):
        """Set up file logging with appropriate format and level."""
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Create file handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(formatter)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # Set up logger
        self.logger = logging.getLogger('training')
        self.logger.setLevel(logging.DEBUG)  # Capture all levels
        
        # Disable propagation to root logger to prevent duplicate messages
        self.logger.propagate = False
        
        # Remove any existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def _should_log(self, level):
        """Check if the given level should be logged based on current log_level."""
        levels = {
            'debug': 0,
            'detailed': 1,
            'info': 2,
            'warning': 3,
            'error': 4
        }
        return levels.get(level, 0) >= levels.get(self.log_level, 0)
    
    def debug(self, message):
        """Log debug message."""
        if self._should_log('debug'):
            self.logger.debug(message)
    
    def detailed(self, message):
        """Log detailed message."""
        if self._should_log('detailed'):
            self.logger.info(f"[DETAILED] {message}")
    
    def info(self, message):
        """Log info message."""
        if self._should_log('info'):
            self.logger.info(message)
    
    def warning(self, message):
        """Log warning message."""
        if self._should_log('warning'):
            self.logger.warning(message)
    
    def error(self, message):
        """Log error message."""
        if self._should_log('error'):
            self.logger.error(message)
    
    def log_batch(self, epoch, batch_idx, n_batches, loss, acc):
        """Log batch metrics."""
        if self._should_log('detailed'):
            self.detailed(
                f"Batch {batch_idx}/{n_batches} - "
                f"Loss: {loss:.6f}, Acc: {acc:.6f}"
            )
        elif self._should_log('info'):
            if batch_idx % self.config['logging']['log_interval'] == 0:
                self.info(
                    f"Epoch {epoch} [{batch_idx}/{n_batches} "
                    f"({100. * batch_idx / n_batches:.0f}%)] - "
                    f"Loss: {loss:.6f}, Acc: {acc:.6f}"
                )
    
    def log_metrics(self, epoch, metrics, split="train"):
        """Log detailed metrics."""
        if not self._should_log('detailed'):
            return
            
        self.detailed(f"\n{split.capitalize()} metrics for epoch {epoch}:")
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                self.detailed(f"  {metric_name}: {metric_value:.4f}")
            elif isinstance(metric_value, np.ndarray):
                self.detailed(f"  {metric_name}:\n{metric_value}")
            elif isinstance(metric_value, dict):
                self.detailed(f"  {metric_name}:")
                for k, v in metric_value.items():
                    if isinstance(v, (int, float)):
                        self.detailed(f"    {k}: {v:.4f}")
                    else:
                        self.detailed(f"    {k}: {v}")
    
    def start_epoch(self, epoch):
        """Log epoch start."""
        self.info(f"Starting epoch {epoch}")
    
    def end_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc, lr):
        """Log epoch end."""
        self.info(
            f"Epoch {epoch} completed - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
            f"LR: {lr:.6f}"
        )
    
    def save_model(self, model, optimizer, epoch, val_loss, val_acc, is_best=False):
        """Save model checkpoint."""
        if not self._should_log('detailed'):
            return
            
        self.detailed(f"Saving model checkpoint for epoch {epoch}")
        self.detailed(f"  Validation Loss: {val_loss:.4f}")
        self.detailed(f"  Validation Accuracy: {val_acc:.4f}")
        self.detailed(f"  Is Best Model: {is_best}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc
        }
        
        # Create checkpoint directory
        checkpoint_dir = self.config['logging']['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save checkpoint
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"{self.experiment_name}_epoch_{epoch}.pt"
        )
        torch.save(checkpoint, checkpoint_path)
        self.detailed(f"  Checkpoint saved to: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_model_path = os.path.join(
                checkpoint_dir,
                f"{self.experiment_name}_best.pt"
            )
            torch.save(checkpoint, best_model_path)
            self.detailed(f"  Best model saved to: {best_model_path}")
    
    def finish(self):
        """Clean up logger resources."""
        # Close and remove all handlers
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
        
        # Close TensorBoard writer
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        
        # Close Weights & Biases
        if self.wandb is not None:
            self.wandb.finish()
            self.wandb = None
        
        # Close any open matplotlib figures
        plt.close('all')
        
        # Shutdown the logging system
        logging.shutdown()
    
    def log_model_summary(self, model):
        """Log model architecture and parameters."""
        if not self._should_log('detailed'):
            return
            
        self.detailed("\nModel Summary:")
        self.detailed(f"Architecture: {model.__class__.__name__}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.detailed(f"Total parameters: {total_params:,}")
        self.detailed(f"Trainable parameters: {trainable_params:,}")
        self.detailed(f"Non-trainable parameters: {total_params - trainable_params:,}")
        
        # Log layer information
        self.detailed("\nLayer Information:")
        for name, module in model.named_children():
            self.detailed(f"\n{name}:")
            self.detailed(f"  Type: {module.__class__.__name__}")
            if hasattr(module, 'out_features'):
                self.detailed(f"  Output features: {module.out_features}")
            if hasattr(module, 'kernel_size'):
                self.detailed(f"  Kernel size: {module.kernel_size}")
            if hasattr(module, 'stride'):
                self.detailed(f"  Stride: {module.stride}")
            if hasattr(module, 'padding'):
                self.detailed(f"  Padding: {module.padding}")


class MetricTracker:
    """Tracker for metrics during training/evaluation."""
    
    def __init__(self):
        """Initialize metric tracker."""
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.loss_sum = 0.0
        self.correct = 0
        self.total = 0
        self.batches = 0
        self.start_time = time.time()
    
    def update(self, loss, outputs, targets):
        """
        Update metrics with batch results.
        
        Args:
            loss (float): Batch loss.
            outputs (torch.Tensor): Model outputs.
            targets (torch.Tensor): Target labels.
        """
        # Update loss
        self.loss_sum += loss
        self.batches += 1
        
        # Update accuracy
        if outputs.shape[1] == 1:  # Binary classification
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()
            self.correct += (preds == targets).sum().item()
        else:  # Multi-class classification
            _, preds = torch.max(outputs, 1)
            self.correct += (preds == targets).sum().item()
        self.total += targets.size(0)
    
    def get_metrics(self):
        """
        Get current metrics.
        
        Returns:
            dict: Dictionary of metrics.
        """
        avg_loss = self.loss_sum / max(1, self.batches)
        accuracy = self.correct / max(1, self.total)
        time_elapsed = time.time() - self.start_time
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'correct': self.correct,
            'total': self.total,
            'time': time_elapsed
        }


def get_logger(config):
    """
    Get logger instance.
    
    Args:
        config (dict): Configuration dictionary.
        
    Returns:
        Logger: Logger instance.
    """
    return Logger(config)