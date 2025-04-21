#!/usr/bin/env python
"""
Logging utilities for training and evaluation.
"""

import os
import json
import logging
from pathlib import Path
import time
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class Logger:
    """Logger class for tracking experiments."""
    
    def __init__(self, config, log_dir="logs"):
        """
        Initialize the logger.
        
        Args:
            config (dict): Experiment configuration.
            log_dir (str): Base directory for logs.
        """
        self.config = config
        
        # Create experiment name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{config['experiment_name']}_{timestamp}"
        
        # Setup directories
        self.log_dir = Path(log_dir)
        self.exp_dir = self.log_dir / "run_logs" / exp_name
        self.tensorboard_dir = self.log_dir / "tensorboard" / exp_name
        
        # Create directories
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup file logger
        self.file_logger = self._setup_file_logger()
        
        # Setup TensorBoard if enabled
        self.tb_writer = None
        if config['logging']['tensorboard']:
            self.tb_writer = SummaryWriter(log_dir=self.tensorboard_dir)
        
        # Setup WandB if enabled
        self.wandb_enabled = config['logging'].get('wandb', False)
        if self.wandb_enabled:
            self._setup_wandb()
        
        # Save config
        with open(self.exp_dir / "config.json", "w") as f:
            json.dump(config, f, indent=4)
        
        # Training metrics
        self.metrics = {
            "train": {"loss": [], "acc": []},
            "val": {"loss": [], "acc": []}
        }
        
        # Timing info
        self.start_time = time.time()
        self.epoch_start_time = None
        
        # Log start
        self.info(f"Starting experiment: {exp_name}")
        self.info(f"Config: {json.dumps(config, indent=2)}")
    
    def _setup_file_logger(self):
        """Setup file logger."""
        logger = logging.getLogger(f"exp_{self.exp_dir.name}")
        logger.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler(self.exp_dir / "experiment.log")
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(file_handler)
        
        return logger
    
    def _setup_wandb(self):
        """Setup Weights & Biases logging."""
        try:
            import wandb
            wandb.init(
                project="transfer_learning_project",
                name=self.exp_dir.name,
                config=self.config
            )
            self.info("WandB logging enabled")
        except ImportError:
            self.warning("WandB not installed. WandB logging disabled.")
            self.wandb_enabled = False
    
    def info(self, message):
        """Log info message."""
        logging.info(message)
        self.file_logger.info(message)
    
    def warning(self, message):
        """Log warning message."""
        logging.warning(message)
        self.file_logger.warning(message)
    
    def error(self, message):
        """Log error message."""
        logging.error(message)
        self.file_logger.error(message)
    
    def start_epoch(self, epoch):
        """Mark the start of an epoch."""
        self.epoch_start_time = time.time()
        self.info(f"Starting epoch {epoch}")
    
    def end_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc, lr=None):
        """
        Mark the end of an epoch and log metrics.
        
        Args:
            epoch (int): Current epoch.
            train_loss (float): Training loss.
            train_acc (float): Training accuracy.
            val_loss (float): Validation loss.
            val_acc (float): Validation accuracy.
            lr (float, optional): Current learning rate.
        """
        # Calculate epoch time
        epoch_time = time.time() - self.epoch_start_time
        total_time = time.time() - self.start_time
        
        # Store metrics
        self.metrics["train"]["loss"].append(train_loss)
        self.metrics["train"]["acc"].append(train_acc)
        self.metrics["val"]["loss"].append(val_loss)
        self.metrics["val"]["acc"].append(val_acc)
        
        # Log metrics
        self.info(
            f"Epoch {epoch} completed in {epoch_time:.2f}s (Total: {total_time:.2f}s) - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}" +
            (f", LR: {lr:.6f}" if lr is not None else "")
        )
        
        # Log to TensorBoard
        if self.tb_writer:
            self.tb_writer.add_scalar("Loss/train", train_loss, epoch)
            self.tb_writer.add_scalar("Accuracy/train", train_acc, epoch)
            self.tb_writer.add_scalar("Loss/val", val_loss, epoch)
            self.tb_writer.add_scalar("Accuracy/val", val_acc, epoch)
            if lr is not None:
                self.tb_writer.add_scalar("LearningRate", lr, epoch)
        
        # Log to WandB
        if self.wandb_enabled:
            try:
                import wandb
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "epoch_time": epoch_time,
                    "lr": lr
                })
            except ImportError:
                pass
    
    def log_batch(self, epoch, batch_idx, n_batches, loss, acc):
        """
        Log batch metrics during training.
        
        Args:
            epoch (int): Current epoch.
            batch_idx (int): Current batch index.
            n_batches (int): Total number of batches.
            loss (float): Batch loss.
            acc (float): Batch accuracy.
        """
        if batch_idx % self.config['logging']['log_interval'] == 0:
            self.info(
                f"Epoch {epoch} [{batch_idx}/{n_batches} "
                f"({100. * batch_idx / n_batches:.0f}%)] - "
                f"Loss: {loss:.6f}, Acc: {acc:.4f}"
            )
    
    def log_metrics(self, epoch, metrics_dict, split="val"):
        """
        Log detailed metrics.
        
        Args:
            epoch (int): Current epoch.
            metrics_dict (dict): Dictionary of metrics.
            split (str): Data split ('train', 'val', or 'test').
        """
        # Log to console and file
        self.info(f"{split.capitalize()} metrics for epoch {epoch}:")
        for metric_name, metric_value in metrics_dict.items():
            if isinstance(metric_value, (int, float)):
                self.info(f"  {metric_name}: {metric_value:.4f}")
            else:
                self.info(f"  {metric_name}: {metric_value}")
        
        # Log to TensorBoard
        if self.tb_writer:
            for metric_name, metric_value in metrics_dict.items():
                if isinstance(metric_value, (int, float)):
                    self.tb_writer.add_scalar(f"{metric_name}/{split}", metric_value, epoch)
        
        # Log to WandB
        if self.wandb_enabled:
            try:
                import wandb
                
                # Create a dictionary with prefixed keys
                prefixed_metrics = {f"{split}_{k}": v for k, v in metrics_dict.items()
                                 if isinstance(v, (int, float))}
                prefixed_metrics["epoch"] = epoch
                
                wandb.log(prefixed_metrics)
                
                # Log confusion matrix if available
                if "confusion_matrix" in metrics_dict:
                    try:
                        wandb.log({
                            f"{split}_confusion_matrix": wandb.plot.confusion_matrix(
                                probs=None,
                                y_true=metrics_dict.get("y_true", []),
                                preds=metrics_dict.get("y_pred", []),
                                class_names=metrics_dict.get("class_names", [])
                            )
                        })
                    except Exception as e:
                        self.warning(f"Failed to log confusion matrix to WandB: {e}")
            except ImportError:
                pass
    
    def log_model_summary(self, model):
        """
        Log model summary.
        
        Args:
            model (nn.Module): Model to summarize.
        """
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        # Log model summary
        self.info(f"Model: {model.__class__.__name__}")
        self.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
                f"({100. * trainable_params / total_params:.2f}%)")
        
        # Log which layers are being fine-tuned
        self.info("Trainable layers:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.info(f"  {name}: {param.numel():,} parameters")
    
    def log_image(self, tag, img_tensor, global_step=0, dataformats='CHW'):
        """
        Log an image to TensorBoard.
        
        Args:
            tag (str): Image tag.
            img_tensor (torch.Tensor): Image tensor.
            global_step (int): Global step.
            dataformats (str): Format of the input tensor ('CHW', 'HWC', etc.)
        """
        if self.tb_writer:
            self.tb_writer.add_image(tag, img_tensor, global_step, dataformats)
    
    def log_histogram(self, tag, values, global_step=0, bins='auto'):
        """
        Log a histogram to TensorBoard.
        
        Args:
            tag (str): Histogram tag.
            values (torch.Tensor): Values to plot.
            global_step (int): Global step.
            bins (str or int): Number of bins or method.
        """
        if self.tb_writer:
            self.tb_writer.add_histogram(tag, values, global_step, bins=bins)
    
    def save_model(self, model, optimizer, epoch, val_loss, val_acc, is_best=False):
        """
        Save model checkpoint.
        
        Args:
            model (nn.Module): Model to save.
            optimizer (torch.optim.Optimizer): Optimizer to save.
            epoch (int): Current epoch.
            val_loss (float): Validation loss.
            val_acc (float): Validation accuracy.
            is_best (bool): Whether this is the best model so far.
        """
        checkpoint_dir = Path(self.config['logging']['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Create checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'config': self.config
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, checkpoint_dir / 'latest_model.pt')
        
        # Save epoch checkpoint (every 5 epochs)
        if epoch % 5 == 0:
            torch.save(checkpoint, checkpoint_dir / f'model_epoch_{epoch}.pt')
        
        # Save best model
        if is_best:
            torch.save(checkpoint, checkpoint_dir / 'best_model.pt')
            self.info(f"New best model saved (val_acc: {val_acc:.4f})")
    
    def finish(self):
        """Finish logging and clean up."""
        # Calculate total time
        total_time = time.time() - self.start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Log completion
        self.info(
            f"Experiment completed in "
            f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
        )
        
        # Close TensorBoard writer
        if self.tb_writer:
            self.tb_writer.close()
        
        # Close WandB
        if self.wandb_enabled:
            try:
                import wandb
                wandb.finish()
            except ImportError:
                pass
        
        # Save final metrics
        with open(self.exp_dir / "metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=4)


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