#!/usr/bin/env python
"""
Main script for running transfer learning experiments.
"""

import os
import argparse
import yaml
import torch
import random
import numpy as np
from pathlib import Path

from src.data.dataset import create_data_loaders
from src.models.transfer_model import create_model
from src.trainers.binary_trainer import BinaryTrainer
from src.trainers.multiclass_trainer import MultiClassTrainer
from src.utils.logger import get_logger


def set_seed(seed):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to the configuration file.
        
    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def main():
    """Main function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Transfer learning project")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--strategy", type=str, help="Override fine-tuning strategy in config")
    parser.add_argument("--eval", action="store_true", help="Evaluate model instead of training")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint for evaluation")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override strategy if provided
    if args.strategy:
        config['training']['strategy'] = args.strategy
    
    # Set random seed
    set_seed(config['seed'])
    
    # Create logger
    logger = get_logger(config)
    
    # Log configuration
    logger.info(f"Configuration: {config}")
    
    # Check if CUDA is available
    if torch.cuda.is_available() and config['device'] == 'cuda':
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("Using CPU")
        config['device'] = 'cpu'
    
    # Create data loaders
    logger.info("Creating data loaders...")
    dataloaders = create_data_loaders(config)
    
    # Create model
    logger.info(f"Creating model: {config['model']['architecture']}")
    model = create_model(config)
    
    # Load checkpoint if provided
    if args.checkpoint:
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=config['device'])
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Select trainer based on number of classes
    if config['model']['num_classes'] == 2:
        trainer_class = BinaryTrainer
    else:
        trainer_class = MultiClassTrainer
    
    # Create trainer
    trainer = trainer_class(
        model=model,
        dataloaders=dataloaders,
        config=config,
        logger=logger
    )
    
    # Evaluate or train
    if args.eval:
        logger.info("Evaluating model...")
        metrics = trainer.evaluate('test')
        logger.info(f"Test accuracy: {metrics['accuracy']:.4f}")
    else:
        logger.info("Training model...")
        metrics = trainer.train()
        logger.info(f"Training completed. Final metrics: {metrics}")
    
    # Finish logging
    logger.finish()


if __name__ == "__main__":
    main()