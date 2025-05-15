#!/usr/bin/env python
"""
Script to run pseudo-labeling training for both binary and multiclass tasks.
"""

import argparse
import yaml
import torch
from pathlib import Path

from src.models.model_factory import create_model
from src.data.dataset import create_data_loaders
from .binary_pseudo_labeling_trainer import BinaryPseudoLabelingTrainer
from .multiclass_pseudo_labeling_trainer import MultiClassPseudoLabelingTrainer
from src.utils.logger import Logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train model with pseudo-labeling')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to configuration file')
    parser.add_argument('--percentage', type=int, default=50,
                      choices=[100, 50, 10, 1],
                      help='Percentage of labeled data to use')
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update percentage in config
    config['data']['percentage_labeled'] = args.percentage
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create logger
    logger = Logger(config['logging'])
    
    # Create data loaders
    dataloaders = create_data_loaders(config['data'])
    
    # Create model
    model = create_model(config['model'])
    model = model.to(device)
    
    # Create trainer based on task
    if config['data']['task'] == 'binary':
        trainer = BinaryPseudoLabelingTrainer(model, dataloaders, config, logger)
    else:  # multiclass
        trainer = MultiClassPseudoLabelingTrainer(model, dataloaders, config, logger)
    
    # Train model
    metrics = trainer.train()
    
    # Log final metrics
    logger.info(f"Training completed. Final metrics: {metrics}")


if __name__ == '__main__':
    main() 