#!/usr/bin/env python
"""
Main script for training models with various configurations.
"""

import argparse
import yaml
import torch
from pathlib import Path

from src.models.transfer_model import create_model
from src.data.dataset import create_data_loaders
from src.trainers.binary_trainer import BinaryTrainer
from src.trainers.multiclass_trainer import MultiClassTrainer
from src.trainers.binary_pseudo_labeling_trainer import BinaryPseudoLabelingTrainer
from src.trainers.multiclass_pseudo_labeling_trainer import MultiClassPseudoLabelingTrainer
from src.utils.logger import Logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train model with various configurations')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to configuration file')
    parser.add_argument('--pseudo-labeling', action='store_true',
                      help='Enable pseudo-labeling training')
    parser.add_argument('--percentage', type=int, default=50,
                      choices=[100, 50, 10, 1],
                      help='Percentage of labeled data to use (only for pseudo-labeling)')
    parser.add_argument('--fast', action='store_true',
                      help='Enable fast training mode (skip detailed metrics during training)')
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update percentage in config if using pseudo-labeling
    if args.pseudo_labeling:
        config['data']['percentage_labeled'] = args.percentage
    
    # Update fast mode in config
    if args.fast:
        config['training']['fast_mode'] = True
        print("Running in fast mode - detailed metrics will only be calculated at the end of training")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create logger
    logger = Logger(config)
    
    # Create data loaders
    dataloaders = create_data_loaders(config['data'])
    
    # Create model
    model = create_model(config)
    model = model.to(device)
    
    # Create trainer based on task and training mode
    if config['data']['task_type'] == 'binary':
        if args.pseudo_labeling:
            trainer = BinaryPseudoLabelingTrainer(model, dataloaders, config, logger)
        else:
            trainer = BinaryTrainer(model, dataloaders, config, logger)
    else:  # multiclass
        if args.pseudo_labeling:
            trainer = MultiClassPseudoLabelingTrainer(model, dataloaders, config, logger)
        else:
            trainer = MultiClassTrainer(model, dataloaders, config, logger)
    
    # Train model
    metrics = trainer.train()
    
    # Clean up logger resources
    logger.finish()
    

if __name__ == '__main__':
    main()