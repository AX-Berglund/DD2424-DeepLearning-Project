#!/usr/bin/env python
"""
Early stopping utility for training.
"""

import numpy as np


class EarlyStopping:
    """
    Early stopping handler to prevent overfitting.
    
    Args:
        patience (int): Number of epochs to wait before stopping.
        min_delta (float): Minimum change in monitored value to qualify as an improvement.
        mode (str): One of 'min' or 'max'. In 'min' mode, training will stop when the quantity monitored has stopped decreasing;
                   in 'max' mode it will stop when the quantity monitored has stopped increasing.
    """
    def __init__(self, patience=7, min_delta=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        # Initialize best_score based on mode
        if mode == 'min':
            self.best_score = float('inf')
        else:  # mode == 'max'
            self.best_score = float('-inf')
    
    def __call__(self, val_metric):
        """
        Check if training should be stopped.
        
        Args:
            val_metric (float): Current validation metric value.
            
        Returns:
            bool: True if training should be stopped, False otherwise.
        """
        if self.mode == 'min':
            score = -val_metric
        else:  # mode == 'max'
            score = val_metric
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.early_stop 