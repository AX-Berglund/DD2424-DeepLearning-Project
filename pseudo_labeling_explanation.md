# Semi-Supervised Learning with Pseudo-Labeling

This document provides a comprehensive explanation of the pseudo-labeling implementation for semi-supervised learning in our transfer learning project.

## Table of Contents
1. [Introduction to Semi-Supervised Learning](#introduction-to-semi-supervised-learning)
2. [Pseudo-Labeling Theory](#pseudo-labeling-theory)
3. [Implementation Details](#implementation-details)
4. [Configuration and Usage](#configuration-and-usage)
5. [Training Process](#training-process)
6. [Key Components](#key-components)

## Introduction to Semi-Supervised Learning

Semi-supervised learning is a machine learning paradigm that leverages both labeled and unlabeled data for training. In our case, we're working with a dataset where we have:
- A small percentage of labeled data (100%, 50%, 10%, or 1%)
- The remaining data as unlabeled

The goal is to use the unlabeled data to improve model performance beyond what would be possible with just the labeled data.

## Pseudo-Labeling Theory

Pseudo-labeling is a simple yet effective semi-supervised learning technique that works as follows:

1. **Initial Training**: Train the model on the available labeled data
2. **Pseudo-Label Generation**: Use the trained model to predict labels for unlabeled data
3. **Confidence Thresholding**: Only use predictions with high confidence as pseudo-labels
4. **Combined Training**: Train the model on both labeled and pseudo-labeled data

The key aspects of pseudo-labeling are:
- **Confidence Threshold**: Only use predictions with confidence above a threshold (e.g., 0.95)
- **Ramp-up Schedule**: Gradually increase the weight of pseudo-labeled loss during training
- **Consistency Regularization**: The model should make consistent predictions for the same input

## Implementation Details

### 1. Data Structure
```
data/processed/semisupervised/
├── 100_percent_labeled/
│   ├── labeled/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── unlabeled/
├── 50_percent_labeled/
├── 10_percent_labeled/
└── 1_percent_labeled/
```

Each percentage folder contains:
- Labeled data split into train/val/test sets
- Unlabeled data for pseudo-labeling

### 2. PseudoLabelingTrainer Class

The `PseudoLabelingTrainer` class extends the base `MultiClassTrainer` and implements pseudo-labeling:

```python
class PseudoLabelingTrainer(MultiClassTrainer):
    def __init__(self, model, dataloaders, config, logger):
        # Initialize base trainer
        super().__init__(model, dataloaders, config, logger)
        
        # Pseudo-labeling specific settings
        self.confidence_threshold = 0.95
        self.rampup_epochs = 5
        self.alpha = 0.5
```

Key methods:
- `get_pseudo_label_weight()`: Implements the ramp-up schedule
- `generate_pseudo_labels()`: Creates pseudo-labels with confidence thresholding
- `train_epoch()`: Combines labeled and pseudo-labeled training

### 3. Training Process

The training process in `train_epoch()` works as follows:

```python
# For each batch:
# 1. Process labeled data
labeled_outputs = self.model(labeled_data)
labeled_loss = F.cross_entropy(labeled_outputs, labeled_targets)

# 2. Generate pseudo-labels
pseudo_labels, mask = self.generate_pseudo_labels(unlabeled_data)

# 3. Process unlabeled data
unlabeled_outputs = self.model(unlabeled_data)
unlabeled_loss = F.cross_entropy(unlabeled_outputs[mask], pseudo_labels[mask])

# 4. Combine losses with ramp-up weight
total_loss = labeled_loss + pseudo_weight * unlabeled_loss
```

## Configuration and Usage

### Configuration File (pseudo_labeling_config.yaml)

Key settings:
```yaml
training:
  pseudo_labeling:
    confidence_threshold: 0.95  # Minimum confidence for pseudo-labels
    rampup_epochs: 5           # Number of epochs to ramp up weight
    alpha: 0.5                 # Maximum weight for pseudo-labeled loss
```

### Running the Training

```bash
# Prepare the semi-supervised dataset
python prepare_semisupervised_data.py

# Train with pseudo-labeling
python main.py --config config/pseudo_labeling_config.yaml --pseudo-labeling
```

## Training Process

1. **Initial Phase (Epochs 0-4)**:
   - Train on labeled data only
   - Gradually increase pseudo-label weight from 0 to 0.5
   - Generate pseudo-labels with confidence > 0.95

2. **Main Phase (Epochs 5-30)**:
   - Train on both labeled and pseudo-labeled data
   - Pseudo-label weight remains at 0.5
   - Continue to use confidence thresholding

3. **Validation and Testing**:
   - Validate on labeled validation set
   - Test on labeled test set
   - Early stopping based on validation accuracy

## Key Components

### 1. Confidence Thresholding
```python
def generate_pseudo_labels(self, unlabeled_data):
    outputs = self.model(unlabeled_data)
    probs = F.softmax(outputs, dim=1)
    max_probs, pseudo_labels = torch.max(probs, dim=1)
    mask = max_probs >= self.confidence_threshold
    return pseudo_labels, mask
```

### 2. Ramp-up Schedule
```python
def get_pseudo_label_weight(self, epoch):
    if epoch < self.rampup_epochs:
        return self.alpha * (epoch / self.rampup_epochs)
    return self.alpha
```

### 3. Loss Combination
```python
total_loss = labeled_loss + pseudo_weight * unlabeled_loss
```

## Best Practices and Tips

1. **Confidence Threshold**:
   - Start with a high threshold (0.95)
   - Adjust based on model performance
   - Consider class-wise thresholds for imbalanced data

2. **Ramp-up Schedule**:
   - Start with a small number of epochs (5)
   - Increase for more stable training
   - Monitor validation performance during ramp-up

3. **Alpha Value**:
   - Typical range: 0.3 to 0.7
   - Higher values give more weight to pseudo-labels
   - Lower values are more conservative

4. **Data Augmentation**:
   - Use strong augmentation for unlabeled data
   - Helps prevent overfitting to pseudo-labels
   - Improves model robustness

## Monitoring and Debugging

1. **Metrics to Watch**:
   - Labeled data accuracy
   - Pseudo-label accuracy (if available)
   - Validation performance
   - Number of pseudo-labels used per batch

2. **Common Issues**:
   - Overfitting to pseudo-labels
   - Confidence threshold too low/high
   - Ramp-up too fast/slow
   - Class imbalance in pseudo-labels

## Future Improvements

1. **Advanced Techniques**:
   - MixMatch-style augmentation
   - Temperature scaling
   - Class-balanced pseudo-labeling
   - Dynamic confidence thresholds

2. **Architecture Improvements**:
   - Dual student networks
   - Mean teacher approach
   - Consistency regularization

## References

1. Lee, D. H. (2013). Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks.
2. Sohn, K., et al. (2020). FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence.
3. Berthelot, D., et al. (2019). MixMatch: A Holistic Approach to Semi-Supervised Learning. 