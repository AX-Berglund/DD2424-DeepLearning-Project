# Transfer Learning Project for DD2424

This repository contains the implementation for Project 1 of the DD2424 course. The project explores transfer learning with pre-trained convolutional neural networks on the Oxford-IIIT Pet Dataset.

## Project Structure

```
DD2424-DeepLearning-Project/
├── config/                     # Configuration files
│   ├── binary_config.yaml      # Binary classification (dog vs cat)
│   └── multiclass_config.yaml  # Multi-class classification (37 breeds)
├── data/                       # Data directory
├── logs/                       # Training logs
├── models/                     # Saved models
├── notebooks/                  # Jupyter notebooks
├── scripts/                    # Utility scripts
├── src/                        # Source code
│   ├── data/                   # Data loading
│   ├── models/                 # Model definitions
│   ├── trainers/               # Training code
│   └── utils/                  # Utility functions
├── main.py                     # Main execution script
└── requirements.txt            # Project dependencies
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/AX-Berglund/DD2424-DeepLearning-Project.git
cd DD2424-DeepLearning-Project
```

### 2. Set up a virtual environment

```bash
python -m venv deep-env
source deep-env/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

```bash
python scripts/download_data.py
```

### 5. Prepare the dataset

```bash
python scripts/prepare_data.py
```

This will:
- Download the Oxford-IIIT Pet Dataset
- Create train/val/test splits
- Organize images into folders for binary and multi-class classification
- Create an imbalanced dataset for experiments

## Running Experiments

### Basic Binary Classification (Dog vs Cat)

```bash
python main.py --config config/binary_config.yaml
```

This will train a model on the binary classification task (dog vs cat) using the configuration specified in `binary_config.yaml`.

### Multi-class Classification (37 Breeds)

```bash
python main.py --config config/multiclass_config.yaml
```

This will train a model on the multi-class classification task (37 breeds) using the configuration specified in `multiclass_config.yaml`.

### Specifying Fine-tuning Strategy

You can override the fine-tuning strategy specified in the config file:

```bash
# Only fine-tune the last layer
python main.py --config config/multiclass_config.yaml --strategy last_layer

# Fine-tune multiple layers simultaneously
python main.py --config config/multiclass_config.yaml --strategy multi_layer

# Gradually unfreeze layers during training
python main.py --config config/multiclass_config.yaml --strategy gradual_unfreeze

# Only fine-tune batch normalization parameters
python main.py --config config/multiclass_config.yaml --strategy batch_norm_only
```

### Evaluating a Trained Model

```bash
python main.py --config config/multiclass_config.yaml --eval --checkpoint models/multiclass/best_model.pt
```

## Experiment Configuration

Configuration files are in YAML format and define all aspects of the experiment:

- `experiment_name`: Name of the experiment
- `seed`: Random seed for reproducibility
- `device`: Device to use ('cuda' or 'cpu')
- `data`: Dataset configuration
  - `dataset_path`: Path to the dataset
  - `image_size`: Size of input images
  - `batch_size`: Batch size
  - `augmentation`: Data augmentation settings
- `model`: Model configuration
  - `architecture`: Model architecture (resnet18, resnet34, etc.)
  - `pretrained`: Whether to use pre-trained weights
  - `num_classes`: Number of output classes
  - `lora`: Low-Rank Adaptation configuration (optional)
  - `masked_finetuning`: Masked fine-tuning configuration (optional)
- `training`: Training configuration
  - `strategy`: Fine-tuning strategy
  - `unfreeze_layers`: Number of layers to unfreeze (for multi_layer strategy)
  - `gradual_unfreeze`: Settings for gradual unfreezing
  - `optimizer`: Optimizer to use (adam, adamw, sgd)
  - `learning_rate`: Base learning rate
  - `weight_decay`: L2 regularization strength
  - `num_epochs`: Maximum number of epochs
  - `early_stopping_patience`: Patience for early stopping
  - `lr_scheduler`: Learning rate scheduler settings
  - `class_imbalance`: Class imbalance handling settings
- `logging`: Logging configuration
  - `log_interval`: Batch logging interval
  - `tensorboard`: Whether to use TensorBoard
  - `wandb`: Whether to use Weights & Biases
  - `save_model`: Whether to save model checkpoints
  - `checkpoint_dir`: Directory to save checkpoints
- `evaluation`: Evaluation settings
  - `evaluate_every`: Evaluate every N epochs
  - `metrics`: List of metrics to compute