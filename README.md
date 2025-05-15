# Transfer Learning Project for DD2424

This repository contains the implementation for Project 1 of the DD2424 course. The project explores transfer learning with pre-trained convolutional neural networks on the Oxford-IIIT Pet Dataset, including semi-supervised learning with pseudo-labeling.

## Project Structure

```
DD2424-DeepLearning-Project/
├── config/                     # Configuration files
│   ├── binary_config.yaml      # Binary classification (dog vs cat)
│   ├── multiclass_config.yaml  # Multi-class classification (37 breeds)
│   ├── binary_pseudo_labeling_config.yaml    # Binary classification with pseudo-labeling
│   └── multiclass_pseudo_labeling_config.yaml # Multi-class classification with pseudo-labeling
├── data/                       # Data directory
├── logs/                       # Training logs
├── checkpoints/                # Model checkpoints
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

#### Using venv (Python's built-in virtual environment)

```bash
# On Unix/macOS
python -m venv deep-learn
source deep-learn/bin/activate

# On Windows
python -m venv deep-learn
deep-learn\Scripts\activate
```

#### Using Conda

```bash
# Create a new conda environment
conda create -n deep-learn python=3.8
conda activate deep-learn
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download and prepare the dataset

```bash
# Download the dataset
python scripts/download_data.py

# Prepare the dataset for supervised learning
python scripts/prepare_data.py

# Prepare the dataset for semi-supervised learning
python scripts/prepare_semisupervised_binary_data.py # For binary classification
python scripts/prepare_semisupervised_multiclass_data.py  # For multi-class classification
```

The data preparation scripts will:
- Download the Oxford-IIIT Pet Dataset
- Create train/val/test splits
- Organize images into folders for binary and multi-class classification
- Create labeled and unlabeled splits for semi-supervised learning

## Running Experiments

### Supervised Learning

#### Basic Binary Classification (Dog vs Cat)

```bash
python main.py --config config/binary_config.yaml
```

#### Multi-class Classification (37 Breeds)

```bash
python main.py --config config/multiclass_config.yaml
```

### Semi-supervised Learning with Pseudo-labeling

#### Binary Classification with Pseudo-labeling

```bash
python main.py --config config/binary_pseudo_labeling_config.yaml --pseudo-labeling --percentage 50
```

#### Multi-class Classification with Pseudo-labeling

```bash
python main.py --config config/multiclass_pseudo_labeling_config.yaml --pseudo-labeling --percentage 50
```

The `--percentage` flag specifies [100, 50, 10, 1] what percentage of the training data should be labeled (the rest will be used as unlabeled data).

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
python main.py --config config/multiclass_config.yaml --eval --checkpoint checkpoints/multiclass/best_model.pt
```

## Experiment Configuration

Configuration files are in YAML format and define all aspects of the experiment:

- `experiment_name`: Name of the experiment
- `seed`: Random seed for reproducibility
- `device`: Device to use ('cuda' or 'cpu')
- `data`: Dataset configuration
  - `task`: Task type ('supervised' or 'semisupervised')
  - `task_type`: Classification type ('binary' or 'multiclass')
  - `dataset_path`: Path to the dataset
  - `image_size`: Size of input images
  - `batch_size`: Batch size
  - `percentage_labeled`: Percentage of labeled data (for semi-supervised)
  - `augmentation`: Data augmentation settings
- `model`: Model configuration
  - `architecture`: Model architecture (resnet18, resnet34, etc.)
  - `pretrained`: Whether to use pre-trained weights
  - `num_classes`: Number of output classes
  - `dropout_rate`: Dropout rate for classification head
- `training`: Training configuration
  - `strategy`: Fine-tuning strategy
  - `unfreeze_layers`: Number of layers to unfreeze
  - `optimizer`: Optimizer to use (adam, adamw, sgd)
  - `learning_rate`: Base learning rate
  - `weight_decay`: L2 regularization strength
  - `num_epochs`: Maximum number of epochs
  - `early_stopping_patience`: Patience for early stopping
  - `pseudo_labeling`: Pseudo-labeling settings
    - `confidence_threshold`: Minimum confidence for pseudo-labels
    - `warmup_epochs`: Number of epochs before starting pseudo-labeling
    - `rampup_epochs`: Number of epochs to ramp up pseudo-label weight
    - `alpha`: Maximum weight for pseudo-labeled loss
- `logging`: Logging configuration
  - `log_interval`: Batch logging interval
  - `tensorboard`: Whether to use TensorBoard
  - `wandb`: Whether to use Weights & Biases
  - `save_model`: Whether to save model checkpoints
  - `checkpoint_dir`: Directory to save checkpoints
- `evaluation`: Evaluation settings
  - `evaluate_every`: Evaluate every N epochs
  - `metrics`: List of metrics to compute

## Results and Analysis

The training results and analysis can be found in the following notebooks:
- `notebooks/01_binary_result_analysis.ipynb`: Analysis of binary classification results
- `notebooks/02_multiclass_result_analysis.ipynb`: Analysis of multi-class classification results
- `notebooks/03_pseudo_labeling_analysis.ipynb`: Analysis of semi-supervised learning results