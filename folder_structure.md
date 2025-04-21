# Project root directory
transfer_learning_project/
├── config/                   # Configuration files
│   ├── binary_config.yaml    # Configuration for binary classification
│   └── multiclass_config.yaml# Configuration for multiclass classification
├── data/                     # Data directory
│   ├── raw/                  # Raw downloaded data
│   ├── processed/            # Processed data
│   └── splits/               # Train/val/test splits
├── logs/                     # Training logs
│   ├── tensorboard/          # Tensorboard logs
│   └── run_logs/             # Text logs
├── models/                   # Saved models
│   ├── binary/               # Binary classification models
│   └── multiclass/           # Multi-class classification models
├── notebooks/                # Jupyter notebooks for exploration and visualization
├── scripts/                  # Utility scripts
│   ├── download_data.py      # Script to download the pet dataset
│   └── prepare_data.py       # Script to prepare and preprocess the data
├── src/                      # Source code
│   ├── __init__.py
│   ├── data/                 # Data loading and processing
│   │   ├── __init__.py
│   │   ├── dataset.py        # Dataset class
│   │   └── transforms.py     # Data transformations
│   ├── models/               # Model definitions
│   │   ├── __init__.py
│   │   └── transfer_model.py # Transfer learning model wrapper
│   ├── trainers/             # Training code
│   │   ├── __init__.py
│   │   ├── binary_trainer.py # Trainer for binary classification
│   │   └── multiclass_trainer.py # Trainer for multiclass classification
│   └── utils/                # Utility functions
│       ├── __init__.py
│       ├── logger.py         # Logging utilities
│       ├── metrics.py        # Evaluation metrics
│       └── visualization.py  # Visualization utilities
├── requirements.txt          # Project dependencies
├── main.py                   # Main execution script
└── README.md                 # Project documentation