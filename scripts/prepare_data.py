#!/usr/bin/env python
"""
Script to prepare the Oxford-IIIT Pet Dataset for training.
This includes:
1. Creating train/val/test splits
2. Organizing images into class folders
3. Creating binary (cat vs dog) and multi-class (breed) datasets
"""

import os
import shutil
import argparse
import random
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET


# Constants
RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
ANNOTATIONS_DIR = RAW_DATA_DIR / "annotations"
IMAGES_DIR = RAW_DATA_DIR / "images"

# Define the class distributions
CAT_BREEDS = [
    "Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair", 
    "Egyptian_Mau", "Maine_Coon", "Persian", "Ragdoll", "Russian_Blue", 
    "Siamese", "Sphynx"
]

DOG_BREEDS = [
    "american_bulldog", "american_pit_bull_terrier", "basset_hound", 
    "beagle", "boxer", "chihuahua", "english_cocker_spaniel", 
    "english_setter", "german_shorthaired", "great_pyrenees", 
    "havanese", "japanese_chin", "keeshond", "leonberger", 
    "miniature_pinscher", "newfoundland", "pomeranian", "pug", 
    "saint_bernard", "samoyed", "scottish_terrier", "shiba_inu", 
    "staffordshire_bull_terrier", "wheaten_terrier", "yorkshire_terrier"
]


def ensure_directory_exists(directory):
    """Create directory if it doesn't exist."""
    if not directory.exists():
        directory.mkdir(parents=True)
        print(f"Created directory: {directory}")


def parse_annotations(annotations_file):
    """Parse the annotations file to get class and species information."""
    data = []
    
    with open(annotations_file, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                parts = line.strip().split()
                if len(parts) >= 3:
                    image_id = parts[0]
                    # Class ID is 1-indexed in the dataset
                    class_id = int(parts[1]) - 1  # Convert to 0-indexed
                    # Species: 1 = Cat, 2 = Dog
                    species = int(parts[2])
                    
                    data.append({
                        'image_id': image_id,
                        'class_id': class_id,
                        'species': 'cat' if species == 1 else 'dog'
                    })
    
    return pd.DataFrame(data)


def create_splits(df, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Create train/val/test splits, stratified by class."""
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Create empty dataframes for each split
    train_df = pd.DataFrame(columns=df.columns)
    val_df = pd.DataFrame(columns=df.columns)
    test_df = pd.DataFrame(columns=df.columns)
    
    # Group by class_id
    for class_id, group in df.groupby('class_id'):
        # Shuffle the group
        group = group.sample(frac=1, random_state=seed).reset_index(drop=True)
        
        # Calculate split indices
        n = len(group)
        train_end = int(train_ratio * n)
        val_end = train_end + int(val_ratio * n)
        
        # Split the data
        train_df = pd.concat([train_df, group[:train_end]])
        val_df = pd.concat([val_df, group[train_end:val_end]])
        test_df = pd.concat([test_df, group[val_end:]])
    
    return train_df, val_df, test_df


def organize_binary_dataset(df, output_dir):
    """Organize images into cat/dog folders for binary classification."""
    binary_dir = output_dir / "binary"
    ensure_directory_exists(binary_dir)
    
    # Create directories for splits
    for split in ['train', 'val', 'test']:
        split_dir = binary_dir / split
        ensure_directory_exists(split_dir)
        
        # Create directories for classes
        for species in ['cat', 'dog']:
            ensure_directory_exists(split_dir / species)
    
    # Copy images to appropriate directories
    for split_name, split_df in [('train', df['train']), ('val', df['val']), ('test', df['test'])]:
        print(f"Processing {split_name} set for binary classification...")
        for _, row in tqdm(split_df.iterrows(), total=len(split_df)):
            image_id = row['image_id']
            species = row['species']
            
            # Source and destination paths
            src_path = IMAGES_DIR / f"{image_id}.jpg"
            dst_path = binary_dir / split_name / species / f"{image_id}.jpg"
            
            # Copy the image
            if src_path.exists():
                shutil.copy(src_path, dst_path)
            else:
                print(f"Warning: Image not found: {src_path}")
    
    print(f"Binary dataset created at {binary_dir}")


def organize_multiclass_dataset(df, output_dir):
    """Organize images into breed folders for multi-class classification."""
    multiclass_dir = output_dir / "multiclass"
    ensure_directory_exists(multiclass_dir)
    
    # Create directories for splits
    for split in ['train', 'val', 'test']:
        split_dir = multiclass_dir / split
        ensure_directory_exists(split_dir)
    
    # Get all class names
    all_breeds = CAT_BREEDS + DOG_BREEDS
    
    # Create class directories
    for split in ['train', 'val', 'test']:
        split_dir = multiclass_dir / split
        for breed in all_breeds:
            ensure_directory_exists(split_dir / breed)
    
    # Copy images to appropriate directories
    for split_name, split_df in [('train', df['train']), ('val', df['val']), ('test', df['test'])]:
        print(f"Processing {split_name} set for multi-class classification...")
        for _, row in tqdm(split_df.iterrows(), total=len(split_df)):
            image_id = row['image_id']
            class_id = row['class_id']
            
            # Get breed name
            breed = all_breeds[class_id]
            
            # Source and destination paths
            src_path = IMAGES_DIR / f"{image_id}.jpg"
            dst_path = multiclass_dir / split_name / breed / f"{image_id}.jpg"
            
            # Copy the image
            if src_path.exists():
                shutil.copy(src_path, dst_path)
            else:
                print(f"Warning: Image not found: {src_path}")
    
    print(f"Multi-class dataset created at {multiclass_dir}")


def create_imbalanced_dataset(df, output_dir, imbalance_ratio=0.2):
    """Create an imbalanced dataset for experimenting with class imbalance."""
    imbalanced_dir = output_dir / "imbalanced"
    ensure_directory_exists(imbalanced_dir)
    
    # Only modify the training set, keep validation and test sets balanced
    train_df = df['train'].copy()
    val_df = df['val']
    test_df = df['test']
    
    # Create imbalanced training set: keep all dog images but limit cat images
    all_breeds = CAT_BREEDS + DOG_BREEDS
    imbalanced_train_df = pd.DataFrame(columns=train_df.columns)
    
    # Group by class_id
    for class_id, group in train_df.groupby('class_id'):
        breed = all_breeds[class_id]
        
        # If it's a cat breed, keep only a fraction of the images
        if breed in CAT_BREEDS:
            group = group.sample(frac=imbalance_ratio, random_state=42).reset_index(drop=True)
        
        imbalanced_train_df = pd.concat([imbalanced_train_df, group])
    
    # Create directories for splits
    for split in ['train', 'val', 'test']:
        split_dir = imbalanced_dir / split
        ensure_directory_exists(split_dir)
        
        # Create class directories
        for breed in all_breeds:
            ensure_directory_exists(split_dir / breed)
    
    # Process and copy images
    print(f"Processing train set for imbalanced classification...")
    for _, row in tqdm(imbalanced_train_df.iterrows(), total=len(imbalanced_train_df)):
        image_id = row['image_id']
        class_id = row['class_id']
        breed = all_breeds[class_id]
        
        src_path = IMAGES_DIR / f"{image_id}.jpg"
        dst_path = imbalanced_dir / 'train' / breed / f"{image_id}.jpg"
        
        if src_path.exists():
            shutil.copy(src_path, dst_path)
    
    # Copy validation and test sets normally
    for split_name, split_df in [('val', val_df), ('test', test_df)]:
        print(f"Processing {split_name} set for imbalanced classification...")
        for _, row in tqdm(split_df.iterrows(), total=len(split_df)):
            image_id = row['image_id']
            class_id = row['class_id']
            breed = all_breeds[class_id]
            
            src_path = IMAGES_DIR / f"{image_id}.jpg"
            dst_path = imbalanced_dir / split_name / breed / f"{image_id}.jpg"
            
            if src_path.exists():
                shutil.copy(src_path, dst_path)
    
    print(f"Imbalanced dataset created at {imbalanced_dir}")
    
    # Save the class distribution for reference
    class_dist = imbalanced_train_df.groupby(['species', 'class_id']).size().reset_index(name='count')
    class_dist.to_csv(imbalanced_dir / 'class_distribution.csv', index=False)
    
    return {
        'train': imbalanced_train_df,
        'val': val_df,
        'test': test_df
    }


def save_splits_info(splits, output_dir):
    """Save information about the splits for future reference."""
    stats_dir = output_dir / "stats"
    ensure_directory_exists(stats_dir)
    
    # Save each split to CSV
    for split_name, split_df in splits.items():
        split_df.to_csv(stats_dir / f"{split_name}_split.csv", index=False)
    
    # Save class distribution for each split
    for split_name, split_df in splits.items():
        class_dist = split_df.groupby(['species', 'class_id']).size().reset_index(name='count')
        class_dist.to_csv(stats_dir / f"{split_name}_class_distribution.csv", index=False)
    
    print(f"Split statistics saved to {stats_dir}")


def main():
    parser = argparse.ArgumentParser(description="Prepare the Oxford-IIIT Pet Dataset for training")
    parser.add_argument("--data-dir", type=str, default=RAW_DATA_DIR, help="Directory with the raw dataset")
    parser.add_argument("--output-dir", type=str, default=PROCESSED_DATA_DIR, help="Directory to store processed data")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Ratio of training data")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Ratio of validation data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--imbalance-ratio", type=float, default=0.2, help="Ratio to reduce cat breeds for imbalanced dataset")
    args = parser.parse_args()
    
    # Convert paths to Path objects
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    # Ensure the output directory exists
    ensure_directory_exists(output_dir)
    
    # Parse annotations
    print("Parsing annotations...")
    annotations_file = data_dir / "annotations" / "trainval.txt"
    df = parse_annotations(annotations_file)
    
    # Create train/val/test splits
    print("Creating data splits...")
    train_df, val_df, test_df = create_splits(
        df, 
        train_ratio=args.train_ratio, 
        val_ratio=args.val_ratio, 
        seed=args.seed
    )
    
    # Organize splits into a dictionary
    splits = {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }
    
    # Save split information
    save_splits_info(splits, output_dir)
    
    # Organize binary (cat vs dog) dataset
    print("Organizing binary classification dataset...")
    organize_binary_dataset(splits, output_dir)
    
    # Organize multi-class (breed) dataset
    print("Organizing multi-class classification dataset...")
    organize_multiclass_dataset(splits, output_dir)
    
    # Create imbalanced dataset for experiments
    print("Creating imbalanced dataset...")
    imbalanced_splits = create_imbalanced_dataset(
        splits, 
        output_dir, 
        imbalance_ratio=args.imbalance_ratio
    )
    
    # Save imbalanced split information
    imbalanced_stats_dir = output_dir / "imbalanced" / "stats"
    ensure_directory_exists(imbalanced_stats_dir)
    for split_name, split_df in imbalanced_splits.items():
        split_df.to_csv(imbalanced_stats_dir / f"{split_name}_split.csv", index=False)
    
    print("\nData preparation complete!")
    print(f"Processed data is available at: {output_dir}")
    print("\nSummary:")
    print(f"  - Training samples: {len(train_df)}")
    print(f"  - Validation samples: {len(val_df)}")
    print(f"  - Test samples: {len(test_df)}")
    print(f"  - Total cat samples: {len(df[df['species'] == 'cat'])}")
    print(f"  - Total dog samples: {len(df[df['species'] == 'dog'])}")
    print(f"  - Number of classes: {df['class_id'].nunique()}")
    print("\nNext step: Train models using main.py")


if __name__ == "__main__":
    main()