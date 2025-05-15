import os
import shutil
import random
from pathlib import Path
import json

# Define paths
RAW_DATA_DIR = Path("/Users/axhome/AX/KTH/Courses/DD2424-DeepLearning-Project/data/raw")
PROCESSED_DATA_DIR = Path("/Users/axhome/AX/KTH/Courses/DD2424-DeepLearning-Project/data/processed")
SEMISUPERVISED_DIR = PROCESSED_DATA_DIR / "semisupervised" / "multiclass"

# Define percentages for labeled data
PERCENTAGES = [100, 50, 10, 1]

# Define train/val/test split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Define breed lists
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

# Combine all breeds
ALL_BREEDS = CAT_BREEDS + DOG_BREEDS

def get_breed_from_filename(img_name):
    """Extract breed information from filename."""
    # Extract breed name from filename (format: breed_123.jpg)
    breed = img_name.split('_')[0]
    
    # Check if breed is in our list
    if breed in ALL_BREEDS:
        return breed
    return None

def create_directory_structure():
    """Create the directory structure for multiclass semi-supervised learning."""
    # Create main multiclass directory
    SEMISUPERVISED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for each percentage
    for percentage in PERCENTAGES:
        percentage_dir = SEMISUPERVISED_DIR / f"{percentage}_percent_labeled"
        percentage_dir.mkdir(exist_ok=True)
        
        # Create labeled and unlabeled subdirectories
        for subset in ["labeled", "unlabeled"]:
            subset_dir = percentage_dir / subset
            subset_dir.mkdir(exist_ok=True)
            
            # For labeled data, create train/val/test splits
            if subset == "labeled":
                for split in ["train", "val", "test"]:
                    split_dir = subset_dir / split
                    split_dir.mkdir(exist_ok=True)
            else:
                # For unlabeled data, just create images directory
                (subset_dir / "images").mkdir(exist_ok=True)

def split_indices(total_size, train_ratio, val_ratio, test_ratio):
    """Split indices into train, validation, and test sets."""
    indices = list(range(total_size))
    random.shuffle(indices)
    
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    return train_indices, val_indices, test_indices

def process_data():
    """Process and split the data according to different percentages."""
    # Get all image files
    image_files = list((RAW_DATA_DIR / "images").glob("*.jpg"))
    image_files.sort()
    
    # Create mapping of images to their breeds
    image_to_breed = {}
    for img_file in image_files:
        breed = get_breed_from_filename(img_file.name)
        if breed:  # Only include images we can classify
            image_to_breed[img_file] = breed
    
    # Process for each percentage
    for percentage in PERCENTAGES:
        percentage_dir = SEMISUPERVISED_DIR / f"{percentage}_percent_labeled"
        
        # Calculate number of labeled samples per breed
        breeds = set(image_to_breed.values())
        num_labeled_per_breed = int(len(image_to_breed) * percentage / 100 / len(breeds))
        
        # For each breed, select labeled samples
        labeled_images = []
        for breed in breeds:
            breed_images = [img for img, b in image_to_breed.items() if b == breed]
            if len(breed_images) > num_labeled_per_breed:
                labeled_images.extend(random.sample(breed_images, num_labeled_per_breed))
            else:
                labeled_images.extend(breed_images)
        
        # Split labeled images into train/val/test
        train_indices, val_indices, test_indices = split_indices(
            len(labeled_images), TRAIN_RATIO, VAL_RATIO, TEST_RATIO
        )
        
        # Create mapping of labeled images to their splits
        labeled_split_map = {}
        for idx in train_indices:
            labeled_split_map[labeled_images[idx]] = "train"
        for idx in val_indices:
            labeled_split_map[labeled_images[idx]] = "val"
        for idx in test_indices:
            labeled_split_map[labeled_images[idx]] = "test"
        
        # Process each image
        for img_file in image_files:
            breed = image_to_breed.get(img_file)
            if breed and img_file in labeled_split_map:
                # Determine which split this labeled image belongs to
                split = labeled_split_map[img_file]
                
                # Create breed directory if it doesn't exist
                breed_dir = percentage_dir / "labeled" / split / breed
                breed_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy to appropriate labeled split directory
                shutil.copy2(img_file, breed_dir / img_file.name)
            elif breed:  # Only copy to unlabeled if we can classify it
                # Copy to unlabeled directory
                unlabeled_dir = percentage_dir / "unlabeled" / "images"
                unlabeled_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img_file, unlabeled_dir / img_file.name)

def main():
    print("Creating directory structure for multiclass classification...")
    create_directory_structure()
    
    print("Processing and splitting multiclass data...")
    process_data()
    
    print("Done! Multiclass data has been organized for semi-supervised learning.")

if __name__ == "__main__":
    main() 