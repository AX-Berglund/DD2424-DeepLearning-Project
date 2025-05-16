#!/usr/bin/env python
"""
Dataset classes for the Oxford-IIIT Pet Dataset.
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import numpy as np
import torchvision.transforms as transforms


class PetDataset(Dataset):
    """Base dataset class for the Oxford-IIIT Pet Dataset."""
    
    def __init__(self, root_dir, split='train', task='binary', transform=None, percentage_labeled=None):
        """
        Initialize the dataset.
        
        Args:
            root_dir (str): Root directory of the processed dataset.
            split (str): Data split to use ('train', 'val', or 'test').
            task (str): Task type ('binary' or 'multiclass').
            transform (callable, optional): Optional transform to be applied to the images.
            percentage_labeled (int, optional): Percentage of labeled data to use (1, 10, 50, or 100).
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.task = task
        self.transform = transform
        
        # Construct the data directory path
        if percentage_labeled is not None:
            percentage_dir = f"{percentage_labeled}_percent_labeled"
            self.data_dir = self.root_dir / percentage_dir / 'labeled' / split
        else:
            self.data_dir = self.root_dir / split
        
        # Get class folders
        self.classes = sorted([d for d in os.listdir(self.data_dir) 
                             if os.path.isdir(os.path.join(self.data_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Get all image paths and labels
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))
    
    def __len__(self):
        """Return the total number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample.
            
        Returns:
            tuple: (image, label) where label is the class index.
        """
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
        
        return image, label


class BinaryPetDataset(PetDataset):
    """Dataset for binary classification (cat vs dog)."""
    
    def __init__(self, root_dir, split='train', transform=None, percentage_labeled=None):
        super().__init__(root_dir, split, 'binary', transform, percentage_labeled)
        
        # Ensure classes is always ['cat', 'dog'] for consistency
        if self.classes != ['cat', 'dog']:
            # If folder names are different, update class mapping
            cat_idx = -1
            dog_idx = -1
            
            for cls_name, idx in self.class_to_idx.items():
                if 'cat' in cls_name.lower():
                    cat_idx = idx
                elif 'dog' in cls_name.lower():
                    dog_idx = idx
            
            # Update samples with correct binary labels
            if cat_idx != -1 and dog_idx != -1:
                self.samples = [(path, 0 if label == cat_idx else 1) 
                                for path, label in self.samples]
                
                # Update class mapping
                self.classes = ['cat', 'dog']
                self.class_to_idx = {'cat': 0, 'dog': 1}


class MultiClassPetDataset(PetDataset):
    """Dataset for multi-class classification (37 breeds)."""
    
    def __init__(self, root_dir, split='train', transform=None, percentage_labeled=None):
        task = 'imbalanced' if percentage_labeled == 'imbalanced' else 'multiclass'
        super().__init__(root_dir, split, task, transform, percentage_labeled)


class SemiSupervisedPetDataset(PetDataset):
    """Dataset for semi-supervised learning with pseudo-labeling."""
    
    def __init__(self, root_dir, split='train', transform=None, task='multiclass', percentage_labeled=None):
        """
        Initialize the dataset.
        
        Args:
            root_dir (str): Root directory of the processed dataset.
            split (str): Data split to use ('train', 'val', or 'test').
            transform (callable, optional): Optional transform to be applied to the images.
            task (str): Task type ('binary' or 'multiclass').
            percentage_labeled (int, optional): Percentage of labeled data to use (1, 10, 50, or 100).
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.task = task
        
        # Construct the data directory path
        percentage_dir = f"{percentage_labeled}_percent_labeled"
        if split == 'unlabeled':
            self.data_dir = self.root_dir / percentage_dir / 'unlabeled' / 'images'  # Added 'images' subdirectory
        else:
            self.data_dir = self.root_dir / percentage_dir / 'labeled' / split
        
        # Get class folders from the appropriate directory
        if task == "binary":
            self.classes = ['cat', 'dog']
            self.class_to_idx = {'cat': 0, 'dog': 1}
        else:
            # For multiclass, get classes from the labeled directory
            labeled_dir = self.root_dir / percentage_dir / 'labeled' / 'train'  # Always use train split for class mapping
            if not labeled_dir.exists():
                raise ValueError(f"Labeled directory not found: {labeled_dir}")
            self.classes = sorted([d for d in os.listdir(labeled_dir) 
                                 if os.path.isdir(os.path.join(labeled_dir, d))])
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Get all image paths and labels
        self.samples = []
        if split == 'unlabeled':
            # For unlabeled data, we don't have class directories
            if not self.data_dir.exists():
                raise ValueError(f"Unlabeled data directory not found: {self.data_dir}")
            for img_name in os.listdir(self.data_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(self.data_dir, img_name)
                    self.samples.append((img_path, -1))  # -1 is the dummy label for unlabeled data
        else:
            # For labeled data, we have class directories
            if not self.data_dir.exists():
                raise ValueError(f"Labeled data directory not found: {self.data_dir}")
            for class_name in self.classes:
                class_dir = os.path.join(self.data_dir, class_name)
                if os.path.exists(class_dir):
                    class_idx = self.class_to_idx[class_name]
                    
                    for img_name in os.listdir(class_dir):
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(class_dir, img_name)
                            self.samples.append((img_path, class_idx))


class UnlabeledPetDataset(Dataset):
    """Dataset for unlabeled data in semi-supervised learning."""
    
    def __init__(self, root_dir, transform=None):
        """
        Initialize the dataset.
        
        Args:
            root_dir (str): Root directory of the processed dataset.
            transform (callable, optional): Optional transform to be applied to the images.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # For unlabeled data, we use the unlabeled directory
        self.data_dir = self.root_dir / "unlabeled" / "images"
        
        # Get all image paths
        self.samples = []
        for img_name in os.listdir(self.data_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(self.data_dir, img_name)
                self.samples.append(img_path)
    
    def __len__(self):
        """Return the total number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample.
            
        Returns:
            tuple: (image, -1) where -1 is a dummy label for unlabeled data.
        """
        img_path = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
        
        return image, -1  # Return -1 as dummy label


class MixupDataset(Dataset):
    """Mixup data augmentation wrapper for a dataset."""
    
    def __init__(self, dataset, alpha=0.2):
        self.dataset = dataset
        self.alpha = alpha
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Get original sample
        img1, label1 = self.dataset[idx]
        
        # Randomly select another sample
        idx2 = np.random.randint(len(self.dataset))
        img2, label2 = self.dataset[idx2]
        
        # Generate lambda from a beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Mix images
        mixed_img = lam * img1 + (1 - lam) * img2
        
        # Return mixed image and both labels with mixing coefficient
        return mixed_img, label1, label2, lam


class CutMixDataset(Dataset):
    """CutMix data augmentation wrapper for a dataset."""
    
    def __init__(self, dataset, alpha=1.0):
        self.dataset = dataset
        self.alpha = alpha
    
    def __len__(self):
        return len(self.dataset)
    
    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # Uniform sampling of center point
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        # Ensure box is within image boundaries
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2
    
    def __getitem__(self, idx):
        # Get original sample
        img1, label1 = self.dataset[idx]
        
        # Randomly select another sample
        idx2 = np.random.randint(len(self.dataset))
        img2, label2 = self.dataset[idx2]
        
        # Generate lambda from a beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Convert images to tensors if they're not already
        if not isinstance(img1, torch.Tensor):
            img1 = transforms.ToTensor()(img1)
        if not isinstance(img2, torch.Tensor):
            img2 = transforms.ToTensor()(img2)
        
        # Add batch dimension
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
        
        # Get bounding box coordinates
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(img1.size(), lam)
        
        # Apply CutMix
        img1[:, :, bbx1:bbx2, bby1:bby2] = img2[:, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda to account for the ratio of pixels replaced
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img1.size()[2] * img1.size()[3]))
        
        # Remove batch dimension
        img1 = img1.squeeze(0)
        
        # Return mixed image and both labels with mixing coefficient
        return img1, label1, label2, lam


def get_transforms(config, split='train'):
    """
    Get transforms for a specific split.
    
    Args:
        config (dict): Configuration dictionary.
        split (str): Data split ('train', 'val', or 'test').
        
    Returns:
        torchvision.transforms.Compose: Transforms for the specified split.
    """
    # Base transforms
    base_transforms = [
        transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    
    # Add augmentation for training
    if split == 'train' and config['data']['augmentation']['use_augmentation']:
        augmentation_transforms = []
        
        # Add augmentations based on config
        aug_config = config['data']['augmentation']
        
        if aug_config.get('horizontal_flip', False):
            augmentation_transforms.append(transforms.RandomHorizontalFlip())
        
        if aug_config.get('rotation_angle', 0) > 0:
            augmentation_transforms.append(
                transforms.RandomRotation(aug_config['rotation_angle'])
            )
        
        if aug_config.get('random_crop', False):
            augmentation_transforms.append(
                transforms.RandomResizedCrop(
                    config['data']['image_size'],
                    scale=(0.8, 1.0),
                    ratio=(0.9, 1.1)
                )
            )
        
        if aug_config.get('color_jitter', False):
            augmentation_transforms.append(
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                )
            )
        
        return transforms.Compose(augmentation_transforms + base_transforms)
    
    return transforms.Compose(base_transforms)


def create_data_loaders(config):
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        config (dict): Configuration dictionary containing data settings.
        
    Returns:
        dict: Dictionary containing data loaders for each split.
    """
    # Get transforms for each split
    transforms = {
        split: get_transforms(config, split)
        for split in ['train', 'val', 'test']
    }
    
    # Create datasets
    if config['data']['task_type'] == 'binary':
        datasets = {
            split: BinaryPetDataset(
                root_dir=config['data']['dataset_path'],
                split=split,
                transform=transforms[split],
                percentage_labeled=config.get('percentage_labeled')
            )
            for split in ['train', 'val', 'test']
        }
    elif config['data']['task_type'] == 'multiclass':
        datasets = {
            split: MultiClassPetDataset(
                root_dir=config['data']['dataset_path'],
                split=split,
                transform=transforms[split],
                percentage_labeled=config.get('percentage_labeled')
            )
            for split in ['train', 'val', 'test']
        }
    elif config['data']['task_type'] == 'semisupervised':
        # Create labeled dataset
        labeled_dataset = SemiSupervisedPetDataset(
            root_dir=config['data']['dataset_path'],
            split='train',
            transform=transforms['train'],
            task=config['data']['task_type'],
            percentage_labeled=config.get('percentage_labeled')
        )
        
        # Create validation and test datasets
        val_dataset = SemiSupervisedPetDataset(
            root_dir=config['data']['dataset_path'],
            split='val',
            transform=transforms['val'],
            task=config['data']['task_type'],
            percentage_labeled=config.get('percentage_labeled')
        )
        test_dataset = SemiSupervisedPetDataset(
            root_dir=config['data']['dataset_path'],
            split='test',
            transform=transforms['test'],
            task=config['data']['task_type'],
            percentage_labeled=config.get('percentage_labeled')
        )
        
        # Create unlabeled dataset if it exists
        percentage_dir = f"{config.get('percentage_labeled')}_percent_labeled"
        unlabeled_dir = Path(config['data']['dataset_path']) / percentage_dir / 'unlabeled'
        if unlabeled_dir.exists():
            unlabeled_dataset = SemiSupervisedPetDataset(
                root_dir=config['data']['dataset_path'],
                split='unlabeled',
                transform=transforms['train'],
                task=config['data']['task_type'],
                percentage_labeled=config.get('percentage_labeled')
            )
        else:
            unlabeled_dataset = None
        
        datasets = {
            'train': labeled_dataset,
            'val': val_dataset,
            'test': test_dataset,
            'unlabeled': unlabeled_dataset
        }
    else:
        raise ValueError(f"Unknown task type: {config['data']['task_type']}")
    
    # Create data loaders
    dataloaders = {
        split: torch.utils.data.DataLoader(
            dataset,
            batch_size=config['data']['batch_size'],
            shuffle=(split == 'train'),
            num_workers=config['num_workers'],
            pin_memory=True
        )
        for split, dataset in datasets.items()
        if dataset is not None
    }
    
    return dataloaders