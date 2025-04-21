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
    
    def __init__(self, root_dir, split='train', task='binary', transform=None):
        """
        Initialize the dataset.
        
        Args:
            root_dir (str): Root directory of the processed dataset.
            split (str): Data split to use ('train', 'val', or 'test').
            task (str): Task type ('binary' or 'multiclass').
            transform (callable, optional): Optional transform to be applied to the images.
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.task = task
        self.transform = transform
        
        # Determine the dataset directory based on task
        self.data_dir = self.root_dir / task / split
        
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
    
    def __init__(self, root_dir, split='train', transform=None):
        super().__init__(root_dir, split, 'binary', transform)
        
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
    
    def __init__(self, root_dir, split='train', transform=None, imbalanced=False):
        task = 'imbalanced' if imbalanced else 'multiclass'
        super().__init__(root_dir, split, task, transform)


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


def get_transforms(config, split):
    """
    Get transforms for a specific split based on the configuration.
    
    Args:
        config (dict): Configuration dictionary.
        split (str): Data split ('train', 'val', or 'test').
        
    Returns:
        transforms.Compose: Composition of transforms to apply.
    """
    mean = [0.485, 0.456, 0.406]  # ImageNet mean
    std = [0.229, 0.224, 0.225]   # ImageNet std
    
    if split == 'train' and config['data']['augmentation']['use_augmentation']:
        # Training transforms with augmentation
        transform_list = [
            transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
        ]
        
        # Add augmentations based on config
        aug_config = config['data']['augmentation']
        
        if aug_config.get('horizontal_flip', False):
            transform_list.append(transforms.RandomHorizontalFlip())
        
        if aug_config.get('rotation_angle', 0) > 0:
            transform_list.append(
                transforms.RandomRotation(aug_config['rotation_angle'])
            )
        
        if aug_config.get('random_crop', False):
            transform_list.extend([
                transforms.RandomResizedCrop(
                    config['data']['image_size'],
                    scale=(0.8, 1.0),
                    ratio=(0.9, 1.1)
                )
            ])
        
        if aug_config.get('color_jitter', False):
            transform_list.append(
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                )
            )
        
        # Add standard transforms
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
        # Add random erasing if specified
        if aug_config.get('random_erasing', False):
            transform_list.append(
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.33))
            )
    else:
        # Validation/Test transforms (no augmentation)
        transform_list = [
            transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
    
    return transforms.Compose(transform_list)


def create_data_loaders(config):
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        config (dict): Configuration dictionary.
        
    Returns:
        dict: Dictionary of data loaders for each split.
    """
    data_loaders = {}
    
    # Determine dataset class based on task
    if config['model']['num_classes'] == 2:
        dataset_class = BinaryPetDataset
    else:
        dataset_class = MultiClassPetDataset
    
    # Create transforms for each split
    transforms_dict = {
        split: get_transforms(config, split)
        for split in ['train', 'val', 'test']
    }
    
    # Check if using imbalanced dataset
    use_imbalanced = False
    if 'class_imbalance' in config['training'] and config['training']['class_imbalance']['enabled']:
        use_imbalanced = True
    
    # Create datasets
    datasets = {}
    for split in ['train', 'val', 'test']:
        datasets[split] = dataset_class(
            root_dir=config['data']['processed_path'],
            split=split,
            transform=transforms_dict[split],
            imbalanced=use_imbalanced if hasattr(dataset_class, 'imbalanced') else False
        )
    
    # Apply mixup or cutmix if specified (only for training)
    if (config['data']['augmentation'].get('mixup', False) and 
        split == 'train'):
        datasets['train'] = MixupDataset(datasets['train'])
    elif (config['data']['augmentation'].get('cutmix', False) and 
          split == 'train'):
        datasets['train'] = CutMixDataset(datasets['train'])
    
    # Create data loaders
    for split, dataset in datasets.items():
        shuffle = (split == 'train')
        batch_size = config['data']['batch_size']
        
        data_loaders[split] = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=config['num_workers'],
            pin_memory=True
        )
    
    return data_loaders