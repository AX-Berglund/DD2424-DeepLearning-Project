#!/usr/bin/env python
"""
Custom transformations for data augmentation.
"""

import random
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image, ImageFilter


class RandomGaussianBlur:
    """Apply Gaussian blur with random sigma."""
    
    def __init__(self, radius_min=0.1, radius_max=2.0, p=0.5):
        self.radius_min = radius_min
        self.radius_max = radius_max
        self.p = p
    
    def __call__(self, img):
        if random.random() < self.p:
            radius = random.uniform(self.radius_min, self.radius_max)
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img


class RandomGrayscale:
    """Convert image to grayscale with probability p."""
    
    def __init__(self, p=0.2):
        self.p = p
    
    def __call__(self, img):
        if random.random() < self.p:
            return F.to_grayscale(img, num_output_channels=3)
        return img


class RandomAutoContrast:
    """Apply auto contrast with probability p."""
    
    def __init__(self, p=0.2):
        self.p = p
    
    def __call__(self, img):
        if random.random() < self.p:
            return F.autocontrast(img)
        return img


class RandomAdjustSharpness:
    """Adjust sharpness with random factor."""
    
    def __init__(self, sharpness_min=0.1, sharpness_max=2.0, p=0.2):
        self.sharpness_min = sharpness_min
        self.sharpness_max = sharpness_max
        self.p = p
    
    def __call__(self, img):
        if random.random() < self.p:
            factor = random.uniform(self.sharpness_min, self.sharpness_max)
            return F.adjust_sharpness(img, factor)
        return img


class AdvancedColorJitter:
    """More advanced color jitter transform."""
    
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.transform = transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )
    
    def __call__(self, img):
        return self.transform(img)


def get_advanced_transforms(config, split):
    """
    Get more advanced transforms based on configuration.
    
    Args:
        config (dict): Configuration dictionary.
        split (str): Data split ('train', 'val', or 'test').
        
    Returns:
        transforms.Compose: Composition of transforms.
    """
    mean = [0.485, 0.456, 0.406]  # ImageNet mean
    std = [0.229, 0.224, 0.225]   # ImageNet std
    
    if split == 'train' and config['data']['augmentation']['use_augmentation']:
        # Advanced training transforms
        aug_config = config['data']['augmentation']
        prob = aug_config.get('prob', 0.5)
        
        transform_list = [
            transforms.Resize((config['data']['image_size'] + 32, config['data']['image_size'] + 32)),
            transforms.RandomCrop(config['data']['image_size']),
        ]
        
        # Add standard augmentations
        if aug_config.get('horizontal_flip', False):
            transform_list.append(transforms.RandomHorizontalFlip(p=prob))
        
        if aug_config.get('rotation_angle', 0) > 0:
            transform_list.append(
                transforms.RandomRotation(aug_config['rotation_angle'])
            )
        
        # Add color augmentations
        if aug_config.get('color_jitter', False):
            transform_list.append(
                AdvancedColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                )
            )
        
        # Add advanced augmentations
        transform_list.extend([
            RandomGrayscale(p=0.2),
            RandomGaussianBlur(radius_min=0.1, radius_max=2.0, p=0.2),
            RandomAutoContrast(p=0.2),
            RandomAdjustSharpness(sharpness_min=0.1, sharpness_max=2.0, p=0.2),
        ])
        
        # Add standard transforms
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
        # Add random erasing if specified
        if aug_config.get('random_erasing', False):
            transform_list.append(
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.2))
            )
            
    else:
        # Validation/Test transforms (no augmentation)
        transform_list = [
            transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
    
    return transforms.Compose(transform_list)


class MixUp:
    """MixUp augmentation applied directly on batches."""
    
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, batch):
        """Apply MixUp to a batch of data."""
        images, labels = batch
        batch_size = len(images)
        
        # Generate indices for mixing
        indices = torch.randperm(batch_size)
        
        # Generate mixing parameters
        lam = np.random.beta(self.alpha, self.alpha, batch_size)
        lam = torch.from_numpy(lam).float().to(images.device)
        lam = lam.view(-1, 1, 1, 1)
        
        # Mix images
        mixed_images = lam * images + (1 - lam) * images[indices]
        
        # Get mixed labels (for one-hot encoded labels)
        lam = lam.squeeze()
        
        return mixed_images, labels, labels[indices], lam


class CutMix:
    """CutMix augmentation applied directly on batches."""
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def rand_bbox(self, size, lam):
        """Generate random bounding box."""
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # Uniform sampling of center point
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        # Bounding box coordinates
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2
    
    def __call__(self, batch):
        """Apply CutMix to a batch of data."""
        images, labels = batch
        batch_size = len(images)
        
        # Generate indices for mixing
        indices = torch.randperm(batch_size)
        
        # Generate mixing parameters
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Get bounding box coordinates
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(images.size(), lam)
        
        # Apply CutMix
        images[:, :, bbx1:bbx2, bby1:bby2] = images[indices, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda to account for the ratio of pixels replaced
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[2] * images.size()[3]))
        
        return images, labels, labels[indices], lam


def get_batch_transforms(config):
    """
    Get transforms that operate on batches of data.
    
    Args:
        config (dict): Configuration dictionary.
        
    Returns:
        callable or None: Batch transform function or None if not used.
    """
    aug_config = config['data']['augmentation']
    
    if aug_config.get('mixup', False):
        return MixUp(alpha=0.2)
    elif aug_config.get('cutmix', False):
        return CutMix(alpha=1.0)
    
    return None


class AlbumentationsTransforms:
    """
    Wrapper for Albumentations transforms.
    Only used if advanced augmentations are needed.
    """
    
    def __init__(self, transform):
        """
        Initialize with an albumentations transform.
        
        Args:
            transform: Albumentations transform pipeline.
        """
        self.transform = transform
    
    def __call__(self, img):
        """
        Apply the transform to the image.
        
        Args:
            img (PIL.Image): Input image.
            
        Returns:
            torch.Tensor: Transformed image.
        """
        # Convert PIL Image to numpy array
        img_np = np.array(img)
        
        # Apply albumentations transforms
        transformed = self.transform(image=img_np)
        img_transformed = transformed["image"]
        
        # Convert back to tensor
        return transforms.ToTensor()(Image.fromarray(img_transformed))


def get_albumentations_transforms(config, split):
    """
    Get transforms using the Albumentations library.
    
    Args:
        config (dict): Configuration dictionary.
        split (str): Data split ('train', 'val', or 'test').
        
    Returns:
        transforms.Compose: Composition of transforms.
    """
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
    except ImportError:
        print("Albumentations not installed. Falling back to torchvision transforms.")
        return get_advanced_transforms(config, split)
    
    mean = [0.485, 0.456, 0.406]  # ImageNet mean
    std = [0.229, 0.224, 0.225]   # ImageNet std
    img_size = config['data']['image_size']
    
    if split == 'train' and config['data']['augmentation']['use_augmentation']:
        # Advanced training transforms with Albumentations
        aug_config = config['data']['augmentation']
        prob = aug_config.get('prob', 0.5)
        
        transform = A.Compose([
            A.Resize(img_size + 32, img_size + 32),
            A.RandomCrop(img_size, img_size),
            A.HorizontalFlip(p=prob if aug_config.get('horizontal_flip', False) else 0),
            A.Rotate(limit=aug_config.get('rotation_angle', 0), p=prob),
            A.OneOf([
                A.ColorJitter(
                    brightness=0.2, 
                    contrast=0.2, 
                    saturation=0.2, 
                    hue=0.1, 
                    p=1.0
                ),
                A.ToGray(p=1.0),
            ], p=prob if aug_config.get('color_jitter', False) else 0),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.RandomBrightnessContrast(p=0.2),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.2),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
        
        if aug_config.get('random_erasing', False):
            # Note: Albumentations doesn't have a direct equivalent to RandomErasing,
            # so we apply it after converting to tensor using torchvision
            return transforms.Compose([
                AlbumentationsTransforms(transform),
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.2))
            ])
        
        return AlbumentationsTransforms(transform)
    else:
        # Validation/Test transforms
        transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
        
        return AlbumentationsTransforms(transform)