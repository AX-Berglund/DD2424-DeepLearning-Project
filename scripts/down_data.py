#!/usr/bin/env python
"""
Script to download and extract the Oxford-IIIT Pet Dataset.
"""

import os
import shutil
import urllib.request
import tarfile
from pathlib import Path
import argparse


# URLs for the Oxford-IIIT Pet Dataset
IMAGES_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
ANNOTATIONS_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"

# Paths
DEFAULT_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")


def download_url(url, output_path):
    """Download a file from a URL."""
    print(f"Downloading {url.split('/')[-1]}...")
    urllib.request.urlretrieve(url, filename=output_path)
    print(f"Download complete: {output_path}")


def extract_tar(tar_path, extract_path):
    """Extract a tar file to the specified directory."""
    print(f"Extracting {tar_path} to {extract_path}...")
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=extract_path)
    print(f"Extraction complete.")


def ensure_directory_exists(directory):
    """Create directory if it doesn't exist."""
    if not directory.exists():
        directory.mkdir(parents=True)
        print(f"Created directory: {directory}")


def download_and_extract_dataset(data_dir=DEFAULT_DATA_DIR):
    """Download and extract the Oxford-IIIT Pet Dataset."""
    data_dir = Path(data_dir)
    ensure_directory_exists(data_dir)
    
    # Download images
    images_tar_path = data_dir / "images.tar.gz"
    if not images_tar_path.exists():
        print(f"Downloading images dataset...")
        download_url(IMAGES_URL, images_tar_path)
    else:
        print(f"Images dataset already downloaded: {images_tar_path}")
    
    # Download annotations
    annotations_tar_path = data_dir / "annotations.tar.gz"
    if not annotations_tar_path.exists():
        print(f"Downloading annotations dataset...")
        download_url(ANNOTATIONS_URL, annotations_tar_path)
    else:
        print(f"Annotations dataset already downloaded: {annotations_tar_path}")
    
    # Extract images
    extract_tar(images_tar_path, data_dir)
    
    # Extract annotations
    extract_tar(annotations_tar_path, data_dir)
    
    print(f"Dataset downloaded and extracted to {data_dir}")


def cleanup_files(data_dir=DEFAULT_DATA_DIR, keep_tars=False):
    """Remove tar files after extraction to save space."""
    data_dir = Path(data_dir)
    
    if not keep_tars:
        for file_name in ["images.tar.gz", "annotations.tar.gz"]:
            file_path = data_dir / file_name
            if file_path.exists():
                file_path.unlink()
                print(f"Removed: {file_path}")


def main():
    parser = argparse.ArgumentParser(description="Download and extract the Oxford-IIIT Pet Dataset")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help="Directory to store the dataset")
    parser.add_argument("--keep-tars", action="store_true", help="Keep tar files after extraction")
    args = parser.parse_args()
    
    download_and_extract_dataset(args.data_dir)
    cleanup_files(args.data_dir, args.keep_tars)
    
    print("\nDataset ready. Next, run prepare_data.py to process the dataset.")


if __name__ == "__main__":
    main()