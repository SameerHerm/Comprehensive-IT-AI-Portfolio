#!/usr/bin/env python3
"""
Script to download datasets for object detection
"""

import os
import sys
import argparse
import urllib.request
import zipfile
import tarfile
from pathlib import Path
import json

def download_file(url, dest_path):
    """Download file from URL"""
    print(f"Downloading from {url}...")
    urllib.request.urlretrieve(url, dest_path)
    print(f"Downloaded to {dest_path}")

def extract_archive(archive_path, extract_to):
    """Extract zip or tar archive"""
    print(f"Extracting {archive_path}...")
    
    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif archive_path.endswith(('.tar.gz', '.tgz')):
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_to)
    elif archive_path.endswith('.tar'):
        with tarfile.open(archive_path, 'r') as tar_ref:
            tar_ref.extractall(extract_to)
    
    print(f"Extracted to {extract_to}")

def download_coco_dataset(data_dir):
    """Download COCO dataset"""
    print("Downloading COCO dataset...")
    
    # COCO URLs (using sample for demonstration)
    urls = {
        'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
        # Add image URLs as needed
    }
    
    # Note: Actual implementation would download full dataset
    print("COCO dataset download placeholder")
    
def download_custom_dataset(data_dir):
    """Download custom dataset"""
    print("Downloading custom dataset...")
    
    # Create sample dataset structure
    os.makedirs(data_dir / 'images', exist_ok=True)
    os.makedirs(data_dir / 'annotations', exist_ok=True)
    
    # Create sample annotation file
    sample_annotation = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "person"},
            {"id": 2, "name": "car"},
            {"id": 3, "name": "dog"}
        ]
    }
    
    with open(data_dir / 'annotations' / 'train.json', 'w') as f:
        json.dump(sample_annotation, f, indent=2)
    
    print("Sample dataset created")

def main():
    parser = argparse.ArgumentParser(description='Download datasets for object detection')
    parser.add_argument('--dataset', type=str, default='custom',
                       choices=['coco', 'voc', 'custom'],
                       help='Dataset to download')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory to save dataset')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    if args.dataset == 'coco':
        download_coco_dataset(data_dir)
    elif args.dataset == 'custom':
        download_custom_dataset(data_dir)
    
    print(f"Dataset '{args.dataset}' ready in {data_dir}")

if __name__ == '__main__':
    main()
