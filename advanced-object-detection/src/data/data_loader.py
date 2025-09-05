import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
import os
import json
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ObjectDetectionDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        annotations_dir: str,
        annotation_format: str = "yolo",  # yolo, coco, pascal_voc
        transform=None,
        class_names: List[str] = None,
        image_size: Tuple[int, int] = (640, 640)
    ):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.annotation_format = annotation_format
        self.transform = transform
        self.class_names = class_names or []
        self.image_size = image_size
        
        self.images = []
        self.annotations = []
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Load dataset based on annotation format"""
        if self.annotation_format == "yolo":
            self._load_yolo_dataset()
        elif self.annotation_format == "coco":
            self._load_coco_dataset()
        elif self.annotation_format == "pascal_voc":
            self._load_pascal_voc_dataset()
        else:
            raise ValueError(f"Unsupported annotation format: {self.annotation_format}")
        
        logger.info(f"Loaded {len(self.images)} images with {len(self.annotations)} annotations")
    
    def _load_yolo_dataset(self):
        """Load YOLO format dataset"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        for filename in os.listdir(self.images_dir):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_path = os.path.join(self.images_dir, filename)
                annotation_path = os.path.join(
                    self.annotations_dir, 
                    os.path.splitext(filename)[0] + '.txt'
                )
                
                if os.path.exists(annotation_path):
                    self.images.append(image_path)
                    self.annotations.append(annotation_path)
    
    def _load_coco_dataset(self):
        """Load COCO format dataset"""
        annotation_file = os.path.join(self.annotations_dir, "annotations.json")
        
        if not os.path.exists(annotation_file):
            raise FileNotFoundError(f"COCO annotation file not found: {annotation_file}")
        
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)
        
        # Create mapping from image_id to annotations
        image_annotations = {}
        for annotation in coco_data['annotations']:
            image_id = annotation['image_id']
            if image_id not in image_annotations:
                image_annotations[image_id] = []
            image_annotations[image_id].append(annotation)
        
        # Load images and their annotations
        for image_info in coco_data['images']:
            image_path = os.path.join(self.images_dir, image_info['file_name'])
            if os.path.exists(image_path):
                self.images.append(image_path)
                self.annotations.append(image_annotations.get(image_info['id'], []))
        
        # Update class names from COCO data
        if not self.class_names:
            self.class_names = [cat['name'] for cat in coco_data['categories']]
    
    def _load_pascal_voc_dataset(self):
        """Load Pascal VOC format dataset"""
        for filename in os.listdir(self.annotations_dir):
            if filename.endswith('.xml'):
                annotation_path = os.path.join(self.annotations_dir, filename)
                image_filename = os.path.splitext(filename)[0] + '.jpg'
                image_path = os.path.join(self.images_dir, image_filename)
                
                if os.path.exists(image_path):
                    self.images.append(image_path)
                    self.annotations.append(annotation_path)
    
    def _parse_yolo_annotation(self, annotation_path: str, image_width: int, image_height: int):
        """Parse YOLO format annotation"""
        boxes = []
        labels = []
        
        if os.path.exists(annotation_path):
            with open(annotation_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1]) * image_width
                        y_center = float(parts[2]) * image_height
                        width = float(parts[3]) * image_width
                        height = float(parts[4]) * image_height
                        
                        # Convert to xyxy format
                        x1 = x_center - width / 2
                        y1 = y_center - height / 2
                        x2 = x_center + width / 2
                        y2 = y_center + height / 2
                        
                        boxes.append([x1, y1, x2, y2])
                        labels.append(class_id)
        
        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)
    
    def _parse_pascal_voc_annotation(self, annotation_path: str):
        """Parse Pascal VOC format annotation"""
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        
        boxes = []
        labels = []
        
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name in self.class_names:
                class_id = self.class_names.index(class_name)
                
                bbox = obj.find('bndbox')
                x1 = float(bbox.find('xmin').text)
                y1 = float(bbox.find('ymin').text)
                x2 = float(bbox.find('xmax').text)
                y2 = float(bbox.find('ymax').text)
                
                boxes.append([x1, y1, x2, y2])
                labels.append(class_id)
        
        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.images[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        original_height, original_width = image.shape[:2]
        
        # Load annotations
        if self.annotation_format == "yolo":
            boxes, labels = self._parse_yolo_annotation(
                self.annotations[idx], original_width, original_height
            )
        elif self.annotation_format == "pascal_voc":
            boxes, labels = self._parse_pascal_voc_annotation(self.annotations[idx])
        else:  # COCO format
            # Implementation for COCO format parsing
            boxes, labels = self._parse_coco_annotation(self.annotations[idx])
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(
                image=image,
                bboxes=boxes,
                class_labels=labels
            )
            image = transformed['image']
            boxes = np.array(transformed['bboxes'])
            labels = np.array(transformed['class_labels'])
        
        # Convert to tensors
        if len(boxes) > 0:
            target = {
                'boxes': torch.tensor(boxes, dtype=torch.float32),
                'labels': torch.tensor(labels, dtype=torch.long),
                'image_id': torch.tensor([idx])
            }
        else:
            target = {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.long),
                'image_id': torch.tensor([idx])
            }
        
        return image, target
    
    def _parse_coco_annotation(self, annotations):
        """Parse COCO format annotation"""
        boxes = []
        labels = []
        
        for ann in annotations:
            bbox = ann['bbox']  # [x, y, width, height]
            x1, y1, w, h = bbox
            x2 = x1 + w
            y2 = y1 + h
            
            boxes.append([x1, y1, x2, y2])
            labels.append(ann['category_id'])
        
        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)

class DataLoaderManager:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_config = self.config['data']
        self.training_config = self.config['training']
        
    def get_transforms(self, is_training: bool = True):
        """Get data transforms for training or validation"""
        if is_training:
            transform = A.Compose([
                A.HorizontalFlip(p=self.data_config['augmentation']['horizontal_flip']),
                A.VerticalFlip(p=self.data_config['augmentation']['vertical_flip']),
                A.Rotate(limit=self.data_config['augmentation']['rotation'], p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=self.data_config['augmentation']['brightness'],
                    contrast_limit=self.data_config['augmentation']['contrast'],
                    p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=int(self.data_config['augmentation']['hue'] * 255),
                    sat_shift_limit=int(self.data_config['augmentation']['saturation'] * 255),
                    val_shift_limit=int(self.data_config['augmentation']['brightness'] * 255),
                    p=0.5
                ),
                A.GaussNoise(var_limit=(10.0, 50.0), p=self.data_config['augmentation']['noise']),
                A.Resize(640, 640),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
        else:
            transform = A.Compose([
                A.Resize(640, 640),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
        
        return transform
    
    def create_data_loaders(self):
        """Create data loaders for training, validation, and testing"""
        # Training dataset
        train_dataset = ObjectDetectionDataset(
            images_dir=self.data_config['paths']['train_images'],
            annotations_dir=self.data_config['paths']['train_labels'],
            annotation_format="yolo",
            transform=self.get_transforms(is_training=True),
            class_names=self.data_config['classes']
        )
        
        # Validation dataset
        val_dataset = ObjectDetectionDataset(
            images_dir=self.data_config['paths']['val_images'],
            annotations_dir=self.data_config['paths']['val_labels'],
            annotation_format="yolo",
            transform=self.get_transforms(is_training=False),
            class_names=self.data_config['classes']
        )
        
        # Test dataset
        test_dataset = ObjectDetectionDataset(
            images_dir=self.data_config['paths']['test_images'],
            annotations_dir=self.data_config['paths']['test_labels'],
            annotation_format="yolo",
            transform=self.get_transforms(is_training=False),
            class_names=self.data_config['classes']
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_config['batch_size'],
            shuffle=True,
            num_workers=4,
            collate_fn=self.collate_fn,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.training_config['batch_size'],
            shuffle=False,
            num_workers=4,
            collate_fn=self.collate_fn,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            collate_fn=self.collate_fn
        )
        
        return train_loader, val_loader, test_loader
    
    @staticmethod
    def collate_fn(batch):
        """Custom collate function for batching"""
        images = []
        targets = []
        
        for image, target in batch:
            images.append(image)
            targets.append(target)
        
        images = torch.stack(images, 0)
        return images, targets

# Synthetic dataset generator for demonstration
class SyntheticDatasetGenerator:
    def __init__(self, num_images: int = 1000, output_dir: str = "data/synthetic"):
        self.num_images = num_images
        self.output_dir = output_dir
        self.classes = [
            "person", "car", "bicycle", "dog", "cat", 
            "bird", "house", "tree", "flower", "ball"
        ]
        
        os.makedirs(f"{output_dir}/images", exist_ok=True)
        os.makedirs(f"{output_dir}/labels", exist_ok=True)
    
    def generate_synthetic_image(self, image_id: int):
        """Generate a synthetic image with random objects"""
        # Create a random background
        image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Add some random shapes as objects
        annotations = []
        num_objects = np.random.randint(1, 6)
        
        for _ in range(num_objects):
            # Random object properties
            class_id = np.random.randint(0, len(self.classes))
            x_center = np.random.randint(50, 590)
            y_center = np.random.randint(50, 590)
            width = np.random.randint(30, 100)
            height = np.random.randint(30, 100)
            
            # Draw rectangle (representing object)
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.rectangle(
                image,
                (x_center - width//2, y_center - height//2),
                (x_center + width//2, y_center + height//2),
                color,
                -1
            )
            
            # Convert to YOLO format (normalized)
            x_norm = x_center / 640
            y_norm = y_center / 640
            w_norm = width / 640
            h_norm = height / 640
            
            annotations.append(f"{class_id} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}")
        
        return image, annotations
    
    def generate_dataset(self):
        """Generate the complete synthetic dataset"""
        logger.info(f"Generating {self.num_images} synthetic images...")
        
        for i in range(self.num_images):
            image, annotations = self.generate_synthetic_image(i)
            
            # Save image
            image_path = f"{self.output_dir}/images/image_{i:06d}.jpg"
            cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
            # Save annotations
            label_path = f"{self.output_dir}/labels/image_{i:06d}.txt"
            with open(label_path, 'w') as f:
                f.write('\n'.join(annotations))
            
            if (i + 1) % 100 == 0:
                logger.info(f"Generated {i + 1}/{self.num_images} images")
        
        logger.info("Synthetic dataset generation completed!")
        
        # Create class names file
        with open(f"{self.output_dir}/classes.txt", 'w') as f:
            f.write('\n'.join(self.classes))

if __name__ == "__main__":
    # Generate synthetic dataset for demonstration
    generator = SyntheticDatasetGenerator(num_images=1000)
    generator.generate_dataset()
    
    # Test data loader
    data_manager = DataLoaderManager()
    train_loader, val_loader, test_loader = data_manager.create_data_loaders()
    
    # Test loading a batch
    for images, targets in train_loader:
        print(f"Batch shape: {images.shape}")
        print(f"Number of targets: {len(targets)}")
        break
