"""
Data augmentation for object detection
"""

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from typing import Dict, List, Tuple, Optional
import random

class DataAugmentor:
    """Advanced data augmentation for object detection"""
    
    def __init__(self, config: Dict = None):
        """Initialize data augmentor with configuration"""
        self.config = config or {}
        self.train_transform = self._build_train_transform()
        self.val_transform = self._build_val_transform()
        
    def _build_train_transform(self):
        """Build training augmentation pipeline"""
        transforms = []
        
        # Geometric transforms
        if self.config.get('horizontal_flip', True):
            transforms.append(A.HorizontalFlip(p=0.5))
            
        if self.config.get('vertical_flip', False):
            transforms.append(A.VerticalFlip(p=0.3))
            
        if self.config.get('rotate', True):
            transforms.append(A.Rotate(limit=15, p=0.5))
            
        if self.config.get('scale', True):
            transforms.append(A.RandomScale(scale_limit=0.2, p=0.5))
            
        # Color transforms
        if self.config.get('color_jitter', True):
            transforms.append(A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.5
            ))
            
        if self.config.get('blur', True):
            transforms.append(A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.3))
            
        # Advanced augmentations
        if self.config.get('cutout', True):
            transforms.append(A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                p=0.5
            ))
            
        if self.config.get('mixup', False):
            # Mixup will be handled separately
            pass
            
        # Normalization
        transforms.append(A.Normalize(
            mean=self.config.get('mean', [0.485, 0.456, 0.406]),
            std=self.config.get('std', [0.229, 0.224, 0.225])
        ))
        
        transforms.append(ToTensorV2())
        
        return A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['class_labels'],
                min_visibility=0.3
            )
        )
    
    def _build_val_transform(self):
        """Build validation augmentation pipeline"""
        return A.Compose([
            A.Normalize(
                mean=self.config.get('mean', [0.485, 0.456, 0.406]),
                std=self.config.get('std', [0.229, 0.224, 0.225])
            ),
            ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['class_labels']
        ))
    
    def augment(self, image: np.ndarray, bboxes: List, labels: List, 
                is_training: bool = True) -> Tuple:
        """Apply augmentation to image and bounding boxes"""
        transform = self.train_transform if is_training else self.val_transform
        
        # Convert to required format
        transformed = transform(
            image=image,
            bboxes=bboxes,
            class_labels=labels
        )
        
        return (
            transformed['image'],
            transformed['bboxes'],
            transformed['class_labels']
        )
    
    def mosaic_augmentation(self, images: List[np.ndarray], 
                          annotations: List[Dict]) -> Tuple:
        """Apply mosaic augmentation (combine 4 images)"""
        assert len(images) == 4, "Mosaic requires exactly 4 images"
        
        # Get output size
        output_size = self.config.get('image_size', 640)
        
        # Create mosaic canvas
        mosaic_img = np.zeros((output_size, output_size, 3), dtype=np.uint8)
        mosaic_bboxes = []
        mosaic_labels = []
        
        # Define quadrants
        center_x = output_size // 2
        center_y = output_size // 2
        
        for i, (img, ann) in enumerate(zip(images, annotations)):
            h, w = img.shape[:2]
            
            # Determine placement
            if i == 0:  # Top-left
                x1, y1 = 0, 0
                x2, y2 = center_x, center_y
            elif i == 1:  # Top-right
                x1, y1 = center_x, 0
                x2, y2 = output_size, center_y
            elif i == 2:  # Bottom-left
                x1, y1 = 0, center_y
                x2, y2 = center_x, output_size
            else:  # Bottom-right
                x1, y1 = center_x, center_y
                x2, y2 = output_size, output_size
            
            # Resize and place image
            patch_h, patch_w = y2 - y1, x2 - x1
            resized = cv2.resize(img, (patch_w, patch_h))
            mosaic_img[y1:y2, x1:x2] = resized
            
            # Adjust bounding boxes
            scale_x = patch_w / w
            scale_y = patch_h / h
            
            for bbox, label in zip(ann['bboxes'], ann['labels']):
                new_bbox = [
                    bbox[0] * scale_x + x1,
                    bbox[1] * scale_y + y1,
                    bbox[2] * scale_x + x1,
                    bbox[3] * scale_y + y1
                ]
                mosaic_bboxes.append(new_bbox)
                mosaic_labels.append(label)
        
        return mosaic_img, mosaic_bboxes, mosaic_labels
    
    def copy_paste_augmentation(self, image: np.ndarray, bboxes: List,
                               labels: List, paste_objects: List) -> Tuple:
        """Copy-paste augmentation for object detection"""
        aug_image = image.copy()
        aug_bboxes = list(bboxes)
        aug_labels = list(labels)
        
        for obj in paste_objects:
            obj_img = obj['image']
            obj_bbox = obj['bbox']
            obj_label = obj['label']
            
            # Random position
            h, w = image.shape[:2]
            obj_h, obj_w = obj_img.shape[:2]
            
            if obj_w < w and obj_h < h:
                x = random.randint(0, w - obj_w)
                y = random.randint(0, h - obj_h)
                
                # Paste object
                aug_image[y:y+obj_h, x:x+obj_w] = obj_img
                
                # Add bbox
                new_bbox = [x, y, x + obj_w, y + obj_h]
                aug_bboxes.append(new_bbox)
                aug_labels.append(obj_label)
        
        return aug_image, aug_bboxes, aug_labels
