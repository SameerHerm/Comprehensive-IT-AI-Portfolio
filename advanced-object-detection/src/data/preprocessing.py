"""
Image preprocessing utilities for object detection
"""

import cv2
import numpy as np
from PIL import Image
import torch
from typing import Tuple, List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """Preprocess images for object detection models"""
    
    def __init__(self, config: Dict = None):
        """Initialize preprocessor with configuration"""
        self.config = config or {}
        self.target_size = self.config.get('image_size', 640)
        self.keep_ratio = self.config.get('keep_aspect_ratio', True)
        self.pad_color = self.config.get('pad_color', [114, 114, 114])
        
    def preprocess(self, image: np.ndarray, 
                  bboxes: Optional[List] = None) -> Tuple:
        """Preprocess image and adjust bounding boxes"""
        # Resize image
        processed_img, scale, pad = self.resize_with_pad(image)
        
        # Adjust bounding boxes if provided
        if bboxes is not None:
            adjusted_bboxes = self.adjust_bboxes(bboxes, scale, pad)
            return processed_img, adjusted_bboxes
        
        return processed_img
    
    def resize_with_pad(self, image: np.ndarray) -> Tuple:
        """Resize image maintaining aspect ratio with padding"""
        h, w = image.shape[:2]
        
        if self.keep_ratio:
            # Calculate scale to fit target size
            scale = min(self.target_size / h, self.target_size / w)
            new_h, new_w = int(h * scale), int(w * scale)
            
            # Resize image
            resized = cv2.resize(image, (new_w, new_h), 
                                interpolation=cv2.INTER_LINEAR)
            
            # Calculate padding
            pad_h = (self.target_size - new_h) // 2
            pad_w = (self.target_size - new_w) // 2
            
            # Apply padding
            top = pad_h
            bottom = self.target_size - new_h - pad_h
            left = pad_w
            right = self.target_size - new_w - pad_w
            
            padded = cv2.copyMakeBorder(
                resized, top, bottom, left, right,
                cv2.BORDER_CONSTANT, value=self.pad_color
            )
            
            return padded, scale, (pad_w, pad_h)
        else:
            # Direct resize without maintaining ratio
            resized = cv2.resize(image, (self.target_size, self.target_size))
            scale_x = self.target_size / w
            scale_y = self.target_size / h
            return resized, (scale_x, scale_y), (0, 0)
    
    def adjust_bboxes(self, bboxes: List, scale: float, 
                     pad: Tuple) -> List:
        """Adjust bounding boxes after preprocessing"""
        adjusted = []
        
        for bbox in bboxes:
            if isinstance(scale, tuple):
                # Different scales for x and y
                x1 = bbox[0] * scale[0]
                y1 = bbox[1] * scale[1]
                x2 = bbox[2] * scale[0]
                y2 = bbox[3] * scale[1]
            else:
                # Uniform scale
                x1 = bbox[0] * scale + pad[0]
                y1 = bbox[1] * scale + pad[1]
                x2 = bbox[2] * scale + pad[0]
                y2 = bbox[3] * scale + pad[1]
            
            adjusted.append([x1, y1, x2, y2])
        
        return adjusted
    
    def normalize(self, image: np.ndarray, 
                 mean: List[float] = None,
                 std: List[float] = None) -> np.ndarray:
        """Normalize image pixels"""
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]
        
        # Convert to float and scale to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Normalize
        image = (image - mean) / std
        
        return image
    
    def denormalize(self, image: np.ndarray,
                   mean: List[float] = None,
                   std: List[float] = None) -> np.ndarray:
        """Denormalize image for visualization"""
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]
        
        # Denormalize
        image = image * std + mean
        
        # Scale back to [0, 255]
        image = (image * 255).clip(0, 255).astype(np.uint8)
        
        return image
    
    def letterbox(self, image: np.ndarray, new_shape: Tuple[int, int],
                 color: Tuple[int, int, int] = (114, 114, 114),
                 auto: bool = True, scaleFill: bool = False,
                 scaleup: bool = True) -> Tuple:
        """Letterbox resize for YOLO models"""
        shape = image.shape[:2]
        
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        
        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:
            r = min(r, 1.0)
        
        # Compute padding
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        
        if auto:
            dw, dh = np.mod(dw, 32), np.mod(dh, 32)
        elif scaleFill:
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
        
        dw /= 2
        dh /= 2
        
        if shape[::-1] != new_unpad:
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        image = cv2.copyMakeBorder(image, top, bottom, left, right, 
                                  cv2.BORDER_CONSTANT, value=color)
        
        return image, ratio, (dw, dh)
