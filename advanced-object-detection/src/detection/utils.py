"""
Detection utilities for visualization and processing
"""

import cv2
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import json

def draw_boxes(image: np.ndarray, detections: Dict, 
               class_names: List[str] = None,
               threshold: float = 0.5) -> np.ndarray:
    """Draw bounding boxes on image"""
    img = image.copy()
    
    boxes = detections.get('boxes', [])
    labels = detections.get('labels', [])
    scores = detections.get('scores', [])
    
    # Define colors for different classes
    colors = generate_colors(len(class_names) if class_names else 80)
    
    for box, label, score in zip(boxes, labels, scores):
        if score < threshold:
            continue
        
        # Convert to integers
        x1, y1, x2, y2 = map(int, box)
        
        # Get color for this class
        color = colors[label % len(colors)]
        
        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label text
        label_text = f"{class_names[label] if class_names else label}: {score:.2f}"
        
        # Draw label background
        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(img, (x1, y1 - label_size[1] - 4), 
                     (x1 + label_size[0], y1), color, -1)
        
        # Draw label text
        cv2.putText(img, label_text, (x1, y1 - 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return img

def generate_colors(num_colors: int) -> List[Tuple[int, int, int]]:
    """Generate distinct colors for visualization"""
    colors = []
    for i in range(num_colors):
        hue = i / num_colors
        rgb = plt.cm.hsv(hue)[:3]
        colors.append(tuple(int(c * 255) for c in rgb))
    return colors

def save_detection_results(detections: Dict, output_path: str,
                          image_path: str = None):
    """Save detection results to JSON file"""
    results = {
        'image_path': image_path,
        'detections': []
    }
    
    boxes = detections.get('boxes', [])
    labels = detections.get('labels', [])
    scores = detections.get('scores', [])
    
    for box, label, score in zip(boxes, labels, scores):
        results['detections'].append({
            'bbox': box.tolist() if torch.is_tensor(box) else list(box),
            'label': int(label),
            'score': float(score)
        })
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

def non_max_suppression(boxes: torch.Tensor, scores: torch.Tensor,
                       iou_threshold: float = 0.5) -> torch.Tensor:
    """Apply Non-Maximum Suppression"""
    if len(boxes) == 0:
        return torch.empty((0,), dtype=torch.int64)
    
    # Sort by scores
    indices = scores.argsort(descending=True)
    
    keep = []
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
        
        current_box = boxes[current]
        other_boxes = boxes[indices[1:]]
        
        # Calculate IoU
        ious = calculate_iou(current_box.unsqueeze(0), other_boxes).squeeze(0)
        
        # Keep boxes with IoU less than threshold
        indices = indices[1:][ious < iou_threshold]
    
    return torch.tensor(keep)

def calculate_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Calculate IoU between two sets of boxes"""
    # Calculate intersection
    x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])
    
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    
    # Calculate areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Calculate union
    union = area1[:, None] + area2[None, :] - intersection
    
    # Calculate IoU
    iou = intersection / (union + 1e-10)
    
    return iou

def resize_boxes(boxes: torch.Tensor, original_size: Tuple[int, int],
                new_size: Tuple[int, int]) -> torch.Tensor:
    """Resize bounding boxes to match new image size"""
    orig_h, orig_w = original_size
    new_h, new_w = new_size
    
    scale_x = new_w / orig_w
    scale_y = new_h / orig_h
    
    boxes_scaled = boxes.clone()
    boxes_scaled[:, [0, 2]] *= scale_x
    boxes_scaled[:, [1, 3]] *= scale_y
    
    return boxes_scaled

def clip_boxes(boxes: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
    """Clip bounding boxes to image boundaries"""
    h, w = image_size
    
    boxes_clipped = boxes.clone()
    boxes_clipped[:, [0, 2]] = boxes_clipped[:, [0, 2]].clamp(0, w)
    boxes_clipped[:, [1, 3]] = boxes_clipped[:, [1, 3]].clamp(0, h)
    
    return boxes_clipped

def convert_bbox_format(boxes: torch.Tensor, 
                       from_format: str = 'xyxy',
                       to_format: str = 'xywh') -> torch.Tensor:
    """Convert between different bbox formats"""
    boxes_converted = boxes.clone()
    
    if from_format == 'xyxy' and to_format == 'xywh':
        boxes_converted[:, 2:] = boxes[:, 2:] - boxes[:, :2]
    elif from_format == 'xywh' and to_format == 'xyxy':
        boxes_converted[:, 2:] = boxes[:, :2] + boxes[:, 2:]
    elif from_format == 'xyxy' and to_format == 'cxcywh':
        boxes_converted[:, :2] = (boxes[:, :2] + boxes[:, 2:]) / 2
        boxes_converted[:, 2:] = boxes[:, 2:] - boxes[:, :2]
    elif from_format == 'cxcywh' and to_format == 'xyxy':
        boxes_converted[:, :2] = boxes[:, :2] - boxes[:, 2:] / 2
        boxes_converted[:, 2:] = boxes[:, :2] + boxes[:, 2:] / 2
    
    return boxes_converted
