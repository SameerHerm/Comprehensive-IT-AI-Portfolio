"""
Evaluation metrics for object detection
"""

import numpy as np
from typing import Dict, List, Tuple
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch
import logging

logger = logging.getLogger(__name__)

class COCOEvaluator:
    """COCO-style evaluator for object detection"""
    
    def __init__(self, coco_gt, iou_types=['bbox']):
        """Initialize COCO evaluator"""
        self.coco_gt = coco_gt
        self.iou_types = iou_types
        self.coco_eval = {}
        
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)
    
    def update(self, predictions):
        """Update with predictions"""
        img_ids = list(predictions.keys())
        
        for iou_type in self.iou_types:
            results = self.prepare_for_coco(predictions, iou_type)
            
            coco_dt = self.coco_gt.loadRes(results) if results else COCO()
            coco_eval = self.coco_eval[iou_type]
            
            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = img_ids
    
    def prepare_for_coco(self, predictions, iou_type):
        """Prepare predictions for COCO evaluation"""
        coco_results = []
        
        for img_id, prediction in predictions.items():
            if iou_type == 'bbox':
                boxes = prediction['boxes']
                scores = prediction['scores']
                labels = prediction['labels']
                
                for box, score, label in zip(boxes, scores, labels):
                    coco_results.append({
                        'image_id': img_id,
                        'category_id': label.item(),
                        'bbox': self.convert_to_xywh(box).tolist(),
                        'score': score.item()
                    })
        
        return coco_results
    
    def convert_to_xywh(self, box):
        """Convert from xyxy to xywh format"""
        x1, y1, x2, y2 = box
        return torch.tensor([x1, y1, x2 - x1, y2 - y1])
    
    def evaluate(self):
        """Run evaluation"""
        for iou_type in self.iou_types:
            self.coco_eval[iou_type].evaluate()
    
    def accumulate(self):
        """Accumulate results"""
        for iou_type in self.iou_types:
            self.coco_eval[iou_type].accumulate()
    
    def summarize(self):
        """Print summary metrics"""
        for iou_type in self.iou_types:
            print(f"IoU metric: {iou_type}")
            self.coco_eval[iou_type].summarize()

def calculate_metrics(predictions: List[Dict], targets: List[Dict],
                     iou_threshold: float = 0.5) -> Dict:
    """Calculate detection metrics"""
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    class_metrics = {}
    
    for pred, target in zip(predictions, targets):
        pred_boxes = pred.get('boxes', torch.empty(0, 4))
        pred_labels = pred.get('labels', torch.empty(0))
        pred_scores = pred.get('scores', torch.empty(0))
        
        target_boxes = target.get('boxes', torch.empty(0, 4))
        target_labels = target.get('labels', torch.empty(0))
        
        # Match predictions with targets
        matched_targets = set()
        
        for i, pred_box in enumerate(pred_boxes):
            pred_label = pred_labels[i]
            best_iou = 0
            best_target_idx = -1
            
            for j, target_box in enumerate(target_boxes):
                if j in matched_targets:
                    continue
                
                if target_labels[j] != pred_label:
                    continue
                
                iou = calculate_iou_single(pred_box, target_box)
                if iou > best_iou:
                    best_iou = iou
                    best_target_idx = j
            
            if best_iou >= iou_threshold:
                total_tp += 1
                matched_targets.add(best_target_idx)
                
                # Update class metrics
                class_id = pred_label.item()
                if class_id not in class_metrics:
                    class_metrics[class_id] = {'tp': 0, 'fp': 0, 'fn': 0}
                class_metrics[class_id]['tp'] += 1
            else:
                total_fp += 1
                
                # Update class metrics
                class_id = pred_label.item()
                if class_id not in class_metrics:
                    class_metrics[class_id] = {'tp': 0, 'fp': 0, 'fn': 0}
                class_metrics[class_id]['fp'] += 1
        
        # Count false negatives
        total_fn += len(target_boxes) - len(matched_targets)
        
        for j, target_label in enumerate(target_labels):
            if j not in matched_targets:
                class_id = target_label.item()
                if class_id not in class_metrics:
                    class_metrics[class_id] = {'tp': 0, 'fp': 0, 'fn': 0}
                class_metrics[class_id]['fn'] += 1
    
    # Calculate overall metrics
    precision = total_tp / (total_tp + total_fp + 1e-10)
    recall = total_tp / (total_tp + total_fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    
    # Calculate per-class metrics
    per_class_metrics = {}
    for class_id, counts in class_metrics.items():
        tp = counts['tp']
        fp = counts['fp']
        fn = counts['fn']
        
        class_precision = tp / (tp + fp + 1e-10)
        class_recall = tp / (tp + fn + 1e-10)
        class_f1 = 2 * class_precision * class_recall / (class_precision + class_recall + 1e-10)
        
        per_class_metrics[class_id] = {
            'precision': class_precision,
            'recall': class_recall,
            'f1': class_f1
        }
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'per_class': per_class_metrics
    }

def calculate_iou_single(box1: torch.Tensor, box2: torch.Tensor) -> float:
    """Calculate IoU between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return float(intersection / (union + 1e-10))
