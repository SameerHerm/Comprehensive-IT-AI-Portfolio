"""
Validation utilities for object detection models
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class Validator:
    """Validator for object detection models"""
    
    def __init__(self, num_classes: int, iou_threshold: float = 0.5):
        """Initialize validator"""
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.reset()
    
    def reset(self):
        """Reset validation metrics"""
        self.predictions = []
        self.ground_truths = []
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
    
    def update(self, predictions: List[Dict], targets: List[Dict]):
        """Update with batch predictions and targets"""
        for pred, target in zip(predictions, targets):
            self.predictions.append(pred)
            self.ground_truths.append(target)
            
            # Update confusion matrix
            self._update_confusion_matrix(pred, target)
    
    def _update_confusion_matrix(self, pred: Dict, target: Dict):
        """Update confusion matrix"""
        pred_boxes = pred.get('boxes', torch.empty(0, 4))
        pred_labels = pred.get('labels', torch.empty(0))
        
        target_boxes = target.get('boxes', torch.empty(0, 4))
        target_labels = target.get('labels', torch.empty(0))
        
        # Match predictions with targets
        if len(pred_boxes) > 0 and len(target_boxes) > 0:
            ious = self._calculate_iou_matrix(pred_boxes, target_boxes)
            
            for i, pred_label in enumerate(pred_labels):
                max_iou_idx = ious[i].argmax()
                max_iou = ious[i, max_iou_idx]
                
                if max_iou >= self.iou_threshold:
                    target_label = target_labels[max_iou_idx]
                    self.confusion_matrix[target_label, pred_label] += 1
    
    def _calculate_iou_matrix(self, boxes1: torch.Tensor, 
                             boxes2: torch.Tensor) -> torch.Tensor:
        """Calculate IoU matrix between two sets of boxes"""
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        iou_matrix = torch.zeros((len(boxes1), len(boxes2)))
        
        for i, box1 in enumerate(boxes1):
            for j, box2 in enumerate(boxes2):
                # Calculate intersection
                x1 = max(box1[0], box2[0])
                y1 = max(box1[1], box2[1])
                x2 = min(box1[2], box2[2])
                y2 = min(box1[3], box2[3])
                
                if x2 > x1 and y2 > y1:
                    intersection = (x2 - x1) * (y2 - y1)
                    union = area1[i] + area2[j] - intersection
                    iou_matrix[i, j] = intersection / union
        
        return iou_matrix
    
    def compute_metrics(self) -> Dict:
        """Compute validation metrics"""
        metrics = {
            'mAP': self._compute_map(),
            'precision': self._compute_precision(),
            'recall': self._compute_recall(),
            'f1_score': self._compute_f1_score()
        }
        
        return metrics
    
    def _compute_map(self) -> float:
        """Compute mean Average Precision"""
        aps = []
        
        for class_id in range(self.num_classes):
            ap = self._compute_ap_for_class(class_id)
            aps.append(ap)
        
        return np.mean(aps)
    
    def _compute_ap_for_class(self, class_id: int) -> float:
        """Compute Average Precision for a specific class"""
        class_predictions = []
        class_ground_truths = []
        
        # Collect predictions and ground truths for this class
        for pred, gt in zip(self.predictions, self.ground_truths):
            pred_mask = pred['labels'] == class_id
            gt_mask = gt['labels'] == class_id
            
            if pred_mask.any():
                class_predictions.extend([
                    {'box': pred['boxes'][i], 'score': pred['scores'][i]}
                    for i in range(len(pred['boxes'])) if pred_mask[i]
                ])
            
            if gt_mask.any():
                class_ground_truths.extend([
                    {'box': gt['boxes'][i]}
                    for i in range(len(gt['boxes'])) if gt_mask[i]
                ])
        
        if not class_ground_truths:
            return 0.0
        
        # Sort predictions by score
        class_predictions.sort(key=lambda x: x['score'], reverse=True)
        
        # Compute precision-recall curve
        tp = np.zeros(len(class_predictions))
        fp = np.zeros(len(class_predictions))
        
        matched_gt = set()
        
        for i, pred in enumerate(class_predictions):
            best_iou = 0
            best_gt_idx = -1
            
            for j, gt in enumerate(class_ground_truths):
                if j in matched_gt:
                    continue
                
                iou = self._calculate_iou(pred['box'], gt['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_iou >= self.iou_threshold:
                tp[i] = 1
                matched_gt.add(best_gt_idx)
            else:
                fp[i] = 1
        
        # Compute precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / len(class_ground_truths)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
        
        # Compute AP
        ap = self._compute_ap_from_pr(precisions, recalls)
        
        return ap
    
    def _calculate_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> float:
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
        
        return intersection / union
    
    def _compute_ap_from_pr(self, precisions: np.ndarray, 
                           recalls: np.ndarray) -> float:
        """Compute AP from precision-recall curve"""
        # Add sentinel values
        precisions = np.concatenate([[0], precisions, [0]])
        recalls = np.concatenate([[0], recalls, [1]])
        
        # Ensure precision is monotonically decreasing
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])
        
        # Compute AP
        ap = 0
        for i in range(1, len(recalls)):
            ap += (recalls[i] - recalls[i - 1]) * precisions[i]
        
        return ap
    
    def _compute_precision(self) -> float:
        """Compute overall precision"""
        tp = np.diag(self.confusion_matrix).sum()
        fp = self.confusion_matrix.sum(axis=0).sum() - tp
        
        return tp / (tp + fp + 1e-10)
    
    def _compute_recall(self) -> float:
        """Compute overall recall"""
        tp = np.diag(self.confusion_matrix).sum()
        fn = self.confusion_matrix.sum(axis=1).sum() - tp
        
        return tp / (tp + fn + 1e-10)
    
    def _compute_f1_score(self) -> float:
        """Compute F1 score"""
        precision = self._compute_precision()
        recall = self._compute_recall()
        
        return 2 * (precision * recall) / (precision + recall + 1e-10)
