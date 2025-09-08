"""
Faster R-CNN implementation for object detection
"""

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class FasterRCNN(nn.Module):
    """Faster R-CNN model for object detection"""
    
    def __init__(self, num_classes: int, config: Dict = None):
        """Initialize Faster R-CNN model"""
        super().__init__()
        self.num_classes = num_classes
        self.config = config or {}
        
        # Build model
        self.model = self._build_model()
        
    def _build_model(self):
        """Build Faster R-CNN model"""
        # Load pretrained model
        pretrained = self.config.get('pretrained', True)
        pretrained_backbone = self.config.get('pretrained_backbone', True)
        
        model = fasterrcnn_resnet50_fpn(
            pretrained=pretrained,
            pretrained_backbone=pretrained_backbone
        )
        
        # Replace the classifier head
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, self.num_classes
        )
        
        # Configure anchor generator
        if 'anchor_sizes' in self.config:
            from torchvision.models.detection.rpn import AnchorGenerator
            anchor_generator = AnchorGenerator(
                sizes=self.config['anchor_sizes'],
                aspect_ratios=self.config.get('aspect_ratios', ((0.5, 1.0, 2.0),))
            )
            model.rpn.anchor_generator = anchor_generator
        
        # Configure RPN parameters
        if 'rpn_pre_nms_top_n' in self.config:
            model.rpn.pre_nms_top_n = self.config['rpn_pre_nms_top_n']
        if 'rpn_post_nms_top_n' in self.config:
            model.rpn.post_nms_top_n = self.config['rpn_post_nms_top_n']
        
        return model
    
    def forward(self, images: torch.Tensor, 
               targets: Optional[List[Dict]] = None):
        """Forward pass"""
        if self.training and targets is not None:
            return self.model(images, targets)
        else:
            return self.model(images)
    
    def predict(self, image: torch.Tensor, 
               confidence_threshold: float = 0.5) -> Dict:
        """Make predictions on a single image"""
        self.eval()
        with torch.no_grad():
            predictions = self.model([image])[0]
        
        # Filter by confidence
        keep = predictions['scores'] > confidence_threshold
        
        return {
            'boxes': predictions['boxes'][keep],
            'labels': predictions['labels'][keep],
            'scores': predictions['scores'][keep]
        }
    
    def postprocess(self, outputs: List[Dict], 
                   image_sizes: List[Tuple]) -> List[Dict]:
        """Postprocess model outputs"""
        processed = []
        
        for output, size in zip(outputs, image_sizes):
            # Scale boxes back to original size
            boxes = output['boxes']
            h, w = size
            
            # Apply NMS if needed
            if self.config.get('apply_nms', True):
                keep = torchvision.ops.nms(
                    boxes, output['scores'],
                    self.config.get('nms_threshold', 0.5)
                )
                output = {
                    'boxes': boxes[keep],
                    'labels': output['labels'][keep],
                    'scores': output['scores'][keep]
                }
            
            processed.append(output)
        
        return processed
    
    def freeze_backbone(self, freeze: bool = True):
        """Freeze or unfreeze backbone layers"""
        for param in self.model.backbone.parameters():
            param.requires_grad = not freeze
    
    def get_trainable_params(self) -> List:
        """Get list of trainable parameters"""
        return [p for p in self.parameters() if p.requires_grad]
