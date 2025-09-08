"""
SSD (Single Shot Detector) implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math
import logging

logger = logging.getLogger(__name__)

class SSDDetector(nn.Module):
    """SSD model for object detection"""
    
    def __init__(self, num_classes: int, config: Dict = None):
        """Initialize SSD model"""
        super().__init__()
        self.num_classes = num_classes
        self.config = config or {}
        
        # Model configuration
        self.image_size = self.config.get('image_size', 300)
        self.feature_maps = self.config.get('feature_maps', [38, 19, 10, 5, 3, 1])
        self.aspect_ratios = self.config.get('aspect_ratios', 
            [[2], [2, 3], [2, 3], [2, 3], [2], [2]])
        
        # Build model components
        self.backbone = self._build_backbone()
        self.extra_layers = self._build_extra_layers()
        self.loc_layers = self._build_loc_layers()
        self.conf_layers = self._build_conf_layers()
        self.priors = self._generate_priors()
        
    def _build_backbone(self):
        """Build VGG backbone"""
        layers = []
        in_channels = 3
        
        # VGG layers
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 
               512, 512, 512, 'M', 512, 512, 512]
        
        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers.extend([conv2d, nn.ReLU(inplace=True)])
                in_channels = v
        
        return nn.ModuleList(layers)
    
    def _build_extra_layers(self):
        """Build extra feature layers"""
        layers = []
        
        # Extra layers configuration
        extra_cfg = [
            (512, 1024, 3, 1, 1),
            (1024, 256, 1, 1, 0),
            (256, 512, 3, 2, 1),
            (512, 128, 1, 1, 0),
            (128, 256, 3, 2, 1),
            (256, 128, 1, 1, 0),
            (128, 256, 3, 1, 0),
            (256, 128, 1, 1, 0),
            (128, 256, 3, 1, 0)
        ]
        
        for in_ch, out_ch, k, s, p in extra_cfg:
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p),
                nn.ReLU(inplace=True)
            ])
        
        return nn.ModuleList(layers)
    
    def _build_loc_layers(self):
        """Build localization layers"""
        loc_layers = []
        
        # Number of boxes per feature map
        num_boxes = [4, 6, 6, 6, 4, 4]
        
        # Input channels for each feature map
        in_channels = [512, 1024, 512, 256, 256, 256]
        
        for in_ch, num_box in zip(in_channels, num_boxes):
            loc_layers.append(
                nn.Conv2d(in_ch, num_box * 4, kernel_size=3, padding=1)
            )
        
        return nn.ModuleList(loc_layers)
    
    def _build_conf_layers(self):
        """Build confidence layers"""
        conf_layers = []
        
        # Number of boxes per feature map
        num_boxes = [4, 6, 6, 6, 4, 4]
        
        # Input channels for each feature map
        in_channels = [512, 1024, 512, 256, 256, 256]
        
        for in_ch, num_box in zip(in_channels, num_boxes):
            conf_layers.append(
                nn.Conv2d(in_ch, num_box * self.num_classes, 
                         kernel_size=3, padding=1)
            )
        
        return nn.ModuleList(conf_layers)
    
    def _generate_priors(self):
        """Generate prior boxes"""
        priors = []
        
        for k, f in enumerate(self.feature_maps):
            for i in range(f):
                for j in range(f):
                    cx = (j + 0.5) / f
                    cy = (i + 0.5) / f
                    
                    # Aspect ratio 1
                    s_k = self._get_scale(k)
                    priors.append([cx, cy, s_k, s_k])
                    
                    # Additional aspect ratios
                    s_k_prime = math.sqrt(s_k * self._get_scale(k + 1))
                    priors.append([cx, cy, s_k_prime, s_k_prime])
                    
                    for ar in self.aspect_ratios[k]:
                        priors.append([cx, cy, s_k * math.sqrt(ar), s_k / math.sqrt(ar)])
                        priors.append([cx, cy, s_k / math.sqrt(ar), s_k * math.sqrt(ar)])
        
        return torch.tensor(priors).clamp(0, 1)
    
    def _get_scale(self, k: int) -> float:
        """Get scale for k-th feature map"""
        s_min = 0.2
        s_max = 0.9
        m = len(self.feature_maps)
        
        if k == 0:
            return s_min
        elif k >= m:
            return s_max
        else:
            return s_min + (s_max - s_min) * k / (m - 1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        batch_size = x.size(0)
        sources = []
        loc = []
        conf = []
        
        # Pass through backbone
        for layer in self.backbone:
            x = layer(x)
            # Save intermediate features
            if x.size(2) == 38:  # First detection layer
                sources.append(x)
        
        # Pass through extra layers
        for layer in self.extra_layers:
            x = layer(x)
            if x.size(2) in self.feature_maps[1:]:
                sources.append(x)
        
        # Apply detection layers
        for (x, l, c) in zip(sources, self.loc_layers, self.conf_layers):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        
        # Reshape outputs
        loc = torch.cat([o.view(batch_size, -1) for o in loc], 1)
        conf = torch.cat([o.view(batch_size, -1) for o in conf], 1)
        
        loc = loc.view(batch_size, -1, 4)
        conf = conf.view(batch_size, -1, self.num_classes)
        
        return loc, conf
    
    def detect(self, loc_data: torch.Tensor, conf_data: torch.Tensor,
              threshold: float = 0.5) -> List[Dict]:
        """Detect objects from network output"""
        batch_size = loc_data.size(0)
        num_priors = self.priors.size(0)
        
        conf_scores = F.softmax(conf_data, dim=2)
        
        output = []
        for i in range(batch_size):
            decoded_boxes = self._decode_boxes(loc_data[i], self.priors)
            conf = conf_scores[i]
            
            # For each class
            detections = []
            for cl in range(1, self.num_classes):  # Skip background
                c_mask = conf[:, cl] > threshold
                if c_mask.sum() == 0:
                    continue
                
                scores = conf[c_mask, cl]
                boxes = decoded_boxes[c_mask]
                
                # Apply NMS
                keep = self._nms(boxes, scores, threshold=0.5)
                
                detections.extend([
                    {'box': boxes[j], 'score': scores[j], 'class': cl}
                    for j in keep
                ])
            
            output.append(detections)
        
        return output
    
    def _decode_boxes(self, loc: torch.Tensor, priors: torch.Tensor) -> torch.Tensor:
        """Decode predicted boxes"""
        boxes = torch.cat([
            priors[:, :2] + loc[:, :2] * 0.1 * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[:, 2:] * 0.2)
        ], 1)
        
        # Convert to corner format
        boxes = torch.cat([
            boxes[:, :2] - boxes[:, 2:] / 2,
            boxes[:, :2] + boxes[:, 2:] / 2
        ], 1)
        
        return boxes.clamp(0, 1)
    
    def _nms(self, boxes: torch.Tensor, scores: torch.Tensor,
            threshold: float = 0.5) -> List[int]:
        """Non-maximum suppression"""
        keep = torchvision.ops.nms(boxes, scores, threshold)
        return keep.tolist()
