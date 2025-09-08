"""
Object detection models module
"""

from .yolo import YOLODetector
from .rcnn import FasterRCNN
from .ssd import SSDDetector
from .efficientdet import EfficientDetector

__all__ = [
    'YOLODetector',
    'FasterRCNN',
    'SSDDetector',
    'EfficientDetector'
]
