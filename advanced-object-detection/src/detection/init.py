"""
Detection module for inference and prediction
"""

from .detector import ObjectDetector
from .utils import draw_boxes, save_detection_results

__all__ = [
    'ObjectDetector',
    'draw_boxes',
    'save_detection_results'
]
