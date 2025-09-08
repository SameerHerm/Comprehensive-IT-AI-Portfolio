"""
Training module for object detection models
"""

from .trainer import ObjectDetectionTrainer
from .validator import Validator

__all__ = [
    'ObjectDetectionTrainer',
    'Validator'
]
