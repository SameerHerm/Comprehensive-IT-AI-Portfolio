"""
Advanced Object Detection System
A comprehensive object detection framework supporting multiple architectures
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from . import data
from . import models
from . import training
from . import detection
from . import evaluation

__all__ = [
    'data',
    'models', 
    'training',
    'detection',
    'evaluation'
]
