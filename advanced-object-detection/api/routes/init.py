"""
API routes initialization
"""

from .detection import detection_bp
from .training import training_bp

__all__ = ['detection_bp', 'training_bp']
