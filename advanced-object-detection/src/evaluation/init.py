"""
Evaluation module for object detection models
"""

from .metrics import COCOEvaluator, calculate_metrics
from .visualization import plot_precision_recall, plot_confusion_matrix

__all__ = [
    'COCOEvaluator',
    'calculate_metrics',
    'plot_precision_recall',
    'plot_confusion_matrix'
]
