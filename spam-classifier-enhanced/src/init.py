"""
Enhanced Spam Classifier Package
"""

__version__ = "2.0.0"
__author__ = "Your Name"

from .data_preprocessing import DataPreprocessor
from .feature_engineering import FeatureEngineer
from .model_training import ModelTrainer
from .model_evaluation import ModelEvaluator

__all__ = [
    'DataPreprocessor',
    'FeatureEngineer', 
    'ModelTrainer',
    'ModelEvaluator'
]
