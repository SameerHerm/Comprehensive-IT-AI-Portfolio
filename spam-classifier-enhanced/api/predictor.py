import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
import logging

logger = logging.getLogger(__name__)

class SpamPredictor:
    """Predictor class for spam classification API"""
    
    def __init__(self, model_dir: str = 'models'):
        self.model_dir = model_dir
        self.models = {}
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.load_models()
    
    def load_models(self):
        """Load all available models"""
        model_files = {
            'naive_bayes': 'naive_bayes_model.pkl',
            'svm': 'svm_model.pkl',
            'random_forest': 'rf_model.pkl',
            'xgboost': 'xgb_model.pkl'
        }
        
        for name, filename in model_files.items():
            filepath = os.path.join(self.model_dir, filename)
            if os.path.exists(filepath):
                try:
                    self.models[name] = joblib.load(filepath)
                    logger.info(f"Loaded model: {name}")
                except Exception as e:
                    logger.error(f"Error loading {name}: {str(e)}")
    
    def is_loaded(self) -> bool:
        """Check if models are loaded"""
        return len(self.models) > 0
    
    def get_available_models(self) -> list:
        """Get list of available models"""
        return list(self.models.keys())
    
    def preprocess_text(self, text: str):
        """Preprocess single text"""
        # Clean text
        cleaned_text = self.preprocessor.clean_text(text)
        
        # Extract features
        features = self.preprocessor.extract_features(text)
        
        # Create TF-IDF features
        tfidf_features = self.feature_engineer.create_tfidf_features([cleaned_text])
        
        return tfidf_features, features
    
    def predict(self, text: str, model_name: str = 'xgboost') -> dict:
        """Make prediction for single text"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")
        
        # Preprocess
        features, extra_features = self.preprocess_text(text)
        
        # Get model
        model = self.models[model_name]
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Get probability if available
        confidence = 0.5
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)[0]
            confidence = float(max(probabilities))
        
        return {
            'prediction': 'spam' if prediction == 1 else 'ham',
            'confidence': confidence,
            'is_spam': bool(prediction == 1)
        }
    
    def batch_predict(self, texts: list, model_name: str = 'xgboost') -> list:
        """Make predictions for multiple texts"""
        results = []
        for text in texts:
            try:
                result = self.predict(text, model_name)
                results.append(result)
            except Exception as e:
                results.append({
                    'prediction': 'error',
                    'confidence': 0.0,
                    'error': str(e)
                })
        
        return results
