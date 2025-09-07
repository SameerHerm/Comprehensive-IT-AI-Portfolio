import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_training import ModelTrainer
from sklearn.datasets import make_classification
import numpy as np

class TestModelTrainer:
    
    @pytest.fixture
    def trainer(self):
        return ModelTrainer()
    
    @pytest.fixture
    def sample_data(self):
        X, y = make_classification(n_samples=100, n_features=20, 
                                  n_classes=2, random_state=42)
        return X, y
    
    def test_train_single_model(self, trainer, sample_data):
        """Test training a single model"""
        X, y = sample_data
        
        model = trainer.train_model('naive_bayes', X, y)
        
        assert model is not None
        assert 'naive_bayes' in trainer.trained_models
        
        # Test prediction
        predictions = model.predict(X[:10])
        assert len(predictions) == 10
    
    def test_train_all_models(self, trainer, sample_data):
        """Test training all models"""
        X, y = sample_data
        
        results = trainer.train_all_models(X, y, use_grid_search=False)
        
        assert len(results) > 0
        assert all('mean_cv_score' in result for result in results.values())
    
    def test_model_save_load(self, trainer, sample_data, tmp_path):
        """Test saving and loading models"""
        X, y = sample_data
        
        # Train model
        trainer.train_model('naive_bayes', X, y)
        
        # Save model
        model_path = tmp_path / "test_model.pkl"
        trainer.save_model('naive_bayes', str(model_path))
        
        assert model_path.exists()
        
        # Load model
        loaded_model = trainer.load_model(str(model_path))
        assert loaded_model is not None
        
        # Test loaded model
        predictions = loaded_model.predict(X[:5])
        assert len(predictions) == 5
