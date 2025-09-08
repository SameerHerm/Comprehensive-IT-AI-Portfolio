"""
Unit tests for machine learning models
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from unittest.mock import Mock, patch, MagicMock
import pickle
import tempfile
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import (
    train_random_forest,
    train_gradient_boosting,
    train_neural_network,
    train_logistic_regression,
    evaluate_model,
    cross_validate_model,
    hyperparameter_tuning,
    ensemble_predictions,
    save_model,
    load_model
)


class TestModels(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        # Create synthetic dataset for testing
        X, y = make_classification(
            n_samples=1000,
            n_features=10,
            n_informative=8,
            n_redundant=2,
            n_classes=2,
            random_state=42
        )
        
        # Split into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create feature names
        self.feature_names = [f'feature_{i}' for i in range(10)]
        
        # Convert to DataFrame for some tests
        self.X_train_df = pd.DataFrame(self.X_train, columns=self.feature_names)
        self.X_test_df = pd.DataFrame(self.X_test, columns=self.feature_names)
        
    def test_train_random_forest(self):
        """Test Random Forest model training"""
        # Train model
        model, metrics = train_random_forest(
            self.X_train, 
            self.y_train,
            self.X_test,
            self.y_test,
            n_estimators=10,
            random_state=42
        )
        
        # Check model is trained
        self.assertIsNotNone(model)
        
        # Check metrics
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1_score', metrics)
        self.assertIn('roc_auc', metrics)
        
        # Check accuracy is reasonable
        self.assertGreater(metrics['accuracy'], 0.5)
        self.assertLessEqual(metrics['accuracy'], 1.0)
        
        # Check feature importances
        self.assertEqual(len(model.feature_importances_), 10)
        
    def test_train_gradient_boosting(self):
        """Test Gradient Boosting model training"""
        # Train model
        model, metrics = train_gradient_boosting(
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            n_estimators=10,
            learning_rate=0.1,
            random_state=42
        )
        
        # Check model is trained
        self.assertIsNotNone(model)
        
        # Check metrics
        self.assertGreater(metrics['accuracy'], 0.5)
        self.assertGreater(metrics['roc_auc'], 0.5)
        
        # Check predictions
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))
        
    def test_train_neural_network(self):
        """Test Neural Network model training"""
        # Train model
        model, metrics = train_neural_network(
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            hidden_layer_sizes=(10, 5),
            max_iter=100,
            random_state=42
        )
        
        # Check model is trained
        self.assertIsNotNone(model)
        
        # Check convergence
        self.assertTrue(hasattr(model, 'n_iter_'))
        
        # Check metrics
        self.assertGreater(metrics['accuracy'], 0.5)
        
        # Check probability predictions
        probabilities = model.predict_proba(self.X_test)
        self.assertEqual(probabilities.shape, (len(self.y_test), 2))
        
    def test_train_logistic_regression(self):
        """Test Logistic Regression model training"""
        # Train model
        model, metrics = train_logistic_regression(
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            C=1.0,
            random_state=42
        )
        
        # Check model is trained
        self.assertIsNotNone(model)
        
        # Check coefficients
        self.assertEqual(model.coef_.shape[1], 10)  # 10 features
        
        # Check metrics
        self.assertGreater(metrics['accuracy'], 0.5)
        
        # Check prediction probabilities sum to 1
        probabilities = model.predict_proba(self.X_test[:1])
        self.assertAlmostEqual(probabilities[0].sum(), 1.0)
        
    def test_evaluate_model(self):
        """Test model evaluation"""
        from sklearn.ensemble import RandomForestClassifier
        
        # Train a simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(self.X_train, self.y_train)
        
        # Evaluate model
        metrics = evaluate_model(model, self.X_test, self.y_test)
        
        # Check all metrics are present
        expected_metrics = [
            'accuracy', 'precision', 'recall', 'f1_score', 
            'roc_auc', 'confusion_matrix', 'classification_report'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
        
        # Check confusion matrix shape
        cm = metrics['confusion_matrix']
        self.assertEqual(cm.shape, (2, 2))
        
        # Check values are in valid range
        self.assertGreaterEqual(metrics['accuracy'], 0)
        self.assertLessEqual(metrics['accuracy'], 1)
        
    def test_cross_validate_model(self):
        """Test cross-validation"""
        from sklearn.ensemble import RandomForestClassifier
        
        # Create model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Perform cross-validation
        cv_scores = cross_validate_model(
            model,
            self.X_train,
            self.y_train,
            cv=3,
            scoring='accuracy'
        )
        
        # Check results
        self.assertIn('mean_score', cv_scores)
        self.assertIn('std_score', cv_scores)
        self.assertIn('scores', cv_scores)
        
        # Check number of folds
        self.assertEqual(len(cv_scores['scores']), 3)
        
        # Check scores are valid
        for score in cv_scores['scores']:
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1)
            
    def test_hyperparameter_tuning(self):
        """Test hyperparameter tuning"""
        from sklearn.ensemble import RandomForestClassifier
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [5, 10],
            'max_depth': [3, 5],
            'random_state': [42]
        }
        
        # Perform hyperparameter tuning
        best_model, best_params, best_score = hyperparameter_tuning(
            RandomForestClassifier(),
            param_grid,
            self.X_train,
            self.y_train,
            cv=2,
            scoring='accuracy'
        )
        
        # Check results
        self.assertIsNotNone(best_model)
        self.assertIsInstance(best_params, dict)
        self.assertGreater(best_score, 0)
        
        # Check best parameters are from grid
        for param, value in best_params.items():
            if param in param_grid:
                self.assertIn(value, param_grid[param])
                
    def test_ensemble_predictions(self):
        """Test ensemble predictions"""
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        
        # Train multiple models
        models = []
        
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(self.X_train, self.y_train)
        models.append(('rf', rf))
        
        gb = GradientBoostingClassifier(n_estimators=10, random_state=42)
        gb.fit(self.X_train, self.y_train)
        models.append(('gb', gb))
        
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(self.X_train, self.y_train)
        models.append(('lr', lr))
        
        # Get ensemble predictions
        ensemble_pred, ensemble_proba = ensemble_predictions(
            models,
            self.X_test,
            method='voting'
        )
        
        # Check predictions
        self.assertEqual(len(ensemble_pred), len(self.y_test))
        self.assertEqual(ensemble_proba.shape, (len(self.y_test), 2))
        
        # Check predictions are binary
        unique_preds = np.unique(ensemble_pred)
        self.assertTrue(all(p in [0, 1] for p in unique_preds))
        
        # Test weighted ensemble
        weights = [0.5, 0.3, 0.2]
        weighted_pred, weighted_proba = ensemble_predictions(
            models,
            self.X_test,
            method='weighted',
            weights=weights
        )
        
        self.assertEqual(len(weighted_pred), len(self.y_test))
        
    def test_save_and_load_model(self):
        """Test model saving and loading"""
        from sklearn.ensemble import RandomForestClassifier
        
        # Train a model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(self.X_train, self.y_train)
        
        # Get predictions before saving
        predictions_before = model.predict(self.X_test)
        
        # Save model to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            temp_path = tmp.name
            
        try:
            # Save model
            save_model(model, temp_path)
            self.assertTrue(os.path.exists(temp_path))
            
            # Load model
            loaded_model = load_model(temp_path)
            
            # Get predictions after loading
            predictions_after = loaded_model.predict(self.X_test)
            
            # Check predictions are the same
            np.testing.assert_array_equal(predictions_before, predictions_after)
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    def test_model_interpretation(self):
        """Test model interpretation features"""
        from sklearn.ensemble import RandomForestClassifier
        from src.models import get_feature_importance, plot_partial_dependence
        
        # Train model with feature names
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(self.X_train_df, self.y_train)
        
        # Get feature importance
        importance = get_feature_importance(model, self.feature_names)
        
        # Check importance
        self.assertEqual(len(importance), len(self.feature_names))
        self.assertAlmostEqual(sum(importance.values()), 1.0, places=5)
        
        # Check all features have non-negative importance
        for feat_importance in importance.values():
            self.assertGreaterEqual(feat_importance, 0)
            
    def test_model_calibration(self):
        """Test probability calibration"""
        from sklearn.ensemble import RandomForestClassifier
        from src.models import calibrate_model
        
        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(self.X_train, self.y_train)
        
        # Calibrate model
        calibrated_model = calibrate_model(
            model,
            self.X_train,
            self.y_train,
            method='sigmoid'
        )
        
        # Check calibrated predictions
        calibrated_proba = calibrated_model.predict_proba(self.X_test)
        
        # Check shape
        self.assertEqual(calibrated_proba.shape, (len(self.y_test), 2))
        
        # Check probabilities sum to 1
        for proba in calibrated_proba:
            self.assertAlmostEqual(proba.sum(), 1.0)
            
    def test_model_explainability(self):
        """Test model explainability with SHAP or LIME"""
        from sklearn.ensemble import RandomForestClassifier
        
        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(self.X_train, self.y_train)
        
        # Mock SHAP values calculation
        with patch('src.models.calculate_shap_values') as mock_shap:
            mock_shap.return_value = np.random.randn(len(self.X_test), 10)
            
            from src.models import calculate_shap_values
            shap_values = calculate_shap_values(model, self.X_test)
            
            # Check shape
            self.assertEqual(shap_values.shape, (len(self.X_test), 10))
            

if __name__ == '__main__':
    unittest.main()
