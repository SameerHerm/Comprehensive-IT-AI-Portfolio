import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib
import logging
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Train and manage multiple ML models"""
    
    def __init__(self):
        self.models = {
            'naive_bayes': MultinomialNB(),
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
            'svm': SVC(kernel='rbf', probability=True, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boost': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'xgboost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        }
        
        self.best_params = {}
        self.trained_models = {}
        
    def train_model(self, model_name: str, X_train, y_train, 
                   use_grid_search: bool = False) -> Any:
        """Train a specific model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        if use_grid_search:
            model = self._grid_search(model, model_name, X_train, y_train)
        else:
            model.fit(X_train, y_train)
        
        self.trained_models[model_name] = model
        logger.info(f"Model {model_name} trained successfully")
        
        return model
    
    def _grid_search(self, model: Any, model_name: str, X_train, y_train) -> Any:
        """Perform grid search for hyperparameter tuning"""
        param_grids = {
            'naive_bayes': {'alpha': [0.1, 0.5, 1.0, 2.0]},
            'logistic_regression': {'C': [0.1, 1.0, 10.0], 'penalty': ['l2']},
            'svm': {'C': [0.1, 1.0, 10.0], 'gamma': ['scale', 'auto']},
            'random_forest': {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, None]},
            'gradient_boost': {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1, 0.2]},
            'xgboost': {'n_estimators': [50, 100], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1]}
        }
        
        if model_name in param_grids:
            grid_search = GridSearchCV(
                model, 
                param_grids[model_name], 
                cv=5, 
                scoring='accuracy',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            self.best_params[model_name] = grid_search.best_params_
            logger.info(f"Best params for {model_name}: {grid_search.best_params_}")
            return grid_search.best_estimator_
        
        model.fit(X_train, y_train)
        return model
    
    def train_all_models(self, X_train, y_train, use_grid_search: bool = False):
        """Train all available models"""
        results = {}
        
        for model_name in self.models.keys():
            logger.info(f"Training {model_name}...")
            try:
                model = self.train_model(model_name, X_train, y_train, use_grid_search)
                
                # Perform cross-validation
                scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                results[model_name] = {
                    'model': model,
                    'cv_scores': scores,
                    'mean_cv_score': scores.mean(),
                    'std_cv_score': scores.std()
                }
                
                logger.info(f"{model_name} - CV Score: {scores.mean():.4f} (+/- {scores.std():.4f})")
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        return results
    
    def save_model(self, model_name: str, filepath: str):
        """Save a trained model"""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        joblib.dump(self.trained_models[model_name], filepath)
        logger.info(f"Model {model_name} saved to {filepath}")
    
    def load_model(self, filepath: str) -> Any:
        """Load a saved model"""
        model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return model
    
    def get_feature_importance(self, model_name: str, feature_names: list) -> pd.DataFrame:
        """Get feature importance for tree-based models"""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.trained_models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            logger.warning(f"Model {model_name} doesn't support feature importance")
            return None
