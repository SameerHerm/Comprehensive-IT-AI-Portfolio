import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, GridSearchCV
import optuna
import joblib
import yaml
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.models = {}
        self.best_models = {}
        
    def initialize_models(self):
        """Initialize base models"""
        self.models = {
            'logistic_regression': LogisticRegression(random_state=42),
            'random_forest': RandomForestClassifier(random_state=42),
            'xgboost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'lightgbm': lgb.LGBMClassifier(random_state=42, verbose=-1),
            'neural_network': MLPClassifier(random_state=42, max_iter=1000)
        }
        
        logger.info("Models initialized")
    
    def train_base_models(self, X_train, y_train):
        """Train base models without hyperparameter tuning"""
        trained_models = {}
        
        for name, model in self.models.items():
            if name in self.config['model']['models_to_train']:
                logger.info(f"Training {name}...")
                
                # Cross-validation scores
                cv_scores = cross_val_score(
                    model, X_train, y_train,
                    cv=self.config['model']['cross_validation_folds'],
                    scoring=self.config['model']['scoring_metric']
                )
                
                logger.info(f"{name} CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                
                # Train final model
                model.fit(X_train, y_train)
                trained_models[name] = model
        
        return trained_models
    
    def optimize_hyperparameters_optuna(self, X_train, y_train):
        """Optimize hyperparameters using Optuna"""
        optimized_models = {}
        
        def objective_rf(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                'random_state': 42
            }
            
            model = RandomForestClassifier(**params)
            score = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc').mean()
            return score
        
        def objective_xgb(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': 42,
                'eval_metric': 'logloss'
            }
            
            model = xgb.XGBClassifier(**params)
            score = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc').mean()
            return score
        
        # Optimize Random Forest
        if 'random_forest' in self.config['model']['models_to_train']:
            logger.info("Optimizing Random Forest hyperparameters...")
            study_rf = optuna.create_study(direction='maximize')
            study_rf.optimize(objective_rf, n_trials=50)
            
            best_rf = RandomForestClassifier(**study_rf.best_params)
            best_rf.fit(X_train, y_train)
            optimized_models['random_forest'] = best_rf
            logger.info(f"Best RF score: {study_rf.best_value:.4f}")
        
        # Optimize XGBoost
        if 'xgboost' in self.config['model']['models_to_train']:
            logger.info("Optimizing XGBoost hyperparameters...")
            study_xgb = optuna.create_study(direction='maximize')
            study_xgb.optimize(objective_xgb, n_trials=50)
            
            best_xgb = xgb.XGBClassifier(**study_xgb.best_params)
            best_xgb.fit(X_train, y_train)
            optimized_models['xgboost'] = best_xgb
            logger.info(f"Best XGB score: {study_xgb.best_value:.4f}")
        
        return optimized_models
    
    def train_ensemble_model(self, X_train, y_train, base_models):
        """Create ensemble model using voting classifier"""
        from sklearn.ensemble import VotingClassifier
        
        # Select top 3 models for ensemble
        model_list = [(name, model) for name, model in base_models.items()]
        
        ensemble_model = VotingClassifier(
            estimators=model_list,
            voting='soft'
        )
        
        ensemble_model.fit(X_train, y_train)
        logger.info("Ensemble model trained")
        
        return ensemble_model
    
    def save_models(self, models):
        """Save trained models"""
        import os
        os.makedirs('models', exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for name, model in models.items():
            filename = f'models/{name}_model_{timestamp}.pkl'
            joblib.dump(model, filename)
            logger.info(f"Saved {name} model to {filename}")
        
        # Save best model (highest CV score)
        # This would be determined by evaluation metrics
        best_model_name = list(models.keys())[0]  # Placeholder
        joblib.dump(models[best_model_name], 'models/best_model.pkl')
        
        with open('models/model_info.txt', 'w') as f:
            f.write(f"Models trained on: {timestamp}\n")
            f.write(f"Best model: {best_model_name}\n")
            f.write(f"Available models: {list(models.keys())}\n")
    
    def train_all_models(self):
        """Complete model training pipeline"""
        # Load processed data
        data = joblib.load('data/processed/processed_data.pkl')
        X_train, y_train = data['X_train'], data['y_train']
        
        # Initialize models
        self.initialize_models()
        
        # Train base models
        trained_models = self.train_base_models(X_train, y_train)
        
        # Hyperparameter optimization
        if self.config['model']['hyperparameter_tuning']:
            optimized_models = self.optimize_hyperparameters_optuna(X_train, y_train)
            trained_models.update(optimized_models)
        
        # Train ensemble model
        ensemble_model = self.train_ensemble_model(X_train, y_train, trained_models)
        trained_models['ensemble'] = ensemble_model
        
        # Save models
        self.save_models(trained_models)
        
        logger.info("Model training completed")
        return trained_models

if __name__ == "__main__":
    trainer = ModelTrainer()
    models = trainer.train_all_models()
