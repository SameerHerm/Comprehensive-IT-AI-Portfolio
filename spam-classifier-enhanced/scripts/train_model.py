#!/usr/bin/env python
"""
Training script for spam classifier models
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import yaml
import logging

from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer
from src.model_evaluation import ModelEvaluator
from src.utils import Utils

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main(args):
    """Main training pipeline"""
    
    # Load configuration
    config = load_config(args.config)
    
    # Create necessary directories
    Utils.create_directories(['models', 'plots', 'reports', 'logs'])
    
    logger.info("Starting training pipeline...")
    
    # Load data
    logger.info(f"Loading data from {config['data']['dataset_path']}...")
    df = pd.read_csv(config['data']['dataset_path'])
    
    # Initialize components
    preprocessor = DataPreprocessor()
    feature_engineer = FeatureEngineer()
    trainer = ModelTrainer()
    evaluator = ModelEvaluator()
    
    # Preprocess data
    logger.info("Preprocessing data...")
    df = preprocessor.preprocess_dataset(df)
    
    # Create features
    logger.info("Creating features...")
    
    # TF-IDF features
    tfidf_features = feature_engineer.create_tfidf_features(
        df['cleaned_text'],
        max_features=config['features']['tfidf_max_features'],
        ngram_range=tuple(config['features']['tfidf_ngram_range'])
    )
    
    # Character features
    char_features = feature_engineer.create_char_features(df['text'])
    
    # Combine features
    from scipy.sparse import hstack
    X = hstack([tfidf_features, char_features])
    y = df['label_encoded']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state'],
        stratify=y
    )
    
    logger.info(f"Train set size: {X_train.shape}")
    logger.info(f"Test set size: {X_test.shape}")
    
    # Train models
    if args.model == 'all':
        logger.info("Training all models...")
        results = trainer.train_all_models(
            X_train, y_train,
            use_grid_search=config['models']['use_grid_search']
        )
        
        # Evaluate all models
        trained_models = {name: result['model'] for name, result in results.items()}
        comparison_df = evaluator.evaluate_all_models(trained_models, X_test, y_test)
        
        print("\nModel Comparison:")
        print(comparison_df.to_string())
        
        # Save models
        if config['models']['save_models']:
            for model_name, result in results.items():
                model_path = f"models/{model_name}_model.pkl"
                trainer.save_model(model_name, model_path)
        
        # Generate plots
        if config['evaluation']['generate_plots']:
            evaluator.plot_roc_curves(trained_models, X_test, y_test, 'plots/roc_curves.png')
            evaluator.plot_precision_recall_curves(trained_models, X_test, y_test)
        
        # Generate report
        if config['evaluation']['generate_report']:
            evaluator.generate_report(config['evaluation']['report_path'])
    
    else:
        # Train specific model
        logger.info(f"Training {args.model}...")
        model = trainer.train_model(
            args.model, X_train, y_train,
            use_grid_search=config['models']['use_grid_search']
        )
        
        # Evaluate model
        metrics = evaluator.evaluate_model(model, X_test, y_test, args.model)
        print(Utils.get_model_metrics_summary(metrics))
        
        # Save model
        if config['models']['save_models']:
            model_path = f"models/{args.model}_model.pkl"
            trainer.save_model(args.model, model_path)
        
        # Plot confusion matrix
        if config['evaluation']['generate_plots']:
            evaluator.plot_confusion_matrix(args.model, f'plots/{args.model}_confusion.png')
    
    # Save feature engineer for inference
    joblib.dump(feature_engineer, 'models/feature_engineer.pkl')
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    
    logger.info("Training pipeline completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train spam classifier models')
    parser.add_argument('--model', type=str, default='all',
                      choices=['all', 'naive_bayes', 'svm', 'random_forest', 
                              'logistic_regression', 'xgboost'],
                      help='Model to train')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                      help='Path to configuration file')
    
    args = parser.parse_args()
    main(args)
