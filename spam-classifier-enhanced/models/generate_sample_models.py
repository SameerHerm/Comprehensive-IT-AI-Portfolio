```python
"""
Script to generate sample model files for testing
This creates placeholder models for demonstration purposes
"""

import joblib
import json
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_sample_models():
    """Create sample model files for testing"""
    
    print("Creating sample models...")
    
    # Create sample data
    X_sample = np.random.rand(100, 20)
    y_sample = np.random.randint(0, 2, 100)
    
    # Initialize models
    models = {
        'naive_bayes_model.pkl': MultinomialNB(),
        'svm_model.pkl': SVC(probability=True),
        'random_forest_model.pkl': RandomForestClassifier(n_estimators=10, random_state=42),
        'logistic_regression_model.pkl': LogisticRegression(random_state=42),
        'xgboost_model.pkl': XGBClassifier(n_estimators=10, random_state=42, use_label_encoder=False, eval_metric='logloss')
    }
    
    # Train and save models
    for filename, model in models.items():
        try:
            # Make X_sample non-negative for MultinomialNB
            if 'naive_bayes' in filename:
                X_train = np.abs(X_sample)
            else:
                X_train = X_sample
            
            model.fit(X_train, y_sample)
            joblib.dump(model, filename)
            print(f"✅ Created {filename}")
        except Exception as e:
            print(f"❌ Error creating {filename}: {e}")
    
    # Save the best model (using Random Forest as example)
    best_model = RandomForestClassifier(n_estimators=10, random_state=42)
    best_model.fit(X_sample, y_sample)
    joblib.dump(best_model, 'best_model.pkl')
    print("✅ Created best_model.pkl")
    
    # Create sample preprocessor
    from src.data_preprocessing import DataPreprocessor
    preprocessor = DataPreprocessor()
    joblib.dump(preprocessor, 'preprocessor.pkl')
    print("✅ Created preprocessor.pkl")
    
    # Create sample feature engineer
    from src.feature_engineering import FeatureEngineer
    feature_engineer = FeatureEngineer()
    
    # Initialize TF-IDF vectorizer with sample data
    sample_texts = [
        "This is a sample email",
        "Another sample text for testing",
        "Spam detection system test"
    ]
    feature_engineer.tfidf_vectorizer = TfidfVectorizer(max_features=100)
    feature_engineer.tfidf_vectorizer.fit(sample_texts)
    
    joblib.dump(feature_engineer, 'feature_engineer.pkl')
    print("✅ Created feature_engineer.pkl")
    
    # Create sample pipeline
    class SamplePipeline:
        def __init__(self):
            self.model = best_model
            self.preprocessor = preprocessor
            self.feature_engineer = feature_engineer
        
        def predict(self, text):
            return {
                'prediction': 'ham',
                'confidence': 0.95,
                'spam_probability': 0.05,
                'ham_probability': 0.95
            }
    
    pipeline = SamplePipeline()
    joblib.dump(pipeline, 'pipeline.pkl')
    print("✅ Created pipeline.pkl")
    
    # Create performance report
    performance_report = {
        "project": "Spam Classifier Enhanced",
        "date": "2024-01-15 12:00:00",
        "dataset_size": 5000,
        "features_count": 1007,
        "best_model": "xgboost",
        "test_accuracy": 0.989,
        "test_f1_score": 0.989,
        "all_models_performance": {
            "accuracy": {
                "xgboost": 0.989,
                "svm": 0.985,
                "random_forest": 0.982,
                "naive_bayes": 0.978,
                "logistic_regression": 0.975
            },
            "precision": {
                "xgboost": 0.990,
                "svm": 0.987,
                "random_forest": 0.984,
                "naive_bayes": 0.982,
                "logistic_regression": 0.978
            },
            "recall": {
                "xgboost": 0.988,
                "svm": 0.983,
                "random_forest": 0.980,
                "naive_bayes": 0.975,
                "logistic_regression": 0.972
            },
            "f1_score": {
                "xgboost": 0.989,
                "svm": 0.985,
                "random_forest": 0.982,
                "naive_bayes": 0.978,
                "logistic_regression": 0.975
            }
        },
        "false_positive_rate": 0.012,
        "false_negative_rate": 0.009
    }
    
    with open('performance_report.json', 'w') as f:
        json.dump(performance_report, f, indent=4)
    print("✅ Created performance_report.json")
    
    print("\n🎉 All sample models created successfully!")
    print("\nNote: These are sample models for testing purposes.")
    print("Run the training script or notebook to generate real trained models.")

if __name__ == "__main__":
    # Change to models directory
    models_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(models_dir)
    
    create_sample_models()
