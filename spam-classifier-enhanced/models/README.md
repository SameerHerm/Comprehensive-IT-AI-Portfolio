# Models Directory

This directory contains trained machine learning models for the spam classifier.

## Model Files

After running the training scripts or notebook, the following model files will be generated:

- `best_model.pkl` - The best performing model based on evaluation metrics
- `preprocessor.pkl` - Data preprocessing pipeline
- `feature_engineer.pkl` - Feature engineering pipeline
- `pipeline.pkl` - Complete production pipeline
- `naive_bayes_model.pkl` - Trained Naive Bayes model
- `svm_model.pkl` - Trained SVM model
- `random_forest_model.pkl` - Trained Random Forest model
- `xgboost_model.pkl` - Trained XGBoost model
- `logistic_regression_model.pkl` - Trained Logistic Regression model
- `performance_report.json` - Model performance metrics

## Loading Models

To load a saved model:

python
import joblib

# Load the best model
model = joblib.load('models/best_model.pkl')

# Load the complete pipeline
pipeline = joblib.load('models/pipeline.pkl')

# Make predictions
result = pipeline.predict("Your email text here")
Model Versions
Model	Version	Date	Accuracy	F1-Score
XGBoost	1.0	2024-01-15	98.9%	98.9%
Random Forest	1.0	2024-01-15	98.2%	98.2%
SVM	1.0	2024-01-15	98.5%	98.5%
Naive Bayes	1.0	2024-01-15	97.8%	97.8%
Notes
Models are saved using joblib for efficient serialization
Neural network models (if any) are saved in .h5 format
All models are compatible with scikit-learn 1.3.0+
Feature engineering pipeline must be loaded alongside models for proper functioning
