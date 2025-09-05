from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class PredictionAPI:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.feature_names = None
        self.load_models()
    
    def load_models(self):
        """Load trained models and preprocessors"""
        try:
            self.model = joblib.load('models/best_model.pkl')
            self.scaler = joblib.load('models/scaler.pkl')
            self.label_encoders = joblib.load('models/label_encoders.pkl')
            
            # Load feature names
            data = joblib.load('data/processed/processed_data.pkl')
            self.feature_names = data['feature_names']
            
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def preprocess_input(self, input_data):
        """Preprocess input data for prediction"""
        try:
            df = pd.DataFrame([input_data])
            
            # Feature engineering (same as training)
            df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
            df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
            
            # Age groups
            age = df['age'].iloc[0]
            if age <= 40:
                df['age_group'] = 'young'
            elif age <= 55:
                df['age_group'] = 'middle_aged'
            elif age <= 70:
                df['age_group'] = 'senior'
            else:
                df['age_group'] = 'elderly'
            
            # BMI categories
            bmi = df['bmi'].iloc[0]
            if bmi < 18.5:
                df['bmi_category'] = 'underweight'
            elif bmi < 25:
                df['bmi_category'] = 'normal'
            elif bmi < 30:
                df['bmi_category'] = 'overweight'
            else:
                df['bmi_category'] = 'obese'
            
            # Risk factors count
            risk_factors = ['smoking', 'alcohol', 'cholesterol', 'glucose']
            df['risk_factors_count'] = df[risk_factors].sum(axis=1)
            
            # Encode categorical features
            categorical_cols = ['gender', 'smoking', 'alcohol', 'physical_activity', 'age_group', 'bmi_category']
            for col in categorical_cols:
                if col in self.label_encoders and col in df.columns:
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
            
            # Scale numerical features
            numerical_cols = ['age', 'height', 'weight', 'systolic_bp', 'diastolic_bp', 
                            'cholesterol', 'glucose', 'bmi', 'pulse_pressure', 'risk_factors_count']
            numerical_cols = [col for col in numerical_cols if col in df.columns]
            df[numerical_cols] = self.scaler.transform(df[numerical_cols])
            
            # Ensure all required features are present
            for feature in self.feature_names:
                if feature not in df.columns:
                    df[feature] = 0
            
            # Reorder columns to match training data
            df = df[self.feature_names]
            
            return df
        
        except Exception as e:
            logger.error(f"Error preprocessing input: {e}")
            raise
    
    def predict(self, input_data):
        """Make prediction"""
        try:
            # Preprocess input
            processed_data = self.preprocess_input(input_data)
            
            # Make prediction
            prediction = self.model.predict(processed_data)[0]
            prediction_proba = self.model.predict_proba(processed_data)[0]
            
            # Calculate risk level
            risk_probability = prediction_proba[1]
            if risk_probability < 0.3:
                risk_level = "Low"
            elif risk_probability < 0.7:
                risk_level = "Medium"
            else:
                risk_level = "High"
            
            result = {
                'prediction': int(prediction),
                'risk_probability': float(risk_probability),
                'risk_level': risk_level,
                'confidence': float(max(prediction_proba)),
                'timestamp': datetime.now().isoformat()
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise

# Initialize prediction API
predictor = PredictionAPI()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/predict', methods=['POST'])
def predict_cardiovascular_risk():
    """Prediction endpoint"""
    try:
        # Get input data
        input_data = request.json
        
        # Validate required fields
        required_fields = ['age', 'gender', 'height', 'weight', 'systolic_bp', 
                          'diastolic_bp', 'cholesterol', 'glucose', 'smoking', 
                          'alcohol', 'physical_activity']
        
        missing_fields = [field for field in required_fields if field not in input_data]
        if missing_fields:
            return jsonify({'error': f'Missing required fields: {missing_fields}'}), 400
        
        # Make prediction
        result = predictor.predict(input_data)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint"""
    try:
        input_data = request.json
        
        if not isinstance(input_data, list):
            return jsonify({'error': 'Input should be a list of objects'}), 400
        
        results = []
        for i, data in enumerate(input_data):
            try:
                result = predictor.predict(data)
                result['id'] = i
                results.append(result)
            except Exception as e:
                results.append({'id': i, 'error': str(e)})
        
        return jsonify(results)
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    try:
        info = {
            'model_type': type(predictor.model).__name__,
            'feature_count': len(predictor.feature_names),
            'features': predictor.feature_names,
            'timestamp': datetime.now().isoformat()
        }
        return jsonify(info)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
