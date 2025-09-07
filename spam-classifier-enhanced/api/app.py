from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.predictor import SpamPredictor
import logging

app = Flask(__name__)
CORS(app)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize predictor
predictor = SpamPredictor()

@app.route('/')
def home():
    return jsonify({
        'message': 'Spam Classifier API',
        'version': '2.0.0',
        'endpoints': {
            '/predict': 'POST - Predict single text',
            '/batch_predict': 'POST - Predict multiple texts',
            '/health': 'GET - Check API health',
            '/models': 'GET - List available models'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'models_loaded': predictor.is_loaded()})

@app.route('/models', methods=['GET'])
def list_models():
    return jsonify({'available_models': predictor.get_available_models()})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        model_name = data.get('model', 'xgboost')
        
        # Make prediction
        result = predictor.predict(text, model_name)
        
        return jsonify({
            'text': text[:100] + '...' if len(text) > 100 else text,
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'model_used': model_name
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        data = request.get_json()
        
        if 'texts' not in data:
            return jsonify({'error': 'No texts provided'}), 400
        
        texts = data['texts']
        model_name = data.get('model', 'xgboost')
        
        # Make predictions
        results = predictor.batch_predict(texts, model_name)
        
        return jsonify({
            'predictions': results,
            'model_used': model_name,
            'total_processed': len(results)
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
