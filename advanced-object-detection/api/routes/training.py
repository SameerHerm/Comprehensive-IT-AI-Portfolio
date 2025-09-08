"""
Training API routes
"""

from flask import Blueprint, request, jsonify
import os
import json
from datetime import datetime
import threading

training_bp = Blueprint('training', __name__)

# Global training status
TRAINING_STATUS = {}

@training_bp.route('/train', methods=['POST'])
def start_training():
    """Start model training"""
    try:
        data = request.get_json()
        
        # Extract parameters
        model_type = data.get('model', 'yolo')
        dataset_path = data.get('dataset_path')
        epochs = data.get('epochs', 100)
        batch_size = data.get('batch_size', 16)
        learning_rate = data.get('learning_rate', 0.001)
        
        # Generate training ID
        training_id = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize training status
        TRAINING_STATUS[training_id] = {
            'status': 'initializing',
            'progress': 0,
            'current_epoch': 0,
            'total_epochs': epochs,
            'metrics': {},
            'start_time': datetime.now().isoformat()
        }
        
        # Start training in background thread
        thread = threading.Thread(
            target=run_training,
            args=(training_id, model_type, dataset_path, epochs, batch_size, learning_rate)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'training_id': training_id,
            'message': 'Training started successfully'
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@training_bp.route('/train/status/<training_id>', methods=['GET'])
def get_training_status(training_id):
    """Get training status"""
    if training_id not in TRAINING_STATUS:
        return jsonify({'error': 'Training ID not found'}), 404
    
    return jsonify(TRAINING_STATUS[training_id]), 200

@training_bp.route('/train/stop/<training_id>', methods=['POST'])
def stop_training(training_id):
    """Stop ongoing training"""
    if training_id not in TRAINING_STATUS:
        return jsonify({'error': 'Training ID not found'}), 404
    
    TRAINING_STATUS[training_id]['status'] = 'stopping'
    
    return jsonify({
        'success': True,
        'message': 'Training stop requested'
    }), 200

@training_bp.route('/models', methods=['GET'])
def list_models():
    """List available trained models"""
    models_dir = 'models/trained'
    models = []
    
    if os.path.exists(models_dir):
        for filename in os.listdir(models_dir):
            if filename.endswith('.pth'):
                model_path = os.path.join(models_dir, filename)
                model_info = {
                    'name': filename,
                    'size': os.path.getsize(model_path),
                    'modified': datetime.fromtimestamp(os.path.getmtime(model_path)).isoformat()
                }
                
                # Load metadata if exists
                meta_path = model_path.replace('.pth', '_meta.json')
                if os.path.exists(meta_path):
                    with open(meta_path, 'r') as f:
                        model_info['metadata'] = json.load(f)
                
                models.append(model_info)
    
    return jsonify({
        'success': True,
        'models': models,
        'count': len(models)
    }), 200

@training_bp.route('/evaluate', methods=['POST'])
def evaluate_model():
    """Evaluate model on test dataset"""
    try:
        data = request.get_json()
        
        model_path = data.get('model_path')
        test_dataset = data.get('test_dataset')
        
        # Perform evaluation (simplified)
        metrics = {
            'mAP': 0.85,
            'precision': 0.88,
            'recall': 0.82,
            'f1_score': 0.85
        }
        
        return jsonify({
            'success': True,
            'metrics': metrics
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def run_training(training_id, model_type, dataset_path, epochs, batch_size, learning_rate):
    """Run training in background"""
    try:
        # Update status
        TRAINING_STATUS[training_id]['status'] = 'running'
        
        # Import training modules
        from src.training.trainer import ObjectDetectionTrainer
        
        # Simulate training (replace with actual training)
        for epoch in range(epochs):
            if TRAINING_STATUS[training_id]['status'] == 'stopping':
                break
            
            # Update progress
            TRAINING_STATUS[training_id]['current_epoch'] = epoch + 1
            TRAINING_STATUS[training_id]['progress'] = ((epoch + 1) / epochs) * 100
            
            # Simulate metrics
            TRAINING_STATUS[training_id]['metrics'] = {
                'loss': 0.5 - (epoch * 0.01),
                'accuracy': 0.7 + (epoch * 0.003),
                'learning_rate': learning_rate * (0.95 ** epoch)
            }
            
            # Sleep to simulate training time
            import time
            time.sleep(2)
        
        # Training complete
        TRAINING_STATUS[training_id]['status'] = 'completed'
        TRAINING_STATUS[training_id]['end_time'] = datetime.now().isoformat()
        
    except Exception as e:
        TRAINING_STATUS[training_id]['status'] = 'failed'
        TRAINING_STATUS[training_id]['error'] = str(e)
