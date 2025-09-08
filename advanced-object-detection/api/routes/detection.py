"""
Detection API routes
"""

from flask import Blueprint, request, jsonify
import numpy as np
import cv2
from werkzeug.utils import secure_filename
import os
import uuid
from datetime import datetime
import torch
from PIL import Image
import io
import base64

detection_bp = Blueprint('detection', __name__)

# Import detection modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.models.yolo import YOLODetector
from src.detection.utils import draw_boxes

# Initialize models (load once)
MODELS = {}

def load_models():
    """Load detection models"""
    global MODELS
    
    # Load YOLO
    MODELS['yolo'] = YOLODetector(num_classes=80)
    MODELS['yolo'].load_state_dict(torch.load('models/yolo.pth', map_location='cpu'))
    MODELS['yolo'].eval()
    
    # Add other models as needed
    
def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@detection_bp.route('/detect', methods=['POST'])
def detect():
    """Main detection endpoint"""
    try:
        # Check if image is provided
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file format'}), 400
        
        # Get model type
        model_type = request.form.get('model', 'yolo')
        confidence_threshold = float(request.form.get('confidence', 0.5))
        
        # Read and process image
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Perform detection
        detections = perform_detection(image_rgb, model_type, confidence_threshold)
        
        # Draw boxes on image
        annotated_image = draw_boxes(image_rgb, detections)
        
        # Convert annotated image to base64
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Prepare response
        response = {
            'success': True,
            'detections': format_detections(detections),
            'annotated_image': f'data:image/jpeg;base64,{image_base64}',
            'model_used': model_type,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@detection_bp.route('/detect-batch', methods=['POST'])
def detect_batch():
    """Batch detection endpoint"""
    try:
        if 'images' not in request.files:
            return jsonify({'error': 'No images provided'}), 400
        
        files = request.files.getlist('images')
        model_type = request.form.get('model', 'yolo')
        confidence_threshold = float(request.form.get('confidence', 0.5))
        
        results = []
        
        for file in files:
            if file and allowed_file(file.filename):
                # Process each image
                image_bytes = file.read()
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Perform detection
                detections = perform_detection(image_rgb, model_type, confidence_threshold)
                
                results.append({
                    'filename': secure_filename(file.filename),
                    'detections': format_detections(detections)
                })
        
        return jsonify({
            'success': True,
            'results': results,
            'total_processed': len(results)
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@detection_bp.route('/realtime-detect', methods=['POST'])
def realtime_detect():
    """Real-time detection endpoint for video frames"""
    try:
        if 'frame' not in request.files:
            return jsonify({'error': 'No frame provided'}), 400
        
        file = request.files['frame']
        model_type = request.form.get('model', 'yolo')
        confidence_threshold = float(request.form.get('confidence', 0.5))
        
        # Read frame
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Perform detection (optimized for speed)
        detections = perform_detection(image_rgb, model_type, confidence_threshold, optimize_speed=True)
        
        return jsonify({
            'success': True,
            'detections': format_detections(detections)
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def perform_detection(image, model_type, confidence_threshold, optimize_speed=False):
    """Perform object detection on image"""
    if model_type not in MODELS:
        raise ValueError(f"Model {model_type} not available")
    
    model = MODELS[model_type]
    
    # Preprocess image
    if optimize_speed:
        # Resize for faster processing
        image = cv2.resize(image, (640, 640))
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0)
    
    # Perform detection
    with torch.no_grad():
        predictions = model(image_tensor)
    
    # Post-process predictions
    detections = {
        'boxes': predictions['boxes'][predictions['scores'] > confidence_threshold],
        'labels': predictions['labels'][predictions['scores'] > confidence_threshold],
        'scores': predictions['scores'][predictions['scores'] > confidence_threshold]
    }
    
    return detections

def format_detections(detections):
    """Format detections for JSON response"""
    formatted = []
    
    for box, label, score in zip(detections['boxes'], 
                                 detections['labels'], 
                                 detections['scores']):
        formatted.append({
            'bbox': box.tolist(),
            'class': int(label),
            'class_name': get_class_name(int(label)),
            'confidence': float(score),
            'area': calculate_bbox_area(box)
        })
    
    return formatted

def get_class_name(class_id):
    """Get class name from ID (COCO classes)"""
    # Simplified - in practice, load from config file
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow'
        # ... add all 80 COCO classes
    ]
    
    if 0 <= class_id < len(COCO_CLASSES):
        return COCO_CLASSES[class_id]
    return f'class_{class_id}'

def calculate_bbox_area(box):
    """Calculate bounding box area"""
    x1, y1, x2, y2 = box
    return float((x2 - x1) * (y2 - y1))

# Initialize models on module load
load_models()
