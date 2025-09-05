import torch
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import time
import logging
from pathlib import Path
import threading
import queue
from abc import ABC, abstractmethod

from ..models.yolo import YOLODetector
# from ..models.rcnn import RCNNDetector  # You'll create this
# from ..models.ssd import SSDDetector    # You'll create this

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseDetector(ABC):
    """Abstract base class for all detectors"""
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Dict]:
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict:
        pass

class MultiModelDetector:
    """Unified detector that can use multiple models"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        import yaml
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.available_models = {}
        self.current_model = None
        self.current_model_name = None
        
        # Initialize available models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize available detection models"""
        try:
            # Initialize YOLO models
            for model_name in ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l']:
                try:
                    self.available_models[model_name] = YOLODetector(model_name=model_name)
                    logger.info(f"Initialized {model_name}")
                except Exception as e:
                    logger.warning(f"Failed to initialize {model_name}: {e}")
            
            # Set default model
            default_model = self.config['models']['default_model']
            if default_model in self.available_models:
                self.set_model(default_model)
            else:
                # Use first available model
                if self.available_models:
                    first_model = list(self.available_models.keys())[0]
                    self.set_model(first_model)
                    logger.info(f"Default model not available, using {first_model}")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
    
    def set_model(self, model_name: str):
        """Set the current detection model"""
        if model_name in self.available_models:
            self.current_model = self.available_models[model_name]
            self.current_model_name = model_name
            logger.info(f"Set current model to {model_name}")
        else:
            raise ValueError(f"Model {model_name} not available. Available models: {list(self.available_models.keys())}")
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return list(self.available_models.keys())
    
    def detect_image(
        self,
        image_input: Union[str, np.ndarray],
        conf_threshold: float = None,
        return_annotated: bool = True
    ) -> Dict:
        """Detect objects in an image"""
        if self.current_model is None:
            raise RuntimeError("No model selected. Use set_model() first.")
        
        try:
            start_time = time.time()
            
            # Handle different input types
            if isinstance(image_input, str):
                # File path
                result = self.current_model.detect_image(
                    image_path=image_input,
                    conf_threshold=conf_threshold,
                    return_annotated=return_annotated
                )
            else:
                # NumPy array
                # Save temporarily and process
                temp_path = "temp_detection_image.jpg"
                cv2.imwrite(temp_path, image_input)
                result = self.current_model.detect_image(
                    image_path=temp_path,
                    conf_threshold=conf_threshold,
                    return_annotated=return_annotated
                )
                # Clean up
                Path(temp_path).unlink(missing_ok=True)
            
            inference_time = time.time() - start_time
            
            # Add metadata
            result['model_name'] = self.current_model_name
            result['inference_time'] = inference_time
            result['timestamp'] = time.time()
            
            return result
            
        except Exception as e:
            logger.error(f"Error during detection: {e}")
            raise
    
    def detect_batch(
        self,
        image_paths: List[str],
        conf_threshold: float = None,
        batch_size: int = 4
    ) -> List[Dict]:
        """Detect objects in multiple images"""
        results = []
        total_images = len(image_paths)
        
        logger.info(f"Processing {total_images} images in batches of {batch_size}")
        
        for i in range(0, total_images, batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_results = []
            
            for image_path in batch_paths:
                try:
                    result = self.detect_image(
                        image_input=image_path,
                        conf_threshold=conf_threshold,
                        return_annotated=False  # Skip annotation for batch processing
                    )
                    result['image_path'] = image_path
                    batch_results.append(result)
                except Exception as e:
                    logger.error(f"Error processing {image_path}: {e}")
                    batch_results.append({
                        'image_path': image_path,
                        'error': str(e),
                        'detections': []
                    })
            
            results.extend(batch_results)
            logger.info(f"Processed batch {i//batch_size + 1}/{(total_images-1)//batch_size + 1}")
        
        return results
    
    def benchmark_models(
        self,
        test_images: List[str],
        conf_threshold: float = 0.25,
        num_runs: int = 3
    ) -> Dict:
        """Benchmark all available models"""
        benchmark_results = {}
        
        logger.info(f"Benchmarking {len(self.available_models)} models on {len(test_images)} images")
        
        for model_name in self.available_models:
            logger.info(f"Benchmarking {model_name}...")
            self.set_model(model_name)
            
            total_time = 0
            total_detections = 0
            inference_times = []
            
            for run in range(num_runs):
                run_start = time.time()
                
                for image_path in test_images:
                    try:
                        start_time = time.time()
                        result = self.detect_image(
                            image_input=image_path,
                            conf_threshold=conf_threshold,
                            return_annotated=False
                        )
                        inference_time = time.time() - start_time
                        inference_times.append(inference_time)
                        total_detections += result['num_detections']
                    except Exception as e:
                        logger.warning(f"Error in {model_name} on {image_path}: {e}")
                
                run_time = time.time() - run_start
                total_time += run_time
            
            # Calculate statistics
            avg_inference_time = np.mean(inference_times) if inference_times else 0
            std_inference_time = np.std(inference_times) if inference_times else 0
            fps = 1 / avg_inference_time if avg_inference_time > 0 else 0
            
            benchmark_results[model_name] = {
                'avg_inference_time': avg_inference_time,
                'std_inference_time': std_inference_time,
                'fps': fps,
                'total_time': total_time,
                'total_detections': total_detections,
                'avg_detections_per_image': total_detections / (len(test_images) * num_runs) if test_images else 0,
                'num_runs': num_runs,
                'num_images': len(test_images)
            }
        
        return benchmark_results

class RealTimeDetector:
    """Real-time detection using webcam or video stream"""
    
    def __init__(self, detector: MultiModelDetector):
        self.detector = detector
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)
        self.fps_counter = 0
        self.fps_timer = time.time()
        
    def start_camera_detection(
        self,
        camera_id: int = 0,
        conf_threshold: float = 0.25,
        display: bool = True,
        save_video: bool = False,
        output_path: str = "output_video.mp4"
    ):
        """Start real-time detection from camera"""
        try:
            cap = cv2.VideoCapture(camera_id)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open camera {camera_id}")
            
            # Video writer setup
            if save_video:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            self.is_running = True
            
            # Start detection thread
            detection_thread = threading.Thread(
                target=self._detection_worker,
                args=(conf_threshold,)
            )
            detection_thread.daemon = True
            detection_thread.start()
            
            logger.info("Starting real-time detection. Press 'q' to quit.")
            
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Add frame to queue for processing
                if not self.frame_queue.full():
                    self.frame_queue.put(frame.copy())
                
                # Get latest result if available
                annotated_frame = frame.copy()
                if not self.result_queue.empty():
                    try:
                        result = self.result_queue.get_nowait()
                        if result['annotated_image'] is not None:
                            annotated_frame = result['annotated_image']
                        
                        # Draw FPS and detection info
                        self._draw_info(annotated_frame, result)
                    except queue.Empty:
                        pass
                
                # Display frame
                if display:
                    cv2.imshow('Real-time Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.is_running = False
                
                # Save video
                if save_video:
                    out.write(annotated_frame)
                
                # Update FPS counter
                self._update_fps()
            
            # Cleanup
            cap.release()
            if save_video:
                out.release()
            if display:
                cv2.destroyAllWindows()
            
            logger.info("Real-time detection stopped")
            
        except Exception as e:
            logger.error(f"Error in real-time detection: {e}")
            self.is_running = False
    
    def _detection_worker(self, conf_threshold: float):
        """Worker thread for running detection"""
        while self.is_running:
            try:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get(timeout=1)
                    
                    # Run detection
                    result = self.detector.detect_image(
                        image_input=frame,
                        conf_threshold=conf_threshold,
                        return_annotated=True
                    )
                    
                    # Add result to queue
                    if not self.result_queue.full():
                        self.result_queue.put(result)
                    else:
                        # Remove old result and add new one
                        try:
                            self.result_queue.get_nowait()
                            self.result_queue.put(result)
                        except queue.Empty:
                            pass
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in detection worker: {e}")
    
    def _draw_info(self, frame: np.ndarray, result: Dict):
        """Draw information overlay on frame"""
        try:
            # Draw detection count
            text = f"Detections: {result['num_detections']}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw model name
            text = f"Model: {result.get('model_name', 'Unknown')}"
            cv2.putText(frame, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw inference time
            inference_time = result.get('inference_time', 0)
            text = f"Inference: {inference_time*1000:.1f}ms"
            cv2.putText(frame, text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw FPS
            text = f"FPS: {self.fps_counter:.1f}"
            cv2.putText(frame, text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        except Exception as e:
            logger.error(f"Error drawing info: {e}")
    
    def _update_fps(self):
        """Update FPS counter"""
        current_time = time.time()
        if current_time - self.fps_timer >= 1.0:
            self.fps_timer = current_time
            self.fps_counter = 0
        else:
            self.fps_counter += 1
    
    def stop(self):
        """Stop real-time detection"""
        self.is_running = False

class DetectionAnalyzer:
    """Analyze detection results and provide insights"""
    
    def __init__(self):
        self.detection_history = []
    
    def analyze_detections(self, detections: List[Dict]) -> Dict:
        """Analyze detection results"""
        if not detections:
            return {'error': 'No detections to analyze'}
        
        analysis = {
            'total_detections': len(detections),
            'class_distribution': {},
            'confidence_stats': {},
            'bbox_stats': {},
            'temporal_analysis': {}
        }
        
        # Class distribution
        classes = [det['class_name'] for det in detections if 'class_name' in det]
        for class_name in set(classes):
            analysis['class_distribution'][class_name] = classes.count(class_name)
        
        # Confidence statistics
        confidences = [det['confidence'] for det in detections if 'confidence' in det]
        if confidences:
            analysis['confidence_stats'] = {
                'mean': np.mean(confidences),
                'std': np.std(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences),
                'median': np.median(confidences)
            }
        
        # Bounding box statistics
        bboxes = [det['bbox'] for det in detections if 'bbox' in det]
        if bboxes:
            areas = [(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) for bbox in bboxes]
            analysis['bbox_stats'] = {
                'mean_area': np.mean(areas),
                'std_area': np.std(areas),
                'min_area': np.min(areas),
                'max_area': np.max(areas)
            }
        
        return analysis
    
    def track_performance(self, result: Dict):
        """Track detection performance over time"""
        timestamp = result.get('timestamp', time.time())
        performance_data = {
            'timestamp': timestamp,
            'inference_time': result.get('inference_time', 0),
            'num_detections': result.get('num_detections', 0),
            'model_name': result.get('model_name', 'unknown')
        }
        
        self.detection_history.append(performance_data)
        
        # Keep only last 1000 entries
        if len(self.detection_history) > 1000:
            self.detection_history = self.detection_history[-1000:]
    
    def get_performance_report(self) -> Dict:
        """Generate performance report"""
        if not self.detection_history:
            return {'error': 'No performance data available'}
        
        recent_data = self.detection_history[-100:]  # Last 100 detections
        
        inference_times = [d['inference_time'] for d in recent_data]
        detection_counts = [d['num_detections'] for d in recent_data]
        
        report = {
            'total_detections_analyzed': len(self.detection_history),
            'recent_performance': {
                'avg_inference_time': np.mean(inference_times),
                'std_inference_time': np.std(inference_times),
                'avg_detections_per_image': np.mean(detection_counts),
                'avg_fps': 1 / np.mean(inference_times) if np.mean(inference_times) > 0 else 0
            },
            'model_usage': {}
        }
        
        # Model usage statistics
        for data in recent_data:
            model = data['model_name']
            if model not in report['model_usage']:
                report['model_usage'][model] = 0
            report['model_usage'][model] += 1
        
        return report

if __name__ == "__main__":
    # Test the detector
    detector = MultiModelDetector()
    
    print("Available models:", detector.get_available_models())
    
    # Set model
    detector.set_model('yolov8s')
    
    # Test real-time detection (uncomment to test with camera)
    # real_time = RealTimeDetector(detector)
    # real_time.start_camera_detection(camera_id=0, display=True)
