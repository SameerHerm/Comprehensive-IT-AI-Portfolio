import torch
import torch.nn as nn
import torchvision.transforms as transforms
from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import yaml
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YOLODetector:
    def __init__(
        self,
        model_name: str = "yolov8s",
        weights_path: Optional[str] = None,
        config_path: str = "config/config.yaml",
        device: str = "auto"
    ):
        self.model_name = model_name
        self.weights_path = weights_path
        self.device = self._get_device(device)
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['models']['yolo']
        self.classes = self.config['data']['classes']
        
        # Initialize model
        self.model = self._load_model()
        
    def _get_device(self, device: str) -> str:
        """Get the appropriate device for inference"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def _load_model(self) -> YOLO:
        """Load YOLO model"""
        try:
            if self.weights_path and Path(self.weights_path).exists():
                model = YOLO(self.weights_path)
                logger.info(f"Loaded custom weights from {self.weights_path}")
            else:
                model = YOLO(f"{self.model_name}.pt")
                logger.info(f"Loaded pretrained {self.model_name} model")
            
            # Move model to device
            model.to(self.device)
            return model
            
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            raise
    
    def train(
        self,
        data_config_path: str,
        epochs: int = 100,
        batch_size: int = 16,
        image_size: int = 640,
        save_dir: str = "models/trained/yolo",
        **kwargs
    ):
        """Train YOLO model"""
        try:
            logger.info(f"Starting YOLO training for {epochs} epochs")
            
            results = self.model.train(
                data=data_config_path,
                epochs=epochs,
                batch=batch_size,
                imgsz=image_size,
                device=self.device,
                project=save_dir,
                name=f"{self.model_name}_training",
                save=True,
                save_period=10,
                val=True,
                plots=True,
                verbose=True,
                **kwargs
            )
            
            logger.info("YOLO training completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error during YOLO training: {e}")
            raise
    
    def validate(
        self,
        data_config_path: str,
        split: str = "val",
        save_dir: str = "results/validation"
    ):
        """Validate YOLO model"""
        try:
            logger.info("Starting YOLO validation")
            
            results = self.model.val(
                data=data_config_path,
                split=split,
                device=self.device,
                project=save_dir,
                name=f"{self.model_name}_validation",
                save_json=True,
                plots=True
            )
            
            logger.info("YOLO validation completed")
            return results
            
        except Exception as e:
            logger.error(f"Error during YOLO validation: {e}")
            raise
    
    def predict(
        self,
        source,
        conf_threshold: float = None,
        iou_threshold: float = None,
        max_detections: int = None,
        save_results: bool = False,
        save_dir: str = "results/predictions"
    ):
        """Make predictions using YOLO model"""
        try:
            # Use config values if not provided
            conf_threshold = conf_threshold or self.model_config['conf_threshold']
            iou_threshold = iou_threshold or self.model_config['iou_threshold']
            max_detections = max_detections or self.model_config['max_detections']
            
            results = self.model.predict(
                source=source,
                conf=conf_threshold,
                iou=iou_threshold,
                max_det=max_detections,
                device=self.device,
                save=save_results,
                project=save_dir if save_results else None,
                name=f"{self.model_name}_predictions" if save_results else None,
                show_labels=True,
                show_conf=True,
                save_txt=save_results,
                save_conf=save_results
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error during YOLO prediction: {e}")
            raise
    
    def detect_image(
        self,
        image_path: str,
        conf_threshold: float = None,
        return_annotated: bool = True
    ) -> Dict:
        """Detect objects in a single image"""
        try:
            results = self.predict(
                source=image_path,
                conf_threshold=conf_threshold
            )
            
            # Extract detection results
            detections = []
            annotated_image = None
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        detection = {
                            'class_id': int(box.cls.item()),
                            'class_name': self.classes[int(box.cls.item())] if int(box.cls.item()) < len(self.classes) else 'unknown',
                            'confidence': float(box.conf.item()),
                            'bbox': box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                            'bbox_normalized': box.xywhn[0].tolist()  # [x_center, y_center, width, height] normalized
                        }
                        detections.append(detection)
                
                if return_annotated:
                    annotated_image = result.plot()
            
            return {
                'detections': detections,
                'annotated_image': annotated_image,
                'num_detections': len(detections)
            }
            
        except Exception as e:
            logger.error(f"Error detecting objects in image: {e}")
            raise
    
    def detect_video(
        self,
        video_path: str,
        output_path: str = None,
        conf_threshold: float = None,
        show_live: bool = False
    ):
        """Detect objects in video"""
        try:
            conf_threshold = conf_threshold or self.model_config['conf_threshold']
            
            cap = cv2.VideoCapture(video_path)
            
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_count = 0
            total_detections = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run detection
                results = self.predict(source=frame, conf_threshold=conf_threshold)
                
                # Process results
                for result in results:
                    annotated_frame = result.plot()
                    
                    if result.boxes is not None:
                        total_detections += len(result.boxes)
                    
                    if output_path:
                        out.write(annotated_frame)
                    
                    if show_live:
                        cv2.imshow('YOLO Detection', annotated_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                
                frame_count += 1
                if frame_count % 30 == 0:
                    logger.info(f"Processed {frame_count} frames, {total_detections} total detections")
            
            cap.release()
            if output_path:
                out.release()
            if show_live:
                cv2.destroyAllWindows()
            
            logger.info(f"Video processing completed. Total frames: {frame_count}, Total detections: {total_detections}")
            
            return {
                'total_frames': frame_count,
                'total_detections': total_detections,
                'avg_detections_per_frame': total_detections / frame_count if frame_count > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            raise
    
    def export_model(
        self,
        format: str = "onnx",
        output_dir: str = "models/exported",
        **kwargs
    ):
        """Export YOLO model to different formats"""
        try:
            logger.info(f"Exporting model to {format} format")
            
            exported_model = self.model.export(
                format=format,
                project=output_dir,
                name=f"{self.model_name}_{format}",
                **kwargs
            )
            
            logger.info(f"Model exported successfully to {exported_model}")
            return exported_model
            
        except Exception as e:
            logger.error(f"Error exporting model: {e}")
            raise
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        try:
            info = {
                'model_name': self.model_name,
                'device': self.device,
                'input_size': self.model_config['input_size'],
                'num_classes': len(self.classes),
                'classes': self.classes,
                'conf_threshold': self.model_config['conf_threshold'],
                'iou_threshold': self.model_config['iou_threshold'],
                'max_detections': self.model_config['max_detections']
            }
            
            # Try to get model parameters count
            try:
                total_params = sum(p.numel() for p in self.model.model.parameters())
                trainable_params = sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)
                info['total_parameters'] = total_params
                info['trainable_parameters'] = trainable_params
            except:
                pass
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {}

class YOLOEnsemble:
    """Ensemble of multiple YOLO models for improved performance"""
    
    def __init__(self, model_configs: List[Dict]):
        self.models = []
        self.weights = []
        
        for config in model_configs:
            model = YOLODetector(**config['params'])
            self.models.append(model)
            self.weights.append(config.get('weight', 1.0))
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        logger.info(f"Initialized YOLO ensemble with {len(self.models)} models")
    
    def predict_ensemble(
        self,
        source,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        ensemble_method: str = "weighted_average"  # weighted_average, max_confidence, voting
    ):
        """Make ensemble predictions"""
        all_predictions = []
        
        # Get predictions from all models
        for i, model in enumerate(self.models):
            results = model.predict(
                source=source,
                conf_threshold=conf_threshold * 0.8,  # Lower threshold for ensemble
                iou_threshold=iou_threshold
            )
            
            for result in results:
                if result.boxes is not None:
                    predictions = []
                    for box in result.boxes:
                        pred = {
                            'bbox': box.xyxy[0].tolist(),
                            'confidence': float(box.conf.item()) * self.weights[i],
                            'class_id': int(box.cls.item()),
                            'model_id': i
                        }
                        predictions.append(pred)
                    all_predictions.extend(predictions)
        
        # Apply ensemble method
        if ensemble_method == "weighted_average":
            final_predictions = self._weighted_average_ensemble(all_predictions, iou_threshold)
        elif ensemble_method == "max_confidence":
            final_predictions = self._max_confidence_ensemble(all_predictions, iou_threshold)
        else:
            final_predictions = self._voting_ensemble(all_predictions, iou_threshold)
        
        # Filter by confidence threshold
        final_predictions = [p for p in final_predictions if p['confidence'] >= conf_threshold]
        
        return final_predictions
    
    def _weighted_average_ensemble(self, predictions: List[Dict], iou_threshold: float):
        """Weighted average ensemble method"""
        # Group predictions by IoU similarity
        groups = []
        for pred in predictions:
            added_to_group = False
            for group in groups:
                if self._calculate_iou(pred['bbox'], group[0]['bbox']) > iou_threshold:
                    group.append(pred)
                    added_to_group = True
                    break
            if not added_to_group:
                groups.append([pred])
        
        # Average predictions in each group
        final_predictions = []
        for group in groups:
            if len(group) == 1:
                final_predictions.append(group[0])
            else:
                # Weighted average
                avg_bbox = [0, 0, 0, 0]
                total_confidence = 0
                class_votes = {}
                
                for pred in group:
                    weight = pred['confidence']
                    for i in range(4):
                        avg_bbox[i] += pred['bbox'][i] * weight
                    total_confidence += weight
                    
                    class_id = pred['class_id']
                    class_votes[class_id] = class_votes.get(class_id, 0) + weight
                
                if total_confidence > 0:
                    avg_bbox = [coord / total_confidence for coord in avg_bbox]
                    best_class = max(class_votes, key=class_votes.get)
                    
                    final_predictions.append({
                        'bbox': avg_bbox,
                        'confidence': total_confidence / len(group),
                        'class_id': best_class
                    })
        
        return final_predictions
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two bounding boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _max_confidence_ensemble(self, predictions: List[Dict], iou_threshold: float):
        """Max confidence ensemble method"""
        # Similar grouping as weighted average but keep highest confidence
        groups = []
        for pred in predictions:
            added_to_group = False
            for group in groups:
                if self._calculate_iou(pred['bbox'], group[0]['bbox']) > iou_threshold:
                    group.append(pred)
                    added_to_group = True
                    break
            if not added_to_group:
                groups.append([pred])
        
        final_predictions = []
        for group in groups:
            best_pred = max(group, key=lambda x: x['confidence'])
            final_predictions.append(best_pred)
        
        return final_predictions
    
    def _voting_ensemble(self, predictions: List[Dict], iou_threshold: float):
        """Voting ensemble method"""
        # Implementation for voting-based ensemble
        # This is a simplified version - you can make it more sophisticated
        return self._max_confidence_ensemble(predictions, iou_threshold)

if __name__ == "__main__":
    # Test YOLO detector
    detector = YOLODetector(model_name="yolov8s")
    
    # Get model info
    info = detector.get_model_info()
    print("Model Info:", info)
    
    # Test ensemble
    ensemble_configs = [
        {'params': {'model_name': 'yolov8s'}, 'weight': 1.0},
        {'params': {'model_name': 'yolov8m'}, 'weight': 1.2}
    ]
    
    # ensemble = YOLOEnsemble(ensemble_configs)
    # ensemble_results = ensemble.predict_ensemble("path/to/image.jpg")
