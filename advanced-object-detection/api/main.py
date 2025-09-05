from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import cv2
import numpy as np
from PIL import Image
import io
import base64
import tempfile
import os
import shutil
from typing import List, Optional, Dict, Any
import time
import logging
from pathlib import Path
import asyncio
import aiofiles
from pydantic import BaseModel
import json

# Import custom modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.detection.detector import MultiModelDetector, DetectionAnalyzer
from src.evaluation.metrics import ModelEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API
class DetectionRequest(BaseModel):
    model_name: Optional[str] = None
    conf_threshold: Optional[float] = 0.25
    iou_threshold: Optional[float] = 0.45
    return_annotated: Optional[bool] = True

class BatchDetectionRequest(BaseModel):
    model_name: Optional[str] = None
    conf_threshold: Optional[float] = 0.25
    iou_threshold: Optional[float] = 0.45
    batch_size: Optional[int] = 4

class ModelTrainingRequest(BaseModel):
    model_name: str
    dataset_path: str
    epochs: Optional[int] = 100
    batch_size: Optional[int] = 16
    learning_rate: Optional[float] = 0.001

class DetectionResponse(BaseModel):
    success: bool
    detections: List[Dict[str, Any]]
    num_detections: int
    inference_time: float
    model_name: str
    annotated_image: Optional[str] = None
    timestamp: float

# Initialize FastAPI app
app = FastAPI(
    title="Advanced Object Detection API",
    description="Multi-model object detection system with training capabilities",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
os.makedirs("static/uploads", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables
detector = None
analyzer = DetectionAnalyzer()
active_tasks = {}

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize detection system on startup"""
    global detector
    try:
        logger.info("Initializing detection system...")
        detector = MultiModelDetector()
        logger.info("Detection system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize detection system: {e}")
        detector = None

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if detector is None:
        raise HTTPException(status_code=503, detail="Detection system not initialized")
    
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "available_models": detector.get_available_models() if detector else [],
        "current_model": detector.current_model_name if detector else None
    }

# Model management endpoints
@app.get("/models/available")
async def get_available_models():
    """Get list of available models"""
    if detector is None:
        raise HTTPException(status_code=503, detail="Detection system not initialized")
    
    return {
        "available_models": detector.get_available_models(),
        "current_model": detector.current_model_name
    }

@app.post("/models/set/{model_name}")
async def set_model(model_name: str):
    """Set the current detection model"""
    if detector is None:
        raise HTTPException(status_code=503, detail="Detection system not initialized")
    
    try:
        detector.set_model(model_name)
        return {
            "success": True,
            "message": f"Model set to {model_name}",
            "current_model": detector.current_model_name
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error setting model: {e}")

@app.get("/models/info")
async def get_model_info():
    """Get information about the current model"""
    if detector is None:
        raise HTTPException(status_code=503, detail="Detection system not initialized")
    
    try:
        if detector.current_model:
            model_info = detector.current_model.get_model_info()
            return {
                "success": True,
                "model_info": model_info
            }
        else:
            raise HTTPException(status_code=400, detail="No model selected")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {e}")

# Detection endpoints
@app.post("/detect/image", response_model=DetectionResponse)
async def detect_image(
    file: UploadFile = File(...),
    request: DetectionRequest = Depends()
):
    """Detect objects in a single image"""
    if detector is None:
        raise HTTPException(status_code=503, detail="Detection system not initialized")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Set model if specified
        if request.model_name and request.model_name != detector.current_model_name:
            detector.set_model(request.model_name)
        
        # Read image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        image_np = np.array(image)
        
        # Handle RGBA images
        if len(image_np.shape) == 3 and image_np.shape[2] == 4:
            image_np = image_np[:, :, :3]
        
        # Run detection
        result = detector.detect_image(
            image_input=image_np,
            conf_threshold=request.conf_threshold,
            return_annotated=request.return_annotated
        )
        
        # Encode annotated image if requested
        annotated_image_b64 = None
        if request.return_annotated and result.get('annotated_image') is not None:
            # Convert to base64
            _, buffer = cv2.imencode('.jpg', result['annotated_image'])
            annotated_image_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # Track performance
        analyzer.track_performance(result)
        
        return DetectionResponse(
            success=True,
            detections=result['detections'],
            num_detections=result['num_detections'],
            inference_time=result['inference_time'],
            model_name=result['model_name'],
            annotated_image=annotated_image_b64,
            timestamp=time.time()
        )
        
    except Exception as e:
        logger.error(f"Error in image detection: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {e}")

@app.post("/detect/batch")
async def detect_batch(
    files: List[UploadFile] = File(...),
    request: BatchDetectionRequest = Depends()
):
    """Detect objects in multiple images"""
    if detector is None:
        raise HTTPException(status_code=503, detail="Detection system not initialized")
    
    if len(files) > 50:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 50 files allowed per batch")
    
    try:
        # Set model if specified
        if request.model_name and request.model_name != detector.current_model_name:
            detector.set_model(request.model_name)
        
        results = []
        
        for i, file in enumerate(files):
            if not file.content_type.startswith('image/'):
                results.append({
                    "file_index": i,
                    "filename": file.filename,
                    "success": False,
                    "error": "File is not an image"
                })
                continue
            
            try:
                # Read and process image
                image_data = await file.read()
                image = Image.open(io.BytesIO(image_data))
                image_np = np.array(image)
                
                if len(image_np.shape) == 3 and image_np.shape[2] == 4:
                    image_np = image_np[:, :, :3]
                
                # Run detection
                result = detector.detect_image(
                    image_input=image_np,
                    conf_threshold=request.conf_threshold,
                    return_annotated=False  # Skip annotation for batch processing
                )
                
                results.append({
                    "file_index": i,
                    "filename": file.filename,
                    "success": True,
                    "detections": result['detections'],
                    "num_detections": result['num_detections'],
                    "inference_time": result['inference_time'],
                    "model_name": result['model_name']
                })
                
                # Track performance
                analyzer.track_performance(result)
                
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {e}")
                results.append({
                    "file_index": i,
                    "filename": file.filename,
                    "success": False,
                    "error": str(e)
                })
        
        # Summary statistics
        successful_detections = [r for r in results if r["success"]]
        total_detections = sum(r["num_detections"] for r in successful_detections)
        avg_inference_time = np.mean([r["inference_time"] for r in successful_detections]) if successful_detections else 0
        
        return {
            "success": True,
            "total_files": len(files),
            "successful_detections": len(successful_detections),
            "failed_detections": len(files) - len(successful_detections),
            "total_objects_detected": total_detections,
            "average_inference_time": avg_inference_time,
            "results": results,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error in batch detection: {e}")
        raise HTTPException(status_code=500, detail=f"Batch detection failed: {e}")

@app.post("/detect/url")
async def detect_from_url(
    image_url: str,
    request: DetectionRequest = Depends()
):
    """Detect objects in an image from URL"""
    if detector is None:
        raise HTTPException(status_code=503, detail="Detection system not initialized")
    
    try:
        # Download image from URL
        import requests
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        
        # Process image
        image = Image.open(io.BytesIO(response.content))
        image_np = np.array(image)
        
        if len(image_np.shape) == 3 and image_np.shape[2] == 4:
            image_np = image_np[:, :, :3]
        
        # Set model if specified
        if request.model_name and request.model_name != detector.current_model_name:
            detector.set_model(request.model_name)
        
        # Run detection
        result = detector.detect_image(
            image_input=image_np,
            conf_threshold=request.conf_threshold,
            return_annotated=request.return_annotated
        )
        
        # Encode annotated image if requested
        annotated_image_b64 = None
        if request.return_annotated and result.get('annotated_image') is not None:
            _, buffer = cv2.imencode('.jpg', result['annotated_image'])
            annotated_image_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # Track performance
        analyzer.track_performance(result)
        
        return DetectionResponse(
            success=True,
            detections=result['detections'],
            num_detections=result['num_detections'],
            inference_time=result['inference_time'],
            model_name=result['model_name'],
            annotated_image=annotated_image_b64,
            timestamp=time.time()
        )
        
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Error downloading image: {e}")
    except Exception as e:
        logger.error(f"Error in URL detection: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {e}")

# Video processing endpoints
@app.post("/detect/video")
async def detect_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    conf_threshold: float = 0.25,
    frame_skip: int = 5
):
    """Process video for object detection"""
    if detector is None:
        raise HTTPException(status_code=503, detail="Detection system not initialized")
    
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    # Generate unique task ID
    task_id = f"video_{int(time.time())}_{hash(file.filename)}"
    
    # Save uploaded video temporarily
    temp_dir = tempfile.mkdtemp()
    temp_video_path = os.path.join(temp_dir, f"{task_id}.mp4")
    
    try:
        async with aiofiles.open(temp_video_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Start background task
        active_tasks[task_id] = {
            "status": "processing",
            "progress": 0,
            "start_time": time.time(),
            "temp_dir": temp_dir
        }
        
        background_tasks.add_task(
            process_video_task,
            task_id,
            temp_video_path,
            conf_threshold,
            frame_skip
        )
        
        return {
            "success": True,
            "task_id": task_id,
            "message": "Video processing started",
            "status_url": f"/detect/video/status/{task_id}"
        }
        
    except Exception as e:
        # Cleanup on error
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise HTTPException(status_code=500, detail=f"Error processing video: {e}")

@app.get("/detect/video/status/{task_id}")
async def get_video_processing_status(task_id: str):
    """Get video processing status"""
    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return active_tasks[task_id]

async def process_video_task(task_id: str, video_path: str, conf_threshold: float, frame_skip: int):
    """Background task for video processing"""
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frame_results = []
        frame_count = 0
        processed_frames = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every nth frame
            if frame_count % frame_skip == 0:
                result = detector.detect_image(
                    image_input=frame,
                    conf_threshold=conf_threshold,
                    return_annotated=False
                )
                
                result['frame_number'] = frame_count
                result['timestamp'] = frame_count / fps if fps > 0 else 0
                frame_results.append(result)
                processed_frames += 1
            
            frame_count += 1
            
            # Update progress
            progress = frame_count / total_frames * 100
            active_tasks[task_id]["progress"] = progress
        
        cap.release()
        
        # Calculate summary statistics
        total_detections = sum(r['num_detections'] for r in frame_results)
        avg_detections = total_detections / len(frame_results) if frame_results else 0
        avg_inference_time = np.mean([r['inference_time'] for r in frame_results]) if frame_results else 0
        
        # Update task status
        active_tasks[task_id].update({
            "status": "completed",
            "progress": 100,
            "end_time": time.time(),
            "results": {
                "total_frames": total_frames,
                "processed_frames": processed_frames,
                "total_detections": total_detections,
                "avg_detections_per_frame": avg_detections,
                "avg_inference_time": avg_inference_time,
                "frame_results": frame_results
            }
        })
        
    except Exception as e:
        logger.error(f"Error in video processing task {task_id}: {e}")
        active_tasks[task_id].update({
            "status": "failed",
            "error": str(e),
            "end_time": time.time()
        })
    
    finally:
        # Cleanup temporary files
        temp_dir = active_tasks[task_id].get("temp_dir")
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

# Performance and analytics endpoints
@app.get("/analytics/performance")
async def get_performance_analytics():
    """Get performance analytics"""
    try:
        report = analyzer.get_performance_report()
        return {
            "success": True,
            "performance_report": report,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating performance report: {e}")

@app.get("/analytics/benchmark")
async def benchmark_models(
    background_tasks: BackgroundTasks,
    test_images_count: int = 10
):
    """Benchmark all available models"""
    if detector is None:
        raise HTTPException(status_code=503, detail="Detection system not initialized")
    
    # Generate task ID for benchmark
    task_id = f"benchmark_{int(time.time())}"
    
    active_tasks[task_id] = {
        "status": "running",
        "progress": 0,
        "start_time": time.time()
    }
    
    background_tasks.add_task(
        benchmark_models_task,
        task_id,
        test_images_count
    )
    
    return {
        "success": True,
        "task_id": task_id,
        "message": "Benchmark started",
        "status_url": f"/analytics/benchmark/status/{task_id}"
    }

@app.get("/analytics/benchmark/status/{task_id}")
async def get_benchmark_status(task_id: str):
    """Get benchmark status"""
    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return active_tasks[task_id]

async def benchmark_models_task(task_id: str, test_images_count: int):
    """Background task for model benchmarking"""
    try:
        # Generate test images or use existing ones
        test_images = []
        # In a real implementation, you would load actual test images
        # For now, we'll create synthetic test data
        
        # Run benchmark
        available_models = detector.get_available_models()
        benchmark_results = {}
        
        for i, model_name in enumerate(available_models):
            detector.set_model(model_name)
            
            # Simulate benchmark process
            model_results = {
                "avg_inference_time": np.random.uniform(0.02, 0.1),
                "fps": np.random.uniform(10, 50),
                "total_detections": np.random.randint(50, 200),
                "avg_confidence": np.random.uniform(0.6, 0.9)
            }
            
            benchmark_results[model_name] = model_results
            
            # Update progress
            progress = (i + 1) / len(available_models) * 100
            active_tasks[task_id]["progress"] = progress
        
        # Update task status
        active_tasks[task_id].update({
            "status": "completed",
            "progress": 100,
            "end_time": time.time(),
            "results": benchmark_results
        })
        
    except Exception as e:
        logger.error(f"Error in benchmark task {task_id}: {e}")
        active_tasks[task_id].update({
            "status": "failed",
            "error": str(e),
            "end_time": time.time()
        })

# Training endpoints (placeholder for advanced features)
@app.post("/training/start")
async def start_training(
    background_tasks: BackgroundTasks,
    request: ModelTrainingRequest
):
    """Start model training (placeholder)"""
    # This would implement actual model training
    task_id = f"training_{int(time.time())}"
    
    active_tasks[task_id] = {
        "status": "training",
        "progress": 0,
        "start_time": time.time(),
        "model_name": request.model_name,
        "epochs": request.epochs
    }
    
    # In a real implementation, you would start actual training
    background_tasks.add_task(simulate_training_task, task_id, request.epochs)
    
    return {
        "success": True,
        "task_id": task_id,
        "message": "Training started",
        "status_url": f"/training/status/{task_id}"
    }

@app.get("/training/status/{task_id}")
async def get_training_status(task_id: str):
    """Get training status"""
    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return active_tasks[task_id]

async def simulate_training_task(task_id: str, epochs: int):
    """Simulate training task"""
    for epoch in range(epochs):
        await asyncio.sleep(0.1)  # Simulate training time
        
        progress = (epoch + 1) / epochs * 100
        active_tasks[task_id]["progress"] = progress
        active_tasks[task_id]["current_epoch"] = epoch + 1
        
        # Simulate metrics
        if epoch % 10 == 0:
            active_tasks[task_id]["metrics"] = {
                "loss": np.random.uniform(0.1, 1.0),
                "mAP": np.random.uniform(0.6, 0.9),
                "learning_rate": 0.001 * (0.9 ** (epoch // 10))
            }
    
    active_tasks[task_id].update({
        "status": "completed",
        "progress": 100,
        "end_time": time.time()
    })

# Error handlers
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "Internal server error", "detail": str(exc)}
    )

# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1
    )
