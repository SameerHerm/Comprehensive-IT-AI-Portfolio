import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import io
import base64
from PIL import Image
import requests
import json
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.detection.detector import MultiModelDetector, DetectionAnalyzer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Advanced Object Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .detection-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    .sidebar-info {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

class ObjectDetectionApp:
    def __init__(self):
        self.detector = None
        self.analyzer = DetectionAnalyzer()
        self.detection_history = []
        
        # Initialize session state
        if 'detection_results' not in st.session_state:
            st.session_state.detection_results = []
        if 'performance_data' not in st.session_state:
            st.session_state.performance_data = []
        
        self._initialize_detector()
    
    def _initialize_detector(self):
        """Initialize the detection system"""
        try:
            with st.spinner("Initializing detection models..."):
                self.detector = MultiModelDetector()
            st.success("✅ Detection system initialized successfully!")
        except Exception as e:
            st.error(f"❌ Failed to initialize detection system: {e}")
            self.detector = None
    
    def main(self):
        """Main application interface"""
        st.markdown('<h1 class="main-header">🔍 Advanced Object Detection System</h1>', 
                   unsafe_allow_html=True)
        
        # Sidebar
        self._create_sidebar()
        
        # Main content
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🖼️ Image Detection", 
            "📹 Video Detection", 
            "📊 Batch Processing", 
            "⚡ Real-time Analysis",
            "📈 Analytics Dashboard"
        ])
        
        with tab1:
            self._image_detection_tab()
        
        with tab2:
            self._video_detection_tab()
        
        with tab3:
            self._batch_processing_tab()
        
        with tab4:
            self._realtime_tab()
        
        with tab5:
            self._analytics_dashboard()
    
    def _create_sidebar(self):
        """Create sidebar with model selection and settings"""
        st.sidebar.markdown("## 🎛️ Settings")
        
        if self.detector:
            # Model selection
            available_models = self.detector.get_available_models()
            selected_model = st.sidebar.selectbox(
                "Select Model",
                available_models,
                index=0 if available_models else None
            )
            
            if selected_model:
                self.detector.set_model(selected_model)
                st.sidebar.success(f"✅ Model: {selected_model}")
            
            # Detection parameters
            st.sidebar.markdown("### Detection Parameters")
            conf_threshold = st.sidebar.slider(
                "Confidence Threshold",
                min_value=0.1,
                max_value=1.0,
                value=0.25,
                step=0.05
            )
            
            iou_threshold = st.sidebar.slider(
                "IoU Threshold",
                min_value=0.1,
                max_value=1.0,
                value=0.45,
                step=0.05
            )
            
            # Store in session state
            st.session_state.conf_threshold = conf_threshold
            st.session_state.iou_threshold = iou_threshold
            
            # Model info
            with st.sidebar.expander("📋 Model Information"):
                try:
                    model_info = self.detector.current_model.get_model_info()
                    st.json(model_info)
                except:
                    st.write("Model info not available")
        
        else:
            st.sidebar.error("❌ Detector not initialized")
        
        # Performance monitoring
        if st.session_state.performance_data:
            st.sidebar.markdown("### 📊 Performance")
            recent_perf = st.session_state.performance_data[-10:]
            avg_time = np.mean([p['inference_time'] for p in recent_perf])
            fps = 1 / avg_time if avg_time > 0 else 0
            
            st.sidebar.metric("Average FPS", f"{fps:.1f}")
            st.sidebar.metric("Avg Inference Time", f"{avg_time*1000:.1f}ms")
    
    def _image_detection_tab(self):
        """Image detection interface"""
        st.header("🖼️ Single Image Detection")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Upload Image")
            
            # Image upload options
            upload_option = st.radio(
                "Choose input method:",
                ["Upload File", "Camera Capture", "URL"]
            )
            
            uploaded_image = None
            
            if upload_option == "Upload File":
                uploaded_file = st.file_uploader(
                    "Choose an image file",
                    type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
                    accept_multiple_files=False
                )
                
                if uploaded_file:
                    uploaded_image = Image.open(uploaded_file)
            
            elif upload_option == "Camera Capture":
                camera_image = st.camera_input("Take a picture")
                if camera_image:
                    uploaded_image = Image.open(camera_image)
            
            elif upload_option == "URL":
                image_url = st.text_input("Enter image URL:")
                if image_url:
                    try:
                        response = requests.get(image_url)
                        uploaded_image = Image.open(io.BytesIO(response.content))
                    except Exception as e:
                        st.error(f"Error loading image from URL: {e}")
            
            if uploaded_image:
                st.image(uploaded_image, caption="Input Image", use_column_width=True)
                
                # Convert PIL to numpy array
                image_np = np.array(uploaded_image)
                if image_np.shape[2] == 4:  # RGBA to RGB
                    image_np = image_np[:, :, :3]
                
                # Detection button
                if st.button("🔍 Detect Objects", type="primary", use_container_width=True):
                    if self.detector:
                        self._run_detection(image_np, col2)
                    else:
                        st.error("Detector not initialized!")
        
        # Detection history
        if st.session_state.detection_results:
            st.subheader("📜 Detection History")
            history_df = pd.DataFrame([
                {
                    'timestamp': result['timestamp'],
                    'model': result.get('model_name', 'Unknown'),
                    'detections': result['num_detections'],
                    'inference_time': f"{result['inference_time']*1000:.1f}ms"
                }
                for result in st.session_state.detection_results[-10:]
            ])
            st.dataframe(history_df, use_container_width=True)
    
    def _run_detection(self, image_np, display_col):
        """Run detection on image and display results"""
        with st.spinner("🔍 Detecting objects..."):
            try:
                start_time = time.time()
                
                # Run detection
                result = self.detector.detect_image(
                    image_input=image_np,
                    conf_threshold=st.session_state.get('conf_threshold', 0.25),
                    return_annotated=True
                )
                
                # Store results
                result['timestamp'] = time.time()
                st.session_state.detection_results.append(result)
                st.session_state.performance_data.append({
                    'inference_time': result['inference_time'],
                    'num_detections': result['num_detections'],
                    'timestamp': result['timestamp']
                })
                
                # Display results
                with display_col:
                    st.subheader("🎯 Detection Results")
                    
                    # Annotated image
                    if result['annotated_image'] is not None:
                        st.image(
                            result['annotated_image'],
                            caption="Detected Objects",
                            use_column_width=True
                        )
                    
                    # Metrics
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Objects Found", result['num_detections'])
                    with col_b:
                        st.metric("Inference Time", f"{result['inference_time']*1000:.1f}ms")
                    with col_c:
                        fps = 1 / result['inference_time'] if result['inference_time'] > 0 else 0
                        st.metric("FPS", f"{fps:.1f}")
                    
                    # Detection details
                    if result['detections']:
                        st.subheader("📋 Detection Details")
                        
                        detection_data = []
                        for i, detection in enumerate(result['detections']):
                            confidence = detection['confidence']
                            if confidence > 0.7:
                                conf_class = "confidence-high"
                            elif confidence > 0.4:
                                conf_class = "confidence-medium"
                            else:
                                conf_class = "confidence-low"
                            
                            detection_data.append({
                                'ID': i + 1,
                                'Class': detection['class_name'],
                                'Confidence': f"{confidence:.3f}",
                                'Bbox': f"({detection['bbox'][0]:.1f}, {detection['bbox'][1]:.1f}, {detection['bbox'][2]:.1f}, {detection['bbox'][3]:.1f})"
                            })
                        
                        df = pd.DataFrame(detection_data)
                        st.dataframe(df, use_container_width=True)
                        
                        # Class distribution chart
                        class_counts = {}
                        for detection in result['detections']:
                            class_name = detection['class_name']
                            class_counts[class_name] = class_counts.get(class_name, 0) + 1
                        
                        if class_counts:
                            fig = px.pie(
                                values=list(class_counts.values()),
                                names=list(class_counts.keys()),
                                title="Class Distribution"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                
                # Track performance
                self.analyzer.track_performance(result)
                
            except Exception as e:
                st.error(f"Detection failed: {e}")
                logger.error(f"Detection error: {e}")
    
    def _video_detection_tab(self):
        """Video detection interface"""
        st.header("📹 Video Detection")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Upload Video")
            
            uploaded_video = st.file_uploader(
                "Choose a video file",
                type=['mp4', 'avi', 'mov', 'mkv'],
                accept_multiple_files=False
            )
            
            if uploaded_video:
                # Save uploaded video temporarily
                temp_video_path = f"temp_video_{int(time.time())}.mp4"
                with open(temp_video_path, "wb") as f:
                    f.write(uploaded_video.getbuffer())
                
                st.video(uploaded_video)
                
                # Video processing options
                st.subheader("Processing Options")
                
                process_every_n_frames = st.slider(
                    "Process every N frames (for speed)",
                    min_value=1,
                    max_value=30,
                    value=5
                )
                
                output_format = st.selectbox(
                    "Output format",
                    ["Annotated Video", "Detection Data Only", "Both"]
                )
                
                if st.button("🎬 Process Video", type="primary"):
                    self._process_video(temp_video_path, col2, process_every_n_frames, output_format)
                
                # Cleanup
                if os.path.exists(temp_video_path):
                    os.unlink(temp_video_path)
    
    def _process_video(self, video_path, display_col, frame_skip, output_format):
        """Process video for object detection"""
        with display_col:
            st.subheader("🎬 Processing Results")
            
            if not self.detector:
                st.error("Detector not initialized!")
                return
            
            try:
                # Video info
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                duration = total_frames / fps if fps > 0 else 0
                
                st.info(f"📊 Video Info: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s")
                
                # Processing
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                frame_results = []
                processed_frames = 0
                
                cap = cv2.VideoCapture(video_path)
                frame_count = 0
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process every nth frame
                    if frame_count % frame_skip == 0:
                        # Run detection
                        result = self.detector.detect_image(
                            image_input=frame,
                            conf_threshold=st.session_state.get('conf_threshold', 0.25),
                            return_annotated=False
                        )
                        
                        result['frame_number'] = frame_count
                        result['timestamp'] = frame_count / fps if fps > 0 else 0
                        frame_results.append(result)
                        
                        processed_frames += 1
                    
                    frame_count += 1
                    
                    # Update progress
                    progress = frame_count / total_frames
                    progress_bar.progress(progress)
                    status_text.text(f"Processing frame {frame_count}/{total_frames} ({progress*100:.1f}%)")
                
                cap.release()
                
                # Results summary
                total_detections = sum(r['num_detections'] for r in frame_results)
                avg_detections = total_detections / len(frame_results) if frame_results else 0
                avg_inference_time = np.mean([r['inference_time'] for r in frame_results]) if frame_results else 0
                
                st.success("✅ Video processing completed!")
                
                # Display metrics
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    st.metric("Frames Processed", processed_frames)
                with col_b:
                    st.metric("Total Detections", total_detections)
                with col_c:
                    st.metric("Avg Detections/Frame", f"{avg_detections:.1f}")
                with col_d:
                    st.metric("Avg Processing Time", f"{avg_inference_time*1000:.1f}ms")
                
                # Temporal analysis
                if frame_results:
                    df = pd.DataFrame([
                        {
                            'frame': r['frame_number'],
                            'timestamp': r['timestamp'],
                            'detections': r['num_detections'],
                            'inference_time': r['inference_time']
                        }
                        for r in frame_results
                    ])
                    
                    # Detection count over time
                    fig1 = px.line(
                        df, 
                        x='timestamp', 
                        y='detections',
                        title='Detections Over Time',
                        labels={'timestamp': 'Time (seconds)', 'detections': 'Number of Detections'}
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                    
                    # Performance over time
                    fig2 = px.line(
                        df,
                        x='timestamp',
                        y='inference_time',
                        title='Inference Time Over Time',
                        labels={'timestamp': 'Time (seconds)', 'inference_time': 'Inference Time (seconds)'}
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                
            except Exception as e:
                st.error(f"Video processing failed: {e}")
                logger.error(f"Video processing error: {e}")
    
    def _batch_processing_tab(self):
        """Batch processing interface"""
        st.header("📊 Batch Processing")
        
        st.info("💡 Upload multiple images for batch processing and analysis")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Choose image files",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} files uploaded")
            
            # Processing options
            col1, col2 = st.columns(2)
            
            with col1:
                batch_size = st.slider("Batch Size", 1, min(10, len(uploaded_files)), 4)
                save_results = st.checkbox("Save detailed results", value=True)
            
            with col2:
                show_preview = st.checkbox("Show image previews", value=True)
                conf_threshold = st.slider(
                    "Confidence Threshold (Batch)",
                    0.1, 1.0, 0.25, 0.05
                )
            
            # Preview images
            if show_preview and len(uploaded_files) <= 10:
                st.subheader("📸 Image Preview")
                cols = st.columns(min(5, len(uploaded_files)))
                for i, file in enumerate(uploaded_files[:10]):
                    with cols[i % 5]:
                        image = Image.open(file)
                        st.image(image, caption=file.name, use_column_width=True)
            
            # Process batch
            if st.button("🚀 Process Batch", type="primary"):
                self._process_batch(uploaded_files, batch_size, conf_threshold, save_results)
    
    def _process_batch(self, uploaded_files, batch_size, conf_threshold, save_results):
        """Process batch of images"""
        if not self.detector:
            st.error("Detector not initialized!")
            return
        
        try:
            total_files = len(uploaded_files)
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            all_results = []
            batch_stats = []
            
            # Process in batches
            for i in range(0, total_files, batch_size):
                batch_files = uploaded_files[i:i + batch_size]
                batch_results = []
                
                for j, file in enumerate(batch_files):
                    # Load image
                    image = Image.open(file)
                    image_np = np.array(image)
                    if len(image_np.shape) == 3 and image_np.shape[2] == 4:
                        image_np = image_np[:, :, :3]
                    
                    # Run detection
                    result = self.detector.detect_image(
                        image_input=image_np,
                        conf_threshold=conf_threshold,
                        return_annotated=False
                    )
                    
                    result['filename'] = file.name
                    result['file_size'] = len(file.getvalue())
                    batch_results.append(result)
                    
                    # Update progress
                    current_progress = (i + j + 1) / total_files
                    progress_bar.progress(current_progress)
                    status_text.text(f"Processing {file.name} ({i + j + 1}/{total_files})")
                
                all_results.extend(batch_results)
                
                # Batch statistics
                batch_detections = sum(r['num_detections'] for r in batch_results)
                batch_avg_time = np.mean([r['inference_time'] for r in batch_results])
                
                batch_stats.append({
                    'batch_number': i // batch_size + 1,
                    'files_processed': len(batch_files),
                    'total_detections': batch_detections,
                    'avg_inference_time': batch_avg_time
                })
            
            # Results summary
            st.success("✅ Batch processing completed!")
            
            # Overall statistics
            total_detections = sum(r['num_detections'] for r in all_results)
            avg_detections_per_image = total_detections / len(all_results) if all_results else 0
            total_processing_time = sum(r['inference_time'] for r in all_results)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Images Processed", len(all_results))
            with col2:
                st.metric("Total Detections", total_detections)
            with col3:
                st.metric("Avg per Image", f"{avg_detections_per_image:.1f}")
            with col4:
                st.metric("Total Time", f"{total_processing_time:.1f}s")
            
            # Detailed results table
            if save_results:
                st.subheader("📋 Detailed Results")
                
                results_df = pd.DataFrame([
                    {
                        'Filename': r['filename'],
                        'Detections': r['num_detections'],
                        'Inference Time (ms)': f"{r['inference_time']*1000:.1f}",
                        'File Size (KB)': f"{r['file_size']/1024:.1f}",
                        'Model': r.get('model_name', 'Unknown')
                    }
                    for r in all_results
                ])
                
                st.dataframe(results_df, use_container_width=True)
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="💾 Download Results as CSV",
                    data=csv,
                    file_name=f"batch_detection_results_{int(time.time())}.csv",
                    mime="text/csv"
                )
            
            # Analysis charts
            self._create_batch_analysis_charts(all_results)
            
        except Exception as e:
            st.error(f"Batch processing failed: {e}")
            logger.error(f"Batch processing error: {e}")
    
    def _create_batch_analysis_charts(self, results):
        """Create analysis charts for batch results"""
        try:
            df = pd.DataFrame(results)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Detection distribution
                fig1 = px.histogram(
                    df,
                    x='num_detections',
                    nbins=20,
                    title='Distribution of Detections per Image',
                    labels={'num_detections': 'Number of Detections', 'count': 'Frequency'}
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Inference time distribution
                fig2 = px.histogram(
                    df,
                    x='inference_time',
                    nbins=20,
                    title='Distribution of Inference Times',
                    labels={'inference_time': 'Inference Time (seconds)', 'count': 'Frequency'}
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            # Correlation analysis
            if len(df) > 1:
                fig3 = px.scatter(
                    df,
                    x='file_size',
                    y='inference_time',
                    size='num_detections',
                    title='File Size vs Inference Time',
                    labels={
                        'file_size': 'File Size (bytes)',
                        'inference_time': 'Inference Time (seconds)',
                        'num_detections': 'Detections'
                    }
                )
                st.plotly_chart(fig3, use_container_width=True)
                
        except Exception as e:
            logger.error(f"Error creating batch analysis charts: {e}")
    
    def _realtime_tab(self):
        """Real-time detection interface"""
        st.header("⚡ Real-time Analysis")
        
        st.info("🚧 Real-time detection via webcam (requires local setup)")
        
        # Webcam detection would require additional setup
        # For demo purposes, showing simulated real-time data
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📹 Live Feed Simulation")
            
            if st.button("🎥 Start Simulation"):
                self._simulate_realtime_detection()
        
        with col2:
            st.subheader("⚙️ Real-time Settings")
            
            camera_id = st.selectbox("Camera Source", [0, 1, 2])
            display_fps = st.checkbox("Show FPS", value=True)
            save_detections = st.checkbox("Save Detection Log", value=False)
            
            detection_frequency = st.slider("Detection Frequency (Hz)", 1, 30, 10)
            
            st.markdown("""
            **Real-time Features:**
            - Live object detection
            - Performance monitoring
            - Detection logging
            - Real-time analytics
            """)
    
    def _simulate_realtime_detection(self):
        """Simulate real-time detection for demo"""
        st.subheader("🔴 Live Detection Simulation")
        
        # Create placeholders
        image_placeholder = st.empty()
        metrics_placeholder = st.empty()
        chart_placeholder = st.empty()
        
        # Simulation data
        detection_data = []
        
        for i in range(30):  # 30 frames simulation
            # Simulate detection results
            num_detections = np.random.poisson(3)
            inference_time = np.random.normal(0.05, 0.01)
            confidence = np.random.uniform(0.3, 0.9)
            
            # Store data
            detection_data.append({
                'frame': i,
                'detections': num_detections,
                'inference_time': inference_time,
                'avg_confidence': confidence
            })
            
            # Update metrics
            with metrics_placeholder.container():
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Frame", i + 1)
                with col2:
                    st.metric("Detections", num_detections)
                with col3:
                    st.metric("FPS", f"{1/inference_time:.1f}")
                with col4:
                    st.metric("Avg Confidence", f"{confidence:.2f}")
            
            # Update chart
            if len(detection_data) > 1:
                with chart_placeholder.container():
                    df = pd.DataFrame(detection_data)
                    fig = px.line(
                        df, 
                        x='frame', 
                        y='detections',
                        title='Real-time Detection Count'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            time.sleep(0.1)  # Simulate processing time
        
        st.success("✅ Simulation completed!")
    
    def _analytics_dashboard(self):
        """Analytics and performance dashboard"""
        st.header("📈 Analytics Dashboard")
        
        if not st.session_state.detection_results:
            st.info("📊 No detection data available. Run some detections first!")
            return
        
        # Overall statistics
        st.subheader("📊 Overall Statistics")
        
        total_detections = len(st.session_state.detection_results)
        total_objects = sum(r['num_detections'] for r in st.session_state.detection_results)
        avg_objects_per_image = total_objects / total_detections if total_detections > 0 else 0
        avg_inference_time = np.mean([r['inference_time'] for r in st.session_state.detection_results])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Images", total_detections)
        with col2:
            st.metric("Total Objects", total_objects)
        with col3:
            st.metric("Avg Objects/Image", f"{avg_objects_per_image:.1f}")
        with col4:
            st.metric("Avg Inference Time", f"{avg_inference_time*1000:.1f}ms")
        
        # Performance trends
        st.subheader("📈 Performance Trends")
        
        if len(st.session_state.detection_results) > 1:
            df = pd.DataFrame([
                {
                    'index': i,
                    'timestamp': r['timestamp'],
                    'detections': r['num_detections'],
                    'inference_time': r['inference_time'],
                    'model': r.get('model_name', 'Unknown')
                }
                for i, r in enumerate(st.session_state.detection_results)
            ])
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = px.line(
                    df,
                    x='index',
                    y='inference_time',
                    title='Inference Time Trend',
                    labels={'index': 'Detection Number', 'inference_time': 'Time (seconds)'}
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                fig2 = px.line(
                    df,
                    x='index',
                    y='detections',
                    title='Detection Count Trend',
                    labels={'index': 'Detection Number', 'detections': 'Objects Detected'}
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            # Model usage
            model_counts = df['model'].value_counts()
            if len(model_counts) > 1:
                fig3 = px.pie(
                    values=model_counts.values,
                    names=model_counts.index,
                    title='Model Usage Distribution'
                )
                st.plotly_chart(fig3, use_container_width=True)
        
        # Class analysis
        st.subheader("🏷️ Class Analysis")
        
        all_detections = []
        for result in st.session_state.detection_results:
            all_detections.extend(result.get('detections', []))
        
        if all_detections:
            # Class distribution
            class_counts = {}
            confidence_by_class = {}
            
            for detection in all_detections:
                class_name = detection.get('class_name', 'Unknown')
                confidence = detection.get('confidence', 0)
                
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                if class_name not in confidence_by_class:
                    confidence_by_class[class_name] = []
                confidence_by_class[class_name].append(confidence)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig4 = px.bar(
                    x=list(class_counts.keys()),
                    y=list(class_counts.values()),
                    title='Class Distribution',
                    labels={'x': 'Class', 'y': 'Count'}
                )
                st.plotly_chart(fig4, use_container_width=True)
            
            with col2:
                # Average confidence by class
                avg_confidence = {
                    class_name: np.mean(confidences)
                    for class_name, confidences in confidence_by_class.items()
                }
                
                fig5 = px.bar(
                    x=list(avg_confidence.keys()),
                    y=list(avg_confidence.values()),
                    title='Average Confidence by Class',
                    labels={'x': 'Class', 'y': 'Average Confidence'}
                )
                st.plotly_chart(fig5, use_container_width=True)
        
        # Download analytics data
        st.subheader("💾 Export Data")
        
        if st.button("📥 Download Analytics Report"):
            self._generate_analytics_report()
    
    def _generate_analytics_report(self):
        """Generate and download analytics report"""
        try:
            # Create comprehensive report
            report_data = {
                'summary': {
                    'total_images': len(st.session_state.detection_results),
                    'total_objects': sum(r['num_detections'] for r in st.session_state.detection_results),
                    'avg_inference_time': np.mean([r['inference_time'] for r in st.session_state.detection_results]),
                    'report_generated': time.strftime('%Y-%m-%d %H:%M:%S')
                },
                'detections': st.session_state.detection_results
            }
            
            # Convert to JSON
            report_json = json.dumps(report_data, indent=2, default=str)
            
            st.download_button(
                label="📄 Download JSON Report",
                data=report_json,
                file_name=f"detection_analytics_{int(time.time())}.json",
                mime="application/json"
            )
            
            st.success("✅ Report ready for download!")
            
        except Exception as e:
            st.error(f"Error generating report: {e}")

if __name__ == "__main__":
    app = ObjectDetectionApp()
    app.main()
