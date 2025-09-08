// Main JavaScript for Object Detection Web App

// Global variables
let selectedFile = null;
let selectedModel = 'yolo';
let isProcessing = false;

// DOM elements
const fileInput = document.getElementById('file-input');
const uploadArea = document.getElementById('upload-area');
const modelSelect = document.getElementById('model-select');
const detectBtn = document.getElementById('detect-btn');
const resultsSection = document.getElementById('results-section');
const loadingDiv = document.getElementById('loading');

// File upload handling
uploadArea.addEventListener('click', () => {
    fileInput.click();
});

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

function handleFileSelect(file) {
    // Validate file type
    const validTypes = ['image/jpeg', 'image/png', 'image/jpg'];
    if (!validTypes.includes(file.type)) {
        showError('Please select a valid image file (JPEG or PNG)');
        return;
    }
    
    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
        showError('File size must be less than 10MB');
        return;
    }
    
    selectedFile = file;
    updateUploadArea(file);
}

function updateUploadArea(file) {
    const uploadText = document.getElementById('upload-text');
    uploadText.innerHTML = `
        <p><strong>Selected:</strong> ${file.name}</p>
        <p><strong>Size:</strong> ${(file.size / 1024).toFixed(2)} KB</p>
        <p class="text-muted">Click or drag to change file</p>
    `;
    
    // Preview image
    const reader = new FileReader();
    reader.onload = (e) => {
        const preview = document.createElement('img');
        preview.src = e.target.result;
        preview.style.maxWidth = '200px';
        preview.style.marginTop = '10px';
        uploadText.appendChild(preview);
    };
    reader.readAsDataURL(file);
}

// Model selection
modelSelect.addEventListener('change', (e) => {
    selectedModel = e.target.value;
});

// Detection
detectBtn.addEventListener('click', performDetection);

async function performDetection() {
    if (!selectedFile) {
        showError('Please select an image first');
        return;
    }
    
    if (isProcessing) {
        return;
    }
    
    isProcessing = true;
    showLoading(true);
    
    const formData = new FormData();
    formData.append('image', selectedFile);
    formData.append('model', selectedModel);
    
    try {
        const response = await fetch('/api/detect', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Detection failed');
        }
        
        const data = await response.json();
        displayResults(data);
        
    } catch (error) {
        showError('Error performing detection: ' + error.message);
    } finally {
        isProcessing = false;
        showLoading(false);
    }
}

function displayResults(data) {
    resultsSection.classList.add('active');
    
    // Display image with bounding boxes
    const imageContainer = document.getElementById('image-container');
    imageContainer.innerHTML = `<img src="${data.image_url}" class="detected-image" alt="Detection Result">`;
    
    // Update statistics
    updateStatistics(data.detections);
    
    // Display detection list
    displayDetectionList(data.detections);
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function updateStatistics(detections) {
    const totalObjects = detections.length;
    const uniqueClasses = [...new Set(detections.map(d => d.class))].length;
    const avgConfidence = detections.reduce((sum, d) => sum + d.confidence, 0) / totalObjects || 0;
    
    document.getElementById('total-objects').textContent = totalObjects;
    document.getElementById('unique-classes').textContent = uniqueClasses;
    document.getElementById('avg-confidence').textContent = (avgConfidence * 100).toFixed(1) + '%';
}

function displayDetectionList(detections) {
    const listContainer = document.getElementById('detection-list');
    listContainer.innerHTML = '';
    
    // Group detections by class
    const grouped = {};
    detections.forEach(det => {
        if (!grouped[det.class]) {
            grouped[det.class] = [];
        }
        grouped[det.class].push(det);
    });
    
    // Display grouped detections
    Object.entries(grouped).forEach(([className, items]) => {
        const groupDiv = document.createElement('div');
        groupDiv.className = 'detection-group';
        
        const title = document.createElement('h4');
        title.textContent = `${className} (${items.length})`;
        groupDiv.appendChild(title);
        
        items.forEach((item, index) => {
            const itemDiv = document.createElement('div');
            itemDiv.className = 'detection-item';
            
            itemDiv.innerHTML = `
                <div class="detection-info">
                    <span>Instance ${index + 1}</span>
                    <span>Confidence: ${(item.confidence * 100).toFixed(1)}%</span>
                </div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${item.confidence * 100}%">
                        ${(item.confidence * 100).toFixed(1)}%
                    </div>
                </div>
            `;
            
            groupDiv.appendChild(itemDiv);
        });
        
        listContainer.appendChild(groupDiv);
    });
}

// Real-time detection
let stream = null;
let isRealtime = false;

async function startRealtime() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        const video = document.getElementById('video-feed');
        video.srcObject = stream;
        isRealtime = true;
        
        // Start detection loop
        detectRealtime();
        
    } catch (error) {
        showError('Cannot access camera: ' + error.message);
    }
}

function stopRealtime() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    isRealtime = false;
}

async function detectRealtime() {
    if (!isRealtime) return;
    
    const video = document.getElementById('video-feed');
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    
    ctx.drawImage(video, 0, 0);
    
    // Convert to blob and send for detection
    canvas.toBlob(async (blob) => {
        const formData = new FormData();
        formData.append('image', blob);
        formData.append('model', selectedModel);
        
        try {
            const response = await fetch('/api/detect-realtime', {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                const data = await response.json();
                drawRealtimeBoxes(data.detections);
            }
        } catch (error) {
            console.error('Realtime detection error:', error);
        }
        
        // Continue detection loop
        if (isRealtime) {
            setTimeout(() => detectRealtime(), 100); // 10 FPS
        }
    });
}

function drawRealtimeBoxes(detections) {
    const canvas = document.getElementById('canvas-overlay');
    const ctx = canvas.getContext('2d');
    const video = document.getElementById('video-feed');
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    detections.forEach(det => {
        // Draw box
        ctx.strokeStyle = '#00FF00';
        ctx.lineWidth = 2;
        ctx.strokeRect(det.x, det.y, det.width, det.height);
        
        // Draw label
        ctx.fillStyle = '#00FF00';
        ctx.font = '16px Arial';
        ctx.fillText(
            `${det.class} ${(det.confidence * 100).toFixed(1)}%`,
            det.x,
            det.y - 5
        );
    });
}

// Utility functions
function showLoading(show) {
    if (show) {
        loadingDiv.classList.add('active');
    } else {
        loadingDiv.classList.remove('active');
    }
}

function showError(message) {
    const toast = document.createElement('div');
    toast.className = 'error-toast';
    toast.textContent = message;
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.remove();
    }, 3000);
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    console.log('Object Detection Web App initialized');
});
