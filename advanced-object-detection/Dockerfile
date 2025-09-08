FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/annotations \
    models/pretrained models/trained models/checkpoints \
    logs web_app/static/uploads

# Set environment variables
ENV PYTHONPATH=/app:$PYTHONPATH
ENV FLASK_APP=web_app/app.py

# Expose ports
EXPOSE 5000 8000

# Default command
CMD ["python", "web_app/app.py"]
