# Comprehensive IT & AI Portfolio

Repository: https://github.com/SameerHerm/Comprehensive-IT-AI-Portfolio

This portfolio showcases four production‑ready projects across Machine Learning, Computer Vision, NLP, and Data Engineering. Each project is modular, reproducible, and includes APIs/UI where relevant.

- Cardiovascular Risk Prediction System (ML + Explainability + Streamlit + Flask + Docker)
- Advanced Object Detection System (YOLOv8/Faster R‑CNN/SSD/EfficientDet + Real-time + API + Web)
- Enhanced ETL Data Pipeline (Shell + Airflow + Kafka + Monitoring)
- Enhanced Spam Classifier (NLP + Multiple Models + REST API + Dashboard)

## Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Projects](#projects)
  - [Cardiovascular Risk Prediction System](#cardiovascular-risk-prediction-system)
  - [Advanced Object Detection System](#advanced-object-detection-system)
  - [Enhanced ETL Data Pipeline (Airflow & Kafka)](#enhanced-etl-data-pipeline-airflow--kafka)
  - [Enhanced Spam Classifier](#enhanced-spam-classifier)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview
- Languages/Frameworks: Python 3.8+, scikit‑learn, XGBoost, LightGBM, Optuna, SHAP, PyTorch, OpenCV, FastAPI/Flask, Streamlit, Apache Airflow, Apache Kafka, Docker.
- Goals: Clear, modular code ready for demos, interviews, and real‑world deployment (APIs, dashboards, containers).
- Datasets: Uses public datasets and/or scripts to generate sample data.

## Repository Structure
```
projects/
  cardio-risk/
  object-detection/
  etl-airflow-kafka/
  spam-classifier/
common/
  docker/            # optional shared compose files
  scripts/           # utilities
docs/
LICENSE
README.md            # this file
```

## Quick Start
- Prerequisites
  - Python 3.8+ (conda or venv recommended)
  - CUDA-capable GPU recommended for object detection
  - Docker (optional) for containerized runs

- Create environment
  ```
  python -m venv .venv
  source .venv/bin/activate   # Windows: .venv\Scripts\activate
  ```

- Install per project
  ```
  cd projects/<project-folder>
  pip install -r requirements.txt
  ```

- Run locally or via Docker (if Dockerfile/compose provided).

## Projects

### Cardiovascular Risk Prediction System
A comprehensive ML system for predicting cardiovascular disease risk using patient health and lifestyle features.

- Features
  - Multiple Models: Random Forest, XGBoost, LightGBM, simple Neural Networks
  - Hyperparameter Optimization: Optuna automated tuning
  - Interpretability: SHAP values for global and per‑prediction insights
  - Interactive Web App: Streamlit UI for clinicians/stakeholders
  - REST API: Flask-based service (single and batch predictions)
  - Evaluation: ROC/PR curves, confusion matrices, detailed reports
  - Docker Support: Containerized UI and API

- Tech
  - Python, scikit‑learn, XGBoost, LightGBM, Optuna, SHAP, Streamlit, Flask, Docker

- Run
  ```
  cd projects/cardio-risk
  pip install -r requirements.txt
  # UI
  streamlit run app.py
  # API
  python api.py
  # Example JSON body for POST /predict:
  # {"age": 54, "bmi": 27.2, "systolic_bp": 138, "cholesterol": 212, "smoker": 1, "...": "..."}
  ```

### Advanced Object Detection System
A production‑ready object detection suite with multiple models, real‑time inference, web interface, and REST API.

- Features
  - Multi‑Model: YOLOv8 (n/s/m/l/x), Faster R‑CNN, SSD MobileNet, EfficientDet
  - Real‑time: Webcam/video stream support
  - Batch: Directory/image‑list inference
  - Benchmarking: Speed vs accuracy comparisons
  - Analytics: Performance reports, monitoring hooks
  - REST API: FastAPI endpoints
  - Web Interface: Streamlit/FastAPI UI
  - Docker Support

- AI/ML Capabilities
  - Custom Training, Data Augmentation, Transfer Learning
  - Ensembles for improved accuracy
  - Model Quantization for edge
  - ONNX Export for cross‑platform inference

- Requirements
  - Python 3.8+, PyTorch/TorchVision, Ultralytics (YOLOv8), OpenCV, FastAPI/Streamlit
  - CUDA GPU recommended (CPU supported for demo scale)

- Run
  ```
  cd projects/object-detection
  pip install -r requirements.txt
  # Inference (YOLOv8)
  python detect.py --model yolov8n.pt --source data/sample.mp4
  # API
  uvicorn api:app --host 0.0.0.0 --port 8000
  # Web
  streamlit run app.py
  ```

### Enhanced ETL Data Pipeline (Airflow & Kafka)
A production‑ready ETL pipeline featuring real‑time streaming, batch processing, data quality checks, and monitoring.

- Features
  - Real‑time Streaming: Kafka‑based ingestion
  - Batch Processing: Apache Airflow DAGs for scheduled jobs
  - Data Quality: Automated validation and QA checks
  - Monitoring & Alerting: Pipeline health, lag, and error alerts
  - Scalable: Docker‑based deployment and easy scaling
  - Multiple Sources: Files, APIs, and databases
  - Error Handling: Retries, dead‑letter queues, structured logging
  - Data Versioning: Lineage and schema versioning

- Architecture (high‑level)
  ```
  [Sources] → [Kafka (Extract)] → [Airflow (Orchestrate)] → [Transform] → [Load → Warehouse]
                                           ↓                     ↓                    ↓
                                     [Monitoring]         [Quality Checks]       [Alerting]
  ```

- Run
  ```
  cd projects/etl-airflow-kafka
  # if docker-compose.yml is provided
  docker compose up -d
  # produce sample data
  bash scripts/producer.sh   # or: python scripts/producer.py
  # trigger Airflow DAG from UI or CLI (etl_dag)
  ```

### Enhanced Spam Classifier
An advanced spam email classifier with multiple algorithms, real‑time API, and a Streamlit dashboard.

- Features
  - Algorithms: Naive Bayes, SVM, Random Forest, XGBoost, and a simple Deep Learning model
  - Text Processing: TF‑IDF, Word2Vec, custom feature engineering
  - Real‑time API: Flask‑based REST service
  - Evaluation: Confusion matrix, ROC curves, detailed metrics/plots
  - Dashboard: Streamlit for interactive exploration
  - Robustness: K‑fold cross‑validation, class‑imbalance handling
  - Engineering: Config‑driven runs, logging, model persistence, batch prediction

- Install
  ```
  cd projects/spam-classifier
  pip install -r requirements.txt
  python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')"
  ```

- Train & Run
  ```
  # Train all models
  python scripts/train_model.py --model all

  # Predict via CLI
  python scripts/predict.py --text "Your email text here"

  # API
  python api/app.py
  # Endpoints:
  #  - POST /predict         → Predict if text is spam
  #  - POST /batch_predict   → Predict multiple texts
  #  - GET  /health          → Service status

  # Tests
  pytest tests/
  ```

- Sample Performance (varies by dataset/splits)
  | Model         | Accuracy | Precision | Recall | F1-Score |
  |---------------|----------|-----------|--------|----------|
  | Naive Bayes   | 97.8%    | 98.2%     | 97.5%  | 97.8%    |
  | SVM           | 98.5%    | 98.7%     | 98.3%  | 98.5%    |
  | Random Forest | 98.2%    | 98.4%     | 98.0%  | 98.2%    |
  | XGBoost       | 98.9%    | 99.0%     | 98.8%  | 98.9%    |

## Contributing
Pull requests are welcome.
- Fork the repo and create a feature branch.
- Follow PEP 8 and add tests where relevant.
- For major changes, open an issue first to discuss scope.

## License
MIT License — see LICENSE.

## Contact
- Portfolio/Repo: https://github.com/SameerHerm/Comprehensive-IT-AI-Portfolio
- Open an issue for questions, suggestions, or collaboration.
