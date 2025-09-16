# Comprehensive IT & AI Portfolio

Repository: https://github.com/SameerHerm/Comprehensive-IT-AI-Portfolio

This monorepo showcases production-ready projects across Machine Learning, Deep Learning, NLP, and Data Engineering. Each project follows clean code practices, offers reproducible environments, and includes APIs/UX where relevant.

- Cardiovascular Risk Prediction System (ML + Explainability + Streamlit + Flask + Docker)
- Advanced Object Detection System (YOLOv8 / Faster R-CNN / SSD / EfficientDet + Real-time + API + Web)
- Enhanced ETL Data Pipeline (Shell + Airflow + Kafka + Monitoring)
- Enhanced Spam Classifier (NLP + Multiple Models + REST API + Dashboard)

Table of Contents
- Overview
- Repo Structure
- Quick Start
- Projects
  - Cardiovascular Risk Prediction System
  - Advanced Object Detection System
  - Enhanced ETL Data Pipeline (Airflow & Kafka)
  - Enhanced Spam Classifier
- Contributing
- License
- Contact

Overview
- Languages/Frameworks: Python 3.8+, scikit-learn, XGBoost, LightGBM, PyTorch, SHAP, Optuna, OpenCV, FastAPI/Flask, Streamlit, Airflow, Kafka, Docker.
- Goals: Clear, modular code ready for demos, interviews, and real-world deployment (APIs, dashboards, containers).
- Datasets: Use public datasets or scripts to generate sample data where applicable.

Repo Structure (suggested)
- projects/
  - cardio-risk/
  - object-detection/
  - etl-airflow-kafka/
  - spam-classifier/
- common/
  - docker/ (optional shared compose files)
  - scripts/ (utility scripts)
- docs/
- LICENSE
- README.md (this file)

Quick Start
- Prerequisites
  - Python 3.8+
  - pip or conda
  - CUDA-capable GPU recommended for object detection
  - Docker (optional) for containerized runs

- Create a Python environment
  - python -m venv .venv
  - source .venv/bin/activate  # Windows: .venv\Scripts\activate

- Install dependencies per project
  - cd projects/<project-folder>
  - pip install -r requirements.txt

- Run locally or via Docker (if Dockerfile is provided in the project)

Projects

1) Cardiovascular Risk Prediction System
A comprehensive machine learning system for predicting cardiovascular disease risk using patient health and lifestyle features.

Key Features
- Multiple Models: Random Forest, XGBoost, LightGBM, Neural Networks
- Hyperparameter Optimization: Optuna automated tuning
- Interpretability: SHAP values for feature importance and individual explanations
- Interactive Web App: Streamlit-based UI for clinicians/stakeholders
- REST API: Flask-based prediction service (single and batch)
- Batch Processing: CSV/JSON batch prediction support
- Evaluation: ROC curves, PR curves, confusion matrices, detailed reports
- Docker Support: Containerized inference service and UI

Tech
- Python, scikit-learn, XGBoost, LightGBM, Optuna, SHAP, Streamlit, Flask, Docker

Run (example)
- Streamlit UI
  - cd projects/cardio-risk
  - pip install -r requirements.txt
  - streamlit run app.py
- REST API
  - python api.py
  - Example (JSON): {"age": 54, "bmi": 27.2, "systolic_bp": 138, "cholesterol": 212, "smoker": 1, "...": "..."}
  - POST /predict for single; POST /batch_predict for multiple
- Docker (if provided)
  - docker build -t cardio-risk .
  - docker run -p 8501:8501 cardio-risk

2) Advanced Object Detection System
A production-ready object detection suite with multiple models, real-time inference, web interface, and REST API.

Features
- Multi-Model Support: YOLOv8 (n/s/m/l/x), Faster R-CNN, SSD MobileNet, EfficientDet
- Real-time Detection: Webcam and video stream processing
- Batch Inference: Directory/image list processing
- Benchmarking: Speed/accuracy comparisons across models
- Analytics: Performance reports and monitoring hooks
- REST API: FastAPI endpoints for image/video inference
- Web Interface: Streamlit/FastAPI UI for interactive testing
- Docker Support: Containerized deployment for reproducibility

AI/ML Capabilities
- Custom Training: Train on your dataset with clear config files
- Data Augmentation and Transfer Learning
- Ensembles: Combine model outputs for improved accuracy
- Model Quantization: For edge deployment
- ONNX Export: Cross-platform inference

Requirements
- Python 3.8+, PyTorch/TorchVision, Ultralytics (YOLOv8), OpenCV, FastAPI/Streamlit
- CUDA-capable GPU recommended; CPU mode supported for demo scale

Run (example)
- cd projects/object-detection
- pip install -r requirements.txt
- Inference (YOLOv8)
  - python detect.py --model yolov8n.pt --source data/sample.mp4
- API
  - uvicorn api:app --host 0.0.0.0 --port 8000
  - POST /predict (multipart/form-data image)
- Web
  - streamlit run app.py
- Docker (if provided)
  - docker build -t object-detection .
  - docker run -p 8000:8000 object-detection

3) Enhanced ETL Data Pipeline with Shell, Airflow & Kafka
A production-ready ETL pipeline featuring real-time streaming, batch processing, data quality checks, and monitoring.

Features
- Real-time Streaming: Kafka-based ingestion for event/data streams
- Batch Processing: Apache Airflow DAGs for scheduled ETL jobs
- Data Quality: Automated validation and QA checks
- Monitoring & Alerting: Pipeline health, lags, and error alerts
- Scalable Architecture: Docker-based for easy scaling and reproducibility
- Multiple Sources: Files, APIs, and databases
- Error Handling: Retries, dead-letter queues, and logging
- Data Versioning: Track lineage and schema versions

Architecture (high-level)
[Sources] → [Kafka (Extract)] → [Airflow (Orchestrate)] → [Transform] → [Load → Warehouse]
                                      ↓                ↓                 ↓
                               [Monitoring]     [Quality Checks]     [Alerting]

Run (example)
- cd projects/etl-airflow-kafka
- Start services
  - docker compose up -d  # starts Zookeeper, Kafka, Airflow (if compose file provided)
- Produce sample data
  - bash scripts/producer.sh  # or python scripts/producer.py
- Trigger Airflow DAG
  - Use Airflow UI or CLI to trigger etl_dag
- Inspect warehouse (e.g., Postgres) for loaded tables

4) Enhanced Spam Classifier
An advanced, production-ready spam email classifier with multiple algorithms, real-time API, and comprehensive evaluation.

Features
- Multiple Algorithms: Naive Bayes, SVM, Random Forest, XGBoost, and Deep Learning
- Text Processing: TF-IDF, Word2Vec, custom feature engineering
- Real-time API: Flask-based REST API for instant predictions
- Evaluation: Confusion matrix, ROC curves, detailed metrics and plots
- Dashboard: Streamlit app for exploration and manual testing
- Robustness: K-fold cross-validation, class imbalance handling
- Engineering: Config-driven runs, logging, model persistence, batch prediction

Installation
- cd projects/spam-classifier
- pip install -r requirements.txt
- Download NLTK data
  - python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')"

Train & Run
- Train all models
  - python scripts/train_model.py --model all
- Predict (CLI)
  - python scripts/predict.py --text "Your email text here"
- API
  - python api/app.py
  - Endpoints:
    - POST /predict  → Predict if text is spam
    - POST /batch_predict → Predict multiple texts
    - GET /health → Service status
- Tests
  - pytest tests/

Model Performance (sample)
| Model         | Accuracy | Precision | Recall | F1-Score |
|---------------|----------|-----------|--------|----------|
| Naive Bayes   | 97.8%    | 98.2%     | 97.5%  | 97.8%    |
| SVM           | 98.5%    | 98.7%     | 98.3%  | 98.5%    |
| Random Forest | 98.2%    | 98.4%     | 98.0%  | 98.2%    |
| XGBoost       | 98.9%    | 99.0%     | 98.8%  | 98.9%    |

Notes
- Results depend on dataset splits and preprocessing; replicate via provided scripts.

Contributing
- Pull requests are welcome. Please:
  - Fork the repo, create a feature branch, and open a PR
  - Follow PEP8 and add tests where relevant
  - For major changes, open an issue first to discuss scope

License
- MIT License — see LICENSE file.

Contact
- Portfolio/Repo: https://github.com/SameerHerm/Comprehensive-IT-AI-Portfolio
- Feel free to open an issue for questions, suggestions, or collaboration.
