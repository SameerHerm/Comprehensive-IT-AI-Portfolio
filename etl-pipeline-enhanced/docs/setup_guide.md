# ETL Pipeline Setup Guide

## Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- Python 3.8+
- 8GB RAM minimum
- 20GB free disk space

## Installation Steps

### 1. Clone Repository
bash
git clone https://github.com/yourusername/etl-pipeline-enhanced.git
cd etl-pipeline-enhanced

2. Environment Configuration
Bash

cp .env.example .env
# Edit .env with your configurations
nano .env
3. Run Setup Script
Bash

chmod +x scripts/setup.sh
./scripts/setup.sh
4. Start Services
Bash

./scripts/start_services.sh
Service URLs
Airflow: http://localhost:8080 (admin/admin)
Kafka UI: http://localhost:9000
Grafana: http://localhost:3000 (admin/admin)
Configuration
Kafka Topics
Create additional topics:

Bash

docker-compose exec kafka kafka-topics \
  --create --topic my-topic \
  --bootstrap-server localhost:9093 \
  --partitions 3 --replication-factor 1
Airflow DAGs
Place DAG files in airflow/dags/ directory.

Database Connections
Configure in Airflow UI under Admin > Connections.

Troubleshooting
Services not starting
Bash

docker-compose logs [service-name]
docker-compose ps
Reset everything
Bash

docker-compose down -v
rm -rf logs/* data/processed/*
./scripts/setup.sh
Production Deployment
Update .env with production values
Configure SSL certificates
Set up monitoring alerts
Enable backups
Configure resource limits
text


## File 45: docs/api_documentation.md
markdown
# ETL Pipeline API Documentation

## REST API Endpoints

### Health Check
GET /api/v1/health

text

Returns service health status.

### Pipeline Status
GET /api/v1/pipeline/status

text

Returns current pipeline status and metrics.

### Trigger Pipeline
POST /api/v1/pipeline/trigger
{
"pipeline_name": "string",
"parameters": {}
}

text


### Data Quality Report
GET /api/v1/quality/report/{date}

text

Returns quality report for specified date.

## Kafka Topics

### etl-streaming-data
Raw streaming data input.
json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {},
  "source": "string"
}
etl-processed-data
Processed data output.

JSON

{
  "timestamp": "2024-01-15T10:30:00Z",
  "processed_data": {},
  "pipeline": "string",
  "status": "success|failed"
}


Database Schema
warehouse.fact_sales
Column	Type	Description
transaction_id	VARCHAR	Unique transaction ID
date	DATE	Transaction date
amount	DECIMAL	Transaction amount
status	VARCHAR	Transaction status
Error Codes
Code	Description
1001	Data validation failed
1002	Pipeline execution error
1003	Database connection error
1004	Kafka producer error
1005	Authentication failed
