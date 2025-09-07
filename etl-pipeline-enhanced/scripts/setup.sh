#!/bin/bash

#############################################
# ETL Pipeline Setup Script
# Sets up the complete ETL pipeline environment
#############################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="etl-pipeline-enhanced"
PYTHON_VERSION="3.9"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   ETL Pipeline Setup Script${NC}"
echo -e "${GREEN}========================================${NC}"

# Function to print colored messages
print_message() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_message "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3 first."
        exit 1
    fi
    
    print_message "All prerequisites are installed ✓"
}

# Create directory structure
create_directories() {
    print_message "Creating directory structure..."
    
    directories=(
        "data/raw"
        "data/processed"
        "data/staging"
        "data/archive"
        "logs"
        "config"
        "models"
        "notebooks"
        "monitoring/dashboards"
        "monitoring/alerts"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        print_message "Created directory: $dir"
    done
    
    # Create .gitkeep files
    for dir in "${directories[@]}"; do
        touch "$dir/.gitkeep"
    done
}

# Setup Python virtual environment
setup_python_env() {
    print_message "Setting up Python virtual environment..."
    
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_message "Virtual environment created"
    else
        print_warning "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_message "Python dependencies installed"
    else
        print_warning "requirements.txt not found"
    fi
}

# Setup environment file
setup_env_file() {
    print_message "Setting up environment file..."
    
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            cp .env.example .env
            print_message "Created .env file from .env.example"
            print_warning "Please update .env with your actual configuration"
        else
            print_error ".env.example not found"
        fi
    else
        print_warning ".env file already exists"
    fi
}

# Initialize database
init_database() {
    print_message "Initializing database..."
    
    # Start only PostgreSQL service
    docker-compose up -d postgres
    
    # Wait for PostgreSQL to be ready
    print_message "Waiting for PostgreSQL to be ready..."
    sleep 10
    
    # Create database schema
    docker-compose exec -T postgres psql -U airflow -d airflow << EOF
CREATE SCHEMA IF NOT EXISTS warehouse;
CREATE SCHEMA IF NOT EXISTS staging;

CREATE TABLE IF NOT EXISTS warehouse.fact_sales (
    id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(50),
    product_id VARCHAR(50),
    product_name VARCHAR(100),
    quantity INTEGER,
    price DECIMAL(10,2),
    total_amount DECIMAL(10,2),
    customer_id VARCHAR(50),
    region VARCHAR(50),
    payment_method VARCHAR(50),
    status VARCHAR(20),
    transaction_date DATE,
    load_date DATE DEFAULT CURRENT_DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS staging.raw_transactions (
    id SERIAL PRIMARY KEY,
    data JSONB,
    source VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

GRANT ALL PRIVILEGES ON SCHEMA warehouse TO airflow;
GRANT ALL PRIVILEGES ON SCHEMA staging TO airflow;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA warehouse TO airflow;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA staging TO airflow;
EOF
    
    print_message "Database initialized"
}

# Setup Kafka topics
setup_kafka_topics() {
    print_message "Setting up Kafka topics..."
    
    # Start Kafka and Zookeeper
    docker-compose up -d zookeeper kafka
    
    # Wait for Kafka to be ready
    print_message "Waiting for Kafka to be ready..."
    sleep 15
    
    # Create topics
    topics=(
        "etl-streaming-data"
        "etl-processed-data"
        "etl-errors"
        "etl-monitoring"
    )
    
    for topic in "${topics[@]}"; do
        docker-compose exec -T kafka kafka-topics \
            --create \
            --if-not-exists \
            --bootstrap-server localhost:9093 \
            --topic "$topic" \
            --partitions 3 \
            --replication-factor 1
        
        print_message "Created Kafka topic: $topic"
    done
}

# Initialize Airflow
init_airflow() {
    print_message "Initializing Airflow..."
    
    # Build Airflow image
    docker-compose build airflow-webserver
    
    # Initialize Airflow database
    docker-compose run --rm airflow-webserver airflow db init
    
    # Create admin user
    docker-compose run --rm airflow-webserver airflow users create \
        --username admin \
        --password admin \
        --firstname Admin \
        --lastname User \
        --role Admin \
        --email admin@example.com
    
    print_message "Airflow initialized with admin user (username: admin, password: admin)"
}

# Generate sample data
generate_sample_data() {
    print_message "Generating sample data..."
    
    python3 << EOF
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Generate sample sales data
dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
products = ['Laptop', 'Phone', 'Tablet', 'Headphones', 'Smartwatch']
regions = ['North', 'South', 'East', 'West', 'Central']

data = []
for date in dates:
    for _ in range(random.randint(50, 200)):
        data.append({
            'date': date.strftime('%Y-%m-%d'),
            'product_id': f'PROD_{random.randint(1000, 9999)}',
            'product_name': random.choice(products),
            'quantity': random.randint(1, 10),
            'price': round(random.uniform(50, 2000), 2),
            'customer_id': f'CUST_{random.randint(10000, 99999)}',
            'region': random.choice(regions),
            'payment_method': random.choice(['credit_card', 'debit_card', 'paypal'])
        })

df = pd.DataFrame(data)
df['total_amount'] = df['quantity'] * df['price']
df.to_csv('data/raw/sales_data.csv', index=False)

print(f"Generated {len(df)} sample records")
EOF
    
    print_message "Sample data generated in data/raw/sales_data.csv"
}

# Setup monitoring
setup_monitoring() {
    print_message "Setting up monitoring..."
    
    # Create Grafana dashboard
    cat > monitoring/dashboards/pipeline_dashboard.json << 'EOF'
{
  "dashboard": {
    "id": null,
    "title": "ETL Pipeline Dashboard",
    "tags": ["etl", "pipeline"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Records Processed",
        "type": "graph"
      },
      {
        "id": 2,
        "title": "Error Rate",
        "type": "graph"
      },
      {
        "id": 3,
        "title": "Pipeline Status",
        "type": "stat"
      }
    ]
  }
}
EOF
    
    print_message "Monitoring configuration created"
}

# Main setup flow
main() {
    print_message "Starting ETL Pipeline setup..."
    
    check_prerequisites
    create_directories
    setup_python_env
    setup_env_file
    init_database
    setup_kafka_topics
    init_airflow
    generate_sample_data
    setup_monitoring
    
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}   Setup Complete! 🎉${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Update .env file with your configuration"
    echo "2. Run 'docker-compose up -d' to start all services"
    echo "3. Access Airflow at http://localhost:8080 (admin/admin)"
    echo "4. Access Kafka UI at http://localhost:9000"
    echo "5. Access Grafana at http://localhost:3000 (admin/admin)"
    echo ""
    echo "To start the ETL pipeline:"
    echo "  ./scripts/start_services.sh"
    echo ""
    echo "Happy ETL-ing! 🚀"
}

# Run main function
main
