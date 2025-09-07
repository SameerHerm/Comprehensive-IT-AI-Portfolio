#!/bin/bash

#############################################
# Monitoring Script
# Monitor ETL pipeline health and metrics
#############################################

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_ROOT/logs"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Functions
check_service() {
    local service=$1
    if docker-compose ps | grep -q "$service.*Up"; then
        echo -e "${GREEN}✓${NC} $service is running"
        return 0
    else
        echo -e "${RED}✗${NC} $service is not running"
        return 1
    fi
}

check_kafka_topics() {
    echo "Checking Kafka topics..."
    docker-compose exec -T kafka kafka-topics --list --bootstrap-server localhost:9093 2>/dev/null || echo "Kafka not available"
}

check_airflow_dags() {
    echo "Checking Airflow DAGs..."
    docker-compose exec -T airflow-webserver airflow dags list 2>/dev/null || echo "Airflow not available"
}

check_database() {
    echo "Checking database connection..."
    docker-compose exec -T postgres psql -U airflow -c "SELECT 1" >/dev/null 2>&1 && \
        echo -e "${GREEN}Database connection OK${NC}" || \
        echo -e "${RED}Database connection failed${NC}"
}

show_metrics() {
    echo -e "\n${GREEN}=== Pipeline Metrics ===${NC}"
    
    # Get record counts
    docker-compose exec -T postgres psql -U airflow -d airflow -t -c \
        "SELECT COUNT(*) as record_count FROM warehouse.fact_sales" 2>/dev/null || echo "0"
    
    # Get latest job status
    docker-compose exec -T airflow-webserver airflow dags state etl_main_pipeline 2>/dev/null || echo "No runs"
}

show_logs() {
    local component=${1:-all}
    
    case $component in
        airflow)
            docker-compose logs --tail=50 airflow-webserver airflow-scheduler
            ;;
        kafka)
            docker-compose logs --tail=50 kafka
            ;;
        all)
            docker-compose logs --tail=20
            ;;
        *)
            echo "Unknown component: $component"
            ;;
    esac
}

# Main
case ${1:-status} in
    status)
        echo -e "${GREEN}=== Service Status ===${NC}"
        check_service "postgres"
        check_service "redis"
        check_service "kafka"
        check_service "zookeeper"
        check_service "airflow-webserver"
        check_service "airflow-scheduler"
        check_service "grafana"
        ;;
    
    health)
        echo -e "${GREEN}=== Health Check ===${NC}"
        check_database
        check_kafka_topics
        check_airflow_dags
        ;;
    
    metrics)
        show_metrics
        ;;
    
    logs)
        show_logs ${2:-all}
        ;;
    
    *)
        echo "Usage: $0 {status|health|metrics|logs} [component]"
        exit 1
        ;;
esac
