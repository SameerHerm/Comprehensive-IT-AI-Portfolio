#!/bin/bash

#############################################
# Start Services Script
# Starts all ETL pipeline services
#############################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   Starting ETL Pipeline Services${NC}"
echo -e "${GREEN}========================================${NC}"

# Function to check if service is running
check_service() {
    local service=$1
    if docker-compose ps | grep -q "$service.*Up"; then
        echo -e "${GREEN}✓${NC} $service is already running"
        return 0
    else
        echo -e "${YELLOW}○${NC} $service is not running"
        return 1
    fi
}

# Function to wait for service
wait_for_service() {
    local service=$1
    local port=$2
    local max_attempts=30
    local attempt=0
    
    echo -n "Waiting for $service to be ready"
    while [ $attempt -lt $max_attempts ]; do
        if nc -z localhost $port 2>/dev/null; then
            echo -e " ${GREEN}✓${NC}"
            return 0
        fi
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    echo -e " ${RED}✗${NC}"
    return 1
}

# Change to project root
cd "$PROJECT_ROOT"

# Check Docker and Docker Compose
echo -e "${GREEN}[1/7]${NC} Checking prerequisites..."
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Error: Docker Compose is not installed${NC}"
    exit 1
fi

# Start Docker if not running
echo -e "${GREEN}[2/7]${NC} Ensuring Docker daemon is running..."
if ! docker info &> /dev/null; then
    echo "Starting Docker daemon..."
    sudo systemctl start docker || sudo service docker start
fi

# Start core services
echo -e "${GREEN}[3/7]${NC} Starting core services..."
docker-compose up -d postgres redis

# Wait for PostgreSQL
wait_for_service "PostgreSQL" 5432

# Wait for Redis
wait_for_service "Redis" 6379

# Start Kafka services
echo -e "${GREEN}[4/7]${NC} Starting Kafka services..."
docker-compose up -d zookeeper
wait_for_service "Zookeeper" 2181

docker-compose up -d kafka
wait_for_service "Kafka" 9092

docker-compose up -d kafka-ui
echo -e "${GREEN}✓${NC} Kafka UI started"

# Initialize Airflow
echo -e "${GREEN}[5/7]${NC} Initializing Airflow..."
if [ ! -f ".airflow_initialized" ]; then
    echo "Running Airflow database initialization..."
    docker-compose run --rm airflow-webserver airflow db init
    
    echo "Creating Airflow admin user..."
    docker-compose run --rm airflow-webserver airflow users create \
        --username admin \
        --password admin \
        --firstname Admin \
        --lastname User \
        --role Admin \
        --email admin@example.com
    
    touch .airflow_initialized
    echo -e "${GREEN}✓${NC} Airflow initialized"
else
    echo -e "${GREEN}✓${NC} Airflow already initialized"
fi

# Start Airflow services
echo -e "${GREEN}[6/7]${NC} Starting Airflow services..."
docker-compose up -d airflow-webserver airflow-scheduler airflow-worker
wait_for_service "Airflow Webserver" 8080

# Start monitoring services
echo -e "${GREEN}[7/7]${NC} Starting monitoring services..."
docker-compose up -d grafana
wait_for_service "Grafana" 3000

# Create Kafka topics
echo -e "\n${GREEN}Creating Kafka topics...${NC}"
sleep 5

topics=(
    "etl-streaming-data"
    "etl-processed-data"
    "etl-errors"
    "transactions"
    "weather-data"
    "fraud-transactions"
)

for topic in "${topics[@]}"; do
    docker-compose exec -T kafka kafka-topics \
        --create \
        --if-not-exists \
        --bootstrap-server localhost:9093 \
        --topic "$topic" \
        --partitions 3 \
        --replication-factor 1 2>/dev/null && \
    echo -e "${GREEN}✓${NC} Created topic: $topic" || \
    echo -e "${YELLOW}○${NC} Topic already exists: $topic"
done

# Show service status
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}   Service Status${NC}"
echo -e "${GREEN}========================================${NC}"

docker-compose ps

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}   Services Started Successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Access points:"
echo "  • Airflow UI: http://localhost:8080 (admin/admin)"
echo "  • Kafka UI: http://localhost:9000"
echo "  • Grafana: http://localhost:3000 (admin/admin)"
echo "  • PostgreSQL: localhost:5432 (airflow/airflow)"
echo ""
echo "To check logs: docker-compose logs -f [service-name]"
echo "To stop services: ./scripts/stop_services.sh"
echo ""
echo -e "${GREEN}Happy ETL-ing! 🚀${NC}"
