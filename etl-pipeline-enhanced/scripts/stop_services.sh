#!/bin/bash

#############################################
# Stop Services Script
# Stops all ETL pipeline services
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

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}   Stopping ETL Pipeline Services${NC}"
echo -e "${YELLOW}========================================${NC}"

# Change to project root
cd "$PROJECT_ROOT"

# Function to stop service gracefully
stop_service() {
    local service=$1
    echo -n "Stopping $service..."
    if docker-compose stop $service 2>/dev/null; then
        echo -e " ${GREEN}✓${NC}"
    else
        echo -e " ${YELLOW}already stopped${NC}"
    fi
}

# Ask for confirmation
read -p "Are you sure you want to stop all services? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Operation cancelled"
    exit 0
fi

# Stop services in reverse order
echo -e "\n${YELLOW}Stopping services...${NC}"

# Stop monitoring
echo -e "\n${YELLOW}[1/4]${NC} Stopping monitoring services..."
stop_service grafana

# Stop Airflow
echo -e "\n${YELLOW}[2/4]${NC} Stopping Airflow services..."
stop_service airflow-worker
stop_service airflow-scheduler
stop_service airflow-webserver

# Stop Kafka
echo -e "\n${YELLOW}[3/4]${NC} Stopping Kafka services..."
stop_service kafka-ui
stop_service kafka
stop_service zookeeper

# Stop core services
echo -e "\n${YELLOW}[4/4]${NC} Stopping core services..."
stop_service redis
stop_service postgres

# Ask if user wants to remove volumes
echo ""
read -p "Do you want to remove data volumes? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}Removing volumes...${NC}"
    docker-compose down -v
    rm -f .airflow_initialized
    echo -e "${GREEN}✓${NC} Volumes removed"
else
    echo "Volumes preserved"
fi

# Show final status
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}   All Services Stopped${NC}"
echo -e "${GREEN}========================================${NC}"

docker-compose ps

echo ""
echo "To restart services: ./scripts/start_services.sh"
echo ""
