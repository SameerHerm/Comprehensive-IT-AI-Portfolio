#!/bin/bash

#############################################
# Backup Script
# Backup processed data and database
#############################################

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKUP_DIR="$PROJECT_ROOT/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p "$BACKUP_DIR"

echo "Starting backup..."

# Backup database
echo "Backing up database..."
docker-compose exec -T postgres pg_dump -U airflow -d airflow | \
    gzip > "$BACKUP_DIR/database_$DATE.sql.gz"

# Backup processed data
echo "Backing up processed data..."
tar -czf "$BACKUP_DIR/data_$DATE.tar.gz" -C "$PROJECT_ROOT" data/processed/

# Backup configurations
echo "Backing up configurations..."
tar -czf "$BACKUP_DIR/config_$DATE.tar.gz" -C "$PROJECT_ROOT" config/

# Clean old backups (keep last 7 days)
find "$BACKUP_DIR" -name "*.gz" -mtime +7 -delete

echo "Backup completed: $BACKUP_DIR/*_$DATE.*"

# Optional: Upload to S3
if [ ! -z "$S3_BUCKET" ]; then
    echo "Uploading to S3..."
    aws s3 cp "$BACKUP_DIR/database_$DATE.sql.gz" "s3://$S3_BUCKET/backups/"
    aws s3 cp "$BACKUP_DIR/data_$DATE.tar.gz" "s3://$S3_BUCKET/backups/"
fi
