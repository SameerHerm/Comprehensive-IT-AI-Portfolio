#!/bin/bash

#############################################
# ETL Processor Script
# Main script for running ETL operations
#############################################

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/data"
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/etl_processor_$(date +%Y%m%d_%H%M%S).log"

# Source environment variables
if [ -f "$PROJECT_ROOT/.env" ]; then
    source "$PROJECT_ROOT/.env"
fi

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Logging functions
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" | tee -a "$LOG_FILE" >&2
}

log_success() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS: $1" | tee -a "$LOG_FILE"
}

# Function to extract data
extract_data() {
    log "Starting data extraction..."
    
    local source_type=$1
    local source_path=$2
    local output_path="$DATA_DIR/staging/extracted_$(date +%Y%m%d_%H%M%S).csv"
    
    case $source_type in
        file)
            if [ -f "$source_path" ]; then
                cp "$source_path" "$output_path"
                log_success "Extracted data from file: $source_path"
            else
                log_error "Source file not found: $source_path"
                return 1
            fi
            ;;
        
        api)
            # Example API extraction
            curl -s "$source_path" > "$output_path"
            log_success "Extracted data from API: $source_path"
            ;;
        
        database)
            # Example database extraction
            psql -h "$POSTGRES_HOST" -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
                -c "\COPY (SELECT * FROM $source_path) TO '$output_path' CSV HEADER"
            log_success "Extracted data from database table: $source_path"
            ;;
        
        *)
            log_error "Unknown source type: $source_type"
            return 1
            ;;
    esac
    
    echo "$output_path"
}

# Function to transform data
transform_data() {
    log "Starting data transformation..."
    
    local input_file=$1
    local output_file="$DATA_DIR/staging/transformed_$(date +%Y%m%d_%H%M%S).csv"
    
    # Run Python transformation script
    python3 << EOF
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('$input_file')

# Data cleaning
df = df.dropna()
df = df.drop_duplicates()

# Data transformation
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day

# Data validation
assert len(df) > 0, "No data after transformation"

# Save transformed data
df.to_csv('$output_file', index=False)
print(f"Transformed {len(df)} records")
EOF
    
    if [ $? -eq 0 ]; then
        log_success "Data transformation completed"
        echo "$output_file"
    else
        log_error "Data transformation failed"
        return 1
    fi
}

# Function to load data
load_data() {
    log "Starting data loading..."
    
    local input_file=$1
    local target_type=$2
    local target_location=$3
    
    case $target_type in
        postgres)
            # Load to PostgreSQL
            psql -h "$POSTGRES_HOST" -U "$POSTGRES_USER" -d "$POSTGRES_DB" << EOF
\COPY warehouse.fact_sales (
    date, product_id, product_name, quantity, price,
    customer_id, region, payment_method, total_amount
) FROM '$input_file' CSV HEADER;
EOF
            log_success "Data loaded to PostgreSQL"
            ;;
        
        file)
            # Save to file
            cp "$input_file" "$DATA_DIR/processed/$(basename "$input_file")"
            log_success "Data saved to processed folder"
            ;;
        
        s3)
            # Upload to S3
            aws s3 cp "$input_file" "s3://$S3_BUCKET/$target_location"
            log_success "Data uploaded to S3"
            ;;
        
        *)
            log_error "Unknown target type: $target_type"
            return 1
            ;;
    esac
}

# Function to run data quality checks
quality_check() {
    log "Running data quality checks..."
    
    local data_file=$1
    
    python3 << EOF
import pandas as pd
import sys

df = pd.read_csv('$data_file')

# Quality checks
checks_passed = True
errors = []

# Check 1: No null values in critical columns
critical_columns = ['product_id', 'quantity', 'price']
for col in critical_columns:
    if col in df.columns and df[col].isnull().any():
        errors.append(f"Null values found in {col}")
        checks_passed = False

# Check 2: Positive values for quantity and price
if 'quantity' in df.columns and (df['quantity'] <= 0).any():
    errors.append("Non-positive quantities found")
    checks_passed = False

if 'price' in df.columns and (df['price'] <= 0).any():
    errors.append("Non-positive prices found")
    checks_passed = False

# Check 3: Data volume check
if len(df) < 10:
    errors.append(f"Insufficient data: only {len(df)} records")
    checks_passed = False

if not checks_passed:
    print("Quality checks failed:")
    for error in errors:
        print(f"  - {error}")
    sys.exit(1)
else:
    print(f"All quality checks passed for {len(df)} records")
EOF
    
    if [ $? -eq 0 ]; then
        log_success "Data quality checks passed"
        return 0
    else
        log_error "Data quality checks failed"
        return 1
    fi
}

# Function to archive processed data
archive_data() {
    log "Archiving processed data..."
    
    local file_to_archive=$1
    local archive_name="archive_$(date +%Y%m%d_%H%M%S).tar.gz"
    
    tar -czf "$DATA_DIR/archive/$archive_name" -C "$(dirname "$file_to_archive")" "$(basename "$file_to_archive")"
    
    if [ $? -eq 0 ]; then
        log_success "Data archived: $archive_name"
        # Optionally remove the original file
        # rm "$file_to_archive"
    else
        log_error "Failed to archive data"
        return 1
    fi
}

# Function to run aggregate operations
run_aggregation() {
    log "Running data aggregation..."
    
    psql -h "$POSTGRES_HOST" -U "$POSTGRES_USER" -d "$POSTGRES_DB" << EOF
-- Daily aggregations
INSERT INTO warehouse.daily_sales_summary
SELECT 
    date,
    region,
    COUNT(*) as transaction_count,
    SUM(quantity) as total_quantity,
    SUM(total_amount) as total_revenue,
    AVG(total_amount) as avg_transaction_value,
    CURRENT_TIMESTAMP as created_at
FROM warehouse.fact_sales
WHERE date = CURRENT_DATE - INTERVAL '1 day'
GROUP BY date, region
ON CONFLICT (date, region) DO UPDATE
SET 
    transaction_count = EXCLUDED.transaction_count,
    total_quantity = EXCLUDED.total_quantity,
    total_revenue = EXCLUDED.total_revenue,
    avg_transaction_value = EXCLUDED.avg_transaction_value,
    created_at = CURRENT_TIMESTAMP;
EOF
    
    log_success "Aggregation completed"
}

# Main ETL pipeline
run_etl_pipeline() {
    local mode=$1
    local source=$2
    local target=$3
    
    log "Starting ETL pipeline in $mode mode"
    
    # Extract
    extracted_file=$(extract_data "file" "$DATA_DIR/raw/sales_data.csv")
    if [ $? -ne 0 ]; then
        log_error "Extraction failed"
        exit 1
    fi
    
    # Quality check on raw data
    quality_check "$extracted_file"
    if [ $? -ne 0 ]; then
        log_error "Quality check failed on raw data"
        exit 1
    fi
    
    # Transform
    transformed_file=$(transform_data "$extracted_file")
    if [ $? -ne 0 ]; then
        log_error "Transformation failed"
        exit 1
    fi
    
    # Quality check on transformed data
    quality_check "$transformed_file"
    if [ $? -ne 0 ]; then
        log_error "Quality check failed on transformed data"
        exit 1
    fi
    
    # Load
    load_data "$transformed_file" "$target" "warehouse.fact_sales"
    if [ $? -ne 0 ]; then
        log_error "Loading failed"
        exit 1
    fi
    
    # Archive
    archive_data "$transformed_file"
    
    log_success "ETL pipeline completed successfully"
}

# Parse command line arguments
MODE=${1:-batch}
SOURCE=${2:-file}
TARGET=${3:-postgres}

case $MODE in
    batch)
        run_etl_pipeline "$MODE" "$SOURCE" "$TARGET"
        ;;
    
    stream)
        log "Starting streaming mode..."
        python3 "$PROJECT_ROOT/kafka/producers/data_producer.py" &
        python3 "$PROJECT_ROOT/kafka/consumers/stream_processor.py"
        ;;
    
    aggregate)
        run_aggregation
        ;;
    
    *)
        echo "Usage: $0 [batch|stream|aggregate] [source] [target]"
        echo "  Modes:"
        echo "    batch     - Run batch ETL pipeline"
        echo "    stream    - Run streaming pipeline"
        echo "    aggregate - Run aggregation jobs"
        echo "  Sources: file, api, database"
        echo "  Targets: postgres, file, s3"
        exit 1
        ;;
esac
