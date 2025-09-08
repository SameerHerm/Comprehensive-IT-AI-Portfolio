#!/bin/bash

#############################################
# Data Generator Script
# Generates sample data for ETL pipeline
#############################################

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/data"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   Generating Sample Data${NC}"
echo -e "${GREEN}========================================${NC}"

# Create data directories
mkdir -p "$DATA_DIR"/{raw,processed,staging,archive}

# Function to generate CSV data
generate_csv_data() {
    local filename=$1
    local rows=$2
    
    echo "Generating $filename with $rows rows..."
    
    python3 << EOF
import csv
import random
from datetime import datetime, timedelta
from faker import Faker

fake = Faker()

# Generate sample data
data = []
for i in range($rows):
    row = {
        'transaction_id': f'TXN_{i+1:06d}',
        'date': (datetime.now() - timedelta(days=random.randint(0, 30))).strftime('%Y-%m-%d'),
        'customer_id': f'CUST_{random.randint(1000, 9999)}',
        'customer_name': fake.name(),
        'product_id': f'PROD_{random.randint(100, 999)}',
        'product_name': random.choice(['Laptop', 'Phone', 'Tablet', 'Headphones', 'Watch']),
        'quantity': random.randint(1, 10),
        'price': round(random.uniform(10, 1000), 2),
        'region': random.choice(['North', 'South', 'East', 'West', 'Central']),
        'payment_method': random.choice(['credit_card', 'debit_card', 'paypal', 'cash']),
        'status': random.choice(['completed', 'pending', 'cancelled'])
    }
    row['total_amount'] = round(row['quantity'] * row['price'], 2)
    data.append(row)

# Write to CSV
with open('$filename', 'w', newline='') as f:
    if data:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

print(f"Generated {len(data)} rows in $filename")
EOF
}

# Function to generate JSON data
generate_json_data() {
    local filename=$1
    local records=$2
    
    echo "Generating $filename with $records records..."
    
    python3 << EOF
import json
import random
from datetime import datetime, timedelta
from faker import Faker

fake = Faker()

# Generate sample data
data = []
for i in range($records):
    record = {
        'id': f'REC_{i+1:06d}',
        'timestamp': (datetime.now() - timedelta(seconds=random.randint(0, 86400))).isoformat(),
        'event_type': random.choice(['click', 'view', 'purchase', 'signup', 'logout']),
        'user_id': f'USER_{random.randint(1000, 9999)}',
        'user_email': fake.email(),
        'ip_address': fake.ipv4(),
        'user_agent': fake.user_agent(),
        'session_id': fake.uuid4(),
        'page_url': fake.url(),
        'referrer': fake.url() if random.random() > 0.3 else None,
        'device_type': random.choice(['desktop', 'mobile', 'tablet']),
        'os': random.choice(['Windows', 'MacOS', 'Linux', 'iOS', 'Android']),
        'browser': random.choice(['Chrome', 'Firefox', 'Safari', 'Edge']),
        'country': fake.country(),
        'city': fake.city(),
        'latitude': float(fake.latitude()),
        'longitude': float(fake.longitude())
    }
    data.append(record)

# Write to JSON
with open('$filename', 'w') as f:
    json.dump(data, f, indent=2, default=str)

print(f"Generated {len(data)} records in $filename")
EOF
}

# Function to generate Parquet data
generate_parquet_data() {
    local filename=$1
    local rows=$2
    
    echo "Generating $filename with $rows rows..."
    
    python3 << EOF
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Generate time series data
dates = pd.date_range(start='2024-01-01', periods=$rows, freq='H')

data = pd.DataFrame({
    'timestamp': dates,
    'sensor_id': [f'SENSOR_{random.randint(1, 100):03d}' for _ in range($rows)],
    'temperature': np.random.normal(25, 5, $rows),
    'humidity': np.random.normal(60, 15, $rows),
    'pressure': np.random.normal(1013, 10, $rows),
    'co2_level': np.random.normal(400, 50, $rows),
    'light_intensity': np.random.uniform(0, 1000, $rows),
    'sound_level': np.random.uniform(30, 80, $rows),
    'motion_detected': np.random.choice([0, 1], $rows, p=[0.8, 0.2]),
    'battery_level': np.random.uniform(20, 100, $rows),
    'signal_strength': np.random.uniform(-90, -30, $rows)
})

# Add some anomalies
anomaly_indices = np.random.choice(range($rows), size=int($rows * 0.05), replace=False)
data.loc[anomaly_indices, 'temperature'] = np.random.uniform(-10, 50, len(anomaly_indices))

# Save to Parquet
data.to_parquet('$filename', engine='pyarrow', compression='snappy')

print(f"Generated {len(data)} rows in $filename")
EOF
}

# Generate different types of data files
echo -e "\n${YELLOW}Generating CSV files...${NC}"
generate_csv_data "$DATA_DIR/raw/sales_data.csv" 10000
generate_csv_data "$DATA_DIR/raw/customer_data.csv" 5000
generate_csv_data "$DATA_DIR/raw/product_data.csv" 1000

echo -e "\n${YELLOW}Generating JSON files...${NC}"
generate_json_data "$DATA_DIR/raw/events_data.json" 5000
generate_json_data "$DATA_DIR/raw/user_activity.json" 3000

echo -e "\n${YELLOW}Generating Parquet files...${NC}"
generate_parquet_data "$DATA_DIR/raw/sensor_data.parquet" 10000
generate_parquet_data "$DATA_DIR/raw/metrics_data.parquet" 5000

# Generate sample configuration
echo -e "\n${YELLOW}Generating sample configuration...${NC}"
cat > "$DATA_DIR/raw/data_manifest.json" << EOF
{
  "generated_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "files": [
    {
      "name": "sales_data.csv",
      "type": "csv",
      "rows": 10000,
      "size_bytes": $(stat -f%z "$DATA_DIR/raw/sales_data.csv" 2>/dev/null || stat -c%s "$DATA_DIR/raw/sales_data.csv" 2>/dev/null || echo "0")
    },
    {
      "name": "customer_data.csv",
      "type": "csv",
      "rows": 5000,
      "size_bytes": $(stat -f%z "$DATA_DIR/raw/customer_data.csv" 2>/dev/null || stat -c%s "$DATA_DIR/raw/customer_data.csv" 2>/dev/null || echo "0")
    },
    {
      "name": "product_data.csv",
      "type": "csv",
      "rows": 1000,
      "size_bytes": $(stat -f%z "$DATA_DIR/raw/product_data.csv" 2>/dev/null || stat -c%s "$DATA_DIR/raw/product_data.csv" 2>/dev/null || echo "0")
    },
    {
      "name": "events_data.json",
      "type": "json",
      "records": 5000
    },
    {
      "name": "user_activity.json",
      "type": "json",
      "records": 3000
    },
    {
      "name": "sensor_data.parquet",
      "type": "parquet",
      "rows": 10000
    },
    {
      "name": "metrics_data.parquet",
      "type": "parquet",
      "rows": 5000
    }
  ],
  "total_files": 7,
  "formats": ["csv", "json", "parquet"]
}
EOF

echo -e "${GREEN}✓${NC} Generated data manifest"

# Display summary
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}   Data Generation Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Generated files in $DATA_DIR/raw/:"
ls -lh "$DATA_DIR/raw/" | grep -v "^total"
echo ""
echo "You can now:"
echo "  1. Run ETL pipeline to process this data"
echo "  2. Start Kafka producers to stream data"
echo "  3. Check Airflow DAGs for batch processing"
echo ""
