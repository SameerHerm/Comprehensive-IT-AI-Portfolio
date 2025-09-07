"""
Main ETL Pipeline DAG
Orchestrates the complete ETL process with data quality checks
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.postgres_operator import PostgresOperator
from airflow.operators.email_operator import EmailOperator
from airflow.providers.http.sensors.http import HttpSensor
from airflow.utils.task_group import TaskGroup
import sys
import os
sys.path.append('/opt/airflow')

from src.extractors.file_extractor import FileExtractor
from src.transformers.data_cleaner import DataCleaner
from src.loaders.database_loader import DatabaseLoader
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Default arguments
default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'email': ['admin@example.com'],
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'catchup': False,
}

# Create DAG
dag = DAG(
    'etl_main_pipeline',
    default_args=default_args,
    description='Main ETL Pipeline with Quality Checks',
    schedule_interval='0 2 * * *',  # Daily at 2 AM
    max_active_runs=1,
    tags=['etl', 'production', 'daily'],
)

def extract_data(**context):
    """Extract data from various sources"""
    logger.info("Starting data extraction...")
    
    extractor = FileExtractor()
    
    # Extract from multiple sources
    file_data = extractor.extract_csv('/opt/airflow/data/raw/sales_data.csv')
    
    # Store in XCom for next task
    context['task_instance'].xcom_push(key='raw_data', value=file_data.to_dict())
    
    logger.info(f"Extracted {len(file_data)} records")
    return True

def validate_data(**context):
    """Validate extracted data"""
    logger.info("Validating data...")
    
    # Retrieve data from XCom
    raw_data = context['task_instance'].xcom_pull(key='raw_data')
    
    # Validation logic
    validation_rules = {
        'required_columns': ['date', 'product_id', 'quantity', 'price'],
        'min_records': 100,
        'date_format': '%Y-%m-%d'
    }
    
    # Perform validation
    is_valid = True
    errors = []
    
    if len(raw_data) < validation_rules['min_records']:
        errors.append(f"Insufficient records: {len(raw_data)}")
        is_valid = False
    
    if errors:
        logger.error(f"Validation failed: {errors}")
        raise ValueError(f"Data validation failed: {errors}")
    
    logger.info("Data validation successful")
    return is_valid

def transform_data(**context):
    """Transform and clean data"""
    logger.info("Starting data transformation...")
    
    # Retrieve data from XCom
    raw_data = context['task_instance'].xcom_pull(key='raw_data')
    
    cleaner = DataCleaner()
    
    # Apply transformations
    cleaned_data = cleaner.clean_data(raw_data)
    enriched_data = cleaner.enrich_data(cleaned_data)
    
    # Store transformed data
    context['task_instance'].xcom_push(key='transformed_data', value=enriched_data)
    
    logger.info(f"Transformed {len(enriched_data)} records")
    return True

def load_data(**context):
    """Load data to destination"""
    logger.info("Starting data loading...")
    
    # Retrieve transformed data
    transformed_data = context['task_instance'].xcom_pull(key='transformed_data')
    
    loader = DatabaseLoader()
    
    # Load to database
    records_loaded = loader.load_to_postgres(
        data=transformed_data,
        table='fact_sales',
        schema='warehouse'
    )
    
    logger.info(f"Successfully loaded {records_loaded} records")
    return records_loaded

def check_data_quality(**context):
    """Perform data quality checks"""
    logger.info("Performing data quality checks...")
    
    quality_checks = [
        {
            'check': 'record_count',
            'sql': 'SELECT COUNT(*) FROM warehouse.fact_sales WHERE load_date = CURRENT_DATE',
            'expected': {'min': 100}
        },
        {
            'check': 'null_check',
            'sql': 'SELECT COUNT(*) FROM warehouse.fact_sales WHERE product_id IS NULL AND load_date = CURRENT_DATE',
            'expected': {'max': 0}
        },
        {
            'check': 'duplicate_check',
            'sql': '''
                SELECT COUNT(*) 
                FROM (
                    SELECT product_id, date, COUNT(*) as cnt
                    FROM warehouse.fact_sales 
                    WHERE load_date = CURRENT_DATE
                    GROUP BY product_id, date
                    HAVING COUNT(*) > 1
                ) t
            ''',
            'expected': {'max': 0}
        }
    ]
    
    failed_checks = []
    
    for check in quality_checks:
        # Execute check (pseudo-code, implement actual DB connection)
        logger.info(f"Running check: {check['check']}")
        # result = execute_sql(check['sql'])
        # if not validate_result(result, check['expected']):
        #     failed_checks.append(check['check'])
    
    if failed_checks:
        raise ValueError(f"Quality checks failed: {failed_checks}")
    
    logger.info("All quality checks passed")
    return True

def send_notification(**context):
    """Send success notification"""
    logger.info("Sending notification...")
    
    # Get metrics from previous tasks
    metrics = {
        'pipeline_name': 'etl_main_pipeline',
        'execution_date': context['execution_date'],
        'status': 'SUCCESS',
        'records_processed': context['task_instance'].xcom_pull(key='records_loaded')
    }
    
    # Send notification (implement actual notification logic)
    logger.info(f"Pipeline completed successfully: {metrics}")
    return True

# Define tasks
start_task = DummyOperator(
    task_id='start',
    dag=dag
)

check_source = HttpSensor(
    task_id='check_data_source',
    http_conn_id='data_api',
    endpoint='health',
    poke_interval=30,
    timeout=300,
    dag=dag
)

extract_task = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
    provide_context=True,
    dag=dag
)

validate_task = PythonOperator(
    task_id='validate_data',
    python_callable=validate_data,
    provide_context=True,
    dag=dag
)

# Transform task group
with TaskGroup('transform_tasks', dag=dag) as transform_group:
    
    clean_task = PythonOperator(
        task_id='clean_data',
        python_callable=transform_data,
        provide_context=True,
        dag=dag
    )
    
    aggregate_task = BashOperator(
        task_id='aggregate_data',
        bash_command='/opt/airflow/scripts/etl_processor.sh --aggregate',
        dag=dag
    )
    
    clean_task >> aggregate_task

load_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    provide_context=True,
    dag=dag
)

quality_check_task = PythonOperator(
    task_id='check_data_quality',
    python_callable=check_data_quality,
    provide_context=True,
    trigger_rule='all_success',
    dag=dag
)

archive_task = BashOperator(
    task_id='archive_raw_data',
    bash_command='/opt/airflow/scripts/backup_data.sh --archive',
    dag=dag
)

notify_task = PythonOperator(
    task_id='send_notification',
    python_callable=send_notification,
    provide_context=True,
    trigger_rule='all_success',
    dag=dag
)

end_task = DummyOperator(
    task_id='end',
    trigger_rule='none_failed_or_skipped',
    dag=dag
)

# Define task dependencies
start_task >> check_source >> extract_task >> validate_task >> transform_group
transform_group >> load_task >> [quality_check_task, archive_task]
[quality_check_task, archive_task] >> notify_task >> end_task
