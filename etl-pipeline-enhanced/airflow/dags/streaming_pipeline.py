"""
Streaming Pipeline DAG
Manages real-time data streaming with Kafka
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
from airflow.providers.apache.kafka.operators.produce import ProduceToTopicOperator
from airflow.providers.apache.kafka.operators.consume import ConsumeFromTopicOperator
import json
import sys
sys.path.append('/opt/airflow')

from kafka.producers.data_producer import DataProducer
from kafka.consumers.stream_processor import StreamProcessor

default_args = {
    'owner': 'streaming-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email': ['streaming@example.com'],
    'retries': 2,
    'retry_delay': timedelta(minutes=2),
}

dag = DAG(
    'streaming_pipeline',
    default_args=default_args,
    description='Real-time streaming pipeline with Kafka',
    schedule_interval=timedelta(minutes=5),
    catchup=False,
    tags=['streaming', 'kafka', 'real-time'],
)

def start_kafka_producer(**context):
    """Start Kafka producer for streaming data"""
    producer = DataProducer()
    
    # Configuration
    config = {
        'topic': 'etl-streaming-data',
        'batch_size': 100,
        'interval': 1  # seconds
    }
    
    # Start producing
    producer.start_streaming(config)
    
    return True

def process_stream_data(**context):
    """Process streaming data"""
    processor = StreamProcessor()
    
    # Process configuration
    config = {
        'input_topic': 'etl-streaming-data',
        'output_topic': 'etl-processed-data',
        'processing_window': 60  # seconds
    }
    
    # Start processing
    results = processor.process_stream(config)
    
    context['task_instance'].xcom_push(key='processed_count', value=results['count'])
    
    return results

def monitor_stream_health(**context):
    """Monitor streaming pipeline health"""
    metrics = {
        'lag': 0,
        'throughput': 0,
        'error_rate': 0
    }
    
    # Check Kafka consumer lag
    # Check processing throughput
    # Check error rates
    
    if metrics['lag'] > 1000:
        raise ValueError(f"High consumer lag detected: {metrics['lag']}")
    
    return metrics

# Define tasks
start_producer = PythonOperator(
    task_id='start_kafka_producer',
    python_callable=start_kafka_producer,
    dag=dag
)

process_stream = PythonOperator(
    task_id='process_stream_data',
    python_callable=process_stream_data,
    dag=dag
)

monitor_health = PythonOperator(
    task_id='monitor_stream_health',
    python_callable=monitor_stream_health,
    dag=dag
)

check_topics = BashOperator(
    task_id='check_kafka_topics',
    bash_command="""
    kafka-topics --list --bootstrap-server kafka:9093 | \
    grep -E "etl-streaming-data|etl-processed-data"
    """,
    dag=dag
)

# Task dependencies
check_topics >> start_producer >> process_stream >> monitor_health
