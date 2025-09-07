"""
Kafka Stream Processor
Consumes and processes streaming data
"""

import json
import logging
from typing import Dict, Any, List
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StreamProcessor:
    """Process streaming data from Kafka"""
    
    def __init__(self, bootstrap_servers: str = None):
        """Initialize stream processor"""
        self.bootstrap_servers = bootstrap_servers or os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
        self.consumer = None
        self.producer = None
        
    def create_consumer(self, topics: List[str], group_id: str) -> KafkaConsumer:
        """Create Kafka consumer"""
        consumer = KafkaConsumer(
            *topics,
            bootstrap_servers=self.bootstrap_servers,
            group_id=group_id,
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            max_poll_records=100,
            session_timeout_ms=30000,
            consumer_timeout_ms=10000
        )
        
        logger.info(f"Consumer created for topics: {topics}")
        return consumer
    
    def create_producer(self) -> KafkaProducer:
        """Create Kafka producer for processed data"""
        producer = KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            acks='all'
        )
        
        return producer
    
    def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process individual message"""
        try:
            # Add processing timestamp
            message['processed_at'] = datetime.now().isoformat()
            
            # Data validation
            if not self.validate_message(message):
                message['status'] = 'invalid'
                return message
            
            # Data enrichment
            message = self.enrich_message(message)
            
            # Data transformation
            message = self.transform_message(message)
            
            message['status'] = 'processed'
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            message['status'] = 'error'
            message['error'] = str(e)
        
        return message
    
    def validate_message(self, message: Dict[str, Any]) -> bool:
        """Validate message data"""
        required_fields = ['transaction_id', 'product_id', 'quantity', 'price']
        
        for field in required_fields:
            if field not in message:
                logger.warning(f"Missing required field: {field}")
                return False
        
        # Validate data types and ranges
        if message['quantity'] <= 0 or message['price'] <= 0:
            logger.warning("Invalid quantity or price")
            return False
        
        return True
    
    def enrich_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich message with additional data"""
        # Add calculated fields
        if 'quantity' in message and 'price' in message:
            message['total_amount'] = message['quantity'] * message['price']
        
        # Add metadata
        message['processing_date'] = datetime.now().strftime('%Y-%m-%d')
        
        # Add derived fields
        if 'region' in message:
            message['region_code'] = message['region'][:2].upper()
        
        return message
    
    def transform_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Transform message data"""
        # Normalize fields
        if 'product_name' in message:
            message['product_name'] = message['product_name'].lower().strip()
        
        # Convert timestamps
        if 'timestamp' in message:
            try:
                dt = datetime.fromisoformat(message['timestamp'])
                message['year'] = dt.year
                message['month'] = dt.month
                message['day'] = dt.day
                message['hour'] = dt.hour
            except:
                pass
        
        return message
    
    def process_stream(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process stream of messages"""
        input_topic = config.get('input_topic', 'etl-streaming-data')
        output_topic = config.get('output_topic', 'etl-processed-data')
        group_id = config.get('group_id', 'etl-processor-group')
        processing_window = config.get('processing_window', 60)
        
        # Create consumer and producer
        self.consumer = self.create_consumer([input_topic], group_id)
        self.producer = self.create_producer()
        
        processed_count = 0
        error_count = 0
        batch = []
        
        logger.info(f"Starting stream processing from {input_topic} to {output_topic}")
        
        try:
            for message in self.consumer:
                # Process message
                data = message.value
                processed_data = self.process_message(data)
                
                # Send to output topic
                if processed_data.get('status') == 'processed':
                    self.producer.send(output_topic, value=processed_data)
                    processed_count += 1
                else:
                    error_count += 1
                
                batch.append(processed_data)
                
                # Process in batches
                if len(batch) >= 100:
                    self.process_batch(batch)
                    batch = []
                
                # Log progress
                if (processed_count + error_count) % 100 == 0:
                    logger.info(f"Processed: {processed_count}, Errors: {error_count}")
        
        except Exception as e:
            logger.error(f"Stream processing error: {e}")
        
        finally:
            # Process remaining batch
            if batch:
                self.process_batch(batch)
            
            self.close()
        
        return {
            'count': processed_count,
            'errors': error_count,
            'success_rate': processed_count / (processed_count + error_count) if (processed_count + error_count) > 0 else 0
        }
    
    def process_batch(self, batch: List[Dict[str, Any]]):
        """Process batch of messages"""
        if not batch:
            return
        
        # Convert to DataFrame for batch operations
        df = pd.DataFrame(batch)
        
        # Perform batch aggregations
        summary = {
            'batch_size': len(df),
            'avg_amount': df['total_amount'].mean() if 'total_amount' in df else 0,
            'total_revenue': df['total_amount'].sum() if 'total_amount' in df else 0,
            'processed_at': datetime.now().isoformat()
        }
        
        # Save batch summary (implement actual storage)
        logger.info(f"Batch processed: {summary}")
    
    def close(self):
        """Close connections"""
        if self.consumer:
            self.consumer.close()
        if self.producer:
            self.producer.close()
        logger.info("Stream processor closed")

def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Kafka Stream Processor')
    parser.add_argument('--input-topic', default='etl-streaming-data', help='Input topic')
    parser.add_argument('--output-topic', default='etl-processed-data', help='Output topic')
    parser.add_argument('--servers', default='localhost:9092', help='Bootstrap servers')
    parser.add_argument('--group-id', default='etl-processor-group', help='Consumer group ID')
    
    args = parser.parse_args()
    
    processor = StreamProcessor(bootstrap_servers=args.servers)
    
    config = {
        'input_topic': args.input_topic,
        'output_topic': args.output_topic,
        'group_id': args.group_id
    }
    
    results = processor.process_stream(config)
    print(f"Processing complete: {results}")

if __name__ == "__main__":
    main()
