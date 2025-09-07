"""
Kafka Data Producer
Generates and sends data to Kafka topics
"""

import json
import time
import random
import logging
from datetime import datetime
from typing import Dict, Any, List
from kafka import KafkaProducer
from kafka.errors import KafkaError
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProducer:
    """Kafka data producer for ETL pipeline"""
    
    def __init__(self, bootstrap_servers: str = None):
        """Initialize Kafka producer"""
        self.bootstrap_servers = bootstrap_servers or os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
        
        self.producer = KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None,
            acks='all',
            retries=3,
            max_in_flight_requests_per_connection=5,
            compression_type='gzip'
        )
        
        logger.info(f"Producer initialized with servers: {self.bootstrap_servers}")
    
    def generate_sample_data(self) -> Dict[str, Any]:
        """Generate sample data record"""
        products = ['laptop', 'phone', 'tablet', 'headphones', 'smartwatch']
        regions = ['north', 'south', 'east', 'west', 'central']
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'transaction_id': f"TXN_{random.randint(100000, 999999)}",
            'product_id': f"PROD_{random.randint(1000, 9999)}",
            'product_name': random.choice(products),
            'quantity': random.randint(1, 10),
            'price': round(random.uniform(10, 1000), 2),
            'customer_id': f"CUST_{random.randint(10000, 99999)}",
            'region': random.choice(regions),
            'payment_method': random.choice(['credit_card', 'debit_card', 'paypal', 'cash']),
            'status': random.choice(['completed', 'pending', 'cancelled'])
        }
        
        data['total_amount'] = round(data['quantity'] * data['price'], 2)
        
        return data
    
    def send_message(self, topic: str, message: Dict[str, Any], key: str = None) -> bool:
        """Send single message to Kafka topic"""
        try:
            future = self.producer.send(
                topic=topic,
                value=message,
                key=key
            )
            
            # Wait for confirmation
            record_metadata = future.get(timeout=10)
            
            logger.debug(f"Message sent to {record_metadata.topic}:{record_metadata.partition} at offset {record_metadata.offset}")
            return True
            
        except KafkaError as e:
            logger.error(f"Failed to send message: {e}")
            return False
    
    def send_batch(self, topic: str, messages: List[Dict[str, Any]]) -> int:
        """Send batch of messages to Kafka"""
        success_count = 0
        
        for message in messages:
            key = message.get('transaction_id', None)
            if self.send_message(topic, message, key):
                success_count += 1
        
        self.producer.flush()
        
        logger.info(f"Sent {success_count}/{len(messages)} messages to {topic}")
        return success_count
    
    def start_streaming(self, config: Dict[str, Any]):
        """Start continuous streaming of data"""
        topic = config.get('topic', 'etl-streaming-data')
        batch_size = config.get('batch_size', 10)
        interval = config.get('interval', 1)
        max_messages = config.get('max_messages', None)
        
        logger.info(f"Starting streaming to topic: {topic}")
        
        message_count = 0
        try:
            while max_messages is None or message_count < max_messages:
                batch = []
                
                for _ in range(batch_size):
                    data = self.generate_sample_data()
                    batch.append(data)
                
                sent = self.send_batch(topic, batch)
                message_count += sent
                
                logger.info(f"Total messages sent: {message_count}")
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("Streaming stopped by user")
        except Exception as e:
            logger.error(f"Streaming error: {e}")
        finally:
            self.close()
    
    def produce_from_file(self, topic: str, filepath: str) -> int:
        """Produce messages from a file"""
        import pandas as pd
        
        try:
            df = pd.read_csv(filepath)
            records = df.to_dict('records')
            
            return self.send_batch(topic, records)
            
        except Exception as e:
            logger.error(f"Error reading file {filepath}: {e}")
            return 0
    
    def close(self):
        """Close producer connection"""
        self.producer.close()
        logger.info("Producer closed")

def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Kafka Data Producer')
    parser.add_argument('--topic', default='etl-streaming-data', help='Kafka topic')
    parser.add_argument('--servers', default='localhost:9092', help='Bootstrap servers')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size')
    parser.add_argument('--interval', type=int, default=1, help='Interval between batches (seconds)')
    parser.add_argument('--max-messages', type=int, help='Maximum messages to send')
    
    args = parser.parse_args()
    
    producer = DataProducer(bootstrap_servers=args.servers)
    
    config = {
        'topic': args.topic,
        'batch_size': args.batch_size,
        'interval': args.interval,
        'max_messages': args.max_messages
    }
    
    producer.start_streaming(config)

if __name__ == "__main__":
    main()
