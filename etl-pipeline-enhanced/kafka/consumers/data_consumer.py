"""
Kafka Data Consumer
Consumes messages from Kafka topics for processing
"""

import json
import logging
from typing import Dict, Any, List, Optional
from kafka import KafkaConsumer
from kafka.errors import KafkaError
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataConsumer:
    """Kafka consumer for ETL pipeline data"""
    
    def __init__(self, bootstrap_servers: str = None, group_id: str = None):
        """Initialize Kafka consumer"""
        self.bootstrap_servers = bootstrap_servers or os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
        self.group_id = group_id or os.getenv('KAFKA_GROUP_ID', 'etl-consumer-group')
        self.consumer = None
        
    def create_consumer(self, topics: List[str], **kwargs) -> KafkaConsumer:
        """Create and configure Kafka consumer"""
        config = {
            'bootstrap_servers': self.bootstrap_servers,
            'group_id': self.group_id,
            'auto_offset_reset': 'earliest',
            'enable_auto_commit': True,
            'auto_commit_interval_ms': 5000,
            'value_deserializer': lambda m: json.loads(m.decode('utf-8')),
            'max_poll_records': 500,
            'session_timeout_ms': 30000,
            'heartbeat_interval_ms': 3000
        }
        config.update(kwargs)
        
        self.consumer = KafkaConsumer(*topics, **config)
        logger.info(f"Consumer created for topics: {topics}")
        
        return self.consumer
    
    def consume_messages(self, topics: List[str], max_messages: int = None,
                        process_func: callable = None) -> List[Dict[str, Any]]:
        """Consume messages from Kafka topics"""
        if not self.consumer:
            self.create_consumer(topics)
        
        messages = []
        processed_count = 0
        
        try:
            for message in self.consumer:
                msg_data = {
                    'topic': message.topic,
                    'partition': message.partition,
                    'offset': message.offset,
                    'timestamp': message.timestamp,
                    'key': message.key.decode('utf-8') if message.key else None,
                    'value': message.value
                }
                
                # Process message if function provided
                if process_func:
                    try:
                        processed_data = process_func(msg_data)
                        msg_data['processed'] = processed_data
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                        msg_data['error'] = str(e)
                
                messages.append(msg_data)
                processed_count += 1
                
                if processed_count % 100 == 0:
                    logger.info(f"Processed {processed_count} messages")
                
                if max_messages and processed_count >= max_messages:
                    break
                    
        except KafkaError as e:
            logger.error(f"Kafka error: {e}")
            raise
        except KeyboardInterrupt:
            logger.info("Consumer interrupted by user")
        finally:
            self.commit()
        
        logger.info(f"Total messages consumed: {processed_count}")
        return messages
    
    def consume_batch(self, topics: List[str], batch_size: int = 100,
                     batch_processor: callable = None) -> int:
        """Consume messages in batches"""
        if not self.consumer:
            self.create_consumer(topics)
        
        total_processed = 0
        batch = []
        
        try:
            for message in self.consumer:
                batch.append({
                    'topic': message.topic,
                    'partition': message.partition,
                    'offset': message.offset,
                    'value': message.value
                })
                
                if len(batch) >= batch_size:
                    if batch_processor:
                        batch_processor(batch)
                    
                    total_processed += len(batch)
                    logger.info(f"Processed batch of {len(batch)} messages")
                    batch = []
                    
                    # Commit after each batch
                    self.commit()
                    
        except Exception as e:
            logger.error(f"Error in batch consumption: {e}")
            raise
        finally:
            # Process remaining messages
            if batch:
                if batch_processor:
                    batch_processor(batch)
                total_processed += len(batch)
            
            self.close()
        
        return total_processed
    
    def seek_to_beginning(self, partitions=None):
        """Seek to beginning of partitions"""
        if self.consumer:
            self.consumer.seek_to_beginning(partitions)
            logger.info("Seeked to beginning of partitions")
    
    def seek_to_end(self, partitions=None):
        """Seek to end of partitions"""
        if self.consumer:
            self.consumer.seek_to_end(partitions)
            logger.info("Seeked to end of partitions")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get consumer metrics"""
        if not self.consumer:
            return {}
        
        metrics = self.consumer.metrics()
        return {
            'connections': len(self.consumer._client._conns),
            'assigned_partitions': len(self.consumer.assignment()),
            'position': {tp: self.consumer.position(tp) for tp in self.consumer.assignment()},
            'committed': {tp: self.consumer.committed(tp) for tp in self.consumer.assignment()}
        }
    
    def commit(self):
        """Manually commit offsets"""
        if self.consumer:
            self.consumer.commit()
            logger.debug("Offsets committed")
    
    def close(self):
        """Close consumer connection"""
        if self.consumer:
            self.consumer.close()
            logger.info("Consumer closed")

def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Kafka Data Consumer')
    parser.add_argument('--topics', nargs='+', required=True, help='Topics to consume')
    parser.add_argument('--servers', default='localhost:9092', help='Bootstrap servers')
    parser.add_argument('--group', default='etl-consumer-group', help='Consumer group')
    parser.add_argument('--max-messages', type=int, help='Maximum messages to consume')
    
    args = parser.parse_args()
    
    consumer = DataConsumer(
        bootstrap_servers=args.servers,
        group_id=args.group
    )
    
    def process_message(msg):
        """Simple message processor"""
        print(f"Processing message from {msg['topic']}: {msg['value']}")
        return True
    
    messages = consumer.consume_messages(
        topics=args.topics,
        max_messages=args.max_messages,
        process_func=process_message
    )
    
    print(f"\nConsumed {len(messages)} messages")
    consumer.close()

if __name__ == "__main__":
    main()
