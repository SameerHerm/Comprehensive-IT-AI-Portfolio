"""
Custom Kafka Hook for Airflow
"""

from airflow.hooks.base_hook import BaseHook
from kafka import KafkaProducer, KafkaConsumer, KafkaAdminClient
from kafka.admin import NewTopic
import json
import logging

logger = logging.getLogger(__name__)

class KafkaHook(BaseHook):
    """
    Hook for interacting with Apache Kafka
    """
    
    def __init__(self, kafka_conn_id='kafka_default'):
        self.kafka_conn_id = kafka_conn_id
        self.connection = self.get_connection(kafka_conn_id)
        self.bootstrap_servers = self.connection.host or 'localhost:9092'
    
    def get_producer(self, **kwargs):
        """Get Kafka producer instance"""
        config = {
            'bootstrap_servers': self.bootstrap_servers,
            'value_serializer': lambda v: json.dumps(v).encode('utf-8'),
            'key_serializer': lambda k: k.encode('utf-8') if k else None
        }
        config.update(kwargs)
        
        return KafkaProducer(**config)
    
    def get_consumer(self, topics, **kwargs):
        """Get Kafka consumer instance"""
        config = {
            'bootstrap_servers': self.bootstrap_servers,
            'auto_offset_reset': 'earliest',
            'enable_auto_commit': True,
            'value_deserializer': lambda m: json.loads(m.decode('utf-8'))
        }
        config.update(kwargs)
        
        if not isinstance(topics, list):
            topics = [topics]
        
        return KafkaConsumer(*topics, **config)
    
    def get_admin_client(self):
        """Get Kafka admin client"""
        return KafkaAdminClient(
            bootstrap_servers=self.bootstrap_servers,
            client_id='airflow-admin'
        )
    
    def create_topic(self, topic_name, num_partitions=1, replication_factor=1):
        """Create a Kafka topic"""
        admin_client = self.get_admin_client()
        
        topic = NewTopic(
            name=topic_name,
            num_partitions=num_partitions,
            replication_factor=replication_factor
        )
        
        try:
            admin_client.create_topics([topic])
            logger.info(f"Topic {topic_name} created successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to create topic {topic_name}: {e}")
            return False
    
    def list_topics(self):
        """List all Kafka topics"""
        consumer = KafkaConsumer(bootstrap_servers=self.bootstrap_servers)
        topics = consumer.topics()
        consumer.close()
        return list(topics)
    
    def send_message(self, topic, message, key=None):
        """Send a single message to Kafka"""
        producer = self.get_producer()
        
        try:
            future = producer.send(topic, value=message, key=key)
            record_metadata = future.get(timeout=10)
            producer.flush()
            
            logger.info(f"Message sent to {record_metadata.topic}:{record_metadata.partition}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False
        finally:
            producer.close()
    
    def consume_messages(self, topics, max_messages=100, timeout_ms=10000):
        """Consume messages from Kafka topics"""
        consumer = self.get_consumer(topics, consumer_timeout_ms=timeout_ms)
        
        messages = []
        try:
            for message in consumer:
                messages.append({
                    'topic': message.topic,
                    'partition': message.partition,
                    'offset': message.offset,
                    'key': message.key,
                    'value': message.value
                })
                
                if len(messages) >= max_messages:
                    break
            
            return messages
            
        finally:
            consumer.close()
