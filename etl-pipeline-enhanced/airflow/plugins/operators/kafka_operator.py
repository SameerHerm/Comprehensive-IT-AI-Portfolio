"""
Custom Kafka Operators for Airflow
"""

from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
from kafka import KafkaProducer, KafkaConsumer
import json
import logging

logger = logging.getLogger(__name__)

class KafkaProducerOperator(BaseOperator):
    """
    Produce messages to Kafka topic
    """
    
    template_fields = ['topic', 'messages']
    
    @apply_defaults
    def __init__(
        self,
        topic,
        messages,
        bootstrap_servers='localhost:9092',
        key=None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.topic = topic
        self.messages = messages
        self.bootstrap_servers = bootstrap_servers
        self.key = key
    
    def execute(self, context):
        """Execute the operator"""
        logger.info(f"Producing messages to topic: {self.topic}")
        
        producer = KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None
        )
        
        try:
            if isinstance(self.messages, list):
                for message in self.messages:
                    future = producer.send(self.topic, value=message, key=self.key)
                    record_metadata = future.get(timeout=10)
                    logger.info(f"Message sent to {record_metadata.topic}:{record_metadata.partition}")
            else:
                future = producer.send(self.topic, value=self.messages, key=self.key)
                record_metadata = future.get(timeout=10)
                logger.info(f"Message sent to {record_metadata.topic}:{record_metadata.partition}")
            
            producer.flush()
            return True
            
        except Exception as e:
            logger.error(f"Failed to produce messages: {e}")
            raise
        finally:
            producer.close()

class KafkaConsumerOperator(BaseOperator):
    """
    Consume messages from Kafka topic
    """
    
    template_fields = ['topics']
    
    @apply_defaults
    def __init__(
        self,
        topics,
        bootstrap_servers='localhost:9092',
        group_id='airflow-consumer',
        max_messages=100,
        timeout_ms=10000,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.topics = topics if isinstance(topics, list) else [topics]
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.max_messages = max_messages
        self.timeout_ms = timeout_ms
    
    def execute(self, context):
        """Execute the operator"""
        logger.info(f"Consuming messages from topics: {self.topics}")
        
        consumer = KafkaConsumer(
            *self.topics,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            consumer_timeout_ms=self.timeout_ms
        )
        
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
                
                if len(messages) >= self.max_messages:
                    break
            
            logger.info(f"Consumed {len(messages)} messages")
            return messages
            
        except Exception as e:
            logger.error(f"Failed to consume messages: {e}")
            raise
        finally:
            consumer.close()
