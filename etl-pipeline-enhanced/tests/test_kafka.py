"""
Tests for Kafka components
"""

import pytest
import json
from unittest.mock import Mock, patch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kafka.producers.data_producer import DataProducer
from kafka.consumers.stream_processor import StreamProcessor

class TestDataProducer:
    
    @patch('kafka.producers.data_producer.KafkaProducer')
    def test_producer_initialization(self, mock_kafka_producer):
        """Test producer initialization"""
        producer = DataProducer(bootstrap_servers='localhost:9092')
        
        assert producer.bootstrap_servers == 'localhost:9092'
        mock_kafka_producer.assert_called_once()
    
    def test_generate_sample_data(self):
        """Test sample data generation"""
        with patch('kafka.producers.data_producer.KafkaProducer'):
            producer = DataProducer()
            data = producer.generate_sample_data()
            
            assert 'timestamp' in data
            assert 'transaction_id' in data
            assert 'product_id' in data
            assert 'quantity' in data
            assert 'price' in data
            assert data['total_amount'] == data['quantity'] * data['price']
    
    @patch('kafka.producers.data_producer.KafkaProducer')
    def test_send_message(self, mock_kafka_producer):
        """Test sending single message"""
        mock_instance = Mock()
        mock_kafka_producer.return_value = mock_instance
        
        mock_future = Mock()
        mock_future.get.return_value = Mock(topic='test', partition=0, offset=1)
        mock_instance.send.return_value = mock_future
        
        producer = DataProducer()
        result = producer.send_message('test-topic', {'test': 'data'})
        
        assert result is True
        mock_instance.send.assert_called_once()

class TestStreamProcessor:
    
    def test_validate_message(self):
        """Test message validation"""
        processor = StreamProcessor()
        
        # Valid message
        valid_msg = {
            'transaction_id': 'TXN_123',
            'product_id': 'PROD_456',
            'quantity': 5,
            'price': 100.0
        }
        assert processor.validate_message(valid_msg) is True
        
        # Invalid message (missing field)
        invalid_msg = {
            'transaction_id': 'TXN_123',
            'quantity': 5
        }
        assert processor.validate_message(invalid_msg) is False
        
        # Invalid message (negative quantity)
        invalid_msg2 = {
            'transaction_id': 'TXN_123',
            'product_id': 'PROD_456',
            'quantity': -1,
            'price': 100.0
        }
        assert processor.validate_message(invalid_msg2) is False
    
    def test_enrich_message(self):
        """Test message enrichment"""
        processor = StreamProcessor()
        
        message = {
            'quantity': 5,
            'price': 100.0,
            'region': 'north'
        }
        
        enriched = processor.enrich_message(message)
        
        assert 'total_amount' in enriched
        assert enriched['total_amount'] == 500.0
        assert 'processing_date' in enriched
        assert 'region_code' in enriched
        assert enriched['region_code'] == 'NO'
