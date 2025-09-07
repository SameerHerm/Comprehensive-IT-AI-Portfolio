"""
Kafka Module for ETL Pipeline
"""

from .producers.data_producer import DataProducer
from .consumers.stream_processor import StreamProcessor

__all__ = ['DataProducer', 'StreamProcessor']
