"""
Monitoring Module
Provides monitoring and metrics collection functionality
"""

import time
import psutil
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from collections import defaultdict
import json
from prometheus_client import Counter, Histogram, Gauge, Summary

logger = logging.getLogger(__name__)

class MetricsCollector:
    """Collect and track metrics for ETL pipeline"""
    
    def __init__(self):
        """Initialize metrics collector"""
        self.metrics = defaultdict(list)
        self.counters = {}
        self.timers = {}
        
        # Prometheus metrics
        self.records_processed = Counter('etl_records_processed_total', 
                                        'Total records processed', 
                                        ['pipeline', 'stage'])
        self.processing_time = Histogram('etl_processing_duration_seconds',
                                        'Processing time in seconds',
                                        ['pipeline', 'stage'])
        self.error_count = Counter('etl_errors_total',
                                  'Total errors',
                                  ['pipeline', 'stage', 'error_type'])
        self.pipeline_status = Gauge('etl_pipeline_status',
                                    'Pipeline status (1=running, 0=stopped)',
                                    ['pipeline'])
        
    def start_timer(self, name: str):
        """Start a timer"""
        self.timers[name] = time.time()
        logger.debug(f"Timer started: {name}")
    
    def stop_timer(self, name: str) -> float:
        """Stop a timer and return elapsed time"""
        if name not in self.timers:
            logger.warning(f"Timer not found: {name}")
            return 0
        
        elapsed = time.time() - self.timers[name]
        del self.timers[name]
        
        self.metrics[f"{name}_duration"].append(elapsed)
        logger.debug(f"Timer stopped: {name}, Duration: {elapsed:.2f}s")
        
        return elapsed
    
    def increment_counter(self, name: str, value: int = 1):
        """Increment a counter"""
        if name not in self.counters:
            self.counters[name] = 0
        
        self.counters[name] += value
        logger.debug(f"Counter incremented: {name} = {self.counters[name]}")
    
    def record_metric(self, name: str, value: Any):
        """Record a metric value"""
        self.metrics[name].append({
            'value': value,
            'timestamp': datetime.now().isoformat()
        })
        logger.debug(f"Metric recorded: {name} = {value}")
    
    def record_pipeline_metrics(self, pipeline: str, stage: str, 
                              records: int, duration: float):
        """Record pipeline-specific metrics"""
        self.records_processed.labels(pipeline=pipeline, stage=stage).inc(records)
        self.processing_time.labels(pipeline=pipeline, stage=stage).observe(duration)
        
        # Calculate throughput
        throughput = records / duration if duration > 0 else 0
        self.record_metric(f"{pipeline}_{stage}_throughput", throughput)
    
    def record_error(self, pipeline: str, stage: str, error_type: str):
        """Record an error"""
        self.error_count.labels(
            pipeline=pipeline,
            stage=stage,
            error_type=error_type
        ).inc()
        
        self.increment_counter('total_errors')
        self.metrics['errors'].append({
            'pipeline': pipeline,
            'stage': stage,
            'error_type': error_type,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        summary = {
            'counters': dict(self.counters),
            'timers': {},
            'metrics': {}
        }
        
        # Summarize timer metrics
        for name, values in self.metrics.items():
            if name.endswith('_duration'):
                if values:
                    durations = [v for v in values if isinstance(v, (int, float))]
                    if durations:
                        summary['timers'][name] = {
                            'count': len(durations),
                            'total': sum(durations),
                            'average': sum(durations) / len(durations),
                            'min': min(durations),
                            'max': max(durations)
                        }
        
        # Summarize other metrics
        for name, values in self.metrics.items():
            if not name.endswith('_duration') and values:
                summary['metrics'][name] = {
                    'count': len(values),
                    'latest': values[-1] if values else None
                }
        
        return summary
    
    def export_metrics(self, filepath: str):
        """Export metrics to file"""
        with open(filepath, 'w') as f:
            json.dump(self.get_summary(), f, indent=2, default=str)
        
        logger.info(f"Metrics exported to {filepath}")
    
    def reset(self):
        """Reset all metrics"""
        self.metrics.clear()
        self.counters.clear()
        self.timers.clear()
        logger.info("Metrics reset")

class PerformanceMonitor:
    """Monitor system performance during ETL operations"""
    
    def __init__(self):
        """Initialize performance monitor"""
        self.start_time = None
        self.samples = []
        self.is_monitoring = False
    
    def start(self):
        """Start monitoring"""
        self.start_time = time.time()
        self.is_monitoring = True
        self.samples = []
        logger.info("Performance monitoring started")
    
    def stop(self) -> Dict[str, Any]:
        """Stop monitoring and return summary"""
        self.is_monitoring = False
        duration = time.time() - self.start_time if self.start_time else 0
        
        summary = self.get_summary()
        summary['duration'] = duration
        
        logger.info("Performance monitoring stopped")
        return summary
    
    def sample(self):
        """Take a performance sample"""
        if not self.is_monitoring:
            return
        
        sample = {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory': {
                'percent': psutil.virtual_memory().percent,
                'used': psutil.virtual_memory().used,
                'available': psutil.virtual_memory().available
            },
            'disk': {
                'read_bytes': psutil.disk_io_counters().read_bytes,
                'write_bytes': psutil.disk_io_counters().write_bytes
            },
            'network': {
                'bytes_sent': psutil.net_io_counters().bytes_sent,
                'bytes_recv': psutil.net_io_counters().bytes_recv
            }
        }
        
        self.samples.append(sample)
        return sample
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current system statistics"""
        return {
            'cpu': {
                'percent': psutil.cpu_percent(interval=0.1),
                'count': psutil.cpu_count()
            },
            'memory': {
                'total': psutil.virtual_memory().total,
                'available': psutil.virtual_memory().available,
                'percent': psutil.virtual_memory().percent,
                'used': psutil.virtual_memory().used
            },
            'disk': {
                'total': psutil.disk_usage('/').total,
                'used': psutil.disk_usage('/').used,
                'free': psutil.disk_usage('/').free,
                'percent': psutil.disk_usage('/').percent
            },
            'network': {
                'connections': len(psutil.net_connections())
            }
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.samples:
            return {}
        
        cpu_samples = [s['cpu_percent'] for s in self.samples]
        memory_samples = [s['memory']['percent'] for s in self.samples]
        
        return {
            'samples_count': len(self.samples),
            'cpu': {
                'average': sum(cpu_samples) / len(cpu_samples),
                'max': max(cpu_samples),
                'min': min(cpu_samples)
            },
            'memory': {
                'average': sum(memory_samples) / len(memory_samples),
                'max': max(memory_samples),
                'min': min(memory_samples)
            },
            'disk_io': {
                'read_bytes': self.samples[-1]['disk']['read_bytes'] - self.samples[0]['disk']['read_bytes'] if len(self.samples) > 1 else 0,
                'write_bytes': self.samples[-1]['disk']['write_bytes'] - self.samples[0]['disk']['write_bytes'] if len(self.samples) > 1 else 0
            },
            'network_io': {
                'bytes_sent': self.samples[-1]['network']['bytes_sent'] - self.samples[0]['network']['bytes_sent'] if len(self.samples) > 1 else 0,
                'bytes_recv': self.samples[-1]['network']['bytes_recv'] - self.samples[0]['network']['bytes_recv'] if len(self.samples) > 1 else 0
            }
        }
    
    def check_resource_limits(self, cpu_threshold: float = 90.0,
                            memory_threshold: float = 90.0) -> List[str]:
        """Check if resource usage exceeds thresholds"""
        warnings = []
        current = self.get_current_stats()
        
        if current['cpu']['percent'] > cpu_threshold:
            warnings.append(f"High CPU usage: {current['cpu']['percent']}%")
        
        if current['memory']['percent'] > memory_threshold:
            warnings.append(f"High memory usage: {current['memory']['percent']}%")
        
        if current['disk']['percent'] > 90:
            warnings.append(f"Low disk space: {current['disk']['percent']}% used")
        
        return warnings
