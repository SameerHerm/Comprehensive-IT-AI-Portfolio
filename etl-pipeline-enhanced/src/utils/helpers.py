"""
Helper Functions Module
Common utility functions for ETL pipeline
"""

import hashlib
import re
import os
import time
import logging
from typing import Dict, Any, Optional, Callable
from functools import wraps
from urllib.parse import urlparse
import json

logger = logging.getLogger(__name__)

def retry_with_backoff(max_retries: int = 3, 
                       backoff_factor: float = 2.0,
                       exceptions: tuple = (Exception,)):
    """Decorator to retry function with exponential backoff"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            retry_count = 0
            wait_time = 1
            
            while retry_count < max_retries:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    retry_count += 1
                    
                    if retry_count >= max_retries:
                        logger.error(f"Max retries ({max_retries}) reached for {func.__name__}")
                        raise
                    
                    logger.warning(f"Attempt {retry_count} failed for {func.__name__}: {e}")
                    logger.info(f"Retrying in {wait_time} seconds...")
                    
                    time.sleep(wait_time)
                    wait_time *= backoff_factor
            
            return None
        
        return wrapper
    return decorator

def parse_connection_string(conn_string: str) -> Dict[str, Any]:
    """Parse database connection string"""
    # Example: postgresql://user:password@host:port/database
    parsed = urlparse(conn_string)
    
    return {
        'scheme': parsed.scheme,
        'user': parsed.username,
        'password': parsed.password,
        'host': parsed.hostname,
        'port': parsed.port,
        'database': parsed.path.lstrip('/') if parsed.path else None,
        'params': dict(param.split('=') for param in parsed.query.split('&')) if parsed.query else {}
    }

def format_bytes(bytes_value: int) -> str:
    """Format bytes to human readable string"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    
    return f"{bytes_value:.2f} PB"

def get_file_hash(filepath: str, algorithm: str = 'md5') -> str:
    """Calculate file hash"""
    hash_algorithms = {
        'md5': hashlib.md5,
        'sha1': hashlib.sha1,
        'sha256': hashlib.sha256,
        'sha512': hashlib.sha512
    }
    
    if algorithm not in hash_algorithms:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    hasher = hash_algorithms[algorithm]()
    
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hasher.update(chunk)
    
    return hasher.hexdigest()

def validate_email(email: str) -> bool:
    """Validate email address format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def sanitize_filename(filename: str) -> str:
    """Sanitize filename by removing invalid characters"""
    # Remove invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading/trailing dots and spaces
    filename = filename.strip('. ')
    
    # Limit length
    max_length = 255
    if len(filename) > max_length:
        name, ext = os.path.splitext(filename)
        filename = name[:max_length - len(ext)] + ext
    
    return filename

def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """Flatten nested dictionary"""
    items = []
    
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    items.extend(flatten_dict(item, f"{new_key}_{i}", sep=sep).items())
                else:
                    items.append((f"{new_key}_{i}", item))
        else:
            items.append((new_key, v))
    
    return dict(items)

def deep_merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries"""
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result

def chunk_list(lst: list, chunk_size: int):
    """Split list into chunks"""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def safe_json_loads(json_string: str, default: Any = None) -> Any:
    """Safely parse JSON string"""
    try:
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError):
        logger.warning(f"Failed to parse JSON: {json_string[:100]}...")
        return default

def get_file_extension(filepath: str) -> str:
    """Get file extension without dot"""
    return os.path.splitext(filepath)[1].lstrip('.')

def ensure_list(value: Any) -> list:
    """Ensure value is a list"""
    if value is None:
        return []
    elif isinstance(value, list):
        return value
    else:
        return [value]

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers"""
    if denominator == 0:
        return default
    return numerator / denominator

def truncate_string(s: str, max_length: int, suffix: str = '...') -> str:
    """Truncate string to maximum length"""
    if len(s) <= max_length:
        return s
    
    return s[:max_length - len(suffix)] + suffix

def generate_batch_id() -> str:
    """Generate unique batch ID"""
    from datetime import datetime
    import uuid
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_id = str(uuid.uuid4())[:8]
    
    return f"batch_{timestamp}_{unique_id}"

def estimate_memory_usage(df) -> str:
    """Estimate memory usage of pandas DataFrame"""
    if hasattr(df, 'memory_usage'):
        memory_bytes = df.memory_usage(deep=True).sum()
        return format_bytes(memory_bytes)
    
    return "N/A"
