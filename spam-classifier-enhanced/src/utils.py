import os
import json
import yaml
import pickle
import logging
from datetime import datetime
from typing import Any, Dict
import hashlib

logger = logging.getLogger(__name__)

class Utils:
    """Utility functions for the spam classifier"""
    
    @staticmethod
    def setup_logging(log_dir: str = 'logs'):
        """Setup logging configuration"""
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        log_file = os.path.join(log_dir, f'spam_classifier_{datetime.now():%Y%m%d_%H%M%S}.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        return logger
    
    @staticmethod
    def load_config(config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    
    @staticmethod
    def save_config(config: Dict, config_path: str):
        """Save configuration to YAML file"""
        with open(config_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)
    
    @staticmethod
    def create_directories(paths: list):
        """Create multiple directories"""
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)
                logger.info(f"Created directory: {path}")
    
    @staticmethod
    def get_file_hash(filepath: str) -> str:
        """Get MD5 hash of a file"""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    @staticmethod
    def save_pickle(obj: Any, filepath: str):
        """Save object as pickle file"""
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
        logger.info(f"Object saved to {filepath}")
    
    @staticmethod
    def load_pickle(filepath: str) -> Any:
        """Load pickle file"""
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        return obj
    
    @staticmethod
    def get_model_metrics_summary(metrics: Dict) -> str:
        """Get formatted summary of model metrics"""
        summary = f"""
        Model Performance Summary:
        -------------------------
        Accuracy:  {metrics.get('accuracy', 0):.4f}
        Precision: {metrics.get('precision', 0):.4f}
        Recall:    {metrics.get('recall', 0):.4f}
        F1-Score:  {metrics.get('f1_score', 0):.4f}
        """
        return summary
    
    @staticmethod
    def validate_email_text(text: str) -> bool:
        """Basic validation of email text"""
        if not text or len(text.strip()) < 10:
            return False
        if len(text) > 50000:  # Too long
            return False
        return True
