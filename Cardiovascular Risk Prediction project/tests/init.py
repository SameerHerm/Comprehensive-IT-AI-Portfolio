"""
Cardiovascular Risk Prediction Tests Package
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Test configuration
TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), 'test_data')
TEST_MODELS_PATH = os.path.join(os.path.dirname(__file__), 'test_models')

# Create test directories if they don't exist
os.makedirs(TEST_DATA_PATH, exist_ok=True)
os.makedirs(TEST_MODELS_PATH, exist_ok=True)
