"""
Unit tests for data preprocessing functions
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import (
    load_data,
    clean_data,
    handle_missing_values,
    normalize_features,
    encode_categorical_variables,
    create_feature_engineering,
    split_data
)


class TestDataPreprocessing(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        # Create sample test data
        self.sample_data = pd.DataFrame({
            'age': [25, 35, 45, 55, 65, np.nan],
            'gender': ['M', 'F', 'M', 'F', 'M', 'F'],
            'systolic_bp': [120, 130, 140, 150, 160, 135],
            'diastolic_bp': [80, 85, 90, 95, 100, np.nan],
            'cholesterol': [180, 200, 220, 240, 260, 210],
            'bmi': [22.5, 25.0, 27.5, 30.0, 32.5, 28.0],
            'smoking': [0, 1, 1, 0, 1, 0],
            'diabetes': [0, 0, 1, 1, 1, 0],
            'cardiovascular_risk': [0, 0, 1, 1, 1, 0]
        })
        
    def test_load_data(self):
        """Test data loading functionality"""
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.return_value = self.sample_data
            
            # Test loading CSV
            data = load_data('test.csv')
            self.assertIsInstance(data, pd.DataFrame)
            mock_read_csv.assert_called_once_with('test.csv')
            
    def test_clean_data(self):
        """Test data cleaning functionality"""
        # Test removing duplicates
        data_with_duplicates = pd.concat([self.sample_data, self.sample_data.iloc[[0]]])
        cleaned_data = clean_data(data_with_duplicates)
        
        self.assertEqual(len(cleaned_data), len(self.sample_data))
        
    def test_handle_missing_values(self):
        """Test missing value handling"""
        # Test with numeric imputation
        imputed_data = handle_missing_values(self.sample_data, strategy='mean')
        
        # Check no missing values remain
        self.assertFalse(imputed_data.isnull().any().any())
        
        # Test with median imputation
        imputed_data_median = handle_missing_values(self.sample_data, strategy='median')
        self.assertFalse(imputed_data_median.isnull().any().any())
        
        # Test with forward fill
        imputed_data_ffill = handle_missing_values(self.sample_data, strategy='ffill')
        self.assertEqual(imputed_data_ffill['age'].isnull().sum(), 1)  # Last value might still be NaN
        
    def test_normalize_features(self):
        """Test feature normalization"""
        numeric_cols = ['age', 'systolic_bp', 'diastolic_bp', 'cholesterol', 'bmi']
        
        # Drop NaN values for this test
        clean_data = self.sample_data.dropna()
        
        # Test standard scaling
        normalized_data = normalize_features(clean_data, numeric_cols, method='standard')
        
        for col in numeric_cols:
            # Check mean is close to 0 and std is close to 1
            self.assertAlmostEqual(normalized_data[col].mean(), 0, places=5)
            self.assertAlmostEqual(normalized_data[col].std(), 1, places=5)
        
        # Test min-max scaling
        normalized_data_minmax = normalize_features(clean_data, numeric_cols, method='minmax')
        
        for col in numeric_cols:
            # Check values are between 0 and 1
            self.assertGreaterEqual(normalized_data_minmax[col].min(), 0)
            self.assertLessEqual(normalized_data_minmax[col].max(), 1)
            
    def test_encode_categorical_variables(self):
        """Test categorical variable encoding"""
        # Test one-hot encoding
        encoded_data = encode_categorical_variables(
            self.sample_data, 
            ['gender'], 
            encoding_type='onehot'
        )
        
        # Check if gender columns are created
        self.assertIn('gender_F', encoded_data.columns)
        self.assertIn('gender_M', encoded_data.columns)
        self.assertNotIn('gender', encoded_data.columns)
        
        # Test label encoding
        encoded_data_label = encode_categorical_variables(
            self.sample_data, 
            ['gender'], 
            encoding_type='label'
        )
        
        # Check if gender is encoded as numeric
        self.assertTrue(pd.api.types.is_numeric_dtype(encoded_data_label['gender']))
        
    def test_create_feature_engineering(self):
        """Test feature engineering"""
        # Clean data first
        clean_data = self.sample_data.dropna()
        
        # Apply feature engineering
        engineered_data = create_feature_engineering(clean_data)
        
        # Check if new features are created
        expected_features = [
            'bp_ratio',
            'bmi_category',
            'age_group',
            'cholesterol_hdl_ratio',
            'metabolic_syndrome_score'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, engineered_data.columns)
        
        # Validate bp_ratio calculation
        expected_bp_ratio = clean_data['systolic_bp'] / clean_data['diastolic_bp']
        np.testing.assert_array_almost_equal(
            engineered_data['bp_ratio'].values,
            expected_bp_ratio.values
        )
        
    def test_split_data(self):
        """Test data splitting"""
        clean_data = self.sample_data.dropna()
        
        # Define features and target
        features = ['age', 'systolic_bp', 'diastolic_bp', 'cholesterol', 'bmi']
        target = 'cardiovascular_risk'
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            clean_data,
            features,
            target,
            test_size=0.2,
            val_size=0.2,
            random_state=42
        )
        
        # Check shapes
        total_samples = len(clean_data)
        
        # Test split should be approximately 20%
        self.assertAlmostEqual(len(X_test) / total_samples, 0.2, places=1)
        
        # Validation split should be approximately 20% of remaining
        self.assertAlmostEqual(len(X_val) / (total_samples - len(X_test)), 0.2, places=1)
        
        # Check that all splits have the same features
        self.assertEqual(X_train.shape[1], len(features))
        self.assertEqual(X_val.shape[1], len(features))
        self.assertEqual(X_test.shape[1], len(features))
        
        # Check no data leakage
        train_indices = set(X_train.index)
        val_indices = set(X_val.index)
        test_indices = set(X_test.index)
        
        self.assertEqual(len(train_indices & val_indices), 0)
        self.assertEqual(len(train_indices & test_indices), 0)
        self.assertEqual(len(val_indices & test_indices), 0)
        
    def test_outlier_detection(self):
        """Test outlier detection and handling"""
        from src.data_preprocessing import detect_outliers, handle_outliers
        
        # Create data with outliers
        data_with_outliers = self.sample_data.copy()
        data_with_outliers.loc[0, 'systolic_bp'] = 300  # Outlier
        
        # Detect outliers
        outliers = detect_outliers(data_with_outliers, ['systolic_bp'], method='iqr')
        self.assertTrue(outliers[0])  # First row should be detected as outlier
        
        # Handle outliers
        handled_data = handle_outliers(
            data_with_outliers, 
            ['systolic_bp'], 
            method='clip'
        )
        
        # Check that outlier is clipped
        self.assertLess(handled_data.loc[0, 'systolic_bp'], 300)
        
    def test_data_validation(self):
        """Test data validation"""
        from src.data_preprocessing import validate_data
        
        # Test with valid data
        is_valid, errors = validate_data(self.sample_data.dropna())
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        
        # Test with invalid data (negative age)
        invalid_data = self.sample_data.copy()
        invalid_data.loc[0, 'age'] = -5
        
        is_valid, errors = validate_data(invalid_data)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
        

if __name__ == '__main__':
    unittest.main()
