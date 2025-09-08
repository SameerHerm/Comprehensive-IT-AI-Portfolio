"""
Tests for Transformer modules
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.transformers.data_cleaner import DataCleaner
from src.transformers.data_validator import DataValidator
from src.transformers.aggregator import Aggregator

class TestDataCleaner:
    
    @pytest.fixture
    def cleaner(self):
        return DataCleaner()
    
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['John', 'Jane', None, 'Bob', 'Alice'],
            'age': [25, 30, 35, None, 40],
            'email': ['john@example.com', 'JANE@EXAMPLE.COM', 'invalid-email', None, 'alice@example.com'],
            'amount': [100.5, 200.0, -50.0, 300.0, None]
        })
    
    def test_clean_data(self, cleaner, sample_df):
        """Test data cleaning"""
        cleaned = cleaner.clean_data(sample_df)
        
        assert cleaned is not None
        assert len(cleaned) <= len(sample_df)
    
    def test_handle_missing_values(self, cleaner, sample_df):
        """Test missing value handling"""
        result = cleaner.handle_missing_values(sample_df, method='mean')
        
        assert result['age'].isnull().sum() == 0
        assert result['amount'].isnull().sum() == 0

class TestDataValidator:
    
    @pytest.fixture
    def validator(self):
        return DataValidator()
    
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'amount': [100, 200, 300, 400, 500],
            'email': ['test@example.com', 'user@test.com', 'invalid', 'admin@site.com', 'info@company.com']
        })
    
    def test_null_check(self, validator, sample_df):
        """Test null value checking"""
        rule = {
            'name': 'no_nulls',
            'type': 'null_check',
            'columns': ['id', 'amount'],
            'threshold': 0
        }
        
        passed, results = validator.validate(sample_df, [rule])
        assert passed
    
    def test_duplicate_check(self, validator):
        """Test duplicate checking"""
        df = pd.DataFrame({
            'id': [1, 2, 2, 3, 4],
            'value': ['a', 'b', 'b', 'c', 'd']
        })
        
        rule = {
            'name': 'no_duplicates',
            'type': 'duplicate_check',
            'columns': ['id'],
            'max_duplicates': 0
        }
        
        passed, results = validator.validate(df, [rule])
        assert not passed
    
    def test_range_check(self, validator, sample_df):
        """Test range checking"""
        rule = {
            'name': 'amount_range',
            'type': 'range_check',
            'column': 'amount',
            'min_value': 0,
            'max_value': 1000
        }
        
        passed, results = validator.validate(sample_df, [rule])
        assert passed

class TestAggregator:
    
    @pytest.fixture
    def aggregator(self):
        return Aggregator()
    
    @pytest.fixture
    def sample_df(self):
        dates = pd.date_range('2024-01-01', periods=100, freq='H')
        return pd.DataFrame({
            'timestamp': dates,
            'category': np.random.choice(['A', 'B', 'C'], 100),
            'value': np.random.randn(100) * 100 + 500,
            'quantity': np.random.randint(1, 10, 100)
        })
    
    def test_basic_aggregation(self, aggregator, sample_df):
        """Test basic aggregation"""
        result = aggregator.aggregate(
            sample_df,
            group_by='category',
            aggregations={
                'value': 'sum',
                'quantity': ['mean', 'sum']
            }
        )
        
        assert 'category' in result.columns
        assert len(result) <= len(sample_df['category'].unique())
    
    def test_time_series_aggregation(self, aggregator, sample_df):
        """Test time series aggregation"""
        result = aggregator.time_series_aggregation(
            sample_df,
            date_column='timestamp',
            value_columns=['value', 'quantity'],
            freq='D',
            agg_func='sum'
        )
        
        assert 'timestamp' in result.columns
        assert 'year' in result.columns
        assert 'month' in result.columns
    
    def test_rolling_aggregation(self, aggregator, sample_df):
        """Test rolling aggregation"""
        result = aggregator.rolling_aggregation(
            sample_df,
            date_column='timestamp',
            value_columns=['value'],
            window=7,
            agg_func='mean'
        )
        
        assert 'value_rolling_7' in result.columns
        assert len(result) == len(sample_df)
