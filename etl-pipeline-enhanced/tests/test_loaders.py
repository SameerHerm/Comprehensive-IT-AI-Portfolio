"""
Tests for Loader modules
"""

import pytest
import pandas as pd
import tempfile
import os
import json
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.loaders.file_loader import FileLoader
from src.loaders.database_loader import DatabaseLoader

class TestFileLoader:
    
    @pytest.fixture
    def loader(self):
        return FileLoader()
    
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'value': [100, 200, 300, 400, 500]
        })
    
    def test_load_csv(self, loader, sample_df):
        """Test CSV loading"""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            loader.load_csv(sample_df, tmp.name)
            
            # Verify file was created
            assert os.path.exists(tmp.name)
            
            # Verify content
            loaded_df = pd.read_csv(tmp.name)
            assert len(loaded_df) == len(sample_df)
            
            os.unlink(tmp.name)
    
    def test_load_json(self, loader, sample_df):
        """Test JSON loading"""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            loader.load_json(sample_df, tmp.name)
            
            assert os.path.exists(tmp.name)
            
            # Verify content
            with open(tmp.name, 'r') as f:
                data = json.load(f)
            assert len(data) == len(sample_df)
            
            os.unlink(tmp.name)
    
    def test_load_parquet(self, loader, sample_df):
        """Test Parquet loading"""
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
            loader.load_parquet(sample_df, tmp.name)
            
            assert os.path.exists(tmp.name)
            
            # Verify content
            loaded_df = pd.read_parquet(tmp.name)
            assert len(loaded_df) == len(sample_df)
            
            os.unlink(tmp.name)
    
    def test_load_partitioned(self, loader):
        """Test partitioned loading"""
        df = pd.DataFrame({
            'year': [2023, 2023, 2024, 2024],
            'month': [1, 2, 1, 2],
            'value': [100, 200, 300, 400]
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            loader.load_partitioned(
                df,
                tmpdir,
                partition_cols=['year', 'month'],
                format='csv'
            )
            
            # Check if partitions were created
            assert os.path.exists(os.path.join(tmpdir, 'year=2023'))
            assert os.path.exists(os.path.join(tmpdir, 'year=2024'))

class TestDatabaseLoader:
    
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Test1', 'Test2', 'Test3'],
            'value': [100, 200, 300]
        })
    
    @pytest.mark.skip(reason="Requires database connection")
    def test_load_to_postgres(self, sample_df):
        """Test PostgreSQL loading"""
        loader = DatabaseLoader()
        
        # This would require actual database connection
        # loader.load_to_postgres(sample_df, 'test_table', 'public')
        pass
