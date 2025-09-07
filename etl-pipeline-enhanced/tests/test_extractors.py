"""
Tests for Extractor modules
"""

import pytest
import pandas as pd
import tempfile
import json
from pathlib import Path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.extractors.file_extractor import FileExtractor

class TestFileExtractor:
    
    @pytest.fixture
    def extractor(self):
        return FileExtractor()
    
    @pytest.fixture
    def sample_csv(self):
        """Create sample CSV file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("id,name,value\n")
            f.write("1,test1,100\n")
            f.write("2,test2,200\n")
            return f.name
    
    @pytest.fixture
    def sample_json(self):
        """Create sample JSON file"""
        data = [
            {"id": 1, "name": "test1", "value": 100},
            {"id": 2, "name": "test2", "value": 200}
        ]
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            return f.name
    
    def test_extract_csv(self, extractor, sample_csv):
        """Test CSV extraction"""
        df = extractor.extract_csv(sample_csv)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ['id', 'name', 'value']
        
        # Cleanup
        Path(sample_csv).unlink()
    
    def test_extract_json(self, extractor, sample_json):
        """Test JSON extraction"""
        df = extractor.extract_json(sample_json)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'name' in df.columns
        
        # Cleanup
        Path(sample_json).unlink()
    
    def test_extract_unsupported_format(self, extractor):
        """Test extraction with unsupported format"""
        with pytest.raises(ValueError, match="Unsupported format"):
            extractor.extract("test.xyz")
    
    def test_extract_nonexistent_file(self, extractor):
        """Test extraction with non-existent file"""
        with pytest.raises(FileNotFoundError):
            extractor.extract("nonexistent.csv")
