import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import DataPreprocessor
import pandas as pd

class TestDataPreprocessor:
    
    @pytest.fixture
    def preprocessor(self):
        return DataPreprocessor()
    
    def test_clean_text(self, preprocessor):
        """Test text cleaning functionality"""
        text = "Check out this link: http://spam.com!!! Call 1234567890 NOW!"
        cleaned = preprocessor.clean_text(text)
        
        assert "http" not in cleaned
        assert "1234567890" not in cleaned
        assert "!" not in cleaned
        assert cleaned.islower()
    
    def test_tokenize_and_lemmatize(self, preprocessor):
        """Test tokenization and lemmatization"""
        text = "running runs ran"
        tokens = preprocessor.tokenize_and_lemmatize(text)
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
    
    def test_extract_features(self, preprocessor):
        """Test feature extraction"""
        text = "URGENT!!! Free money NOW!!!"
        features = preprocessor.extract_features(text)
        
        assert 'length' in features
        assert 'num_exclamation' in features
        assert features['num_exclamation'] == 6
        assert features['capital_ratio'] > 0
    
    def test_preprocess_dataset(self, preprocessor):
        """Test dataset preprocessing"""
        df = pd.DataFrame({
            'text': ['spam message', 'ham message'],
            'label': ['spam', 'ham']
        })
        
        processed_df = preprocessor.preprocess_dataset(df)
        
        assert 'cleaned_text' in processed_df.columns
        assert 'label_encoded' in processed_df.columns
        assert processed_df['label_encoded'].iloc[0] == 1  # spam = 1
        assert processed_df['label_encoded'].iloc[1] == 0  # ham = 0
