import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.model_selection import train_test_split
from typing import Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Advanced text preprocessing for spam classification"""
    
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))
        
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        """Tokenize and lemmatize text"""
        tokens = nltk.word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        return tokens
    
    def extract_features(self, text: str) -> dict:
        """Extract additional features from text"""
        features = {
            'length': len(text),
            'num_words': len(text.split()),
            'num_capitals': sum(1 for c in text if c.isupper()),
            'num_exclamation': text.count('!'),
            'num_question': text.count('?'),
            'num_dots': text.count('.'),
            'capital_ratio': sum(1 for c in text if c.isupper()) / (len(text) + 1),
            'special_char_ratio': len(re.findall(r'[^a-zA-Z0-9\s]', text)) / (len(text) + 1)
        }
        return features
    
    def preprocess_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess entire dataset"""
        logger.info("Starting preprocessing...")
        
        # Clean text
        df['cleaned_text'] = df['text'].apply(self.clean_text)
        
        # Extract features
        feature_df = pd.DataFrame(df['text'].apply(self.extract_features).tolist())
        df = pd.concat([df, feature_df], axis=1)
        
        # Encode labels
        df['label_encoded'] = df['label'].map({'ham': 0, 'spam': 1})
        
        logger.info(f"Preprocessing complete. Shape: {df.shape}")
        return df
    
    def split_data(self, df: pd.DataFrame, test_size: float = 0.2, 
                   random_state: int = 42) -> Tuple:
        """Split data into train and test sets"""
        X = df.drop(['label', 'label_encoded'], axis=1, errors='ignore')
        y = df['label_encoded']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        return X_train, X_test, y_train, y_test
