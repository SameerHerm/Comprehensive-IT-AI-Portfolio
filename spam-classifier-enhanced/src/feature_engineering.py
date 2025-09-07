import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from gensim.models import Word2Vec
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Advanced feature engineering for text classification"""
    
    def __init__(self):
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
        self.word2vec_model = None
        self.svd = None
        
    def create_tfidf_features(self, texts, max_features=5000, ngram_range=(1, 3)):
        """Create TF-IDF features"""
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                stop_words='english',
                min_df=2,
                max_df=0.95
            )
            features = self.tfidf_vectorizer.fit_transform(texts)
        else:
            features = self.tfidf_vectorizer.transform(texts)
        
        logger.info(f"TF-IDF features shape: {features.shape}")
        return features
    
    def create_count_features(self, texts, max_features=3000):
        """Create count-based features"""
        if self.count_vectorizer is None:
            self.count_vectorizer = CountVectorizer(
                max_features=max_features,
                stop_words='english',
                binary=True
            )
            features = self.count_vectorizer.fit_transform(texts)
        else:
            features = self.count_vectorizer.transform(texts)
        
        return features
    
    def create_word2vec_features(self, tokenized_texts, vector_size=100):
        """Create Word2Vec embeddings"""
        if self.word2vec_model is None:
            self.word2vec_model = Word2Vec(
                sentences=tokenized_texts,
                vector_size=vector_size,
                window=5,
                min_count=2,
                workers=4,
                epochs=10
            )
        
        # Average word vectors for each document
        features = []
        for tokens in tokenized_texts:
            vectors = [self.word2vec_model.wv[token] 
                      for token in tokens 
                      if token in self.word2vec_model.wv]
            if vectors:
                features.append(np.mean(vectors, axis=0))
            else:
                features.append(np.zeros(vector_size))
        
        return np.array(features)
    
    def create_char_features(self, texts):
        """Create character-level features"""
        features = []
        for text in texts:
            feat = {
                'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0,
                'std_word_length': np.std([len(word) for word in text.split()]) if len(text.split()) > 1 else 0,
                'num_unique_words': len(set(text.split())),
                'lexical_diversity': len(set(text.split())) / (len(text.split()) + 1),
                'num_sentences': text.count('.') + text.count('!') + text.count('?'),
            }
            features.append(list(feat.values()))
        
        return np.array(features)
    
    def apply_dimensionality_reduction(self, features, n_components=100):
        """Apply SVD for dimensionality reduction"""
        if self.svd is None:
            self.svd = TruncatedSVD(n_components=min(n_components, features.shape[1]-1))
            reduced_features = self.svd.fit_transform(features)
        else:
            reduced_features = self.svd.transform(features)
        
        logger.info(f"Reduced features from {features.shape[1]} to {reduced_features.shape[1]}")
        return reduced_features
    
    def combine_features(self, *feature_sets):
        """Combine multiple feature sets"""
        from scipy.sparse import hstack, csr_matrix
        
        # Convert all to sparse matrices if needed
        sparse_features = []
        for features in feature_sets:
            if not hasattr(features, 'toarray'):
                sparse_features.append(csr_matrix(features))
            else:
                sparse_features.append(features)
        
        combined = hstack(sparse_features)
        logger.info(f"Combined features shape: {combined.shape}")
        return combined
