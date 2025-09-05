import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self):
        """Load raw cardiovascular data"""
        # Generate synthetic data if no real data available
        np.random.seed(42)
        n_samples = 10000
        
        data = {
            'age': np.random.randint(30, 80, n_samples),
            'gender': np.random.choice([0, 1], n_samples),
            'height': np.random.normal(170, 10, n_samples),
            'weight': np.random.normal(75, 15, n_samples),
            'systolic_bp': np.random.normal(130, 20, n_samples),
            'diastolic_bp': np.random.normal(80, 15, n_samples),
            'cholesterol': np.random.choice([1, 2, 3], n_samples, p=[0.5, 0.3, 0.2]),
            'glucose': np.random.choice([1, 2, 3], n_samples, p=[0.6, 0.25, 0.15]),
            'smoking': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'alcohol': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
            'physical_activity': np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
        }
        
        df = pd.DataFrame(data)
        
        # Create target variable based on risk factors
        risk_score = (
            (df['age'] - 30) * 0.1 +
            df['systolic_bp'] * 0.02 +
            df['diastolic_bp'] * 0.03 +
            df['cholesterol'] * 2 +
            df['glucose'] * 1.5 +
            df['smoking'] * 3 +
            (1 - df['physical_activity']) * 2
        )
        
        # Add some noise and create binary target
        risk_score += np.random.normal(0, 2, n_samples)
        df['cardio_disease'] = (risk_score > np.percentile(risk_score, 70)).astype(int)
        
        logger.info(f"Generated synthetic dataset with {len(df)} samples")
        return df
    
    def clean_data(self, df):
        """Clean and validate data"""
        # Remove outliers using IQR method
        numerical_cols = self.config['features']['numerical_features']
        
        for col in numerical_cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower) & (df[col] <= upper)]
        
        # Handle missing values
        df = df.dropna()
        
        logger.info(f"Data cleaned. Remaining samples: {len(df)}")
        return df
    
    def feature_engineering(self, df):
        """Create new features"""
        # BMI calculation
        df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
        
        # Pulse pressure
        df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
        
        # Age groups
        df['age_group'] = pd.cut(df['age'], bins=[0, 40, 55, 70, 100], 
                                labels=['young', 'middle_aged', 'senior', 'elderly'])
        
        # BMI categories
        df['bmi_category'] = pd.cut(df['bmi'], 
                                   bins=[0, 18.5, 25, 30, 100],
                                   labels=['underweight', 'normal', 'overweight', 'obese'])
        
        # Risk factors count
        risk_factors = ['smoking', 'alcohol', 'cholesterol', 'glucose']
        df['risk_factors_count'] = df[risk_factors].sum(axis=1)
        
        logger.info("Feature engineering completed")
        return df
    
    def encode_categorical_features(self, df, fit=True):
        """Encode categorical features"""
        categorical_cols = self.config['features']['categorical_features'] + ['age_group', 'bmi_category']
        
        for col in categorical_cols:
            if col in df.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    if col in self.label_encoders:
                        df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        return df
    
    def scale_numerical_features(self, df, fit=True):
        """Scale numerical features"""
        numerical_cols = self.config['features']['numerical_features'] + ['bmi', 'pulse_pressure', 'risk_factors_count']
        numerical_cols = [col for col in numerical_cols if col in df.columns]
        
        if fit:
            df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        else:
            df[numerical_cols] = self.scaler.transform(df[numerical_cols])
        
        return df
    
    def handle_class_imbalance(self, X, y):
        """Handle class imbalance using SMOTE"""
        smote = SMOTE(random_state=self.config['data']['random_state'])
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        logger.info(f"Applied SMOTE. New shape: {X_resampled.shape}")
        return X_resampled, y_resampled
    
    def process_data(self):
        """Complete data processing pipeline"""
        # Load data
        df = self.load_data()
        
        # Clean data
        df = self.clean_data(df)
        
        # Feature engineering
        df = self.feature_engineering(df)
        
        # Separate features and target
        X = df.drop('cardio_disease', axis=1)
        y = df['cardio_disease']
        
        # Encode categorical features
        X = self.encode_categorical_features(X, fit=True)
        
        # Scale numerical features
        X = self.scale_numerical_features(X, fit=True)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config['data']['test_size'],
            random_state=self.config['data']['random_state'],
            stratify=y
        )
        
        # Handle class imbalance on training set
        X_train_balanced, y_train_balanced = self.handle_class_imbalance(X_train, y_train)
        
        # Save processed data
        import os
        os.makedirs('data/processed', exist_ok=True)
        
        processed_data = {
            'X_train': X_train_balanced,
            'X_test': X_test,
            'y_train': y_train_balanced,
            'y_test': y_test,
            'feature_names': X.columns.tolist()
        }
        
        import joblib
        joblib.dump(processed_data, 'data/processed/processed_data.pkl')
        joblib.dump(self.scaler, 'models/scaler.pkl')
        joblib.dump(self.label_encoders, 'models/label_encoders.pkl')
        
        logger.info("Data processing completed and saved")
        return processed_data

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    data = preprocessor.process_data()
