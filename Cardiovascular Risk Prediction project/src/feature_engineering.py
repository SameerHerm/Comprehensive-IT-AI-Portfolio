"""
Feature Engineering Module for Cardiovascular Risk Prediction
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Feature engineering for cardiovascular risk prediction
    """
    
    def __init__(self):
        """Initialize feature engineer"""
        self.feature_names = []
        self.polynomial_features = None
        self.feature_selector = None
        
    def create_age_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create age-based features"""
        logger.info("Creating age-based features...")
        
        if 'age' in df.columns:
            # Age groups
            df['age_group'] = pd.cut(df['age'], 
                                     bins=[0, 30, 40, 50, 60, 70, 100],
                                     labels=['<30', '30-40', '40-50', '50-60', '60-70', '70+'])
            
            # Age risk categories
            df['age_risk'] = df['age'].apply(lambda x: 
                'low' if x < 45 else 'medium' if x < 65 else 'high')
            
            # Age squared (non-linear relationship)
            df['age_squared'] = df['age'] ** 2
            
            # Age normalized
            df['age_normalized'] = (df['age'] - df['age'].mean()) / df['age'].std()
            
        return df
    
    def create_blood_pressure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create blood pressure related features"""
        logger.info("Creating blood pressure features...")
        
        if 'sysBP' in df.columns and 'diaBP' in df.columns:
            # Pulse pressure
            df['pulse_pressure'] = df['sysBP'] - df['diaBP']
            
            # Mean arterial pressure
            df['mean_arterial_pressure'] = df['diaBP'] + (df['pulse_pressure'] / 3)
            
            # Hypertension stages
            df['hypertension_stage'] = df.apply(self._classify_hypertension, axis=1)
            
            # BP ratio
            df['bp_ratio'] = df['sysBP'] / (df['diaBP'] + 1)  # Avoid division by zero
            
        return df
    
    def _classify_hypertension(self, row) -> str:
        """Classify hypertension stage based on BP values"""
        if pd.isna(row['sysBP']) or pd.isna(row['diaBP']):
            return 'unknown'
        
        sysBP = row['sysBP']
        diaBP = row['diaBP']
        
        if sysBP < 120 and diaBP < 80:
            return 'normal'
        elif sysBP < 130 and diaBP < 80:
            return 'elevated'
        elif sysBP < 140 or diaBP < 90:
            return 'stage1'
        else:
            return 'stage2'
    
    def create_bmi_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create BMI-related features"""
        logger.info("Creating BMI features...")
        
        if 'BMI' in df.columns:
            # BMI categories
            df['bmi_category'] = pd.cut(df['BMI'],
                                        bins=[0, 18.5, 25, 30, 35, 100],
                                        labels=['underweight', 'normal', 'overweight', 
                                               'obese_1', 'obese_2'])
            
            # BMI risk score
            df['bmi_risk'] = df['BMI'].apply(lambda x: 
                0 if x < 25 else 1 if x < 30 else 2 if x < 35 else 3)
            
            # BMI squared for non-linear relationships
            df['bmi_squared'] = df['BMI'] ** 2
            
        return df
    
    def create_cholesterol_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create cholesterol-related features"""
        logger.info("Creating cholesterol features...")
        
        if 'totChol' in df.columns:
            # Cholesterol risk categories
            df['chol_risk'] = pd.cut(df['totChol'],
                                     bins=[0, 200, 240, 1000],
                                     labels=['normal', 'borderline', 'high'])
            
            # Log transformation for skewed distribution
            df['totChol_log'] = np.log1p(df['totChol'])
            
        return df
    
    def create_smoking_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create smoking-related features"""
        logger.info("Creating smoking features...")
        
        if 'cigsPerDay' in df.columns:
            # Smoking intensity categories
            df['smoking_intensity'] = pd.cut(df['cigsPerDay'],
                                            bins=[-1, 0, 10, 20, 100],
                                            labels=['non_smoker', 'light', 'moderate', 'heavy'])
            
            # Binary smoker flag
            df['is_smoker'] = (df['cigsPerDay'] > 0).astype(int)
            
            # Pack years (if age available)
            if 'age' in df.columns:
                # Assuming started smoking at 18
                df['pack_years'] = df.apply(
                    lambda x: (x['cigsPerDay'] / 20) * max(0, x['age'] - 18) 
                    if x['cigsPerDay'] > 0 else 0, axis=1
                )
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between important variables"""
        logger.info("Creating interaction features...")
        
        # Age and smoking interaction
        if 'age' in df.columns and 'currentSmoker' in df.columns:
            df['age_smoker_interaction'] = df['age'] * df['currentSmoker']
        
        # Age and diabetes interaction
        if 'age' in df.columns and 'diabetes' in df.columns:
            df['age_diabetes_interaction'] = df['age'] * df['diabetes']
        
        # BMI and smoking interaction
        if 'BMI' in df.columns and 'currentSmoker' in df.columns:
            df['bmi_smoker_interaction'] = df['BMI'] * df['currentSmoker']
        
        # Hypertension and diabetes interaction
        if 'prevalentHyp' in df.columns and 'diabetes' in df.columns:
            df['hyp_diabetes_interaction'] = df['prevalentHyp'] * df['diabetes']
        
        # Cholesterol and age interaction
        if 'totChol' in df.columns and 'age' in df.columns:
            df['chol_age_interaction'] = df['totChol'] * df['age'] / 100
        
        return df
    
    def create_risk_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create composite risk scores"""
        logger.info("Creating composite risk scores...")
        
        # Framingham Risk Score components
        risk_score = 0
        
        if 'age' in df.columns:
            df['age_risk_score'] = df['age'].apply(
                lambda x: 0 if x < 35 else 2 if x < 45 else 4 if x < 55 else 6 if x < 65 else 8
            )
            risk_score = df['age_risk_score']
        
        if 'totChol' in df.columns:
            df['chol_risk_score'] = df['totChol'].apply(
                lambda x: 0 if x < 200 else 1 if x < 240 else 2
            )
            risk_score = risk_score + df['chol_risk_score']
        
        if 'sysBP' in df.columns:
            df['bp_risk_score'] = df['sysBP'].apply(
                lambda x: 0 if x < 120 else 1 if x < 140 else 2 if x < 160 else 3
            )
            risk_score = risk_score + df['bp_risk_score']
        
        df['composite_risk_score'] = risk_score
        
        # Metabolic syndrome indicator
        metabolic_conditions = 0
        if 'BMI' in df.columns:
            metabolic_conditions += (df['BMI'] >= 30).astype(int)
        if 'glucose' in df.columns:
            metabolic_conditions += (df['glucose'] >= 100).astype(int)
        if 'sysBP' in df.columns:
            metabolic_conditions += (df['sysBP'] >= 130).astype(int)
        
        df['metabolic_syndrome_score'] = metabolic_conditions
        
        return df
    
    def create_polynomial_features(self, df: pd.DataFrame, 
                                  columns: list = None,
                                  degree: int = 2) -> pd.DataFrame:
        """Create polynomial features for selected columns"""
        logger.info(f"Creating polynomial features of degree {degree}...")
        
        if columns is None:
            columns = ['age', 'BMI', 'sysBP', 'diaBP', 'totChol', 'glucose']
            columns = [col for col in columns if col in df.columns]
        
        if columns:
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            poly_features = poly.fit_transform(df[columns])
            
            # Get feature names
            feature_names = poly.get_feature_names_out(columns)
            
            # Create DataFrame with polynomial features
            poly_df = pd.DataFrame(poly_features, columns=feature_names, index=df.index)
            
            # Add only interaction terms (not original features)
            new_features = [col for col in poly_df.columns if col not in columns]
            for feature in new_features:
                df[f'poly_{feature}'] = poly_df[feature]
            
            self.polynomial_features = poly
        
        return df
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       method: str = 'kbest', k: int = 20) -> pd.DataFrame:
        """Select most important features"""
        logger.info(f"Selecting top {k} features using {method} method...")
        
        if method == 'kbest':
            selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
            X_selected = selector.fit_transform(X, y)
            
            # Get selected feature names
            selected_features = X.columns[selector.get_support()].tolist()
            
        elif method == 'rfe':
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            selector = RFE(estimator, n_features_to_select=min(k, X.shape[1]))
            X_selected = selector.fit_transform(X, y)
            
            # Get selected feature names
            selected_features = X.columns[selector.get_support()].tolist()
        
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        self.feature_selector = selector
        self.selected_features = selected_features
        
        logger.info(f"Selected features: {selected_features}")
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps"""
        logger.info("Starting feature engineering pipeline...")
        
        # Create various feature groups
        df = self.create_age_features(df)
        df = self.create_blood_pressure_features(df)
        df = self.create_bmi_features(df)
        df = self.create_cholesterol_features(df)
        df = self.create_smoking_features(df)
        df = self.create_interaction_features(df)
        df = self.create_risk_scores(df)
        
        # Create polynomial features for numeric columns
        numeric_cols = ['age', 'BMI', 'sysBP', 'diaBP', 'totChol']
        numeric_cols = [col for col in numeric_cols if col in df.columns]
        if numeric_cols:
            df = self.create_polynomial_features(df, numeric_cols, degree=2)
        
        logger.info(f"Feature engineering complete. Total features: {df.shape[1]}")
        
        return df
    
    def get_feature_names(self) -> list:
        """Get list of all engineered feature names"""
        return self.feature_names
