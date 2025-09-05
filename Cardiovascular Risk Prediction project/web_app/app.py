import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import requests
import json
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Cardiovascular Risk Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #e74c3c;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #e74c3c;
        margin: 1rem 0;
    }
    .risk-low {
        color: #27ae60;
        font-weight: bold;
    }
    .risk-medium {
        color: #f39c12;
        font-weight: bold;
    }
    .risk-high {
        color: #e74c3c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class CardiovascularRiskApp:
    def __init__(self):
        self.api_url = "http://localhost:5000"  # Update if API is hosted elsewhere
        
    def main(self):
        st.markdown('<h1 class="main-header">❤️ Cardiovascular Risk Prediction System</h1>', 
                   unsafe_allow_html=True)
        
        # Sidebar navigation
        page = st.sidebar.selectbox(
            "Navigate to:",
            ["Individual Prediction", "Batch Prediction", "Model Analytics", "About"]
        )
        
        if page == "Individual Prediction":
            self.individual_prediction_page()
        elif page == "Batch Prediction":
            self.batch_prediction_page()
        elif page == "Model Analytics":
            self.analytics_page()
        else:
            self.about_page()
    
    def individual_prediction_page(self):
        st.header("Individual Risk Assessment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Basic Information")
            age = st.slider("Age", 18, 100, 45)
            gender = st.selectbox("Gender", ["Male", "Female"])
            height = st.slider("Height (cm)", 120, 220, 170)
            weight = st.slider("Weight (kg)", 30, 200, 70)
            
            st.subheader("Vital Signs")
            systolic_bp = st.slider("Systolic Blood Pressure", 70, 250, 120)
            diastolic_bp = st.slider("Diastolic Blood Pressure", 40, 150, 80)
        
        with col2:
            st.subheader("Health Indicators")
            cholesterol = st.selectbox("Cholesterol Level", 
                                     ["Normal", "Above Normal", "Well Above Normal"])
            glucose = st.selectbox("Glucose Level", 
                                 ["Normal", "Above Normal", "Well Above Normal"])
            
            st.subheader("Lifestyle Factors")
            smoking = st.selectbox("Smoking", ["No", "Yes"])
            alcohol = st.selectbox("Alcohol Consumption", ["No", "Yes"])
            physical_activity = st.selectbox("Regular Physical Activity", ["No", "Yes"])
        
        # Convert inputs to API format
        input_data = {
            "age": age,
            "gender": 1 if gender == "Male" else 0,
            "height": height,
            "weight": weight,
            "systolic_bp": systolic_bp,
            "diastolic_bp": diastolic_bp,
            "cholesterol": {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}[cholesterol],
            "glucose": {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}[glucose],
            "smoking": 1 if smoking == "Yes" else 0,
            "alcohol": 1 if alcohol == "Yes" else 0,
            "physical_activity": 1 if physical_activity == "Yes" else 0
        }
        
        if st.button("Predict Risk", type="primary"):
            self.make_prediction(input_data)
    
    def make_prediction(self, input_data):
        try:
            # Try API first
            response = requests.post(f"{self.api_url}/predict", json=input_data, timeout=10)
            if response.status_code == 200:
                result = response.json()
                self.display_prediction_result(result)
            else:
                st.error("API request failed. Using local model...")
                self.local_prediction(input_data)
        except:
            st.warning("API not available. Using local model...")
            self.local_prediction(input_data)
    
    def local_prediction(self, input_data):
        """Fallback to local model if API is not available"""
        try:
            # Load local model (simplified version)
            model = joblib.load('models/best_model.pkl')
            
            # Simple preprocessing (you might need to adjust this)
            df = pd.DataFrame([input_data])
            prediction_proba = model.predict_proba(df)[0][1]
            
            result = {
                'risk_probability': prediction_proba,
                'risk_level': 'High' if prediction_proba > 0.7 else 'Medium' if prediction_proba > 0.3 else 'Low',
                'prediction': 1 if prediction_proba > 0.5 else 0
            }
            
            self.display_prediction_result(result)
        except Exception as e:
            st.error(f"Error making prediction: {e}")
    
    def display_prediction_result(self, result):
        st.markdown("---")
        st.subheader("Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            risk_prob = result['risk_probability']
            st.metric("Risk Probability", f"{risk_prob:.2%}")
            
            # Create gauge chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = risk_prob * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Risk Score"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            risk_level = result['risk_level']
            risk_class = f"risk-{risk_level.lower()}"
            st.markdown(f'<p class="{risk_class}">Risk Level: {risk_level}</p>', 
                       unsafe_allow_html=True)
            
            if risk_level == "Low":
                st.success("Low risk of cardiovascular disease")
                recommendations = [
                    "Maintain current healthy lifestyle",
                    "Regular check-ups with healthcare provider",
                    "Continue balanced diet and exercise"
                ]
            elif risk_level == "Medium":
                st.warning("Moderate risk of cardiovascular disease")
                recommendations = [
                    "Consult with healthcare provider",
                    "Consider lifestyle modifications",
                    "Monitor blood pressure and cholesterol regularly"
                ]
            else:
                st.error("High risk of cardiovascular disease")
                recommendations = [
                    "Seek immediate medical consultation",
                    "Consider comprehensive health evaluation",
                    "Implement strict lifestyle changes"
                ]
            
            st.subheader("Recommendations:")
            for rec in recommendations:
                st.write(f"• {rec}")
        
        with col3:
            st.subheader("Risk Factors Analysis")
            
            # Create risk factors visualization
            risk_factors = {
                'Age': min(100, max(0, (input_data['age'] - 30) * 2)),
                'Blood Pressure': min(100, max(0, (input_data['systolic_bp'] - 120) * 2)),
                'Lifestyle': (input_data['smoking'] * 30 + (1-input_data['physical_activity']) * 20),
                'Health Markers': (input_data['cholesterol'] - 1) * 25 + (input_data['glucose'] - 1) * 25
            }
            
            fig = px.bar(
                x=list(risk_factors.keys()),
                y=list(risk_factors.values()),
                title="Risk Factor Contributions",
                color=list(risk_factors.values()),
                color_continuous_scale="Reds"
            )
            
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    def batch_prediction_page(self):
        st.header("Batch Risk Assessment")
        
        st.write("Upload a CSV file with patient data for batch prediction.")
        
        # File upload
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("Data preview:")
                st.dataframe(df.head())
                
                if st.button("Process Batch Predictions"):
                    # Convert dataframe to list of dictionaries
                    data_list = df.to_dict('records')
                    
                    try:
                        response = requests.post(f"{self.api_url}/predict/batch", 
                                               json=data_list, timeout=30)
                        if response.status_code == 200:
                            results = response.json()
                            
                            # Create results dataframe
                            results_df = pd.DataFrame(results)
                            
                            # Display results
                            st.subheader("Batch Prediction Results")
                            st.dataframe(results_df)
                            
                            # Download button
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="Download Results as CSV",
                                data=csv,
                                file_name=f"cardiovascular_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                            
                            # Summary statistics
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                high_risk_count = sum(1 for r in results if r.get('risk_level') == 'High')
                                st.metric("High Risk Patients", high_risk_count)
                            
                            with col2:
                                medium_risk_count = sum(1 for r in results if r.get('risk_level') == 'Medium')
                                st.metric("Medium Risk Patients", medium_risk_count)
                            
                            with col3:
                                low_risk_count = sum(1 for r in results if r.get('risk_level') == 'Low')
                                st.metric("Low Risk Patients", low_risk_count)
                        
                        else:
                            st.error("Batch prediction failed")
                    
                    except Exception as e:
                        st.error(f"Error during batch prediction: {e}")
            
            except Exception as e:
                st.error(f"Error reading file: {e}")
    
    def analytics_page(self):
        st.header("Model Analytics Dashboard")
        
        # Model information
        try:
            response = requests.get(f"{self.api_url}/model/info", timeout=10)
            if response.status_code == 200:
                model_info = response.json()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Model Information")
                    st.write(f"**Model Type:** {model_info.get('model_type', 'Unknown')}")
                    st.write(f"**Feature Count:** {model_info.get('feature_count', 'Unknown')}")
                
                with col2:
                    st.subheader("Features Used")
                    features = model_info.get('features', [])
                    for feature in features[:10]:  # Show first 10 features
                        st.write(f"• {feature}")
                    if len(features) > 10:
                        st.write(f"... and {len(features) - 10} more")
        
        except:
            st.warning("Could not fetch model information from API")
        
        # Sample analytics visualizations
        st.subheader("Risk Distribution Analysis")
        
        # Generate sample data for visualization
        np.random.seed(42)
        sample_data = {
            'Age Group': ['18-30', '31-45', '46-60', '61-75', '75+'] * 100,
            'Risk Level': np.random.choice(['Low', 'Medium', 'High'], 500, p=[0.6, 0.25, 0.15])
        }
        
        sample_df = pd.DataFrame(sample_data)
        
        # Risk distribution by age group
        fig = px.histogram(sample_df, x='Age Group', color='Risk Level', 
                          title="Risk Distribution by Age Group")
        st.plotly_chart(fig, use_container_width=True)
    
    def about_page(self):
        st.header("About Cardiovascular Risk Prediction System")
        
        st.markdown("""
        ## Overview
        This system uses machine learning to predict cardiovascular disease risk based on various health indicators and lifestyle factors.
        
        ## Features
        - **Individual Predictions**: Get personalized risk assessments
        - **Batch Processing**: Analyze multiple patients at once
        - **Interactive Dashboard**: Visualize risk factors and model performance
        - **Real-time API**: RESTful API for integration with other systems
        
        ## Model Information
        The system uses ensemble learning with multiple algorithms including:
        - Random Forest
        - XGBoost
        - LightGBM
        - Logistic Regression
        - Neural Networks
        
        ## Risk Factors Considered
        - Age and gender
        - Blood pressure (systolic and diastolic)
        - Cholesterol levels
        - Glucose levels
        - Smoking status
        - Alcohol consumption
        - Physical activity level
        - BMI and other derived metrics
        
        ## Disclaimer
        This tool is for educational and research purposes only. Always consult with healthcare professionals for medical decisions.
        """)

if __name__ == "__main__":
    app = CardiovascularRiskApp()
    app.main()
