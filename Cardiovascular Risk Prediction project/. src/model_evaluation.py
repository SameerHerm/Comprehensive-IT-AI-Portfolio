import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, accuracy_score,
    precision_score, recall_score, f1_score
)
import shap
import joblib
import logging
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self):
        self.results = {}
        self.feature_importance = {}
        
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate a single model"""
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        self.results[model_name] = {
            'metrics': metrics,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        logger.info(f"{model_name} - ROC AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1_score']:.4f}")
        
        return metrics
    
    def plot_confusion_matrix(self, model_name, save_path='reports/confusion_matrices'):
        """Plot confusion matrix"""
        os.makedirs(save_path, exist_ok=True)
        
        cm = self.results[model_name]['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Disease', 'Disease'],
                   yticklabels=['No Disease', 'Disease'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'{save_path}/confusion_matrix_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_roc_curves(self, y_test, save_path='reports'):
        """Plot ROC curves for all models"""
        os.makedirs(save_path, exist_ok=True)
        
        plt.figure(figsize=(10, 8))
        
        for model_name, results in self.results.items():
            fpr, tpr, _ = roc_curve(y_test, results['y_pred_proba'])
            auc_score = results['metrics']['roc_auc']
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.6)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_path}/roc_curves_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_precision_recall_curves(self, y_test, save_path='reports'):
        """Plot Precision-Recall curves for all models"""
        os.makedirs(save_path, exist_ok=True)
        
        plt.figure(figsize=(10, 8))
        
        for model_name, results in self.results.items():
            precision, recall, _ = precision_recall_curve(y_test, results['y_pred_proba'])
            plt.plot(recall, precision, label=f'{model_name}')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_path}/precision_recall_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_interactive_dashboard(self, y_test, feature_names, save_path='reports'):
        """Create interactive Plotly dashboard"""
        os.makedirs(save_path, exist_ok=True)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Model Performance Comparison', 'ROC Curves', 
                          'Feature Importance', 'Prediction Distribution'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "histogram"}]]
        )
        
        # Model performance comparison
        models = list(self.results.keys())
        metrics_df = pd.DataFrame({
            model: self.results[model]['metrics'] 
            for model in models
        }).T
        
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
            fig.add_trace(
                go.Bar(name=metric, x=models, y=metrics_df[metric]),
                row=1, col=1
            )
        
        # ROC Curves
        for model_name, results in self.results.items():
            fpr, tpr, _ = roc_curve(y_test, results['y_pred_proba'])
            fig.add_trace(
                go.Scatter(x=fpr, y=tpr, name=f'{model_name} ROC', mode='lines'),
                row=1, col=2
            )
        
        # Add diagonal line for ROC
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                      line=dict(dash='dash', color='gray'), 
                      showlegend=False),
            row=1, col=2
        )
        
        fig.update_layout(height=800, showlegend=True, 
                         title_text="Cardiovascular Risk Prediction - Model Evaluation Dashboard")
        fig.write_html(f'{save_path}/interactive_dashboard.html')
    
    def analyze_feature_importance(self, models, feature_names, X_test):
        """Analyze feature importance using SHAP"""
        for model_name, model in models.items():
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importance = model.feature_importances_
                self.feature_importance[model_name] = dict(zip(feature_names, importance))
            
            # SHAP analysis for selected models
            if model_name in ['random_forest', 'xgboost']:
                try:
                    if model_name == 'random_forest':
                        explainer = shap.TreeExplainer(model)
                    else:
                        explainer = shap.Explainer(model)
                    
                    shap_values = explainer.shap_values(X_test[:100])  # Sample for speed
                    
                    # Save SHAP plots
                    os.makedirs('reports/shap_plots', exist_ok=True)
                    
                    plt.figure()
                    if isinstance(shap_values, list):
                        shap.summary_plot(shap_values[1], X_test[:100], 
                                        feature_names=feature_names, show=False)
                    else:
                        shap.summary_plot(shap_values, X_test[:100], 
                                        feature_names=feature_names, show=False)
                    plt.tight_layout()
                    plt.savefig(f'reports/shap_plots/shap_summary_{model_name}.png', 
                               dpi=300, bbox_inches='tight')
                    plt.close()
                    
                except Exception as e:
                    logger.warning(f"Could not generate SHAP plots for {model_name}: {e}")
    
    def generate_evaluation_report(self, y_test, feature_names):
        """Generate comprehensive evaluation report"""
        report_data = {
            'evaluation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_test_samples': len(y_test),
            'positive_samples': int(y_test.sum()),
            'negative_samples': int(len(y_test) - y_test.sum()),
            'models_evaluated': list(self.results.keys()),
            'detailed_results': {}
        }
        
        for model_name, results in self.results.items():
            report_data['detailed_results'][model_name] = {
                'metrics': results['metrics'],
                'classification_report': classification_report(
                    y_test, results['y_pred'], output_dict=True
                )
            }
        
        # Save report
        os.makedirs('reports', exist_ok=True)
        import json
        with open('reports/evaluation_report.json', 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Create markdown report
        with open('reports/evaluation_report.md', 'w') as f:
            f.write("# Cardiovascular Risk Prediction - Model Evaluation Report\n\n")
            f.write(f"**Evaluation Date:** {report_data['evaluation_date']}\n\n")
            f.write(f"**Dataset Info:**\n")
            f.write(f"- Total samples: {report_data['total_test_samples']}\n")
            f.write(f"- Positive samples: {report_data['positive_samples']}\n")
            f.write(f"- Negative samples: {report_data['negative_samples']}\n\n")
            
            f.write("## Model Performance Summary\n\n")
            f.write("| Model | Accuracy | Precision | Recall | F1-Score | ROC AUC |\n")
            f.write("|-------|----------|-----------|--------|----------|----------|\n")
            
            for model_name, results in self.results.items():
                metrics = results['metrics']
                f.write(f"| {model_name} | {metrics['accuracy']:.4f} | "
                       f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | "
                       f"{metrics['f1_score']:.4f} | {metrics['roc_auc']:.4f} |\n")
        
        logger.info("Evaluation report generated")
    
    def evaluate_all_models(self):
        """Complete evaluation pipeline"""
        # Load data and models
        data = joblib.load('data/processed/processed_data.pkl')
        X_test, y_test = data['X_test'], data['y_test']
        feature_names = data['feature_names']
        
        # Load trained models
        models = {}
        model_files = [f for f in os.listdir('models') if f.endswith('.pkl') and 'model' in f]
        
        for file in model_files:
            if 'best_model' in file:
                continue
            model_name = file.split('_model_')[0]
            models[model_name] = joblib.load(f'models/{file}')
        
        # Evaluate each model
        for model_name, model in models.items():
            self.evaluate_model(model, X_test, y_test, model_name)
        
        # Generate visualizations
        for model_name in self.results.keys():
            self.plot_confusion_matrix(model_name)
        
        self.plot_roc_curves(y_test)
        self.plot_precision_recall_curves(y_test)
        self.create_interactive_dashboard(y_test, feature_names)
        
        # Feature importance analysis
        self.analyze_feature_importance(models, feature_names, X_test)
        
        # Generate comprehensive report
        self.generate_evaluation_report(y_test, feature_names)
        
        logger.info("Model evaluation completed")
        return self.results

if __name__ == "__main__":
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_all_models()
