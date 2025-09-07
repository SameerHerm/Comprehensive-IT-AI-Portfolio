import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve
)
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation and visualization"""
    
    def __init__(self):
        self.results = {}
        
    def evaluate_model(self, model: Any, X_test, y_test, model_name: str) -> Dict:
        """Evaluate a single model"""
        y_pred = model.predict(X_test)
        y_pred_proba = None
        
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred),
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        self.results[model_name] = metrics
        logger.info(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}")
        
        return metrics
    
    def evaluate_all_models(self, models: Dict, X_test, y_test) -> pd.DataFrame:
        """Evaluate all models and return comparison"""
        comparison = []
        
        for model_name, model in models.items():
            metrics = self.evaluate_model(model, X_test, y_test, model_name)
            comparison.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score']
            })
        
        comparison_df = pd.DataFrame(comparison)
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        
        return comparison_df
    
    def plot_confusion_matrix(self, model_name: str, save_path: str = None):
        """Plot confusion matrix"""
        if model_name not in self.results:
            raise ValueError(f"Model {model_name} not evaluated yet")
        
        cm = self.results[model_name]['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Ham', 'Spam'],
                   yticklabels=['Ham', 'Spam'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_roc_curves(self, models: Dict, X_test, y_test, save_path: str = None):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))
        
        for model_name, model in models.items():
            if hasattr(model, 'predict_proba'):
                y_score = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_score)
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_precision_recall_curves(self, models: Dict, X_test, y_test):
        """Plot precision-recall curves"""
        plt.figure(figsize=(10, 8))
        
        for model_name, model in models.items():
            if hasattr(model, 'predict_proba'):
                y_score = model.predict_proba(X_test)[:, 1]
                precision, recall, _ = precision_recall_curve(y_test, y_score)
                
                plt.plot(recall, precision, label=model_name)
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def create_interactive_comparison(self, comparison_df: pd.DataFrame):
        """Create interactive comparison chart using Plotly"""
        fig = go.Figure()
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for metric in metrics:
            fig.add_trace(go.Bar(
                name=metric,
                x=comparison_df['Model'],
                y=comparison_df[metric],
                text=comparison_df[metric].round(3),
                textposition='auto',
            ))
        
        fig.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Model',
            yaxis_title='Score',
            barmode='group',
            height=500,
            hovermode='x unified'
        )
        
        return fig
    
    def generate_report(self, output_path: str = 'evaluation_report.html'):
        """Generate comprehensive HTML report"""
        html_content = """
        <html>
        <head>
            <title>Spam Classifier Evaluation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #333; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
                th { background-color: #4CAF50; color: white; }
                tr:hover { background-color: #f5f5f5; }
            </style>
        </head>
        <body>
            <h1>Spam Classifier Evaluation Report</h1>
            <h2>Model Performance Summary</h2>
        """
        
        # Add metrics table
        if self.results:
            html_content += "<table>"
            html_content += "<tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1-Score</th></tr>"
            
            for model_name, metrics in self.results.items():
                html_content += f"""
                <tr>
                    <td>{model_name}</td>
                    <td>{metrics['accuracy']:.4f}</td>
                    <td>{metrics['precision']:.4f}</td>
                    <td>{metrics['recall']:.4f}</td>
                    <td>{metrics['f1_score']:.4f}</td>
                </tr>
                """
            
            html_content += "</table>"
        
        html_content += """
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Report saved to {output_path}")
