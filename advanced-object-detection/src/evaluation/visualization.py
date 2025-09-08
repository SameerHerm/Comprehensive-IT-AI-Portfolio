"""
Visualization utilities for evaluation
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional
import torch

def plot_precision_recall(precisions: List[float], recalls: List[float],
                         ap: float = None, class_name: str = None,
                         save_path: str = None):
    """Plot precision-recall curve"""
    plt.figure(figsize=(10, 6))
    
    plt.plot(recalls, precisions, 'b-', linewidth=2)
    plt.fill_between(recalls, precisions, alpha=0.2)
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    
    title = 'Precision-Recall Curve'
    if class_name:
        title += f' - {class_name}'
    if ap is not None:
        title += f' (AP = {ap:.3f})'
    
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    plt.show()

def plot_confusion_matrix(confusion_matrix: np.ndarray,
                         class_names: List[str] = None,
                         normalize: bool = True,
                         save_path: str = None):
    """Plot confusion matrix"""
    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(12, 10))
    
    sns.heatmap(confusion_matrix, annot=True, fmt='.2f' if normalize else 'd',
                cmap='Blues', square=True,
                xticklabels=class_names,
                yticklabels=class_names)
    
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''), fontsize=14)
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    plt.show()

def plot_loss_curves(train_losses: List[float], val_losses: List[float] = None,
                    save_path: str = None):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(12, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    if val_losses:
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Progress', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    plt.show()

def plot_learning_rate(learning_rates: List[float], save_path: str = None):
    """Plot learning rate schedule"""
    plt.figure(figsize=(10, 6))
    
    plt.plot(learning_rates, linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Learning Rate Schedule', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    plt.show()

def visualize_predictions(image: np.ndarray, predictions: Dict,
                         targets: Dict = None, class_names: List[str] = None,
                         threshold: float = 0.5, save_path: str = None):
    """Visualize predictions and ground truth"""
    fig, axes = plt.subplots(1, 2 if targets else 1, figsize=(15, 8))
    
    if not targets:
        axes = [axes]
    
    # Plot predictions
    ax = axes[0]
    ax.imshow(image)
    ax.set_title('Predictions', fontsize=14)
    ax.axis('off')
    
    # Draw prediction boxes
    for box, label, score in zip(predictions['boxes'], 
                                 predictions['labels'],
                                 predictions['scores']):
        if score < threshold:
            continue
        
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                            linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        
        label_text = f"{class_names[label] if class_names else label}: {score:.2f}"
        ax.text(x1, y1 - 5, label_text, color='red', fontsize=10,
               bbox=dict(facecolor='white', alpha=0.7))
    
    # Plot ground truth if provided
    if targets:
        ax = axes[1]
        ax.imshow(image)
        ax.set_title('Ground Truth', fontsize=14)
        ax.axis('off')
        
        for box, label in zip(targets['boxes'], targets['labels']):
            x1, y1, x2, y2 = box
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                linewidth=2, edgecolor='green', facecolor='none')
            ax.add_patch(rect)
            
            label_text = class_names[label] if class_names else str(label)
            ax.text(x1, y1 - 5, label_text, color='green', fontsize=10,
                   bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    plt.show()
