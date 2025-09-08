#!/usr/bin/env python3
"""
Script to train object detection models
"""

import argparse
import yaml
from pathlib import Path
import torch
from src.training.trainer import ObjectDetectionTrainer
from src.models.yolo import YOLODetector

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description='Train object detection model')
    parser.add_argument('--model', type=str, default='yolo',
                       choices=['yolo', 'rcnn', 'ssd', 'efficientdet'],
                       help='Model architecture')
    parser.add_argument('--config', type=str, 
                       default='config/model_configs/yolo_config.yaml',
                       help='Configuration file')
    parser.add_argument('--data', type=str, default='data',
                       help='Dataset directory')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    config['training']['epochs'] = args.epochs
    config['training']['batch_size'] = args.batch_size
    
    # Initialize model
    if args.model == 'yolo':
        model = YOLODetector(num_classes=config['model']['num_classes'])
    
    # Initialize trainer
    trainer = ObjectDetectionTrainer(model, config)
    
    # Load checkpoint if resuming
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    print(f"Starting training for {args.epochs} epochs...")
    # trainer.train(train_loader, val_loader, args.epochs)
    
    print("Training complete!")

if __name__ == '__main__':
    main()
