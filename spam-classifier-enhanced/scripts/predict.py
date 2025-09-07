#!/usr/bin/env python
"""
Prediction script for spam classifier
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import joblib
from api.predictor import SpamPredictor

def main(args):
    """Main prediction function"""
    
    # Initialize predictor
    predictor = SpamPredictor(model_dir='models')
    
    if args.file:
        # Read text from file
        with open(args.file, 'r') as f:
            text = f.read()
    else:
        text = args.text
    
    if not text:
        print("Error: No text provided")
        return
    
    # Make prediction
    result = predictor.predict(text, args.model)
    
    # Display results
    print("\n" + "="*50)
    print("SPAM CLASSIFICATION RESULT")
    print("="*50)
    print(f"Text: {text[:200]}..." if len(text) > 200 else f"Text: {text}")
    print(f"Model: {args.model}")
    print(f"Prediction: {result['prediction'].upper()}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Is Spam: {'Yes' if result['is_spam'] else 'No'}")
    print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict spam for given text')
    parser.add_argument('--text', type=str, help='Text to classify')
    parser.add_argument('--file', type=str, help='File containing text to classify')
    parser.add_argument('--model', type=str, default='xgboost',
                      choices=['naive_bayes', 'svm', 'random_forest', 'xgboost'],
                      help='Model to use for prediction')
    
    args = parser.parse_args()
    
    if not args.text and not args.file:
        parser.error('Either --text or --file must be provided')
    
    main(args)
