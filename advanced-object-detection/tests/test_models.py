"""
Tests for object detection models
"""

import unittest
import torch
import numpy as np
from src.models.yolo import YOLODetector
from src.models.rcnn import FasterRCNN
from src.models.ssd import SSDDetector

class TestObjectDetectionModels(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.num_classes = 20
        self.batch_size = 2
        self.image_size = 640
        
    def test_yolo_model(self):
        """Test YOLO model initialization and forward pass"""
        model = YOLODetector(num_classes=self.num_classes)
        
        # Create dummy input
        x = torch.randn(self.batch_size, 3, self.image_size, self.image_size)
        
        # Forward pass
        output = model(x)
        
        # Check output
        self.assertIsNotNone(output)
        self.assertEqual(len(output), 3)  # Should have 3 scales
        
    def test_faster_rcnn_model(self):
        """Test Faster R-CNN model"""
        model = FasterRCNN(num_classes=self.num_classes)
        
        # Create dummy input
        images = [torch.randn(3, 300, 400) for _ in range(self.batch_size)]
        
        # Test inference mode
        model.eval()
        with torch.no_grad():
            outputs = model(images)
        
        # Check outputs
        self.assertEqual(len(outputs), self.batch_size)
        for output in outputs:
            self.assertIn('boxes', output)
            self.assertIn('labels', output)
            self.assertIn('scores', output)
    
    def test_ssd_model(self):
        """Test SSD model"""
        model = SSDDetector(num_classes=self.num_classes)
        
        # Create dummy input
        x = torch.randn(self.batch_size, 3, 300, 300)
        
        # Forward pass
        loc, conf = model(x)
        
        # Check outputs
        self.assertEqual(loc.shape[0], self.batch_size)
        self.assertEqual(conf.shape[0], self.batch_size)
        self.assertEqual(conf.shape[2], self.num_classes)

if __name__ == '__main__':
    unittest.main()
