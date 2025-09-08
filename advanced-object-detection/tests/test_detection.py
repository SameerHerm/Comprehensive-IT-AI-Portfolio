"""
Tests for detection functionality
"""

import unittest
import numpy as np
import torch
from src.detection.utils import (
    non_max_suppression,
    calculate_iou,
    resize_boxes,
    clip_boxes
)

class TestDetectionUtils(unittest.TestCase):
    
    def test_non_max_suppression(self):
        """Test NMS functionality"""
        boxes = torch.tensor([
            [10, 10, 50, 50],
            [15, 15, 55, 55],  # Overlapping
            [100, 100, 150, 150],  # Non-overlapping
        ]).float()
        
        scores = torch.tensor([0.9, 0.8, 0.7])
        
        keep = non_max_suppression(boxes, scores, iou_threshold=0.5)
        
        # Should keep first and third box
        self.assertEqual(len(keep), 2)
        
    def test_calculate_iou(self):
        """Test IoU calculation"""
        boxes1 = torch.tensor([[10, 10, 50, 50]]).float()
        boxes2 = torch.tensor([[20, 20, 60, 60]]).float()
        
        iou = calculate_iou(boxes1, boxes2)
        
        self.assertEqual(iou.shape, (1, 1))
        self.assertGreater(iou[0, 0], 0)
        self.assertLess(iou[0, 0], 1)
    
    def test_resize_boxes(self):
        """Test box resizing"""
        boxes = torch.tensor([[10, 10, 50, 50]]).float()
        original_size = (100, 100)
        new_size = (200, 200)
        
        resized = resize_boxes(boxes, original_size, new_size)
        
        # Boxes should be scaled by 2x
        expected = torch.tensor([[20, 20, 100, 100]]).float()
        torch.testing.assert_close(resized, expected)
    
    def test_clip_boxes(self):
        """Test box clipping"""
        boxes = torch.tensor([
            [-10, -10, 50, 50],
            [90, 90, 110, 110]  # Extends beyond image
        ]).float()
        
        image_size = (100, 100)
        clipped = clip_boxes(boxes, image_size)
        
        # Check clipping
        self.assertEqual(clipped[0, 0], 0)  # Negative x clipped to 0
        self.assertEqual(clipped[0, 1], 0)  # Negative y clipped to 0
        self.assertEqual(clipped[1, 2], 100)  # x2 clipped to image width
        self.assertEqual(clipped[1, 3], 100)  # y2 clipped to image height

if __name__ == '__main__':
    unittest.main()
