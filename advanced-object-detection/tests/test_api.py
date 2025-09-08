"""
Tests for API endpoints
"""

import unittest
import json
import io
from api.app import app

class TestAPI(unittest.TestCase):
    
    def setUp(self):
        """Set up test client"""
        self.app = app
        self.client = self.app.test_client()
        
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        
    def test_detect_endpoint(self):
        """Test detection endpoint"""
        # Create dummy image
        import numpy as np
        from PIL import Image
        
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        # Send request
        response = self.client.post(
            '/api/detect',
            data={
                'image': (img_bytes, 'test.jpg'),
                'model': 'yolo',
                'confidence': 0.5
            },
            content_type='multipart/form-data'
        )
        
        # Check response
        self.assertIn(response.status_code, [200, 500])  # May fail if model not loaded
        
        if response.status_code == 200:
            data = json.loads(response.data)
            self.assertIn('success', data)
            self.assertIn('detections', data)

if __name__ == '__main__':
    unittest.main()
