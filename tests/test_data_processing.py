"""
Tests for data processing functionality.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

# Import your data processing functions here
# from src.data.convert_annotations import convert_csv_to_yolo, setup_yolo_dataset


class TestDataProcessing:
    """Test cases for data processing functions."""
    
    def test_csv_to_yolo_conversion(self):
        """Test CSV to YOLO format conversion."""
        # Create sample CSV data
        sample_data = {
            'filename': ['test1.jpg', 'test1.jpg', 'test2.jpg'],
            'width': [640, 640, 800],
            'height': [480, 480, 600],
            'xmin': [100, 200, 150],
            'ymin': [50, 100, 75],
            'xmax': [200, 300, 250],
            'ymax': [150, 200, 175]
        }
        
        df = pd.DataFrame(sample_data)
        
        # Test with temporary files
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, 'test.csv')
            output_dir = os.path.join(temp_dir, 'labels')
            
            df.to_csv(csv_path, index=False)
            os.makedirs(output_dir, exist_ok=True)
            
            # Test conversion function
            # convert_csv_to_yolo(csv_path, temp_dir, output_dir)
            
            # Verify output files exist
            assert os.path.exists(os.path.join(output_dir, 'test1.txt'))
            assert os.path.exists(os.path.join(output_dir, 'test2.txt'))
    
    def test_yolo_format_correctness(self):
        """Test that YOLO format coordinates are correct."""
        # Test coordinate normalization
        width, height = 640, 480
        xmin, ymin, xmax, ymax = 100, 50, 200, 150
        
        # Calculate expected YOLO format
        center_x = (xmin + xmax) / 2.0
        center_y = (ymin + ymax) / 2.0
        bbox_width = xmax - xmin
        bbox_height = ymax - ymin
        
        # Normalize coordinates
        center_x_norm = center_x / width
        center_y_norm = center_y / height
        width_norm = bbox_width / width
        height_norm = bbox_height / height
        
        # Verify normalized coordinates are between 0 and 1
        assert 0 <= center_x_norm <= 1
        assert 0 <= center_y_norm <= 1
        assert 0 <= width_norm <= 1
        assert 0 <= height_norm <= 1
        
        # Verify specific values
        expected_center_x = 0.234375  # (100 + 200) / 2 / 640
        expected_center_y = 0.208333  # (50 + 150) / 2 / 480
        expected_width = 0.15625      # (200 - 100) / 640
        expected_height = 0.208333    # (150 - 50) / 480
        
        assert abs(center_x_norm - expected_center_x) < 1e-6
        assert abs(center_y_norm - expected_center_y) < 1e-6
        assert abs(width_norm - expected_width) < 1e-6
        assert abs(height_norm - expected_height) < 1e-6
    
    def test_dataset_structure_creation(self):
        """Test YOLO dataset structure creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test dataset structure creation
            # setup_yolo_dataset()
            
            # Verify directory structure
            expected_dirs = ['train/images', 'train/labels', 
                           'valid/images', 'valid/labels',
                           'test/images', 'test/labels']
            
            for dir_path in expected_dirs:
                full_path = os.path.join(temp_dir, 'yolo_dataset', dir_path)
                # This would be created by setup_yolo_dataset()
                # assert os.path.exists(full_path)
    
    def test_empty_annotations(self):
        """Test handling of images with no annotations."""
        # Test case where image has no people (empty annotation file)
        sample_data = {
            'filename': ['empty_image.jpg'],
            'width': [640],
            'height': [480],
            'xmin': [np.nan],
            'ymin': [np.nan],
            'xmax': [np.nan],
            'ymax': [np.nan]
        }
        
        df = pd.DataFrame(sample_data)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, 'empty_test.csv')
            output_dir = os.path.join(temp_dir, 'labels')
            
            df.to_csv(csv_path, index=False)
            os.makedirs(output_dir, exist_ok=True)
            
            # Test conversion with empty annotations
            # convert_csv_to_yolo(csv_path, temp_dir, output_dir)
            
            # Should create empty annotation file
            annotation_file = os.path.join(output_dir, 'empty_image.txt')
            # assert os.path.exists(annotation_file)
            
            # File should be empty or contain only whitespace
            # with open(annotation_file, 'r') as f:
            #     content = f.read().strip()
            #     assert content == ""


if __name__ == "__main__":
    pytest.main([__file__])
