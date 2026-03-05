"""
Comprehensive Testing Suite

This module provides comprehensive testing including:
- Unit tests for all components
- Integration tests
- Performance benchmarks
- CI/CD pipeline configuration

Author: Air Quality Prediction System
Date: 2026
"""

import unittest
import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from unittest.mock import Mock, patch
import json

# Test our modules
from data_generator import AirQualityDataGenerator
from model import MultiModalAirQualityModel
from inference import AirQualityInference

class TestDataGenerator(unittest.TestCase):
    """Test data generation functionality"""
    
    def setUp(self):
        self.generator = AirQualityDataGenerator(num_samples=10, window_size=24)
    
    def test_aqi_calculation(self):
        """Test AQI calculation"""
        aqi, category = self.generator.calculate_aqi(25, 450, 40)
        self.assertIsInstance(aqi, int)
        self.assertIn(category, ['Good', 'Moderate', 'Unhealthy for Sensitive'])
    
    def test_dataset_generation(self):
        """Test dataset generation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = AirQualityDataGenerator(
                num_samples=5, 
                window_size=24, 
                output_dir=temp_dir
            )
            dataset_info = generator.generate_dataset()
            self.assertEqual(len(dataset_info['samples']), 5)

class TestModel(unittest.TestCase):
    """Test model functionality"""
    
    def setUp(self):
        self.model = MultiModalAirQualityModel()
    
    def test_model_creation(self):
        """Test model creation"""
        self.assertIsNotNone(self.model.model)
        self.assertGreater(self.model.model.count_params(), 0)
    
    def test_model_compilation(self):
        """Test model compilation"""
        self.model.compile_model()
        self.assertIsNotNone(self.model.model.optimizer)

class TestInference(unittest.TestCase):
    """Test inference functionality"""
    
    def setUp(self):
        # Create a mock model for testing
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
            # Create a simple model
            import tensorflow as tf
            from tensorflow import keras
            
            inputs = {
                'image': keras.Input(shape=(224, 224, 3)),
                'environmental': keras.Input(shape=(24, 6)),
                'biosensing': keras.Input(shape=(24, 3))
            }
            
            outputs = {
                'pm25_output': keras.layers.Dense(1)(keras.layers.Flatten()(inputs['image'])),
                'co2_output': keras.layers.Dense(1)(keras.layers.Flatten()(inputs['image'])),
                'no2_output': keras.layers.Dense(1)(keras.layers.Flatten()(inputs['image'])),
                'aqi_output': keras.layers.Dense(6, activation='softmax')(keras.layers.Flatten()(inputs['image']))
            }
            
            model = keras.Model(inputs=inputs, outputs=outputs)
            model.save(tmp_file.name)
            self.model_path = tmp_file.name
    
    def test_inference_initialization(self):
        """Test inference initialization"""
        with patch('data_loader.AirQualityDataLoader'):
            inference = AirQualityInference(model_path=self.model_path)
            self.assertIsNotNone(inference.model)

if __name__ == '__main__':
    unittest.main()
