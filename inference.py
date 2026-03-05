"""
Inference Script for Multi-Modal Air Quality Prediction

This script handles model inference including:
- Loading trained models
- Processing new data samples
- Making predictions
- Interpreting AQI results
- Risk assessment and recommendations

Author: Air Quality Prediction System
Date: 2026
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse
from typing import Dict, List, Tuple, Optional, Union
import json

import tensorflow as tf
from PIL import Image

from data_loader import AirQualityDataLoader
from model import MultiModalAirQualityModel

class AirQualityInference:
    """
    Inference engine for air quality prediction
    """
    
    def __init__(self, 
                 model_path: str,
                 data_dir: str = "data",
                 image_size: Tuple[int, int] = (224, 224),
                 window_size: int = 24):
        """
        Initialize inference engine
        
        Args:
            model_path: Path to trained model file
            data_dir: Directory containing dataset (for scalers)
            image_size: Input image size
            window_size: Time window size
        """
        self.model_path = model_path
        self.data_dir = data_dir
        self.image_size = image_size
        self.window_size = window_size
        
        # Load model and data loader
        self._load_model()
        self._load_data_loader()
        
        # AQI categories and risk levels
        self.aqi_categories = ['Good', 'Moderate', 'Unhealthy for Sensitive', 
                              'Unhealthy', 'Very Unhealthy', 'Hazardous']
        
        self.risk_recommendations = {
            'Good': {
                'color': 'green',
                'description': 'Air quality is satisfactory',
                'recommendations': [
                    'Enjoy outdoor activities',
                    'Open windows for ventilation',
                    'No health precautions needed'
                ]
            },
            'Moderate': {
                'color': 'yellow',
                'description': 'Air quality is acceptable',
                'recommendations': [
                    'Unusually sensitive people should consider limiting prolonged outdoor exertion',
                    'Generally safe for outdoor activities',
                    'Monitor if you have respiratory conditions'
                ]
            },
            'Unhealthy for Sensitive': {
                'color': 'orange',
                'description': 'Members of sensitive groups may experience health effects',
                'recommendations': [
                    'Children, elderly, and people with heart/lung disease should limit prolonged outdoor exertion',
                    'Consider wearing masks outdoors',
                    'Keep windows closed during high pollution hours'
                ]
            },
            'Unhealthy': {
                'color': 'red',
                'description': 'Everyone may begin to experience health effects',
                'recommendations': [
                    'Limit outdoor activities',
                    'Wear N95 masks when outdoors',
                    'Use air purifiers indoors',
                    'Avoid prolonged exertion'
                ]
            },
            'Very Unhealthy': {
                'color': 'purple',
                'description': 'Health warnings of emergency conditions',
                'recommendations': [
                    'Avoid all outdoor activities',
                    'Stay indoors with air purifiers',
                    'Wear high-quality masks if going outside is necessary',
                    'Seek medical attention if experiencing symptoms'
                ]
            },
            'Hazardous': {
                'color': 'maroon',
                'description': 'Emergency conditions - entire population affected',
                'recommendations': [
                    'Remain indoors at all times',
                    'Use multiple air purifiers',
                    'Seal windows and doors',
                    'Seek immediate medical attention for any symptoms',
                    'Consider evacuation if advised by authorities'
                ]
            }
        }
    
    def _load_model(self):
        """Load trained model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Create model instance
        self.model = MultiModalAirQualityModel(
            image_size=self.image_size,
            window_size=self.window_size
        )
        
        # Load weights
        self.model.load_model(self.model_path)
        print(f"Model loaded from {self.model_path}")
    
    def _load_data_loader(self):
        """Load data loader for preprocessing"""
        self.data_loader = AirQualityDataLoader(
            data_dir=self.data_dir,
            image_size=self.image_size,
            window_size=self.window_size,
            batch_size=1,
            shuffle=False
        )
    
    def predict_from_sample(self, sample_id: int) -> Dict:
        """
        Make prediction for a sample from the dataset
        
        Args:
            sample_id: Sample ID from the dataset
            
        Returns:
            Prediction results with interpretation
        """
        # Get sample data
        sample_data = self.data_loader.get_sample_for_inference(sample_id)
        
        # Make prediction
        predictions = self._predict_raw(sample_data)
        
        # Inverse transform to original scale
        predicted_values = self.data_loader.inverse_transform_labels(predictions)
        
        # Calculate AQI from predicted pollutants
        calculated_aqi, _ = self._calculate_aqi(
            predicted_values['pm25'],
            predicted_values['co2'],
            predicted_values['no2']
        )
        
        # Interpret results
        interpretation = self._interpret_results(
            predicted_values,
            calculated_aqi,
            sample_data['sample_info']
        )
        
        return interpretation
    
    def predict_from_files(self,
                          image_path: str,
                          env_sensor_path: str,
                          bio_sensor_path: str) -> Dict:
        """
        Make prediction from external data files
        
        Args:
            image_path: Path to environmental image
            env_sensor_path: Path to environmental sensor CSV
            bio_sensor_path: Path to biosensor CSV
            
        Returns:
            Prediction results with interpretation
        """
        # Load and preprocess data
        sample_data = self._load_external_data(image_path, env_sensor_path, bio_sensor_path)
        
        # Make prediction
        predictions = self._predict_raw(sample_data)
        
        # Inverse transform to original scale
        predicted_values = self.data_loader.inverse_transform_labels(predictions)
        
        # Calculate AQI from predicted pollutants
        calculated_aqi, _ = self._calculate_aqi(
            predicted_values['pm25'],
            predicted_values['co2'],
            predicted_values['no2']
        )
        
        # Interpret results
        interpretation = self._interpret_results(
            predicted_values,
            calculated_aqi,
            {'image_path': image_path, 'env_sensor_path': env_sensor_path, 'bio_sensor_path': bio_sensor_path}
        )
        
        return interpretation
    
    def _load_external_data(self,
                           image_path: str,
                           env_sensor_path: str,
                           bio_sensor_path: str) -> Dict:
        """
        Load and preprocess external data files
        
        Args:
            image_path: Path to image file
            env_sensor_path: Path to environmental sensor CSV
            bio_sensor_path: Path to biosensor CSV
            
        Returns:
            Preprocessed sample data
        """
        # Process image
        image = self.data_loader._load_and_preprocess_image(image_path)
        
        # Process environmental sensor data
        env_data_raw = self.data_loader._load_and_preprocess_sensors(
            env_sensor_path, self.data_loader.env_scaler
        )
        env_data = self.data_loader.env_scaler.transform(env_data_raw)
        
        # Process biosensor data
        bio_data_raw = self.data_loader._load_and_preprocess_sensors(
            bio_sensor_path, self.data_loader.bio_scaler
        )
        bio_data = self.data_loader.bio_scaler.transform(bio_data_raw)
        
        return {
            'image': np.expand_dims(image, axis=0),
            'environmental': np.expand_dims(env_data, axis=0),
            'biosensing': np.expand_dims(bio_data, axis=0)
        }
    
    def _predict_raw(self, sample_data: Dict) -> Dict:
        """
        Make raw prediction from preprocessed data
        
        Args:
            sample_data: Preprocessed sample data
            
        Returns:
            Raw prediction outputs
        """
        # Prepare inputs
        inputs = {
            'image': sample_data['image'],
            'environmental': sample_data['environmental'],
            'biosensing': sample_data['biosensing']
        }
        
        # Make prediction
        predictions = self.model.model.predict(inputs, verbose=0)
        
        # Extract outputs
        output_dict = {
            'pm25_output': predictions[0][0],
            'co2_output': predictions[1][0],
            'no2_output': predictions[2][0],
            'aqi_output': predictions[3][0]
        }
        
        return output_dict
    
    def _calculate_aqi(self, pm25: float, co2: float, no2: float) -> Tuple[int, str]:
        """
        Calculate AQI from pollutant concentrations
        
        Args:
            pm25: PM2.5 concentration in μg/m³
            co2: CO₂ concentration in ppm
            no2: NO₂ concentration in ppb
            
        Returns:
            Tuple of (AQI value, AQI category)
        """
        # PM2.5 breakpoints (US EPA standards)
        pm25_breakpoints = [
            (0, 12, 0, 50),      # Good
            (12.1, 35.4, 51, 100),  # Moderate
            (35.5, 55.4, 101, 150), # Unhealthy for Sensitive
            (55.5, 150.4, 151, 200), # Unhealthy
            (150.5, 250.4, 201, 300), # Very Unhealthy
            (250.5, 500.4, 301, 500)  # Hazardous
        ]
        
        # Calculate PM2.5 sub-index
        pm25_aqi = 0
        for bp_low, bp_high, aqi_low, aqi_high in pm25_breakpoints:
            if bp_low <= pm25 <= bp_high:
                pm25_aqi = ((aqi_high - aqi_low) / (bp_high - bp_low)) * (pm25 - bp_low) + aqi_low
                break
        
        aqi_value = int(pm25_aqi)
        
        # Determine category
        if aqi_value <= 50:
            category = "Good"
        elif aqi_value <= 100:
            category = "Moderate"
        elif aqi_value <= 150:
            category = "Unhealthy for Sensitive"
        elif aqi_value <= 200:
            category = "Unhealthy"
        elif aqi_value <= 300:
            category = "Very Unhealthy"
        else:
            category = "Hazardous"
        
        return aqi_value, category
    
    def _interpret_results(self,
                          predicted_values: Dict,
                          calculated_aqi: int,
                          sample_info: Dict) -> Dict:
        """
        Interpret prediction results with risk assessment
        
        Args:
            predicted_values: Predicted pollutant values
            calculated_aqi: Calculated AQI value
            sample_info: Sample information
            
        Returns:
            Complete interpretation with recommendations
        """
        # Get AQI category
        _, aqi_category = self._calculate_aqi(
            predicted_values['pm25'],
            predicted_values['co2'],
            predicted_values['no2']
        )
        
        # Get risk information
        risk_info = self.risk_recommendations[aqi_category]
        
        # Create interpretation
        interpretation = {
            'timestamp': datetime.now().isoformat(),
            'predictions': {
                'pm25': {
                    'value': float(predicted_values['pm25']),
                    'unit': 'μg/m³',
                    'status': self._get_pollutant_status('pm25', predicted_values['pm25'])
                },
                'co2': {
                    'value': float(predicted_values['co2']),
                    'unit': 'ppm',
                    'status': self._get_pollutant_status('co2', predicted_values['co2'])
                },
                'no2': {
                    'value': float(predicted_values['no2']),
                    'unit': 'ppb',
                    'status': self._get_pollutant_status('no2', predicted_values['no2'])
                }
            },
            'aqi': {
                'value': calculated_aqi,
                'category': aqi_category,
                'predicted_category': self.aqi_categories[np.argmax(predicted_values['aqi_probabilities'])],
                'confidence': float(np.max(predicted_values['aqi_probabilities'])),
                'probabilities': {
                    category: float(prob) 
                    for category, prob in zip(self.aqi_categories, predicted_values['aqi_probabilities'])
                }
            },
            'risk_assessment': {
                'level': aqi_category,
                'color': risk_info['color'],
                'description': risk_info['description'],
                'recommendations': risk_info['recommendations']
            },
            'sample_info': sample_info
        }
        
        return interpretation
    
    def _get_pollutant_status(self, pollutant: str, value: float) -> str:
        """Get status description for pollutant value"""
        thresholds = {
            'pm25': [(12, 'Good'), (35, 'Moderate'), (55, 'Unhealthy for Sensitive'), 
                    (150, 'Unhealthy'), (250, 'Very Unhealthy')],
            'co2': [(400, 'Good'), (1000, 'Moderate'), (2000, 'Unhealthy'), 
                    (5000, 'Very Unhealthy')],
            'no2': [(53, 'Good'), (100, 'Moderate'), (360, 'Unhealthy for Sensitive'), 
                    (649, 'Unhealthy'), (1249, 'Very Unhealthy')]
        }
        
        for threshold, status in thresholds.get(pollutant, []):
            if value <= threshold:
                return status
        return 'Hazardous'
    
    def visualize_prediction(self, interpretation: Dict, save_path: Optional[str] = None):
        """
        Visualize prediction results
        
        Args:
            interpretation: Prediction interpretation
            save_path: Path to save visualization
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Air Quality Prediction Results', fontsize=16, fontweight='bold')
        
        # Pollutant concentrations
        pollutants = ['PM2.5', 'CO₂', 'NO₂']
        values = [
            interpretation['predictions']['pm25']['value'],
            interpretation['predictions']['co2']['value'],
            interpretation['predictions']['no2']['value']
        ]
        units = [
            interpretation['predictions']['pm25']['unit'],
            interpretation['predictions']['co2']['unit'],
            interpretation['predictions']['no2']['unit']
        ]
        
        colors = ['red' if v > 50 else 'orange' if v > 35 else 'green' for v in [values[0], values[1]/20, values[2]*5]]
        
        bars = axes[0, 0].bar(pollutants, values, color=colors, alpha=0.7)
        axes[0, 0].set_title('Predicted Pollutant Concentrations')
        axes[0, 0].set_ylabel('Concentration')
        
        # Add value labels on bars
        for bar, value, unit in zip(bars, values, units):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.1f} {unit}', ha='center', va='bottom')
        
        # AQI gauge visualization
        aqi_value = interpretation['aqi']['value']
        aqi_category = interpretation['aqi']['category']
        risk_color = interpretation['risk_assessment']['color']
        
        # Create simple gauge
        theta = np.linspace(0, np.pi, 100)
        r = 1
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        axes[0, 1].plot(x, y, 'k-', linewidth=2)
        axes[0, 1].fill_between(x, y, 0, alpha=0.3)
        
        # Add AQI needle
        aqi_angle = np.pi * (1 - min(aqi_value / 500, 1))
        needle_x = [0, 0.9 * np.cos(aqi_angle)]
        needle_y = [0, 0.9 * np.sin(aqi_angle)]
        axes[0, 1].plot(needle_x, needle_y, 'r-', linewidth=3)
        
        axes[0, 1].set_xlim(-1.2, 1.2)
        axes[0, 1].set_ylim(-0.2, 1.2)
        axes[0, 1].set_aspect('equal')
        axes[0, 1].set_title(f'AQI: {aqi_value} ({aqi_category})')
        axes[0, 1].axis('off')
        
        # AQI probabilities
        categories = list(interpretation['aqi']['probabilities'].keys())
        probabilities = list(interpretation['aqi']['probabilities'].values())
        
        bars = axes[1, 0].barh(categories, probabilities, color='skyblue', alpha=0.7)
        axes[1, 0].set_title('AQI Classification Probabilities')
        axes[1, 0].set_xlabel('Probability')
        axes[1, 0].set_xlim(0, 1)
        
        # Highlight predicted category
        pred_category = interpretation['aqi']['predicted_category']
        pred_idx = categories.index(pred_category)
        bars[pred_idx].set_color('orange')
        
        # Risk recommendations
        recommendations = interpretation['risk_assessment']['recommendations']
        risk_text = f"Risk Level: {interpretation['risk_assessment']['level']}\n\n"
        risk_text += "Recommendations:\n"
        for i, rec in enumerate(recommendations, 1):
            risk_text += f"{i}. {rec}\n"
        
        axes[1, 1].text(0.05, 0.95, risk_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor=risk_color, alpha=0.3))
        axes[1, 1].set_title('Risk Assessment & Recommendations')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def batch_predict(self, sample_ids: List[int]) -> List[Dict]:
        """
        Make predictions for multiple samples
        
        Args:
            sample_ids: List of sample IDs
            
        Returns:
            List of prediction interpretations
        """
        results = []
        for sample_id in sample_ids:
            try:
                result = self.predict_from_sample(sample_id)
                results.append(result)
            except Exception as e:
                print(f"Error processing sample {sample_id}: {e}")
                continue
        
        return results

def main():
    """Main function for inference"""
    parser = argparse.ArgumentParser(description='Air Quality Prediction Inference')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--sample_id', type=int, help='Sample ID from dataset')
    parser.add_argument('--image_path', type=str, help='Path to image file')
    parser.add_argument('--env_path', type=str, help='Path to environmental sensor CSV')
    parser.add_argument('--bio_path', type=str, help='Path to biosensor CSV')
    parser.add_argument('--output_dir', type=str, default='inference_results', help='Output directory')
    parser.add_argument('--visualize', action='store_true', help='Generate visualization')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize inference engine
    inference = AirQualityInference(
        model_path=args.model_path,
        data_dir=args.data_dir
    )
    
    # Make prediction
    if args.sample_id is not None:
        # Predict from dataset sample
        result = inference.predict_from_sample(args.sample_id)
        print(f"\n=== Prediction for Sample {args.sample_id} ===")
    elif all([args.image_path, args.env_path, args.bio_path]):
        # Predict from external files
        result = inference.predict_from_files(args.image_path, args.env_path, args.bio_path)
        print(f"\n=== Prediction from External Files ===")
    else:
        print("Error: Either provide --sample_id or all three file paths (--image_path, --env_path, --bio_path)")
        return
    
    # Print results
    print(f"Timestamp: {result['timestamp']}")
    print(f"\nPredicted Pollutants:")
    for pollutant, info in result['predictions'].items():
        print(f"  {pollutant.upper()}: {info['value']:.1f} {info['unit']} ({info['status']})")
    
    print(f"\nAQI Assessment:")
    print(f"  Value: {result['aqi']['value']}")
    print(f"  Category: {result['aqi']['category']}")
    print(f"  Confidence: {result['aqi']['confidence']:.3f}")
    
    print(f"\nRisk Assessment:")
    print(f"  Level: {result['risk_assessment']['level']}")
    print(f"  Description: {result['risk_assessment']['description']}")
    print(f"  Recommendations:")
    for i, rec in enumerate(result['risk_assessment']['recommendations'], 1):
        print(f"    {i}. {rec}")
    
    # Save results
    results_file = f"{args.output_dir}/prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    # Generate visualization
    if args.visualize:
        viz_file = f"{args.output_dir}/visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        inference.visualize_prediction(result, viz_file)

if __name__ == "__main__":
    main()
