"""
Data Loading and Preprocessing Module for Multi-Modal Air Quality Prediction

This module handles loading, preprocessing, and batching of multi-modal data
for training and inference.

Author: Air Quality Prediction System
Date: 2026
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class AirQualityDataLoader:
    """
    Data loader for multi-modal air quality prediction
    """
    
    def __init__(self, 
                 data_dir: str = "data",
                 image_size: Tuple[int, int] = (224, 224),
                 window_size: int = 24,
                 batch_size: int = 32,
                 shuffle: bool = True):
        """
        Initialize data loader
        
        Args:
            data_dir: Directory containing the dataset
            image_size: Target image size for resizing
            window_size: Time window size for sensor data
            batch_size: Batch size for training
            shuffle: Whether to shuffle the dataset
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.window_size = window_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Initialize scalers
        self.env_scaler = StandardScaler()
        self.bio_scaler = StandardScaler()
        self.label_scaler = StandardScaler()
        self.aqi_encoder = LabelEncoder()
        
        # Load dataset master file
        self.master_df = self._load_master_dataframe()
        
        # Fit scalers on training data
        self._fit_scalers()
        
        # AQI categories
        self.aqi_categories = ['Good', 'Moderate', 'Unhealthy for Sensitive', 
                              'Unhealthy', 'Very Unhealthy', 'Hazardous']
        
    def _load_master_dataframe(self) -> pd.DataFrame:
        """Load the master dataset file"""
        master_path = os.path.join(self.data_dir, "dataset_master.csv")
        if not os.path.exists(master_path):
            raise FileNotFoundError(f"Master dataset file not found: {master_path}")
        
        df = pd.read_csv(master_path)
        print(f"Loaded {len(df)} samples from master dataset")
        return df
    
    def _fit_scalers(self):
        """Fit preprocessing scalers on the dataset"""
        print("Fitting preprocessing scalers...")
        
        # Collect all environmental sensor data
        all_env_data = []
        all_bio_data = []
        all_labels = []
        all_aqi_categories = []
        
        for _, row in self.master_df.iterrows():
            # Load environmental data
            env_data = pd.read_csv(row['environmental_path'])
            env_features = env_data[['temperature', 'humidity', 'pressure', 'pm25', 'co2', 'no2']].values
            all_env_data.append(env_features)
            
            # Load biosensor data
            bio_data = pd.read_csv(row['biosensing_path'])
            bio_features = bio_data[['heart_rate', 'spo2', 'skin_temperature']].values
            all_bio_data.append(bio_features)
            
            # Collect labels
            labels = [row['pm25'], row['co2'], row['no2']]
            all_labels.append(labels)
            all_aqi_categories.append(row['aqi_category'])
        
        # Fit scalers
        all_env_data = np.vstack(all_env_data)
        all_bio_data = np.vstack(all_bio_data)
        all_labels = np.array(all_labels)
        
        self.env_scaler.fit(all_env_data)
        self.bio_scaler.fit(all_bio_data)
        self.label_scaler.fit(all_labels)
        self.aqi_encoder.fit(self.aqi_categories)
        
        print("Scalers fitted successfully")
    
    def _load_and_preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess a single image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image array
        """
        try:
            # Load image
            img = Image.open(image_path).convert('RGB')
            
            # Resize
            img = img.resize(self.image_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array and normalize
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            return img_array
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a blank image as fallback
            return np.zeros((*self.image_size, 3), dtype=np.float32)
    
    def _load_and_preprocess_sensors(self, sensor_path: str, scaler: StandardScaler) -> np.ndarray:
        """
        Load and preprocess sensor time-series data
        
        Args:
            sensor_path: Path to the sensor CSV file
            scaler: Fitted scaler for normalization
            
        Returns:
            Preprocessed sensor data
        """
        try:
            # Load sensor data
            data = pd.read_csv(sensor_path)
            
            # Handle missing values
            data = data.fillna(method='ffill').fillna(method='bfill')
            
            # Ensure we have the correct window size
            if len(data) < self.window_size:
                # Pad with repeated last values
                last_row = data.iloc[-1:].values
                padding = np.tile(last_row, (self.window_size - len(data), 1))
                data = pd.concat([data, pd.DataFrame(padding, columns=data.columns)], ignore_index=True)
            elif len(data) > self.window_size:
                # Take the last window_size values
                data = data.tail(self.window_size).reset_index(drop=True)
            
            return data.values
            
        except Exception as e:
            print(f"Error loading sensor data {sensor_path}: {e}")
            # Return zeros as fallback
            return np.zeros((self.window_size, data.shape[1] if 'data' in locals() else 6))
    
    def _preprocess_sample(self, row: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess a single sample
        
        Args:
            row: Row from master dataframe
            
        Returns:
            Tuple of (image, env_data, bio_data, regression_labels, classification_label)
        """
        # Load and preprocess image
        image = self._load_and_preprocess_image(row['image_path'])
        
        # Load and preprocess environmental sensors
        env_data_raw = self._load_and_preprocess_sensors(row['environmental_path'], self.env_scaler)
        env_data = self.env_scaler.transform(env_data_raw)
        
        # Load and preprocess biosensors
        bio_data_raw = self._load_and_preprocess_sensors(row['biosensing_path'], self.bio_scaler)
        bio_data = self.bio_scaler.transform(bio_data_raw)
        
        # Prepare labels
        regression_labels = np.array([row['pm25'], row['co2'], row['no2']], dtype=np.float32)
        regression_labels = self.label_scaler.transform(regression_labels.reshape(1, -1)).flatten()
        
        # Encode AQI category
        classification_label = self.aqi_encoder.transform([row['aqi_category']])[0]
        classification_label = tf.keras.utils.to_categorical(classification_label, num_classes=len(self.aqi_categories))
        
        return image, env_data, bio_data, regression_labels, classification_label
    
    def create_dataset(self, split: str = 'train', validation_split: float = 0.15, test_split: float = 0.15) -> tf.data.Dataset:
        """
        Create TensorFlow dataset for training/validation/testing
        
        Args:
            split: Dataset split ('train', 'val', 'test')
            validation_split: Fraction of data for validation
            test_split: Fraction of data for testing
            
        Returns:
            TensorFlow dataset
        """
        # Calculate split indices
        total_samples = len(self.master_df)
        test_size = int(total_samples * test_split)
        val_size = int(total_samples * validation_split)
        train_size = total_samples - val_size - test_size
        
        if split == 'train':
            start_idx = 0
            end_idx = train_size
        elif split == 'val':
            start_idx = train_size
            end_idx = train_size + val_size
        elif split == 'test':
            start_idx = train_size + val_size
            end_idx = total_samples
        else:
            raise ValueError(f"Invalid split: {split}")
        
        # Get subset of data
        subset_df = self.master_df.iloc[start_idx:end_idx].copy()
        
        if self.shuffle and split == 'train':
            subset_df = subset_df.sample(frac=1).reset_index(drop=True)
        
        print(f"Creating {split} dataset with {len(subset_df)} samples")
        
        # Create dataset generator
        def data_generator():
            for _, row in subset_df.iterrows():
                try:
                    image, env_data, bio_data, reg_labels, cls_label = self._preprocess_sample(row)
                    yield {
                        'image': image,
                        'environmental': env_data,
                        'biosensing': bio_data
                    }, {
                        'pm25_output': reg_labels[0],
                        'co2_output': reg_labels[1],
                        'no2_output': reg_labels[2],
                        'aqi_output': cls_label
                    }
                except Exception as e:
                    print(f"Error processing sample {row['sample_id']}: {e}")
                    continue
        
        # Create TensorFlow dataset
        output_signature = (
            {
                'image': tf.TensorSpec(shape=(*self.image_size, 3), dtype=tf.float32),
                'environmental': tf.TensorSpec(shape=(self.window_size, 6), dtype=tf.float32),
                'biosensing': tf.TensorSpec(shape=(self.window_size, 3), dtype=tf.float32)
            },
            {
                'pm25_output': tf.TensorSpec(shape=(), dtype=tf.float32),
                'co2_output': tf.TensorSpec(shape=(), dtype=tf.float32),
                'no2_output': tf.TensorSpec(shape=(), dtype=tf.float32),
                'aqi_output': tf.TensorSpec(shape=(len(self.aqi_categories)), dtype=tf.float32)
            }
        )
        
        dataset = tf.data.Dataset.from_generator(
            data_generator,
            output_signature=output_signature
        )
        
        # Batch and prefetch
        dataset = dataset.batch(self.batch_size)
        if self.shuffle and split == 'train':
            dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def get_sample_for_inference(self, sample_id: int) -> Dict:
        """
        Get a single sample for inference
        
        Args:
            sample_id: Sample ID
            
        Returns:
            Dictionary with preprocessed inputs
        """
        sample_row = self.master_df[self.master_df['sample_id'] == sample_id]
        if len(sample_row) == 0:
            raise ValueError(f"Sample {sample_id} not found")
        
        row = sample_row.iloc[0]
        image, env_data, bio_data, _, _ = self._preprocess_sample(row)
        
        return {
            'image': np.expand_dims(image, axis=0),
            'environmental': np.expand_dims(env_data, axis=0),
            'biosensing': np.expand_dims(bio_data, axis=0),
            'sample_info': row.to_dict()
        }
    
    def inverse_transform_labels(self, predictions: Dict) -> Dict:
        """
        Inverse transform predicted labels to original scale
        
        Args:
            predictions: Dictionary with predicted values
            
        Returns:
            Dictionary with inverse transformed values
        """
        # Inverse transform regression predictions
        reg_predictions = np.array([
            predictions['pm25_output'],
            predictions['co2_output'],
            predictions['no2_output']
        ]).reshape(1, -1)
        
        reg_original = self.label_scaler.inverse_transform(reg_predictions).flatten()
        
        # Get AQI category
        aqi_probs = predictions['aqi_output']
        aqi_class_idx = np.argmax(aqi_probs)
        aqi_category = self.aqi_encoder.inverse_transform([aqi_class_idx])[0]
        
        return {
            'pm25': reg_original[0],
            'co2': reg_original[1],
            'no2': reg_original[2],
            'aqi_category': aqi_category,
            'aqi_probabilities': aqi_probs
        }
    
    def get_dataset_info(self) -> Dict:
        """
        Get dataset information and statistics
        
        Returns:
            Dictionary with dataset statistics
        """
        stats_path = os.path.join(self.data_dir, "dataset_statistics.csv")
        if os.path.exists(stats_path):
            stats_df = pd.read_csv(stats_path)
            stats = stats_df.iloc[0].to_dict()
        else:
            # Calculate basic statistics from master dataframe
            stats = {
                'total_samples': len(self.master_df),
                'pm25_mean': self.master_df['pm25'].mean(),
                'pm25_std': self.master_df['pm25'].std(),
                'co2_mean': self.master_df['co2'].mean(),
                'co2_std': self.master_df['co2'].std(),
                'no2_mean': self.master_df['no2'].mean(),
                'no2_std': self.master_df['no2'].std(),
                'aqi_mean': self.master_df['aqi_value'].mean(),
                'aqi_std': self.master_df['aqi_value'].std()
            }
        
        # Add AQI distribution
        aqi_dist = self.master_df['aqi_category'].value_counts().to_dict()
        stats['aqi_distribution'] = aqi_dist
        
        return stats

def main():
    """Main function to test data loader"""
    print("Testing Air Quality Data Loader...")
    
    # Initialize data loader
    loader = AirQualityDataLoader(
        data_dir="data",
        image_size=(224, 224),
        window_size=24,
        batch_size=16,
        shuffle=True
    )
    
    # Get dataset info
    print("\n=== Dataset Information ===")
    info = loader.get_dataset_info()
    for key, value in info.items():
        print(f"{key}: {value}")
    
    # Create datasets
    print("\n=== Creating Datasets ===")
    train_dataset = loader.create_dataset('train')
    val_dataset = loader.create_dataset('val')
    test_dataset = loader.create_dataset('test')
    
    # Test data loading
    print("\n=== Testing Data Loading ===")
    for batch_inputs, batch_outputs in train_dataset.take(1):
        print(f"Batch shapes:")
        print(f"  Image: {batch_inputs['image'].shape}")
        print(f"  Environmental: {batch_inputs['environmental'].shape}")
        print(f"  Biosensing: {batch_inputs['biosensing'].shape}")
        print(f"  PM2.5 output: {batch_outputs['pm25_output'].shape}")
        print(f"  AQI output: {batch_outputs['aqi_output'].shape}")
        break
    
    # Test single sample inference
    print("\n=== Testing Single Sample ===")
    sample = loader.get_sample_for_inference(0)
    print(f"Sample input shapes:")
    print(f"  Image: {sample['image'].shape}")
    print(f"  Environmental: {sample['environmental'].shape}")
    print(f"  Biosensing: {sample['biosensing'].shape}")
    
    print("\nData loader test completed successfully!")

if __name__ == "__main__":
    main()
