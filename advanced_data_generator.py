"""
Advanced Data Generation with Real-time Streaming and Augmentation

This module provides enhanced data generation capabilities including:
- Advanced image augmentation techniques
- Real-time data streaming simulation
- Dynamic weather and pollution patterns
- Seasonal variations and trends
- Anomaly detection and injection

Author: Air Quality Prediction System
Date: 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import warnings
import json
import threading
import time
import queue
from typing import Dict, List, Tuple, Optional, Generator
import random
warnings.filterwarnings('ignore')

class AdvancedDataAugmentation:
    """Advanced image augmentation for environmental data"""
    
    def __init__(self):
        self.augmentation_methods = [
            self._add_weather_effects,
            self._simulate_lighting_conditions,
            self._add_atmospheric_perspective,
            self._simulate_camera_artifacts,
            self._add_noise_and_blur
        ]
    
    def _add_weather_effects(self, image: Image.Image, severity: float = 0.5) -> Image.Image:
        """Add weather effects like rain, fog, snow"""
        img_array = np.array(image)
        
        # Rain effect
        if random.random() < 0.3:
            rain_mask = np.random.random(img_array.shape[:2]) < (0.01 * severity)
            img_array[rain_mask] = [200, 200, 220]  # Light blue rain drops
        
        # Fog effect
        if random.random() < 0.4:
            fog_layer = np.ones_like(img_array) * 255
            fog_alpha = np.random.uniform(0.1, 0.4) * severity
            img_array = (1 - fog_alpha) * img_array + fog_alpha * fog_layer
        
        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
    
    def _simulate_lighting_conditions(self, image: Image.Image, severity: float = 0.5) -> Image.Image:
        """Simulate different lighting conditions"""
        enhancer = ImageEnhance.Brightness(image)
        
        # Random brightness adjustment
        brightness_factor = np.random.uniform(0.7 - 0.3*severity, 1.3 + 0.3*severity)
        image = enhancer.enhance(brightness_factor)
        
        # Color temperature adjustment
        if random.random() < 0.5:
            # Warm light
            r, g, b = image.split()
            r = ImageEnhance.Contrast(r).enhance(1.1)
            image = Image.merge('RGB', (r, g, b))
        
        return image
    
    def _add_atmospheric_perspective(self, image: Image.Image, severity: float = 0.5) -> Image.Image:
        """Add atmospheric perspective effect"""
        img_array = np.array(image, dtype=np.float32)
        
        # Create depth gradient
        height, width = img_array.shape[:2]
        depth_gradient = np.linspace(0, severity, height).reshape(-1, 1)
        depth_gradient = np.repeat(depth_gradient, width, axis=1)
        depth_gradient = np.stack([depth_gradient] * 3, axis=2)
        
        # Apply atmospheric scattering
        atmospheric_color = np.array([200, 200, 210])  # Light blue-gray
        scattered = atmospheric_color * depth_gradient
        img_array = img_array * (1 - depth_gradient) + scattered
        
        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
    
    def _simulate_camera_artifacts(self, image: Image.Image, severity: float = 0.5) -> Image.Image:
        """Simulate camera artifacts like lens flare, chromatic aberration"""
        img_array = np.array(image)
        
        # Lens flare
        if random.random() < 0.2:
            h, w = img_array.shape[:2]
            flare_center = (random.randint(w//4, 3*w//4), random.randint(h//4, 3*h//4))
            flare_radius = random.randint(20, 50)
            
            y, x = np.ogrid[:h, :w]
            mask = (x - flare_center[0])**2 + (y - flare_center[1])**2 <= flare_radius**2
            img_array[mask] = np.minimum(img_array[mask] + 50, 255)
        
        # Chromatic aberration
        if random.random() < 0.3:
            r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
            shift = int(2 * severity)
            r = np.roll(r, shift, axis=1)
            b = np.roll(b, -shift, axis=1)
            img_array = np.stack([r, g, b], axis=2)
        
        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
    
    def _add_noise_and_blur(self, image: Image.Image, severity: float = 0.5) -> Image.Image:
        """Add noise and blur effects"""
        img_array = np.array(image, dtype=np.float32)
        
        # Gaussian noise
        noise = np.random.normal(0, 10 * severity, img_array.shape)
        img_array = img_array + noise
        
        # Motion blur
        if random.random() < 0.3:
            blur_radius = random.uniform(0.5, 2.0) * severity
            image = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
            image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            return image
        
        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
    
    def augment_image(self, image: Image.Image, augmentation_strength: float = 0.5) -> Image.Image:
        """Apply random augmentations to image"""
        num_augmentations = random.randint(1, 3)
        selected_methods = random.sample(self.augmentation_methods, num_augmentations)
        
        for method in selected_methods:
            image = method(image, augmentation_strength)
        
        return image

class RealTimeDataStream:
    """Simulates real-time streaming of sensor data"""
    
    def __init__(self, update_interval: float = 1.0):
        """
        Initialize real-time data stream
        
        Args:
            update_interval: Update interval in seconds
        """
        self.update_interval = update_interval
        self.data_queue = queue.Queue()
        self.is_streaming = False
        self.stream_thread = None
        
        # Initialize sensor values
        self.current_values = {
            'temperature': 22.0,
            'humidity': 50.0,
            'pressure': 1013.25,
            'pm25': 25.0,
            'co2': 450.0,
            'no2': 40.0,
            'heart_rate': 75,
            'spo2': 98,
            'skin_temperature': 34.5
        }
        
        # Trend parameters
        self.trends = {key: 0.0 for key in self.current_values.keys()}
        self.volatility = {key: 0.1 for key in self.current_values.keys()}
    
    def _update_sensor_values(self):
        """Update sensor values with realistic patterns"""
        for sensor, value in self.current_values.items():
            # Apply trend
            trend_change = np.random.normal(0, 0.01)
            self.trends[sensor] = np.clip(self.trends[sensor] + trend_change, -0.5, 0.5)
            
            # Apply volatility
            noise = np.random.normal(0, self.volatility[sensor])
            
            # Update value
            new_value = value + self.trends[sensor] + noise
            
            # Apply sensor-specific constraints
            if sensor == 'temperature':
                new_value = np.clip(new_value, -10, 40)
                # Add daily cycle
                hour = datetime.now().hour
                daily_cycle = 5 * np.sin((hour - 6) * np.pi / 12)
                new_value += daily_cycle * 0.1
            elif sensor == 'humidity':
                new_value = np.clip(new_value, 20, 90)
            elif sensor == 'pressure':
                new_value = np.clip(new_value, 980, 1030)
            elif sensor == 'pm25':
                new_value = max(0, new_value)
                # Add pollution spikes
                if np.random.random() < 0.05:
                    new_value += np.random.uniform(10, 50)
            elif sensor == 'co2':
                new_value = max(350, new_value)
            elif sensor == 'no2':
                new_value = max(0, new_value)
            elif sensor == 'heart_rate':
                new_value = np.clip(new_value, 60, 100)
            elif sensor == 'spo2':
                new_value = np.clip(new_value, 95, 100)
            elif sensor == 'skin_temperature':
                new_value = np.clip(new_value, 32, 37)
            
            self.current_values[sensor] = new_value
        
        # Create data packet
        timestamp = datetime.now().isoformat()
        data_packet = {
            'timestamp': timestamp,
            'values': self.current_values.copy()
        }
        
        # Add to queue
        try:
            self.data_queue.put_nowait(data_packet)
        except queue.Full:
            # Remove oldest item if queue is full
            try:
                self.data_queue.get_nowait()
                self.data_queue.put_nowait(data_packet)
            except queue.Empty:
                pass
    
    def start_streaming(self):
        """Start real-time data streaming"""
        if self.is_streaming:
            return
        
        self.is_streaming = True
        self.stream_thread = threading.Thread(target=self._stream_loop)
        self.stream_thread.daemon = True
        self.stream_thread.start()
        print(f"Started real-time data streaming (interval: {self.update_interval}s)")
    
    def stop_streaming(self):
        """Stop real-time data streaming"""
        self.is_streaming = False
        if self.stream_thread:
            self.stream_thread.join(timeout=2)
        print("Stopped real-time data streaming")
    
    def _stream_loop(self):
        """Main streaming loop"""
        while self.is_streaming:
            self._update_sensor_values()
            time.sleep(self.update_interval)
    
    def get_latest_data(self) -> Optional[Dict]:
        """Get latest data from stream"""
        try:
            return self.data_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_window_data(self, window_size: int = 24) -> List[Dict]:
        """Get sliding window of recent data"""
        window_data = []
        temp_queue = queue.Queue()
        
        # Collect recent data
        while not self.data_queue.empty() and len(window_data) < window_size:
            try:
                data = self.data_queue.get_nowait()
                window_data.append(data)
                temp_queue.put(data)
            except queue.Empty:
                break
        
        # Restore data to queue
        while not temp_queue.empty():
            self.data_queue.put(temp_queue.get())
        
        return window_data[-window_size:] if window_data else []

class AdvancedAirQualityDataGenerator:
    """Advanced data generator with seasonal patterns and anomalies"""
    
    def __init__(self, 
                 num_samples: int = 1000,
                 window_size: int = 24,
                 output_dir: str = "data",
                 enable_augmentation: bool = True,
                 enable_seasonal_patterns: bool = True,
                 enable_anomalies: bool = True):
        """
        Initialize advanced data generator
        
        Args:
            num_samples: Number of samples to generate
            window_size: Time window size
            output_dir: Output directory
            enable_augmentation: Enable advanced image augmentation
            enable_seasonal_patterns: Enable seasonal variations
            enable_anomalies: Enable anomaly injection
        """
        self.num_samples = num_samples
        self.window_size = window_size
        self.output_dir = output_dir
        self.enable_augmentation = enable_augmentation
        self.enable_seasonal_patterns = enable_seasonal_patterns
        self.enable_anomalies = enable_anomalies
        
        # Initialize augmentation
        self.augmentor = AdvancedDataAugmentation()
        
        # Create output directories
        self.create_output_directories()
        
        # Define realistic ranges
        self.setup_parameter_ranges()
        
        # Seasonal parameters
        self.seasonal_params = {
            'spring': {'temp_bias': 2, 'humidity_bias': 5, 'pollution_factor': 0.8},
            'summer': {'temp_bias': 8, 'humidity_bias': -10, 'pollution_factor': 1.2},
            'fall': {'temp_bias': -2, 'humidity_bias': 0, 'pollution_factor': 1.0},
            'winter': {'temp_bias': -8, 'humidity_bias': 10, 'pollution_factor': 0.6}
        }
    
    def create_output_directories(self):
        """Create necessary output directories"""
        dirs = [
            self.output_dir,
            f"{self.output_dir}/images",
            f"{self.output_dir}/environmental",
            f"{self.output_dir}/biosensing",
            f"{self.output_dir}/labels",
            f"{self.output_dir}/metadata"
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def setup_parameter_ranges(self):
        """Setup realistic parameter ranges"""
        self.pm25_range = (5, 150)
        self.co2_range = (350, 2000)
        self.no2_range = (10, 200)
        self.temp_range = (-10, 40)
        self.humidity_range = (20, 90)
        self.pressure_range = (980, 1030)
        self.heart_rate_range = (60, 100)
        self.spo2_range = (95, 100)
        self.skin_temp_range = (32, 37)
    
    def get_seasonal_adjustment(self, date: datetime) -> Dict:
        """Get seasonal parameter adjustments"""
        month = date.month
        
        if month in [3, 4, 5]:
            season = 'spring'
        elif month in [6, 7, 8]:
            season = 'summer'
        elif month in [9, 10, 11]:
            season = 'fall'
        else:
            season = 'winter'
        
        return self.seasonal_params[season]
    
    def inject_anomalies(self, data: np.ndarray, anomaly_type: str = 'spike') -> np.ndarray:
        """Inject anomalies into sensor data"""
        if not self.enable_anomalies:
            return data
        
        if anomaly_type == 'spike':
            # Random spike
            spike_pos = np.random.randint(0, len(data))
            spike_magnitude = np.random.uniform(2, 5)
            data[spike_pos] *= spike_magnitude
        
        elif anomaly_type == 'drift':
            # Gradual drift
            drift_start = np.random.randint(0, len(data) // 2)
            drift_amount = np.random.uniform(-0.3, 0.3)
            drift = np.linspace(0, drift_amount, len(data) - drift_start)
            data[drift_start:] += drift
        
        elif anomaly_type == 'dropout':
            # Sensor dropout
            dropout_pos = np.random.randint(0, len(data))
            dropout_length = np.random.randint(1, 5)
            data[dropout_pos:dropout_pos+dropout_length] = np.nan
        
        return data
    
    def generate_advanced_environmental_image(self, 
                                          pm25_level: float,
                                          date: datetime,
                                          sample_id: int) -> Image.Image:
        """Generate advanced environmental image with seasonal effects"""
        # Create base image
        img_size = (224, 224)
        img = Image.new('RGB', img_size, color='lightblue')
        draw = ImageDraw.Draw(img)
        
        # Apply seasonal sky color
        seasonal_adj = self.get_seasonal_adjustment(date)
        if date.month in [12, 1, 2]:  # Winter
            img = Image.new('RGB', img_size, color='lightgray')
        elif date.month in [6, 7, 8]:  # Summer
            img = Image.new('RGB', img_size, color='skyblue')
        
        # Add sun/moon with seasonal position
        hour = date.hour
        if 6 <= hour <= 18:  # Daytime
            sun_angle = (hour - 6) * np.pi / 12
            sun_x = int(112 + 80 * np.cos(sun_angle))
            sun_y = int(112 - 80 * np.sin(sun_angle))
            draw.ellipse([sun_x-15, sun_y-15, sun_x+15, sun_y+15], 
                        fill='yellow', outline='orange')
        
        # Add clouds with seasonal density
        cloud_density = 2 if date.month in [6, 7, 8] else 4
        for _ in range(cloud_density):
            cloud_x = int(np.random.random() * 180) + 20
            cloud_y = int(np.random.random() * 80) + 20
            for i in range(3):
                draw.ellipse([cloud_x+i*20, cloud_y, cloud_x+i*20+30, cloud_y+20], 
                           fill='white', outline='lightgray')
        
        # Apply pollution effects
        if pm25_level > 35:
            haze = Image.new('RGB', img_size, color='gray')
            haze_alpha = min((pm25_level / 150) * 200, 200)
            haze.putalpha(haze_alpha)
            img = Image.alpha_composite(img.convert('RGBA'), haze.convert('RGBA')).convert('RGB')
            
            if pm25_level > 100:
                blur_radius = 1 + (pm25_level - 100) / 50
                img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        # Apply augmentation if enabled
        if self.enable_augmentation:
            img = self.augmentor.augment_image(img, augmentation_strength=0.3)
        
        return img
    
    def generate_advanced_sensor_timeseries(self, 
                                         base_values: Dict,
                                         date: datetime,
                                         sensor_type: str = 'environmental') -> pd.DataFrame:
        """Generate advanced sensor time-series with seasonal patterns"""
        timestamps = pd.date_range(
            start=date - timedelta(hours=self.window_size),
            periods=self.window_size,
            freq='H'
        )
        
        data = {'timestamp': timestamps}
        seasonal_adj = self.get_seasonal_adjustment(date)
        
        for sensor, base_value in base_values.items():
            # Apply seasonal adjustment
            if sensor == 'temperature':
                base_value += seasonal_adj['temp_bias']
            elif sensor == 'humidity':
                base_value += seasonal_adj['humidity_bias']
            
            # Generate base signal
            signal = np.full(self.window_size, base_value)
            
            # Add daily cycles
            if sensor == 'temperature':
                hours = np.array([ts.hour for ts in timestamps])
                daily_cycle = 5 * np.sin((hours - 6) * np.pi / 12)
                signal += daily_cycle
            elif sensor == 'humidity':
                hours = np.array([ts.hour for ts in timestamps])
                daily_cycle = -10 * np.sin((hours - 6) * np.pi / 12)
                signal += daily_cycle
            
            # Add trends and noise
            trend = np.linspace(0, np.random.uniform(-5, 5), self.window_size)
            noise = np.random.normal(0, base_value * 0.05, self.window_size)
            
            signal += trend + noise
            
            # Apply pollution factor for pollutants
            if sensor in ['pm25', 'co2', 'no2']:
                signal *= seasonal_adj['pollution_factor']
            
            # Inject anomalies
            if self.enable_anomalies and np.random.random() < 0.1:
                anomaly_types = ['spike', 'drift', 'dropout']
                anomaly_type = np.random.choice(anomaly_types)
                signal = self.inject_anomalies(signal, anomaly_type)
            
            # Apply constraints
            if sensor_type == 'environmental':
                constraints = {
                    'temperature': self.temp_range,
                    'humidity': self.humidity_range,
                    'pressure': self.pressure_range,
                    'pm25': (0, self.pm25_range[1]),
                    'co2': (350, self.co2_range[1]),
                    'no2': (0, self.no2_range[1])
                }
            else:  # biosensing
                constraints = {
                    'heart_rate': self.heart_rate_range,
                    'spo2': self.spo2_range,
                    'skin_temperature': self.skin_temp_range
                }
            
            if sensor in constraints:
                min_val, max_val = constraints[sensor]
                signal = np.clip(signal, min_val, max_val)
            
            data[sensor] = signal
        
        return pd.DataFrame(data)
    
    def generate_advanced_dataset(self) -> Dict:
        """Generate complete advanced dataset"""
        print(f"Generating {self.num_samples} advanced samples...")
        
        dataset_info = {
            'samples': [],
            'statistics': {},
            'metadata': {
                'generation_time': datetime.now().isoformat(),
                'augmentation_enabled': self.enable_augmentation,
                'seasonal_patterns_enabled': self.enable_seasonal_patterns,
                'anomalies_enabled': self.enable_anomalies,
                'window_size': self.window_size
            }
        }
        
        # Generate samples across different times and seasons
        for sample_id in range(self.num_samples):
            if sample_id % 100 == 0:
                print(f"Generated {sample_id}/{self.num_samples} samples...")
            
            # Generate random date across the year
            days_offset = np.random.randint(0, 365)
            hours_offset = np.random.randint(0, 24)
            sample_date = datetime.now() - timedelta(days=days_offset, hours=hours_offset)
            
            # Generate pollutant concentrations
            pm25 = np.random.uniform(*self.pm25_range)
            co2 = 350 + (pm25 / 150) * 1650 + np.random.normal(0, 100)
            co2 = np.clip(co2, *self.co2_range)
            no2 = 10 + (pm25 / 150) * 190 + np.random.normal(0, 20)
            no2 = np.clip(no2, *self.no2_range)
            
            # Calculate AQI
            aqi_value, aqi_category = self.calculate_aqi(pm25, co2, no2)
            
            # Generate advanced image
            img = self.generate_advanced_environmental_image(pm25, sample_date, sample_id)
            img_path = f"{self.output_dir}/images/advanced_sample_{sample_id:06d}.png"
            img.save(img_path)
            
            # Generate environmental sensor data
            env_base_values = {
                'temperature': np.random.uniform(*self.temp_range),
                'humidity': np.random.uniform(*self.humidity_range),
                'pressure': np.random.uniform(*self.pressure_range),
                'pm25': pm25,
                'co2': co2,
                'no2': no2
            }
            env_data = self.generate_advanced_sensor_timeseries(env_base_values, sample_date, 'environmental')
            env_path = f"{self.output_dir}/environmental/advanced_sample_{sample_id:06d}.csv"
            env_data.to_csv(env_path, index=False)
            
            # Generate biosensor data
            bio_base_values = {
                'heart_rate': np.random.uniform(*self.heart_rate_range),
                'spo2': np.random.uniform(*self.spo2_range),
                'skin_temperature': np.random.uniform(*self.skin_temp_range)
            }
            bio_data = self.generate_advanced_sensor_timeseries(bio_base_values, sample_date, 'biosensing')
            bio_path = f"{self.output_dir}/biosensing/advanced_sample_{sample_id:06d}.csv"
            bio_data.to_csv(bio_path, index=False)
            
            # Create labels
            labels = {
                'sample_id': sample_id,
                'pm25': pm25,
                'co2': co2,
                'no2': no2,
                'aqi_value': aqi_value,
                'aqi_category': aqi_category,
                'date': sample_date.isoformat(),
                'season': self.get_seasonal_adjustment(sample_date),
                'image_path': img_path,
                'environmental_path': env_path,
                'biosensing_path': bio_path
            }
            
            labels_df = pd.DataFrame([labels])
            labels_path = f"{self.output_dir}/labels/advanced_sample_{sample_id:06d}.csv"
            labels_df.to_csv(labels_path, index=False)
            
            dataset_info['samples'].append(labels)
        
        # Save master dataset
        master_df = pd.DataFrame(dataset_info['samples'])
        master_path = f"{self.output_dir}/advanced_dataset_master.csv"
        master_df.to_csv(master_path, index=False)
        
        # Save metadata
        metadata_path = f"{self.output_dir}/metadata/dataset_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(dataset_info['metadata'], f, indent=2)
        
        print(f"Advanced dataset generation complete!")
        print(f"Data saved to: {self.output_dir}")
        print(f"Master file: {master_path}")
        
        return dataset_info
    
    def calculate_aqi(self, pm25: float, co2: float, no2: float) -> Tuple[int, str]:
        """Calculate AQI from pollutant concentrations"""
        pm25_breakpoints = [
            (0, 12, 0, 50),
            (12.1, 35.4, 51, 100),
            (35.5, 55.4, 101, 150),
            (55.5, 150.4, 151, 200),
            (150.5, 250.4, 201, 300),
            (250.5, 500.4, 301, 500)
        ]
        
        pm25_aqi = 0
        for bp_low, bp_high, aqi_low, aqi_high in pm25_breakpoints:
            if bp_low <= pm25 <= bp_high:
                pm25_aqi = ((aqi_high - aqi_low) / (bp_high - bp_low)) * (pm25 - bp_low) + aqi_low
                break
        
        aqi_value = int(pm25_aqi)
        
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

def main():
    """Main function to test advanced data generation"""
    print("Testing Advanced Data Generation...")
    
    # Test augmentation
    print("\n=== Testing Image Augmentation ===")
    augmentor = AdvancedDataAugmentation()
    
    # Create test image
    test_img = Image.new('RGB', (224, 224), color='skyblue')
    augmented = augmentor.augment_image(test_img, augmentation_strength=0.5)
    augmented.save("test_augmentation.png")
    print("Augmented test image saved as test_augmentation.png")
    
    # Test real-time streaming
    print("\n=== Testing Real-time Streaming ===")
    stream = RealTimeDataStream(update_interval=0.5)
    stream.start_streaming()
    
    print("Collecting data for 5 seconds...")
    time.sleep(5)
    
    latest_data = stream.get_latest_data()
    if latest_data:
        print(f"Latest data: {latest_data}")
    
    window_data = stream.get_window_data(5)
    print(f"Window data points: {len(window_data)}")
    
    stream.stop_streaming()
    
    # Test advanced dataset generation
    print("\n=== Testing Advanced Dataset Generation ===")
    generator = AdvancedAirQualityDataGenerator(
        num_samples=50,
        window_size=24,
        output_dir="advanced_data",
        enable_augmentation=True,
        enable_seasonal_patterns=True,
        enable_anomalies=True
    )
    
    dataset_info = generator.generate_advanced_dataset()
    print(f"Generated {len(dataset_info['samples'])} advanced samples")
    
    print("\nAdvanced data generation test completed!")

if __name__ == "__main__":
    main()
