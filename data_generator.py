"""
Data Generation Script for Multi-Modal Air Quality Prediction

This script generates simulated datasets for:
1. Environmental images (sky/haze/smoke conditions)
2. Environmental sensor time-series data
3. Human biosensing time-series data
4. Ground truth labels (PM2.5, CO₂, NO₂, AQI)

Author: Air Quality Prediction System
Date: 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFilter
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class AirQualityDataGenerator:
    """
    Generates synthetic multi-modal dataset for air quality prediction
    """
    
    def __init__(self, num_samples=1000, window_size=24, output_dir="data"):
        """
        Initialize data generator
        
        Args:
            num_samples: Number of data samples to generate
            window_size: Time window size for sensor data
            output_dir: Directory to save generated data
        """
        self.num_samples = num_samples
        self.window_size = window_size
        self.output_dir = output_dir
        self.create_output_directories()
        
        # Define realistic ranges for air quality parameters
        self.pm25_range = (5, 150)  # μg/m³
        self.co2_range = (350, 2000)  # ppm
        self.no2_range = (10, 200)  # ppb
        
        # Environmental sensor ranges
        self.temp_range = (-10, 40)  # °C
        self.humidity_range = (20, 90)  # %
        self.pressure_range = (980, 1030)  # hPa
        
        # Biosensing ranges
        self.heart_rate_range = (60, 100)  # bpm
        self.spo2_range = (95, 100)  # %
        self.skin_temp_range = (32, 37)  # °C
        
    def create_output_directories(self):
        """Create necessary output directories"""
        dirs = [
            self.output_dir,
            f"{self.output_dir}/images",
            f"{self.output_dir}/environmental",
            f"{self.output_dir}/biosensing",
            f"{self.output_dir}/labels"
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            
    def calculate_aqi(self, pm25, co2, no2):
        """
        Calculate Air Quality Index based on pollutant concentrations
        
        Args:
            pm25: PM2.5 concentration in μg/m³
            co2: CO₂ concentration in ppm
            no2: NO₂ concentration in ppb
            
        Returns:
            AQI value and category
        """
        # Simplified AQI calculation (US EPA standards)
        # PM2.5 breakpoints
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
        
        # Simplified: use PM2.5 as primary indicator
        aqi_value = int(pm25_aqi)
        
        # Determine AQI category
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
    
    def generate_environmental_image(self, pm25_level, haze_intensity, sample_id):
        """
        Generate synthetic environmental image based on pollution levels
        
        Args:
            pm25_level: PM2.5 concentration
            haze_intensity: Haze intensity (0-1)
            sample_id: Sample identifier
            
        Returns:
            PIL Image object
        """
        # Create base sky image
        img_size = (224, 224)
        img = Image.new('RGB', img_size, color='lightblue')
        draw = ImageDraw.Draw(img)
        
        # Add sun/moon based on time simulation
        sun_pos = (50 + int(np.random.random() * 124), 30 + int(np.random.random() * 40))
        draw.ellipse([sun_pos[0]-15, sun_pos[1]-15, sun_pos[0]+15, sun_pos[1]+15], 
                    fill='yellow', outline='orange')
        
        # Add clouds
        for _ in range(int(np.random.random() * 3) + 1):
            cloud_x = int(np.random.random() * 180) + 20
            cloud_y = int(np.random.random() * 80) + 20
            for i in range(3):
                draw.ellipse([cloud_x+i*20, cloud_y, cloud_x+i*20+30, cloud_y+20], 
                           fill='white', outline='lightgray')
        
        # Apply haze/smoke effect based on PM2.5 level
        if pm25_level > 35:  # Moderate to poor air quality
            # Create haze overlay
            haze = Image.new('RGB', img_size, color='gray')
            haze_alpha = min(haze_intensity * 255, 200)
            haze.putalpha(haze_alpha)
            
            # Blend with original image
            img = Image.alpha_composite(img.convert('RGBA'), haze.convert('RGBA')).convert('RGB')
            
            # Add blur for heavy pollution
            if pm25_level > 100:
                img = img.filter(ImageFilter.GaussianBlur(radius=1 + (pm25_level - 100) / 50))
        
        # Add noise and texture
        img_array = np.array(img)
        noise = np.random.normal(0, 5, img_array.shape)
        img_array = np.clip(img_array + noise, 0, 255)
        img = Image.fromarray(img_array.astype(np.uint8))
        
        return img
    
    def generate_sensor_timeseries(self, base_values, trend=0, noise_level=0.1):
        """
        Generate realistic sensor time-series data
        
        Args:
            base_values: Dictionary of base values for each sensor
            trend: Trend direction (-1, 0, 1)
            noise_level: Amount of noise to add
            
        Returns:
            DataFrame with time-series data
        """
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(hours=self.window_size),
            periods=self.window_size,
            freq='H'
        )
        
        data = {'timestamp': timestamps}
        
        for sensor, (min_val, max_val) in base_values.items():
            # Generate base signal with trend
            base_signal = np.linspace(
                base_values[sensor][0], 
                base_values[sensor][0] + trend * (max_val - min_val) * 0.2,
                self.window_size
            )
            
            # Add daily cycle for temperature
            if sensor == 'temperature':
                daily_cycle = 5 * np.sin(np.linspace(0, 2*np.pi, self.window_size))
                base_signal += daily_cycle
            
            # Add noise
            noise = np.random.normal(0, noise_level * (max_val - min_val), self.window_size)
            
            # Add some spikes for realism
            if np.random.random() < 0.3:  # 30% chance of spikes
                spike_pos = np.random.randint(0, self.window_size)
                spike_magnitude = np.random.uniform(0.1, 0.3) * (max_val - min_val)
                base_signal[spike_pos] += spike_magnitude
            
            # Ensure values stay within realistic bounds
            final_values = base_signal + noise
            final_values = np.clip(final_values, min_val, max_val)
            
            data[sensor] = final_values
        
        return pd.DataFrame(data)
    
    def generate_biosensor_timeseries(self, aqi_category, base_hr=75, base_spo2=98, base_temp=34):
        """
        Generate biosensor data based on air quality impact
        
        Args:
            aqi_category: Air quality category
            base_hr: Base heart rate
            base_spo2: Base SpO2 level
            base_temp: Base skin temperature
            
        Returns:
            DataFrame with biosensor data
        """
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(hours=self.window_size),
            periods=self.window_size,
            freq='H'
        )
        
        data = {'timestamp': timestamps}
        
        # Adjust base values based on air quality
        aqi_impact = {
            "Good": 0,
            "Moderate": 0.05,
            "Unhealthy for Sensitive": 0.1,
            "Unhealthy": 0.2,
            "Very Unhealthy": 0.3,
            "Hazardous": 0.4
        }
        
        impact_factor = aqi_impact.get(aqi_category, 0)
        
        # Heart rate increases with poor air quality
        hr_base = base_hr + impact_factor * 20
        hr_values = np.random.normal(hr_base, 5, self.window_size)
        hr_values = np.clip(hr_values, self.heart_rate_range[0], self.heart_rate_range[1])
        data['heart_rate'] = hr_values
        
        # SpO2 decreases with poor air quality
        spo2_base = base_spo2 - impact_factor * 3
        spo2_values = np.random.normal(spo2_base, 1, self.window_size)
        spo2_values = np.clip(spo2_values, self.spo2_range[0], self.spo2_range[1])
        data['spo2'] = spo2_values
        
        # Skin temperature slightly increases with poor air quality
        temp_base = base_temp + impact_factor * 0.5
        temp_values = np.random.normal(temp_base, 0.5, self.window_size)
        temp_values = np.clip(temp_values, self.skin_temp_range[0], self.skin_temp_range[1])
        data['skin_temperature'] = temp_values
        
        return pd.DataFrame(data)
    
    def generate_dataset(self):
        """
        Generate complete multi-modal dataset
        
        Returns:
            Dictionary with dataset information
        """
        print(f"Generating {self.num_samples} samples...")
        
        dataset_info = {
            'samples': [],
            'statistics': {}
        }
        
        # Collect all pollutant values for statistics
        all_pm25 = []
        all_co2 = []
        all_no2 = []
        all_aqi = []
        
        for sample_id in range(self.num_samples):
            if sample_id % 100 == 0:
                print(f"Generated {sample_id}/{self.num_samples} samples...")
            
            # Generate pollutant concentrations with correlation
            pm25 = np.random.uniform(*self.pm25_range)
            
            # CO₂ and NO₂ correlated with PM2.5
            co2 = 350 + (pm25 / 150) * 1650 + np.random.normal(0, 100)
            co2 = np.clip(co2, *self.co2_range)
            
            no2 = 10 + (pm25 / 150) * 190 + np.random.normal(0, 20)
            no2 = np.clip(no2, *self.no2_range)
            
            # Calculate AQI
            aqi_value, aqi_category = self.calculate_aqi(pm25, co2, no2)
            
            # Calculate haze intensity based on PM2.5
            haze_intensity = min(pm25 / 150, 1.0)
            
            # Generate environmental image
            img = self.generate_environmental_image(pm25, haze_intensity, sample_id)
            img_path = f"{self.output_dir}/images/sample_{sample_id:06d}.png"
            img.save(img_path)
            
            # Generate environmental sensor data
            env_base_values = {
                'temperature': (np.random.uniform(*self.temp_range), self.temp_range[1]),
                'humidity': (np.random.uniform(*self.humidity_range), self.humidity_range[1]),
                'pressure': (np.random.uniform(*self.pressure_range), self.pressure_range[1]),
                'pm25': (pm25, self.pm25_range[1]),
                'co2': (co2, self.co2_range[1]),
                'no2': (no2, self.no2_range[1])
            }
            
            trend = np.random.choice([-1, 0, 1])  # Random trend
            env_data = self.generate_sensor_timeseries(env_base_values, trend, 0.1)
            env_path = f"{self.output_dir}/environmental/sample_{sample_id:06d}.csv"
            env_data.to_csv(env_path, index=False)
            
            # Generate biosensor data
            bio_data = self.generate_biosensor_timeseries(aqi_category)
            bio_path = f"{self.output_dir}/biosensing/sample_{sample_id:06d}.csv"
            bio_data.to_csv(bio_path, index=False)
            
            # Save labels
            labels = {
                'sample_id': sample_id,
                'pm25': pm25,
                'co2': co2,
                'no2': no2,
                'aqi_value': aqi_value,
                'aqi_category': aqi_category,
                'image_path': img_path,
                'environmental_path': env_path,
                'biosensing_path': bio_path
            }
            
            labels_df = pd.DataFrame([labels])
            labels_path = f"{self.output_dir}/labels/sample_{sample_id:06d}.csv"
            labels_df.to_csv(labels_path, index=False)
            
            # Store sample info
            dataset_info['samples'].append(labels)
            
            # Collect statistics
            all_pm25.append(pm25)
            all_co2.append(co2)
            all_no2.append(no2)
            all_aqi.append(aqi_value)
        
        # Calculate statistics
        dataset_info['statistics'] = {
            'pm25_mean': np.mean(all_pm25),
            'pm25_std': np.std(all_pm25),
            'co2_mean': np.mean(all_co2),
            'co2_std': np.std(all_co2),
            'no2_mean': np.mean(all_no2),
            'no2_std': np.std(all_no2),
            'aqi_mean': np.mean(all_aqi),
            'aqi_std': np.std(all_aqi),
            'aqi_distribution': pd.Series([self.calculate_aqi(pm25, co2, no2)[1] 
                                          for pm25, co2, no2 in zip(all_pm25, all_co2, all_no2)]).value_counts().to_dict()
        }
        
        # Save master dataset info
        master_df = pd.DataFrame(dataset_info['samples'])
        master_path = f"{self.output_dir}/dataset_master.csv"
        master_df.to_csv(master_path, index=False)
        
        # Save statistics
        stats_df = pd.DataFrame([dataset_info['statistics']])
        stats_path = f"{self.output_dir}/dataset_statistics.csv"
        stats_df.to_csv(stats_path, index=False)
        
        print(f"Dataset generation complete!")
        print(f"Data saved to: {self.output_dir}")
        print(f"Master file: {master_path}")
        print(f"Statistics: {stats_path}")
        
        return dataset_info
    
    def visualize_samples(self, num_samples=5):
        """
        Visualize sample data from generated dataset
        
        Args:
            num_samples: Number of samples to visualize
        """
        master_df = pd.read_csv(f"{self.output_dir}/dataset_master.csv")
        
        fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(min(num_samples, len(master_df))):
            sample = master_df.iloc[i]
            
            # Load and display image
            img = Image.open(sample['image_path'])
            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f"Sample {sample['sample_id']}: Image")
            axes[i, 0].axis('off')
            
            # Load and plot environmental data
            env_data = pd.read_csv(sample['environmental_path'])
            axes[i, 1].plot(env_data['pm25'], label='PM2.5', color='red')
            axes[i, 1].plot(env_data['co2']/20, label='CO₂/20', color='blue')  # Scaled for visibility
            axes[i, 1].plot(env_data['no2']*5, label='NO₂×5', color='green')  # Scaled for visibility
            axes[i, 1].set_title("Environmental Sensors")
            axes[i, 1].legend()
            axes[i, 1].tick_params(axis='x', rotation=45)
            
            # Load and plot biosensor data
            bio_data = pd.read_csv(sample['biosensing_path'])
            axes[i, 2].plot(bio_data['heart_rate'], label='Heart Rate', color='red')
            axes[i, 2].plot(bio_data['spo2']*2, label='SpO₂×2', color='blue')  # Scaled for visibility
            axes[i, 2].plot(bio_data['skin_temperature']*10, label='Skin Temp×10', color='green')  # Scaled for visibility
            axes[i, 2].set_title("Biosensors")
            axes[i, 2].legend()
            axes[i, 2].tick_params(axis='x', rotation=45)
            
            # Display labels
            label_text = f"PM2.5: {sample['pm25']:.1f} μg/m³\n"
            label_text += f"CO₂: {sample['co2']:.1f} ppm\n"
            label_text += f"NO₂: {sample['no2']:.1f} ppb\n"
            label_text += f"AQI: {sample['aqi_value']} ({sample['aqi_category']})"
            axes[i, 3].text(0.1, 0.5, label_text, fontsize=12, verticalalignment='center')
            axes[i, 3].set_title("Ground Truth Labels")
            axes[i, 3].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/sample_visualization.png", dpi=150, bbox_inches='tight')
        plt.show()

def main():
    """Main function to generate dataset"""
    # Initialize data generator
    generator = AirQualityDataGenerator(
        num_samples=1000,
        window_size=24,
        output_dir="data"
    )
    
    # Generate dataset
    dataset_info = generator.generate_dataset()
    
    # Visualize some samples
    generator.visualize_samples(num_samples=3)
    
    # Print summary statistics
    print("\n=== Dataset Summary ===")
    stats = dataset_info['statistics']
    print(f"PM2.5: {stats['pm25_mean']:.1f} ± {stats['pm25_std']:.1f} μg/m³")
    print(f"CO₂: {stats['co2_mean']:.1f} ± {stats['co2_std']:.1f} ppm")
    print(f"NO₂: {stats['no2_mean']:.1f} ± {stats['no2_std']:.1f} ppb")
    print(f"AQI: {stats['aqi_mean']:.1f} ± {stats['aqi_std']:.1f}")
    print("\nAQI Distribution:")
    for category, count in stats['aqi_distribution'].items():
        print(f"  {category}: {count}")

if __name__ == "__main__":
    main()
