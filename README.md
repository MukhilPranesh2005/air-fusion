# Multi-Modal Air Quality Prediction System

A comprehensive deep learning system for predicting air quality using multi-modal data fusion. This prototype combines environmental images, sensor time-series data, and human biosensing data to predict PM2.5, CO₂, NO₂ concentrations and Air Quality Index (AQI).

## 🌟 Features

- **Multi-Modal Data Integration**: Processes images, environmental sensors, and biosensors
- **Advanced Architecture**: CNN + Temporal models with attention-based fusion
- **Comprehensive Pipeline**: End-to-end data generation, training, and evaluation
- **Risk Assessment**: AQI interpretation with health recommendations
- **Modular Design**: Easy to extend for real hardware integration

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Data Generation](#data-generation)
- [Model Training](#model-training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## 🚀 Installation

### Prerequisites

- Python 3.9+
- TensorFlow 2.10+
- CUDA (optional, for GPU acceleration)

### Setup

1. **Clone or download the project**
   ```bash
   # If using git
   git clone <repository-url>
   cd air-quality-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python model.py  # Should run without errors
   ```

## ⚡ Quick Start

### 1. Generate Dataset

```bash
python data_generator.py
```

This creates a synthetic dataset with:
- 1000 samples of environmental images
- Environmental sensor time-series (24-hour windows)
- Biosensor time-series (24-hour windows)
- Ground truth labels (PM2.5, CO₂, NO₂, AQI)

### 2. Train Model

```bash
python train.py --epochs 100 --batch_size 32
```

Training includes:
- Phase 1: Frozen backbone training
- Phase 2: Fine-tuning with unfrozen backbone
- Automatic checkpointing and early stopping

### 3. Make Predictions

```bash
# Predict on dataset sample
python inference.py --model_path models/best_model.h5 --sample_id 0 --visualize

# Predict on external files
python inference.py --model_path models/best_model.h5 \
    --image_path path/to/image.jpg \
    --env_path path/to/sensors.csv \
    --bio_path path/to/biosensors.csv \
    --visualize
```

### 4. Evaluate Model

```bash
python evaluation.py --model_path models/best_model.h5
```

## 📁 Project Structure

```
Air Fusion/
├── README.md                    # This file
├── requirements.txt              # Python dependencies
├── PROJECT_OVERVIEW.md          # Project documentation
├── SYSTEM_ARCHITECTURE.md       # System architecture details
├── data_generator.py            # Dataset generation script
├── data_loader.py               # Data loading and preprocessing
├── model.py                     # Multi-modal model architecture
├── train.py                     # Training pipeline
├── inference.py                 # Inference and prediction
├── evaluation.py                # Model evaluation
├── data/                        # Generated dataset
│   ├── images/                  # Environmental images
│   ├── environmental/           # Sensor CSV files
│   ├── biosensing/              # Biosensor CSV files
│   ├── labels/                  # Ground truth labels
│   ├── dataset_master.csv       # Master dataset file
│   └── dataset_statistics.csv   # Dataset statistics
├── models/                      # Trained models
├── logs/                        # Training logs and plots
└── evaluation_results/          # Evaluation reports
```

## 📊 Data Generation

### Synthetic Dataset Features

The `data_generator.py` script creates realistic synthetic data:

#### Environmental Images
- Sky/haze/smoke conditions correlated with pollution levels
- Dynamic weather effects (sun, clouds, haze)
- Image size: 224x224 RGB
- Automatic haze intensity based on PM2.5 levels

#### Environmental Sensors
- Temperature, humidity, pressure
- PM2.5, CO₂, NO₂ concentrations
- 24-hour time windows with hourly readings
- Realistic trends and daily cycles
- Missing value handling

#### Biosensors
- Heart rate, SpO₂, skin temperature
- Physiological responses to air quality
- Correlated with AQI categories
- Individual variation patterns

#### Ground Truth Labels
- PM2.5 (μg/m³): 5-150 range
- CO₂ (ppm): 350-2000 range  
- NO₂ (ppb): 10-200 range
- AQI: 6 categories (Good to Hazardous)

### Custom Data Generation

```python
from data_generator import AirQualityDataGenerator

# Create custom dataset
generator = AirQualityDataGenerator(
    num_samples=2000,
    window_size=48,  # 48-hour windows
    output_dir="custom_data"
)

# Generate dataset
dataset_info = generator.generate_dataset()
```

## 🧠 Model Training

### Training Pipeline

The training process uses a two-phase approach:

#### Phase 1: Feature Learning (Epochs 1-50)
- CNN backbone frozen (EfficientNetB0)
- Training fusion and output layers
- Learning rate: 0.001

#### Phase 2: Fine-Tuning (Epochs 51-100)
- Unfreeze last 20 layers of backbone
- Lower learning rate: 0.0001
- Full model optimization

### Training Commands

```bash
# Basic training
python train.py

# Custom training
python train.py \
    --epochs 150 \
    --batch_size 16 \
    --learning_rate 0.0005 \
    --experiment_name "custom_experiment" \
    --fine_tune_after 75
```

### Training Monitoring

- **TensorBoard**: Real-time training visualization
- **CSV Logs**: Detailed metrics logging
- **Checkpoints**: Best model saving
- **Early Stopping**: Prevents overfitting

### Hyperparameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Image Size | 224x224 | 128-512 | Input image resolution |
| Window Size | 24 | 12-48 | Time series window |
| Batch Size | 32 | 8-64 | Training batch size |
| Learning Rate | 0.001 | 0.0001-0.01 | Optimizer learning rate |
| Dropout Rate | 0.3 | 0.1-0.5 | Regularization strength |

## 🔮 Inference

### Single Sample Prediction

```python
from inference import AirQualityInference

# Load inference engine
inference = AirQualityInference(
    model_path="models/best_model.h5",
    data_dir="data"
)

# Predict on dataset sample
result = inference.predict_from_sample(sample_id=0)

# Print results
print(f"PM2.5: {result['predictions']['pm25']['value']:.1f} μg/m³")
print(f"AQI: {result['aqi']['category']} (Confidence: {result['aqi']['confidence']:.2f})")
print(f"Risk Level: {result['risk_assessment']['level']}")
```

### External Data Prediction

```python
# Predict from custom files
result = inference.predict_from_files(
    image_path="my_image.jpg",
    env_sensor_path="my_sensors.csv",
    bio_sensor_path="my_biosensors.csv"
)

# Generate visualization
inference.visualize_prediction(result, "prediction_viz.png")
```

### Risk Assessment

The system provides detailed risk assessment:

- **Good**: Safe for all activities
- **Moderate**: Generally safe, sensitive groups should monitor
- **Unhealthy for Sensitive**: Limit outdoor exertion for sensitive groups
- **Unhealthy**: Everyone should limit outdoor activities
- **Very Unhealthy**: Avoid outdoor activities
- **Hazardous**: Emergency conditions

## 📈 Evaluation

### Comprehensive Metrics

#### Regression Metrics (Pollutants)
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error  
- **R²**: Coefficient of Determination
- **MAPE**: Mean Absolute Percentage Error

#### Classification Metrics (AQI)
- **Accuracy**: Overall classification accuracy
- **Precision/Recall**: Per-class performance
- **F1-Score**: Harmonic mean of precision and recall
- **AUC**: Area Under ROC Curve

### Running Evaluation

```bash
python evaluation.py --model_path models/best_model.h5
```

This generates:
- **Evaluation Report**: Detailed markdown report
- **Performance Plots**: Visual analysis
- **Raw Results**: JSON with all metrics

### Sample Evaluation Output

```
=== Evaluation Summary ===
Test samples: 150

Regression Metrics:
  PM2.5: R² = 0.847, RMSE = 12.34
  CO₂: R² = 0.812, RMSE = 145.67
  NO₂: R² = 0.789, RMSE = 18.92

Classification Metrics:
  Accuracy: 0.833 (83.3%)
  F1-Score: 0.821
```

## 🏗️ Architecture

### Multi-Modal Design

```
┌─────────────────┬─────────────────┬─────────────────┐
│   Image Input   │  Environmental  │   Biosensing    │
│   (224×224×3)   │   Sensors       │   Sensors       │
│                 │   (24×6)        │   (24×3)        │
└─────────────────┴─────────────────┴─────────────────┘
         │                 │                 │
         ▼                 ▼                 ▼
┌─────────────────┬─────────────────┬─────────────────┐
│   CNN Branch    │  Temporal       │  Temporal       │
│ EfficientNetB0  │  1D-CNN+BiLSTM  │  1D-CNN+LSTM    │
└─────────────────┴─────────────────┴─────────────────┘
         │                 │                 │
         └─────────────────┼─────────────────┘
                           ▼
                 ┌─────────────────┐
                 │  Fusion Layer   │
                 │ Multi-Head      │
                 │ Attention       │
                 └─────────────────┘
                           │
                           ▼
                 ┌─────────────────┐
                 │  Output Heads   │
                 │ PM2.5, CO₂, NO₂│
                 │ + AQI Class     │
                 └─────────────────┘
```

### Key Components

#### CNN Branch (Images)
- **Backbone**: EfficientNetB0 (pre-trained on ImageNet)
- **Features**: 256-dim feature vectors
- **Processing**: Global average pooling + dense layers

#### Temporal Branches (Sensors)
- **Environmental**: 1D-CNN + BiLSTM (128 units)
- **Biosensing**: 1D-CNN + LSTM (64 units)
- **Windowing**: 24-hour sliding windows

#### Fusion Layer
- **Mechanism**: Multi-head attention (8 heads)
- **Features**: Cross-modal relationships
- **Output**: 512-dim fused representation

#### Output Heads
- **Regression**: PM2.5, CO₂, NO₂ (linear activation)
- **Classification**: AQI (softmax, 6 classes)

## ⚙️ Configuration

### Model Configuration

```python
# In model.py
model = MultiModalAirQualityModel(
    image_size=(224, 224),
    window_size=24,
    n_env_features=6,
    n_bio_features=3,
    n_aqi_classes=6,
    dropout_rate=0.3
)
```

### Training Configuration

```python
# In train.py
trainer = AirQualityTrainer(
    data_dir="data",
    model_dir="models",
    logs_dir="logs",
    batch_size=32,
    learning_rate=0.001
)
```

### Data Configuration

```python
# In data_generator.py
generator = AirQualityDataGenerator(
    num_samples=1000,
    window_size=24,
    output_dir="data"
)
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Memory Errors
```bash
# Reduce batch size
python train.py --batch_size 16

# Reduce image size
# Edit model.py: image_size=(128, 128)
```

#### 2. Training Slow
```bash
# Use GPU (if available)
export CUDA_VISIBLE_DEVICES=0

# Reduce dataset size
python data_generator.py  # Edit num_samples=500
```

#### 3. Poor Performance
```bash
# Increase training epochs
python train.py --epochs 200

# Adjust learning rate
python train.py --learning_rate 0.0005
```

#### 4. Data Loading Errors
```bash
# Regenerate dataset
rm -rf data/
python data_generator.py

# Check file permissions
ls -la data/
```

### Performance Optimization

#### GPU Acceleration
```python
# Verify GPU usage
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

#### Memory Management
```python
# Enable mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')
```

#### Data Pipeline Optimization
```python
# Increase prefetch buffer
dataset = dataset.prefetch(tf.data.AUTOTUNE)
```

## 📚 Advanced Usage

### Custom Model Architecture

```python
# Modify model.py
class CustomMultiModalModel(MultiModalAirQualityModel):
    def _build_cnn_branch(self, input_shape):
        # Custom CNN implementation
        pass
    
    def _build_fusion_layer(self, ...):
        # Custom fusion strategy
        pass
```

### Real Data Integration

```python
# Extend data_loader.py for real sensors
class RealDataLoader(AirQualityDataLoader):
    def load_real_sensor_data(self, sensor_id):
        # Connect to real sensors
        pass
```

### Deployment

```python
# Save model for deployment
model.model.save("deployment_model")

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_saved_model("deployment_model")
tflite_model = converter.convert()
```

## 📄 License

This project is provided as a prototype for educational and research purposes.

## 🤝 Contributing

1. Fork the project
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the system architecture documentation
3. Examine the code comments
4. Create an issue with detailed information

---

**Note**: This is a software-only prototype designed for research and development. For production use, integrate with real sensors and validate extensively.
