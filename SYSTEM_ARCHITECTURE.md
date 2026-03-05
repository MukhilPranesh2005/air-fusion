# System Architecture

## Text Block Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           DATA SOURCES LAYER                                   │
├─────────────────┬─────────────────┬─────────────────┬─────────────────────────┤
│   Image Data    │ Environmental   │   Biosensing    │      Ground Truth       │
│   (RGB Images)  │   Sensors       │   Sensors       │      (Labels)           │
│                 │   (CSV Files)   │   (CSV Files)   │                         │
│ • Sky/Haze      │ • Temperature   │ • Heart Rate    │ • PM2.5 (μg/m³)         │
│ • Smoke Scenes  │ • Humidity      │ • SpO₂          │ • CO₂ (ppm)             │
│ • Weather       │ • Pressure      │ • Skin Temp     │ • NO₂ (ppb)             │
│   Conditions    │ • PM2.5         │                 │ • AQI Class             │
│                 │ • CO₂           │                 │                         │
│                 │ • NO₂           │                 │                         │
└─────────────────┴─────────────────┴─────────────────┴─────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        PREPROCESSING LAYER                                     │
├─────────────────┬─────────────────┬─────────────────┬─────────────────────────┤
│   Image         │   Environmental │   Biosensing    │      Label Processing    │
│   Preprocessing │   Sensor        │   Preprocessing │                         │
│                 │   Preprocessing │                 │                         │
│ • Resize (224x224)│ • Windowing    │ • Windowing     │ • AQI Calculation       │
│ • Normalization │ • Scaling       │ • Scaling       │ • One-hot Encoding       │
│ • Augmentation  │ • Feature Eng.  │ • Feature Eng.  │ • Normalization          │
│ • Channel Std.  │ • Missing Val.  │ • Missing Val.  │                         │
└─────────────────┴─────────────────┴─────────────────┴─────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          MODEL ARCHITECTURE                                    │
├─────────────────┬─────────────────┬─────────────────┬─────────────────────────┤
│   CNN BRANCH    │   TEMPORAL      │   BIOSIGNAL     │      FUSION LAYER       │
│   (Images)      │   SENSOR BRANCH │   BRANCH        │                         │
│                 │                 │                 │                         │
│ Input:          │ Input:          │ Input:          │ ┌─────────────────────┐ │
│ (224, 224, 3)   │ (window, 6)     │ (window, 3)     │ │   Multi-Head        │ │
│                 │                 │                 │ │   Attention         │ │
│ EfficientNetB0  │ 1D-CNN (64)     │ 1D-CNN (32)     │ │   Fusion            │ │
│ ↓               │ ↓               │ ↓               │ │   ↓                 │ │
│ GlobalAvgPool   │ MaxPool         │ MaxPool         │ │ Concatenation       │ │
│ ↓               │ ↓               │ ↓               │ │ ↓                   │ │
│ Dense (256)     │ BiLSTM (128)    │ LSTM (64)       │ │ Dense (512)         │ │
│ ↓               │ ↓               │ ↓               │ │ ↓                   │ │
│ Dropout (0.3)   │ Dense (128)     │ Dense (64)      │ │ Dropout (0.4)       │ │
│ ↓               │ ↓               │ ↓               │ │ ↓                   │ │
│ Feature Vec     │ Feature Vec     │ Feature Vec     │ │ Fused Features      │ │
└─────────────────┴─────────────────┴─────────────────┴─────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          OUTPUT LAYER                                          │
├─────────────────┬─────────────────┬─────────────────┬─────────────────────────┤
│   PM2.5         │     CO₂         │     NO₂         │        AQI              │
│   Regression    │   Regression    │   Regression    │    Classification       │
│                 │                 │                 │                         │
│ Dense (64)      │ Dense (64)      │ Dense (64)      │ Dense (128)             │
│ ↓               │ ↓               │ ↓               │ ↓                       │
│ Dropout (0.2)   │ Dropout (0.2)   │ Dropout (0.2)   │ Dropout (0.3)           │
│ ↓               │ ↓               │ ↓               │ ↓                       │
│ Dense (1)       │ Dense (1)       │ Dense (1)       │ Dense (6)               │
│ (Linear)        │ (Linear)        │ (Linear)        │ (Softmax)               │
└─────────────────┴─────────────────┴─────────────────┴─────────────────────────┘
```

## Fusion Strategy Justification

### Chosen Strategy: **Attention-Based Late Fusion**

**Why Attention-Based Fusion?**

1. **Adaptive Weighting**: Different modalities have varying importance for different air quality scenarios. Attention mechanisms allow the model to dynamically weight contributions:
   - Images become more important during visible pollution events (haze, smoke)
   - Environmental sensors dominate during gradual pollution changes
   - Biosensing data provides human impact context during health-relevant conditions

2. **Cross-Modal Relationships**: Multi-head attention captures complex relationships between modalities:
   - Visual haze levels correlated with PM2.5 readings
   - Temperature/humidity patterns affecting both sensor readings and human physiology
   - Temporal dependencies across all modalities

3. **Interpretability**: Attention weights provide insights into which modalities drive predictions for specific conditions

4. **Scalability**: Architecture easily accommodates additional modalities in future hardware integration

### Alternative Considerations:

- **Early Fusion (Concatenation)**: Simple but loses modality-specific processing
- **Late Fusion (Weighted Average)**: Preserves modality features but misses cross-modal interactions
- **Hybrid Fusion**: Complex to implement and tune

## Data Flow Architecture

### Input Shapes and Processing:

1. **Image Stream**:
   - Input: `(batch_size, 224, 224, 3)`
   - Processing: EfficientNetB0 → GlobalAvgPool → Dense(256)
   - Output: `(batch_size, 256)`

2. **Environmental Sensor Stream**:
   - Input: `(batch_size, window_size, 6)`
   - Processing: 1D-CNN → BiLSTM → Dense(128)
   - Output: `(batch_size, 128)`

3. **Biosensing Stream**:
   - Input: `(batch_size, window_size, 3)`
   - Processing: 1D-CNN → LSTM → Dense(64)
   - Output: `(batch_size, 64)`

### Fusion Processing:

1. **Multi-Head Attention**:
   - Query/Key/Value dimensions: 64
   - Number of heads: 8
   - Input: Concatenated features `(batch_size, 448, 64)`
   - Output: Attended features `(batch_size, 448, 64)`

2. **Post-Fusion Processing**:
   - Global average pooling → Dense(512) → Dropout(0.4)
   - Output: `(batch_size, 512)`

### Output Heads:

1. **Regression Heads** (PM2.5, CO₂, NO₂):
   - Dense(64) → Dropout(0.2) → Dense(1)
   - Activation: Linear
   - Loss: Mean Squared Error

2. **Classification Head** (AQI):
   - Dense(128) → Dropout(0.3) → Dense(6)
   - Activation: Softmax
   - Loss: Categorical Crossentropy

## Training Architecture

### Loss Function:
```
Total Loss = α * MSE_PM2.5 + β * MSE_CO₂ + γ * MSE_NO₂ + δ * CE_AQI
where α=β=γ=0.2, δ=0.4
```

### Optimization Strategy:
- **Optimizer**: Adam (lr=0.001)
- **Learning Rate Schedule**: ReduceLROnPlateau
- **Batch Size**: 32
- **Epochs**: Up to 100 with early stopping
- **Validation**: 15% of training data
- **Regularization**: Dropout + L2 weight decay

### Performance Monitoring:
- Training/Validation loss curves
- Individual output metrics
- Attention weight visualization
- Gradient flow analysis

## Deployment Architecture

### Inference Pipeline:
1. **Data Loading**: Load image and time-series windows
2. **Preprocessing**: Apply same transformations as training
3. **Model Inference**: Single forward pass through all branches
4. **Post-processing**: AQI class interpretation, confidence scores
5. **Output**: Structured prediction with risk assessment

### Scalability Considerations:
- **Batch Processing**: Support for multiple samples simultaneously
- **Memory Management**: Efficient tensor operations
- **Model Optimization**: Potential for TensorFlow Lite conversion
- **Caching**: Precomputed feature embeddings for repeated inputs
