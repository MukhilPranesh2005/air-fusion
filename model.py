"""
Multi-Modal Air Quality Prediction Model

This module implements the complete deep learning architecture with:
- CNN branch for image processing
- Temporal branches for sensor and biosensing data
- Attention-based fusion layer
- Multi-output heads for regression and classification

Author: Air Quality Prediction System
Date: 2026
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, optimizers, losses, metrics
from tensorflow.keras.applications import EfficientNetB0
from typing import Dict, Tuple, Optional
import numpy as np

class MultiModalAirQualityModel:
    """
    Multi-modal deep learning model for air quality prediction
    """
    
    def __init__(self,
                 image_size: Tuple[int, int] = (224, 224),
                 window_size: int = 24,
                 n_env_features: int = 6,
                 n_bio_features: int = 3,
                 n_aqi_classes: int = 6,
                 dropout_rate: float = 0.3):
        """
        Initialize the multi-modal model
        
        Args:
            image_size: Input image size (height, width)
            window_size: Time window size for sensor data
            n_env_features: Number of environmental sensor features
            n_bio_features: Number of biosensing features
            n_aqi_classes: Number of AQI classification classes
            dropout_rate: Dropout rate for regularization
        """
        self.image_size = image_size
        self.window_size = window_size
        self.n_env_features = n_env_features
        self.n_bio_features = n_bio_features
        self.n_aqi_classes = n_aqi_classes
        self.dropout_rate = dropout_rate
        
        # Build the model
        self.model = self._build_model()
        
    def _build_cnn_branch(self, input_shape: Tuple[int, int, int]) -> Model:
        """
        Build CNN branch for image processing
        
        Args:
            input_shape: Input image shape
            
        Returns:
            CNN branch model
        """
        inputs = layers.Input(shape=input_shape, name='image_input')
        
        # Use EfficientNetB0 as backbone (pretrained on ImageNet)
        base_model = EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_tensor=inputs,
            pooling=None
        )
        
        # Freeze the backbone initially
        base_model.trainable = False
        
        # Add custom layers on top
        x = base_model.output
        x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
        x = layers.Dense(256, activation='relu', name='cnn_dense_1')(x)
        x = layers.BatchNormalization(name='cnn_bn_1')(x)
        x = layers.Dropout(self.dropout_rate, name='cnn_dropout_1')(x)
        x = layers.Dense(128, activation='relu', name='cnn_dense_2')(x)
        x = layers.BatchNormalization(name='cnn_bn_2')(x)
        x = layers.Dropout(self.dropout_rate * 0.5, name='cnn_dropout_2')(x)
        
        # Create branch model
        cnn_model = Model(inputs=inputs, outputs=x, name='cnn_branch')
        
        return cnn_model
    
    def _build_temporal_branch(self, 
                              input_shape: Tuple[int, int],
                              branch_name: str,
                              filters: int = 64,
                              lstm_units: int = 128) -> Model:
        """
        Build temporal branch for time-series data
        
        Args:
            input_shape: Input time-series shape (timesteps, features)
            branch_name: Name of the branch
            filters: Number of CNN filters
            lstm_units: Number of LSTM units
            
        Returns:
            Temporal branch model
        """
        inputs = layers.Input(shape=input_shape, name=f'{branch_name}_input')
        
        # 1D CNN layers for feature extraction
        x = layers.Conv1D(filters, kernel_size=3, activation='relu', 
                         padding='same', name=f'{branch_name}_conv1')(inputs)
        x = layers.BatchNormalization(name=f'{branch_name}_bn1')(x)
        x = layers.MaxPooling1D(pool_size=2, name=f'{branch_name}_maxpool1')(x)
        x = layers.Dropout(self.dropout_rate * 0.5, name=f'{branch_name}_dropout1')(x)
        
        x = layers.Conv1D(filters * 2, kernel_size=3, activation='relu',
                         padding='same', name=f'{branch_name}_conv2')(x)
        x = layers.BatchNormalization(name=f'{branch_name}_bn2')(x)
        x = layers.MaxPooling1D(pool_size=2, name=f'{branch_name}_maxpool2')(x)
        x = layers.Dropout(self.dropout_rate * 0.5, name=f'{branch_name}_dropout2')(x)
        
        # Bidirectional LSTM for temporal modeling
        x = layers.Bidirectional(
            layers.LSTM(lstm_units, return_sequences=True, name=f'{branch_name}_lstm1')
        )(x)
        x = layers.Dropout(self.dropout_rate, name=f'{branch_name}_lstm_dropout1')(x)
        
        x = layers.Bidirectional(
            layers.LSTM(lstm_units // 2, return_sequences=False, name=f'{branch_name}_lstm2')
        )(x)
        x = layers.Dropout(self.dropout_rate * 0.5, name=f'{branch_name}_lstm_dropout2')(x)
        
        # Dense layers
        x = layers.Dense(lstm_units // 2, activation='relu', name=f'{branch_name}_dense')(x)
        x = layers.BatchNormalization(name=f'{branch_name}_final_bn')(x)
        x = layers.Dropout(self.dropout_rate * 0.5, name=f'{branch_name}_final_dropout')(x)
        
        # Create branch model
        temporal_model = Model(inputs=inputs, outputs=x, name=f'{branch_name}_branch')
        
        return temporal_model
    
    def _build_fusion_layer(self, 
                          cnn_features: int,
                          env_features: int,
                          bio_features: int,
                          fusion_dim: int = 512) -> Model:
        """
        Build attention-based fusion layer
        
        Args:
            cnn_features: Number of CNN features
            env_features: Number of environmental features
            bio_features: Number of biosensing features
            fusion_dim: Fusion layer dimension
            
        Returns:
            Fusion layer model
        """
        # Input layers from different branches
        cnn_input = layers.Input(shape=(cnn_features,), name='cnn_features')
        env_input = layers.Input(shape=(env_features,), name='env_features')
        bio_input = layers.Input(shape=(bio_features,), name='bio_features')
        
        # Stack features for attention
        stacked_features = layers.Stack(name='stack_features')([cnn_input, env_input, bio_input])
        
        # Add dimension for attention mechanism
        expanded_features = layers.Reshape((3, -1), name='reshape_for_attention')(stacked_features)
        
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=8,
            key_dim=64,
            dropout=self.dropout_rate,
            name='multi_head_attention'
        )(expanded_features, expanded_features)
        
        # Add residual connection
        attention_output = layers.Add(name='attention_residual')([expanded_features, attention_output])
        attention_output = layers.LayerNormalization(name='attention_norm')(attention_output)
        
        # Global average pooling across attention dimension
        attended_features = layers.GlobalAveragePooling1D(name='global_avg_pool_attention')(attention_output)
        
        # Fusion dense layers
        x = layers.Dense(fusion_dim, activation='relu', name='fusion_dense_1')(attended_features)
        x = layers.BatchNormalization(name='fusion_bn_1')(x)
        x = layers.Dropout(self.dropout_rate, name='fusion_dropout_1')(x)
        
        x = layers.Dense(fusion_dim // 2, activation='relu', name='fusion_dense_2')(x)
        x = layers.BatchNormalization(name='fusion_bn_2')(x)
        x = layers.Dropout(self.dropout_rate * 0.5, name='fusion_dropout_2')(x)
        
        # Create fusion model
        fusion_model = Model(
            inputs=[cnn_input, env_input, bio_input],
            outputs=x,
            name='fusion_layer'
        )
        
        return fusion_model
    
    def _build_output_heads(self, fusion_features: int) -> Dict[str, Model]:
        """
        Build output heads for multi-task learning
        
        Args:
            fusion_features: Number of fusion features
            
        Returns:
            Dictionary of output head models
        """
        output_heads = {}
        
        # Common input for all heads
        common_input = layers.Input(shape=(fusion_features,), name='fusion_features_input')
        
        # PM2.5 regression head
        pm25_x = layers.Dense(64, activation='relu', name='pm25_dense_1')(common_input)
        pm25_x = layers.BatchNormalization(name='pm25_bn_1')(pm25_x)
        pm25_x = layers.Dropout(self.dropout_rate * 0.5, name='pm25_dropout')(pm25_x)
        pm25_output = layers.Dense(1, activation='linear', name='pm25_output')(pm25_x)
        output_heads['pm25'] = Model(inputs=common_input, outputs=pm25_output, name='pm25_head')
        
        # CO₂ regression head
        co2_x = layers.Dense(64, activation='relu', name='co2_dense_1')(common_input)
        co2_x = layers.BatchNormalization(name='co2_bn_1')(co2_x)
        co2_x = layers.Dropout(self.dropout_rate * 0.5, name='co2_dropout')(co2_x)
        co2_output = layers.Dense(1, activation='linear', name='co2_output')(co2_x)
        output_heads['co2'] = Model(inputs=common_input, outputs=co2_output, name='co2_head')
        
        # NO₂ regression head
        no2_x = layers.Dense(64, activation='relu', name='no2_dense_1')(common_input)
        no2_x = layers.BatchNormalization(name='no2_bn_1')(no2_x)
        no2_x = layers.Dropout(self.dropout_rate * 0.5, name='no2_dropout')(no2_x)
        no2_output = layers.Dense(1, activation='linear', name='no2_output')(no2_x)
        output_heads['no2'] = Model(inputs=common_input, outputs=no2_output, name='no2_head')
        
        # AQI classification head
        aqi_x = layers.Dense(128, activation='relu', name='aqi_dense_1')(common_input)
        aqi_x = layers.BatchNormalization(name='aqi_bn_1')(aqi_x)
        aqi_x = layers.Dropout(self.dropout_rate, name='aqi_dropout_1')(aqi_x)
        aqi_x = layers.Dense(64, activation='relu', name='aqi_dense_2')(aqi_x)
        aqi_x = layers.BatchNormalization(name='aqi_bn_2')(aqi_x)
        aqi_x = layers.Dropout(self.dropout_rate * 0.5, name='aqi_dropout_2')(aqi_x)
        aqi_output = layers.Dense(self.n_aqi_classes, activation='softmax', name='aqi_output')(aqi_x)
        output_heads['aqi'] = Model(inputs=common_input, outputs=aqi_output, name='aqi_head')
        
        return output_heads
    
    def _build_model(self) -> Model:
        """
        Build the complete multi-modal model
        
        Returns:
            Complete Keras model
        """
        # Define inputs
        image_input = layers.Input(shape=(*self.image_size, 3), name='image')
        env_input = layers.Input(shape=(self.window_size, self.n_env_features), name='environmental')
        bio_input = layers.Input(shape=(self.window_size, self.n_bio_features), name='biosensing')
        
        # Build branches
        cnn_branch = self._build_cnn_branch((*self.image_size, 3))
        env_branch = self._build_temporal_branch(
            (self.window_size, self.n_env_features), 
            'environmental',
            filters=64,
            lstm_units=128
        )
        bio_branch = self._build_temporal_branch(
            (self.window_size, self.n_bio_features),
            'biosensing', 
            filters=32,
            lstm_units=64
        )
        
        # Extract features from branches
        cnn_features = cnn_branch(image_input)
        env_features = env_branch(env_input)
        bio_features = bio_branch(bio_input)
        
        # Fusion layer
        fusion_layer = self._build_fusion_layer(
            cnn_features=128,  # Output size of CNN branch
            env_features=64,   # Output size of environmental branch
            bio_features=32,  # Output size of biosensing branch
            fusion_dim=512
        )
        
        fusion_features = fusion_layer([cnn_features, env_features, bio_features])
        
        # Output heads
        output_heads = self._build_output_heads(fusion_features.shape[1])
        
        # Generate outputs
        pm25_output = output_heads['pm25'](fusion_features)
        co2_output = output_heads['co2'](fusion_features)
        no2_output = output_heads['no2'](fusion_features)
        aqi_output = output_heads['aqi'](fusion_features)
        
        # Create complete model
        model = Model(
            inputs=[image_input, env_input, bio_input],
            outputs=[pm25_output, co2_output, no2_output, aqi_output],
            name='multi_modal_air_quality_model'
        )
        
        return model
    
    def compile_model(self, 
                     learning_rate: float = 0.001,
                     loss_weights: Optional[Dict[str, float]] = None):
        """
        Compile the model with losses and optimizer
        
        Args:
            learning_rate: Learning rate for optimizer
            loss_weights: Weights for different loss components
        """
        if loss_weights is None:
            loss_weights = {
                'pm25_output': 0.2,
                'co2_output': 0.2,
                'no2_output': 0.2,
                'aqi_output': 0.4
            }
        
        # Define losses
        losses = {
            'pm25_output': losses.MeanSquaredError(),
            'co2_output': losses.MeanSquaredError(),
            'no2_output': losses.MeanSquaredError(),
            'aqi_output': losses.CategoricalCrossentropy()
        }
        
        # Define metrics
        metrics_dict = {
            'pm25_output': [metrics.MeanAbsoluteError(), metrics.RootMeanSquaredError()],
            'co2_output': [metrics.MeanAbsoluteError(), metrics.RootMeanSquaredError()],
            'no2_output': [metrics.MeanAbsoluteError(), metrics.RootMeanSquaredError()],
            'aqi_output': [metrics.CategoricalAccuracy(), metrics.TopKCategoricalAccuracy(k=3)]
        }
        
        # Compile model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss=losses,
            loss_weights=loss_weights,
            metrics=metrics_dict
        )
        
        print("Model compiled successfully!")
    
    def unfreeze_backbone(self, unfreeze_from_layer: int = -20):
        """
        Unfreeze the CNN backbone for fine-tuning
        
        Args:
            unfreeze_from_layer: Number of layers from the end to unfreeze
        """
        # Find the CNN backbone
        cnn_branch = self.model.get_layer('cnn_branch')
        backbone = cnn_branch.get_layer('efficientnetb0')
        
        # Unfreeze layers
        backbone.trainable = True
        for layer in backbone.layers[:unfreeze_from_layer]:
            layer.trainable = False
        
        print(f"Unfroze last {abs(unfreeze_from_layer)} layers of EfficientNetB0 backbone")
    
    def get_model_summary(self) -> str:
        """Get model summary as string"""
        import io
        import sys
        
        # Capture summary output
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        self.model.summary()
        sys.stdout = old_stdout
        
        return buffer.getvalue()
    
    def save_model(self, filepath: str):
        """Save the model"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a saved model"""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")

def create_model(image_size: Tuple[int, int] = (224, 224),
                window_size: int = 24,
                learning_rate: float = 0.001) -> MultiModalAirQualityModel:
    """
    Factory function to create a complete model
    
    Args:
        image_size: Input image size
        window_size: Time window size
        learning_rate: Learning rate for optimizer
        
    Returns:
        Compiled multi-modal model
    """
    # Create model instance
    model_instance = MultiModalAirQualityModel(
        image_size=image_size,
        window_size=window_size
    )
    
    # Compile model
    model_instance.compile_model(learning_rate=learning_rate)
    
    return model_instance

def main():
    """Main function to test model creation"""
    print("Creating Multi-Modal Air Quality Model...")
    
    # Create model
    model_instance = create_model(
        image_size=(224, 224),
        window_size=24,
        learning_rate=0.001
    )
    
    # Print model summary
    print("\n=== Model Architecture ===")
    print(model_instance.get_model_summary())
    
    # Test model with dummy data
    print("\n=== Testing Model with Dummy Data ===")
    batch_size = 2
    
    # Create dummy inputs
    dummy_inputs = {
        'image': np.random.random((batch_size, 224, 224, 3)),
        'environmental': np.random.random((batch_size, 24, 6)),
        'biosensing': np.random.random((batch_size, 24, 3))
    }
    
    # Forward pass
    outputs = model_instance.model.predict(dummy_inputs, verbose=0)
    
    print(f"Input shapes:")
    for name, arr in dummy_inputs.items():
        print(f"  {name}: {arr.shape}")
    
    print(f"\nOutput shapes:")
    output_names = ['PM2.5', 'CO₂', 'NO₂', 'AQI']
    for i, (name, output) in enumerate(zip(output_names, outputs)):
        print(f"  {name}: {output.shape}")
    
    print("\nModel test completed successfully!")

if __name__ == "__main__":
    main()
