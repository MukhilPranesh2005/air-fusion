"""
Ensemble Models and Hyperparameter Optimization

This module provides advanced ensemble techniques and automated hyperparameter
optimization including:
- Multiple model architectures (CNN variants, Transformers)
- Ensemble methods (Voting, Stacking, Bagging)
- Automated hyperparameter optimization (Optuna, Bayesian)
- Cross-validation and model selection
- Performance comparison and analysis

Author: Air Quality Prediction System
Date: 2026
"""

import os
import numpy as np
import pandas as pd
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
import pickle

# Deep learning frameworks
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, optimizers, losses
from tensorflow.keras.applications import EfficientNetB0, ResNet50, MobileNetV2

# Hyperparameter optimization
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# Ensemble and evaluation
from sklearn.ensemble import VotingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedCNNModel:
    """Advanced CNN architectures for image processing"""
    
    def __init__(self, 
                 architecture: str = 'efficientnet',
                 input_shape: Tuple[int, int, int] = (224, 224, 3),
                 num_outputs: int = 4):
        """
        Initialize advanced CNN model
        
        Args:
            architecture: CNN architecture type
            input_shape: Input image shape
            num_outputs: Number of output features
        """
        self.architecture = architecture
        self.input_shape = input_shape
        self.num_outputs = num_outputs
        self.model = None
    
    def build_model(self, learning_rate: float = 0.001, dropout_rate: float = 0.3) -> Model:
        """Build CNN model with specified architecture"""
        
        if self.architecture == 'efficientnet':
            return self._build_efficientnet(learning_rate, dropout_rate)
        elif self.architecture == 'resnet':
            return self._build_resnet(learning_rate, dropout_rate)
        elif self.architecture == 'mobilenet':
            return self._build_mobilenet(learning_rate, dropout_rate)
        elif self.architecture == 'custom_cnn':
            return self._build_custom_cnn(learning_rate, dropout_rate)
        else:
            raise ValueError(f"Unsupported architecture: {self.architecture}")
    
    def _build_efficientnet(self, learning_rate: float, dropout_rate: float) -> Model:
        """Build EfficientNet-based model"""
        inputs = layers.Input(shape=self.input_shape)
        
        # EfficientNet backbone
        base_model = EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_tensor=inputs
        )
        
        # Freeze backbone initially
        base_model.trainable = False
        
        # Custom head
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(dropout_rate * 0.5)(x)
        outputs = layers.Dense(self.num_outputs)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _build_resnet(self, learning_rate: float, dropout_rate: float) -> Model:
        """Build ResNet-based model"""
        inputs = layers.Input(shape=self.input_shape)
        
        # ResNet backbone
        base_model = ResNet50(
            include_top=False,
            weights='imagenet',
            input_tensor=inputs
        )
        
        base_model.trainable = False
        
        # Custom head
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(dropout_rate * 0.5)(x)
        outputs = layers.Dense(self.num_outputs)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _build_mobilenet(self, learning_rate: float, dropout_rate: float) -> Model:
        """Build MobileNet-based model"""
        inputs = layers.Input(shape=self.input_shape)
        
        # MobileNet backbone
        base_model = MobileNetV2(
            include_top=False,
            weights='imagenet',
            input_tensor=inputs
        )
        
        base_model.trainable = False
        
        # Custom head
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(self.num_outputs)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _build_custom_cnn(self, learning_rate: float, dropout_rate: float) -> Model:
        """Build custom CNN architecture"""
        inputs = layers.Input(shape=self.input_shape)
        
        # Block 1
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Block 2
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Block 3
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Dense layers
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate * 0.5)(x)
        outputs = layers.Dense(self.num_outputs)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model

class TransformerModel:
    """Transformer-based model for temporal data"""
    
    def __init__(self, 
                 sequence_length: int = 24,
                 feature_dim: int = 6,
                 d_model: int = 128,
                 num_heads: int = 8,
                 num_layers: int = 4):
        """
        Initialize Transformer model
        
        Args:
            sequence_length: Input sequence length
            feature_dim: Feature dimension
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
        """
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.model = None
    
    def build_model(self, learning_rate: float = 0.001) -> Model:
        """Build Transformer model"""
        inputs = layers.Input(shape=(self.sequence_length, self.feature_dim))
        
        # Input projection
        x = layers.Dense(self.d_model)(inputs)
        
        # Positional encoding
        positions = tf.range(start=0, limit=self.sequence_length, delta=1)
        position_embedding = layers.Embedding(
            input_dim=self.sequence_length, 
            output_dim=self.d_model
        )(positions)
        x = x + position_embedding
        
        # Transformer layers
        for _ in range(self.num_layers):
            # Multi-head attention
            attn_output = layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.d_model // self.num_heads
            )(x, x)
            x = layers.Add()([x, attn_output])
            x = layers.LayerNormalization()(x)
            
            # Feed-forward network
            ffn_output = layers.Dense(self.d_model * 4, activation='relu')(x)
            ffn_output = layers.Dense(self.d_model)(ffn_output)
            x = layers.Add()([x, ffn_output])
            x = layers.LayerNormalization()(x)
        
        # Global pooling and output
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model

class EnsembleModel:
    """Ensemble model combining multiple models"""
    
    def __init__(self, models: List[Model], weights: Optional[List[float]] = None):
        """
        Initialize ensemble model
        
        Args:
            models: List of individual models
            weights: Weights for each model (optional)
        """
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        self.ensemble_type = 'weighted_average'
    
    def predict(self, inputs: Dict) -> np.ndarray:
        """Make ensemble prediction"""
        predictions = []
        
        for model in self.models:
            pred = model.predict(inputs, verbose=0)
            predictions.append(pred)
        
        # Weighted average
        weighted_predictions = np.average(predictions, axis=0, weights=self.weights)
        
        return weighted_predictions
    
    def save_ensemble(self, filepath: str):
        """Save ensemble model"""
        ensemble_data = {
            'models': [],
            'weights': self.weights,
            'ensemble_type': self.ensemble_type
        }
        
        # Save individual models
        for i, model in enumerate(self.models):
            model_path = f"{filepath}_model_{i}.h5"
            model.save(model_path)
            ensemble_data['models'].append(model_path)
        
        # Save ensemble metadata
        with open(f"{filepath}_metadata.json", 'w') as f:
            json.dump(ensemble_data, f, indent=2)

class HyperparameterOptimizer:
    """Hyperparameter optimization using Optuna"""
    
    def __init__(self, 
                 study_name: str = "air_quality_optimization",
                 n_trials: int = 100,
                 timeout: Optional[int] = None):
        """
        Initialize hyperparameter optimizer
        
        Args:
            study_name: Study name for optimization
            n_trials: Number of trials
            timeout: Timeout in seconds
        """
        self.study_name = study_name
        self.n_trials = n_trials
        self.timeout = timeout
        
        # Create study
        self.study = optuna.create_study(
            study_name=study_name,
            direction='minimize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        )
    
    def optimize_cnn_hyperparameters(self, 
                                 train_data: Tuple,
                                 val_data: Tuple,
                                 input_shape: Tuple[int, int, int]) -> Dict:
        """Optimize CNN hyperparameters"""
        
        def objective(trial):
            # Hyperparameters to optimize
            architecture = trial.suggest_categorical('architecture', 
                                                 ['efficientnet', 'resnet', 'mobilenet', 'custom_cnn'])
            learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
            dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
            epochs = trial.suggest_int('epochs', 10, 50)
            
            # Build model
            cnn = AdvancedCNNModel(architecture=architecture, input_shape=input_shape)
            model = cnn.build_model(learning_rate=learning_rate, dropout_rate=dropout_rate)
            
            # Train model
            X_train, y_train = train_data
            X_val, y_val = val_data
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
                ]
            )
            
            # Evaluate
            val_loss = min(history.history['val_loss'])
            
            # Report intermediate values
            trial.report(val_loss, epoch=epochs)
            
            # Prune if necessary
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
            return val_loss
        
        # Run optimization
        self.study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        
        # Get best parameters
        best_params = self.study.best_params
        best_value = self.study.best_value
        
        logger.info(f"Best CNN parameters: {best_params}")
        logger.info(f"Best validation loss: {best_value}")
        
        return {
            'best_params': best_params,
            'best_value': best_value,
            'study': self.study
        }
    
    def optimize_transformer_hyperparameters(self, 
                                         train_data: Tuple,
                                         val_data: Tuple,
                                         sequence_length: int = 24,
                                         feature_dim: int = 6) -> Dict:
        """Optimize Transformer hyperparameters"""
        
        def objective(trial):
            # Hyperparameters to optimize
            d_model = trial.suggest_categorical('d_model', [64, 128, 256])
            num_heads = trial.suggest_categorical('num_heads', [4, 8, 16])
            num_layers = trial.suggest_int('num_layers', 2, 6)
            learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
            dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
            epochs = trial.suggest_int('epochs', 10, 50)
            
            # Build model
            transformer = TransformerModel(
                sequence_length=sequence_length,
                feature_dim=feature_dim,
                d_model=d_model,
                num_heads=num_heads,
                num_layers=num_layers
            )
            model = transformer.build_model(learning_rate=learning_rate)
            
            # Train model
            X_train, y_train = train_data
            X_val, y_val = val_data
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
                ]
            )
            
            # Evaluate
            val_loss = min(history.history['val_loss'])
            
            # Report intermediate values
            trial.report(val_loss, epoch=epochs)
            
            # Prune if necessary
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
            return val_loss
        
        # Run optimization
        self.study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        
        # Get best parameters
        best_params = self.study.best_params
        best_value = self.study.best_value
        
        logger.info(f"Best Transformer parameters: {best_params}")
        logger.info(f"Best validation loss: {best_value}")
        
        return {
            'best_params': best_params,
            'best_value': best_value,
            'study': self.study
        }

class MultiModalEnsemble:
    """Multi-modal ensemble combining different architectures"""
    
    def __init__(self, 
                 model_configs: List[Dict],
                 ensemble_method: str = 'stacking'):
        """
        Initialize multi-modal ensemble
        
        Args:
            model_configs: List of model configurations
            ensemble_method: Ensemble method ('voting', 'stacking', 'bagging')
        """
        self.model_configs = model_configs
        self.ensemble_method = ensemble_method
        self.models = []
        self.meta_model = None
        
    def build_ensemble(self) -> EnsembleModel:
        """Build ensemble model"""
        # Build individual models
        for config in self.model_configs:
            if config['type'] == 'cnn':
                model = AdvancedCNNModel(
                    architecture=config['architecture'],
                    input_shape=config['input_shape']
                ).build_model(
                    learning_rate=config['learning_rate'],
                    dropout_rate=config['dropout_rate']
                )
            elif config['type'] == 'transformer':
                model = TransformerModel(
                    sequence_length=config['sequence_length'],
                    feature_dim=config['feature_dim'],
                    d_model=config['d_model'],
                    num_heads=config['num_heads'],
                    num_layers=config['num_layers']
                ).build_model(learning_rate=config['learning_rate'])
            
            self.models.append(model)
        
        # Create ensemble
        if self.ensemble_method == 'voting':
            return EnsembleModel(self.models)
        elif self.ensemble_method == 'stacking':
            return self._build_stacking_ensemble()
        elif self.ensemble_method == 'bagging':
            return self._build_bagging_ensemble()
        else:
            raise ValueError(f"Unsupported ensemble method: {self.ensemble_method}")
    
    def _build_stacking_ensemble(self) -> EnsembleModel:
        """Build stacking ensemble"""
        # For stacking, we'll use a simple meta-learner
        # In practice, this would require more sophisticated implementation
        return EnsembleModel(self.models)
    
    def _build_bagging_ensemble(self) -> EnsembleModel:
        """Build bagging ensemble"""
        # For bagging, we'll create multiple instances of the same model
        # with different initializations
        bagged_models = []
        
        for config in self.model_configs:
            for _ in range(3):  # Create 3 instances of each model
                if config['type'] == 'cnn':
                    model = AdvancedCNNModel(
                        architecture=config['architecture'],
                        input_shape=config['input_shape']
                    ).build_model(
                        learning_rate=config['learning_rate'],
                        dropout_rate=config['dropout_rate']
                    )
                bagged_models.append(model)
        
        return EnsembleModel(bagged_models)

class EnsembleTrainer:
    """Trainer for ensemble models"""
    
    def __init__(self, 
                 data_dir: str = "data",
                 output_dir: str = "ensemble_models"):
        """
        Initialize ensemble trainer
        
        Args:
            data_dir: Data directory
            output_dir: Output directory for models
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def train_ensemble(self, 
                       ensemble_config: Dict,
                       cv_folds: int = 5) -> Dict:
        """
        Train ensemble model with cross-validation
        
        Args:
            ensemble_config: Ensemble configuration
            cv_folds: Number of cross-validation folds
            
        Returns:
            Training results
        """
        logger.info("Starting ensemble training...")
        
        # Load data
        train_data, val_data, test_data = self._load_data()
        
        # Create ensemble
        ensemble = MultiModalEnsemble(
            model_configs=ensemble_config['models'],
            ensemble_method=ensemble_config['method']
        )
        
        # Build ensemble
        ensemble_model = ensemble.build_ensemble()
        
        # Cross-validation
        cv_scores = []
        for fold in range(cv_folds):
            logger.info(f"Training fold {fold + 1}/{cv_folds}")
            
            # Split data for this fold
            fold_train_data, fold_val_data = self._get_fold_data(
                train_data, val_data, fold, cv_folds
            )
            
            # Train ensemble
            fold_score = self._train_fold(ensemble_model, fold_train_data, fold_val_data)
            cv_scores.append(fold_score)
        
        # Train on full dataset
        logger.info("Training on full dataset...")
        final_score = self._train_fold(ensemble_model, train_data, val_data)
        
        # Evaluate on test set
        test_score = self._evaluate_ensemble(ensemble_model, test_data)
        
        # Save ensemble
        ensemble_model.save_ensemble(f"{self.output_dir}/final_ensemble")
        
        results = {
            'cv_scores': cv_scores,
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'final_score': final_score,
            'test_score': test_score,
            'ensemble_config': ensemble_config
        }
        
        # Save results
        with open(f"{self.output_dir}/training_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Ensemble training completed. CV Mean: {results['cv_mean']:.4f}, Test: {results['test_score']:.4f}")
        
        return results
    
    def _load_data(self) -> Tuple:
        """Load training data"""
        # This would load actual data from the data directory
        # For now, return mock data
        X_train = np.random.random((1000, 224, 224, 3))
        y_train = np.random.random((1000, 4))
        X_val = np.random.random((200, 224, 224, 3))
        y_val = np.random.random((200, 4))
        X_test = np.random.random((200, 224, 224, 3))
        y_test = np.random.random((200, 4))
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def _get_fold_data(self, 
                     train_data: Tuple, 
                     val_data: Tuple, 
                     fold: int, 
                     cv_folds: int) -> Tuple:
        """Get data for specific fold"""
        # Simplified fold splitting
        return train_data, val_data
    
    def _train_fold(self, 
                   ensemble_model: EnsembleModel, 
                   train_data: Tuple, 
                   val_data: Tuple) -> float:
        """Train ensemble on fold data"""
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        # Train each model in ensemble
        for model in ensemble_model.models:
            model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=10,
                batch_size=32,
                verbose=0
            )
        
        # Evaluate ensemble
        predictions = ensemble_model.predict(X_val)
        score = mean_squared_error(y_val, predictions)
        
        return score
    
    def _evaluate_ensemble(self, 
                         ensemble_model: EnsembleModel, 
                         test_data: Tuple) -> float:
        """Evaluate ensemble on test data"""
        X_test, y_test = test_data
        predictions = ensemble_model.predict(X_test)
        score = mean_squared_error(y_test, predictions)
        return score

def main():
    """Main function to test ensemble models"""
    print("Testing Ensemble Models and Hyperparameter Optimization...")
    
    # Test CNN architectures
    print("\n=== Testing CNN Architectures ===")
    cnn = AdvancedCNNModel(architecture='efficientnet')
    model = cnn.build_model()
    print(f"Built EfficientNet model with {model.count_params():,} parameters")
    
    # Test Transformer
    print("\n=== Testing Transformer Model ===")
    transformer = TransformerModel(sequence_length=24, feature_dim=6)
    model = transformer.build_model()
    print(f"Built Transformer model with {model.count_params():,} parameters")
    
    # Test hyperparameter optimization
    print("\n=== Testing Hyperparameter Optimization ===")
    optimizer = HyperparameterOptimizer(n_trials=5)  # Small number for testing
    
    # Create mock data
    X_train = np.random.random((100, 224, 224, 3))
    y_train = np.random.random((100, 4))
    X_val = np.random.random((20, 224, 224, 3))
    y_val = np.random.random((20, 4))
    
    # Optimize CNN hyperparameters
    cnn_results = optimizer.optimize_cnn_hyperparameters(
        (X_train, y_train), (X_val, y_val), (224, 224, 3)
    )
    print(f"Best CNN parameters: {cnn_results['best_params']}")
    
    # Test ensemble
    print("\n=== Testing Ensemble Model ===")
    ensemble_config = {
        'models': [
            {
                'type': 'cnn',
                'architecture': 'efficientnet',
                'input_shape': (224, 224, 3),
                'learning_rate': 0.001,
                'dropout_rate': 0.3
            },
            {
                'type': 'cnn',
                'architecture': 'resnet',
                'input_shape': (224, 224, 3),
                'learning_rate': 0.001,
                'dropout_rate': 0.3
            }
        ],
        'method': 'voting'
    }
    
    ensemble = MultiModalEnsemble(ensemble_config['models'], 'voting')
    ensemble_model = ensemble.build_ensemble()
    print(f"Built ensemble with {len(ensemble_model.models)} models")
    
    print("\nEnsemble models and hyperparameter optimization test completed!")

if __name__ == "__main__":
    main()
