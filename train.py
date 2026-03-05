"""
Training Pipeline for Multi-Modal Air Quality Prediction

This script handles the complete training process including:
- Data loading and preprocessing
- Model training with callbacks
- Model checkpointing and logging
- Performance monitoring and visualization

Author: Air Quality Prediction System
Date: 2026
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import argparse
from typing import Dict, List, Tuple, Optional

import tensorflow as tf
from tensorflow.keras import callbacks

from data_loader import AirQualityDataLoader
from model import create_model, MultiModalAirQualityModel

class AirQualityTrainer:
    """
    Training pipeline for air quality prediction model
    """
    
    def __init__(self,
                 data_dir: str = "data",
                 model_dir: str = "models",
                 logs_dir: str = "logs",
                 image_size: Tuple[int, int] = (224, 224),
                 window_size: int = 24,
                 batch_size: int = 32,
                 learning_rate: float = 0.001):
        """
        Initialize trainer
        
        Args:
            data_dir: Directory containing dataset
            model_dir: Directory to save models
            logs_dir: Directory to save logs
            image_size: Input image size
            window_size: Time window size
            batch_size: Training batch size
            learning_rate: Learning rate
        """
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.logs_dir = logs_dir
        self.image_size = image_size
        self.window_size = window_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # Create directories
        self._create_directories()
        
        # Initialize data loader
        self.data_loader = AirQualityDataLoader(
            data_dir=data_dir,
            image_size=image_size,
            window_size=window_size,
            batch_size=batch_size
        )
        
        # Initialize model
        self.model = create_model(
            image_size=image_size,
            window_size=window_size,
            learning_rate=learning_rate
        )
        
        # Training history
        self.history = None
        
        # Setup logging
        self._setup_logging()
    
    def _create_directories(self):
        """Create necessary directories"""
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(f"{self.logs_dir}/tensorboard", exist_ok=True)
        os.makedirs(f"{self.logs_dir}/plots", exist_ok=True)
    
    def _setup_logging(self):
        """Setup logging configuration"""
        import logging
        
        log_file = f"{self.logs_dir}/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Training pipeline initialized")
    
    def _create_callbacks(self, experiment_name: str) -> List[callbacks.Callback]:
        """
        Create training callbacks
        
        Args:
            experiment_name: Name for this training experiment
            
        Returns:
            List of Keras callbacks
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        callback_list = []
        
        # Model checkpoint
        checkpoint_path = f"{self.model_dir}/{experiment_name}_{timestamp}_best.h5"
        checkpoint_callback = callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=1
        )
        callback_list.append(checkpoint_callback)
        
        # Early stopping
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            mode='min',
            verbose=1
        )
        callback_list.append(early_stopping)
        
        # Learning rate reduction
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            mode='min',
            verbose=1
        )
        callback_list.append(reduce_lr)
        
        # TensorBoard logging
        tensorboard_callback = callbacks.TensorBoard(
            log_dir=f"{self.logs_dir}/tensorboard/{experiment_name}_{timestamp}",
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch'
        )
        callback_list.append(tensorboard_callback)
        
        # CSV logging
        csv_logger = callbacks.CSVLogger(
            filename=f"{self.logs_dir}/{experiment_name}_{timestamp}_training_log.csv",
            append=True
        )
        callback_list.append(csv_logger)
        
        # Custom callback for additional metrics
        class MetricsLogger(callbacks.Callback):
            def __init__(self, log_dir):
                super().__init__()
                self.log_dir = log_dir
                
            def on_epoch_end(self, epoch, logs=None):
                if logs is None:
                    logs = {}
                
                # Log additional metrics
                metrics_log = {
                    'epoch': epoch,
                    'timestamp': datetime.now().isoformat(),
                    **logs
                }
                
                # Save to JSON
                with open(f"{self.log_dir}/epoch_metrics.json", 'a') as f:
                    json.dump(metrics_log, f)
                    f.write('\n')
        
        metrics_logger = MetricsLogger(self.logs_dir)
        callback_list.append(metrics_logger)
        
        return callback_list
    
    def train(self,
              epochs: int = 100,
              experiment_name: str = "air_quality_model",
              fine_tune_after: int = 50,
              validation_split: float = 0.15,
              test_split: float = 0.15) -> Dict:
        """
        Train the model
        
        Args:
            epochs: Maximum number of training epochs
            experiment_name: Name for this experiment
            fine_tune_after: Epoch after which to unfreeze backbone
            validation_split: Validation data fraction
            test_split: Test data fraction
            
        Returns:
            Training history and results
        """
        self.logger.info(f"Starting training for {epochs} epochs")
        
        # Create datasets
        train_dataset = self.data_loader.create_dataset('train', validation_split, test_split)
        val_dataset = self.data_loader.create_dataset('val', validation_split, test_split)
        
        # Get dataset info
        dataset_info = self.data_loader.get_dataset_info()
        self.logger.info(f"Dataset info: {dataset_info}")
        
        # Create callbacks
        callback_list = self._create_callbacks(experiment_name)
        
        # Phase 1: Train with frozen backbone
        self.logger.info("Phase 1: Training with frozen backbone")
        
        history_phase1 = self.model.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=min(fine_tune_after, epochs),
            callbacks=callback_list,
            verbose=1
        )
        
        # Phase 2: Fine-tuning with unfrozen backbone
        if epochs > fine_tune_after:
            self.logger.info("Phase 2: Fine-tuning with unfrozen backbone")
            
            # Unfreeze backbone
            self.model.unfreeze_backbone(unfreeze_from_layer=-20)
            
            # Recompile with lower learning rate
            self.model.compile_model(learning_rate=self.learning_rate * 0.1)
            
            # Continue training
            history_phase2 = self.model.model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=epochs - fine_tune_after,
                callbacks=callback_list,
                verbose=1
            )
            
            # Combine histories
            combined_history = {}
            for key in history_phase1.history:
                combined_history[key] = history_phase1.history[key] + history_phase2.history[key]
            
            self.history = combined_history
        else:
            self.history = history_phase1.history
        
        # Save final model
        final_model_path = f"{self.model_dir}/{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_final.h5"
        self.model.save_model(final_model_path)
        
        # Plot training history
        self._plot_training_history(experiment_name)
        
        # Evaluate on test set
        test_results = self._evaluate_on_test(test_split)
        
        results = {
            'history': self.history,
            'test_results': test_results,
            'final_model_path': final_model_path,
            'dataset_info': dataset_info
        }
        
        self.logger.info("Training completed successfully!")
        
        return results
    
    def _evaluate_on_test(self, test_split: float) -> Dict:
        """
        Evaluate model on test set
        
        Args:
            test_split: Test data fraction
            
        Returns:
            Test evaluation results
        """
        self.logger.info("Evaluating model on test set...")
        
        # Create test dataset
        test_dataset = self.data_loader.create_dataset('test', 0, test_split)
        
        # Evaluate
        test_results = self.model.model.evaluate(test_dataset, verbose=1)
        
        # Map results to metric names
        metric_names = self.model.model.metrics_names
        results_dict = dict(zip(metric_names, test_results))
        
        self.logger.info(f"Test results: {results_dict}")
        
        return results_dict
    
    def _plot_training_history(self, experiment_name: str):
        """
        Plot training history
        
        Args:
            experiment_name: Name of the experiment
        """
        if self.history is None:
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Training History - {experiment_name}', fontsize=16)
        
        # Loss plots
        axes[0, 0].plot(self.history['loss'], label='Training Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Overall Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Individual output losses
        output_names = ['pm25_output', 'co2_output', 'no2_output', 'aqi_output']
        for i, output_name in enumerate(output_names[:3]):
            row, col = (0, i + 1) if i < 2 else (1, 0)
            if f'{output_name}_loss' in self.history:
                axes[row, col].plot(self.history[f'{output_name}_loss'], label=f'Training {output_name}')
                axes[row, col].plot(self.history[f'val_{output_name}_loss'], label=f'Validation {output_name}')
                axes[row, col].set_title(f'{output_name.replace("_", " ").title()} Loss')
                axes[row, col].set_xlabel('Epoch')
                axes[row, col].set_ylabel('Loss')
                axes[row, col].legend()
                axes[row, col].grid(True)
        
        # AQI accuracy
        if 'aqi_output_categorical_accuracy' in self.history:
            axes[1, 1].plot(self.history['aqi_output_categorical_accuracy'], label='Training Accuracy')
            axes[1, 1].plot(self.history['val_aqi_output_categorical_accuracy'], label='Validation Accuracy')
            axes[1, 1].set_title('AQI Classification Accuracy')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        # Learning rate
        if 'lr' in self.history:
            axes[1, 2].plot(self.history['lr'])
            axes[1, 2].set_title('Learning Rate')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Learning Rate')
            axes[1, 2].set_yscale('log')
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = f"{self.logs_dir}/plots/{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_training_history.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        self.logger.info(f"Training history plot saved to {plot_path}")
    
    def save_training_summary(self, results: Dict, experiment_name: str):
        """
        Save training summary to file
        
        Args:
            results: Training results
            experiment_name: Name of the experiment
        """
        summary = {
            'experiment_name': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'hyperparameters': {
                'image_size': self.image_size,
                'window_size': self.window_size,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate
            },
            'dataset_info': results['dataset_info'],
            'test_results': results['test_results'],
            'final_model_path': results['final_model_path']
        }
        
        # Save summary
        summary_path = f"{self.logs_dir}/{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Training summary saved to {summary_path}")

def main():
    """Main function for training"""
    parser = argparse.ArgumentParser(description='Train Air Quality Prediction Model')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--model_dir', type=str, default='models', help='Model directory')
    parser.add_argument('--logs_dir', type=str, default='logs', help='Logs directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--experiment_name', type=str, default='air_quality_model', help='Experiment name')
    parser.add_argument('--fine_tune_after', type=int, default=50, help='Epoch to start fine-tuning')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = AirQualityTrainer(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        logs_dir=args.logs_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    # Train model
    results = trainer.train(
        epochs=args.epochs,
        experiment_name=args.experiment_name,
        fine_tune_after=args.fine_tune_after
    )
    
    # Save training summary
    trainer.save_training_summary(results, args.experiment_name)
    
    print("\n=== Training Completed ===")
    print(f"Final model saved to: {results['final_model_path']}")
    print(f"Test results: {results['test_results']}")

if __name__ == "__main__":
    main()
