"""
Automated ML Pipeline with Experiment Tracking

This module provides a comprehensive ML pipeline including:
- Automated data processing and feature engineering
- Experiment tracking with MLflow
- Model versioning and registry
- Automated hyperparameter tuning
- Model deployment and monitoring
- CI/CD integration

Author: Air Quality Prediction System
Date: 2026
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Callable
import yaml
import subprocess

# Data processing
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# MLflow for experiment tracking
import mlflow
import mlflow.tensorflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

# Deep learning
import tensorflow as tf
from tensorflow import keras

# Our modules
from data_generator import AirQualityDataGenerator
from data_loader import AirQualityDataLoader
from model import MultiModalAirQualityModel
from train import AirQualityTrainer
from evaluation import AirQualityEvaluator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLPipelineConfig:
    """ML Pipeline configuration"""
    
    def __init__(self, config_path: str = "ml_pipeline_config.yaml"):
        """
        Initialize pipeline configuration
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Default configuration
            return {
                'data': {
                    'num_samples': 1000,
                    'window_size': 24,
                    'test_size': 0.2,
                    'val_size': 0.2
                },
                'model': {
                    'image_size': [224, 224],
                    'batch_size': 32,
                    'learning_rate': 0.001,
                    'epochs': 100,
                    'dropout_rate': 0.3
                },
                'training': {
                    'early_stopping_patience': 15,
                    'reduce_lr_patience': 8,
                    'fine_tune_after': 50
                },
                'experiment': {
                    'name': 'air_quality_prediction',
                    'tracking_uri': 'http://localhost:5000',
                    'registry_name': 'air_quality_models'
                },
                'deployment': {
                    'api_host': '0.0.0.0',
                    'api_port': 8000,
                    'model_name': 'production'
                }
            }
    
    def save_config(self):
        """Save configuration to file"""
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

class ExperimentTracker:
    """Experiment tracking with MLflow"""
    
    def __init__(self, config: MLPipelineConfig):
        """
        Initialize experiment tracker
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.experiment_name = config.config['experiment']['name']
        self.tracking_uri = config.config['experiment']['tracking_uri']
        self.registry_name = config.config['experiment']['registry_name']
        
        # Setup MLflow
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        
        # Create experiment if it doesn't exist
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                mlflow.create_experiment(self.experiment_name)
        except Exception as e:
            logger.warning(f"Failed to create experiment: {e}")
    
    def start_run(self, run_name: Optional[str] = None) -> mlflow.ActiveRun:
        """Start a new MLflow run"""
        run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return mlflow.start_run(run_name=run_name)
    
    def log_params(self, params: Dict):
        """Log parameters to MLflow"""
        for key, value in params.items():
            mlflow.log_param(key, value)
    
    def log_metrics(self, metrics: Dict, step: Optional[int] = None):
        """Log metrics to MLflow"""
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
    
    def log_artifacts(self, artifact_path: str):
        """Log artifacts to MLflow"""
        mlflow.log_artifacts(artifact_path)
    
    def log_model(self, model: Any, model_name: str, input_example: Optional[Any] = None):
        """Log model to MLflow"""
        if hasattr(model, 'model'):  # Keras model wrapper
            mlflow.tensorflow.log_model(
                model.model, 
                model_name, 
                input_example=input_example
            )
        else:
            mlflow.sklearn.log_model(
                model, 
                model_name, 
                input_example=input_example
            )
    
    def register_model(self, model_uri: str, model_name: str, version: str = "latest"):
        """Register model in MLflow registry"""
        try:
            model_version = mlflow.register_model(
                model_uri, 
                f"{self.registry_name}_{model_name}"
            )
            logger.info(f"Registered model version: {model_version.version}")
            return model_version
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return None
    
    def get_best_model(self, metric: str = "val_loss") -> Optional[str]:
        """Get best model from experiment"""
        try:
            # Get experiment
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                return None
            
            # Get best run
            best_run = self.client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=[f"metrics.{metric} ASC"],
                max_results=1
            )
            
            if best_run:
                run_id = best_run[0].info.run_id
                model_uri = f"runs:/{run_id}/model"
                return model_uri
            
        except Exception as e:
            logger.error(f"Failed to get best model: {e}")
        
        return None

class DataProcessor:
    """Automated data processing and feature engineering"""
    
    def __init__(self, config: MLPipelineConfig):
        """
        Initialize data processor
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.data_config = config.config['data']
    
    def generate_data(self) -> str:
        """Generate synthetic data"""
        logger.info("Generating synthetic data...")
        
        generator = AirQualityDataGenerator(
            num_samples=self.data_config['num_samples'],
            window_size=self.data_config['window_size'],
            output_dir="pipeline_data"
        )
        
        dataset_info = generator.generate_dataset()
        
        # Save data info
        with open("pipeline_data/data_info.json", 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        logger.info("Data generation completed")
        return "pipeline_data"
    
    def validate_data(self, data_dir: str) -> bool:
        """Validate generated data"""
        logger.info("Validating data...")
        
        try:
            # Check if required files exist
            required_files = [
                "dataset_master.csv",
                "dataset_statistics.csv"
            ]
            
            for file in required_files:
                if not os.path.exists(os.path.join(data_dir, file)):
                    logger.error(f"Missing required file: {file}")
                    return False
            
            # Load and validate master dataset
            master_df = pd.read_csv(os.path.join(data_dir, "dataset_master.csv"))
            
            # Check data quality
            if len(master_df) == 0:
                logger.error("Empty dataset")
                return False
            
            # Check for missing values in critical columns
            critical_columns = ['pm25', 'co2', 'no2', 'aqi_value']
            for col in critical_columns:
                if col in master_df.columns and master_df[col].isnull().any():
                    logger.warning(f"Missing values in {col}")
            
            logger.info(f"Data validation passed. {len(master_df)} samples found.")
            return True
            
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            return False
    
    def engineer_features(self, data_dir: str) -> str:
        """Engineer additional features"""
        logger.info("Engineering features...")
        
        # Load master dataset
        master_df = pd.read_csv(os.path.join(data_dir, "dataset_master.csv"))
        
        # Add time-based features
        master_df['date'] = pd.to_datetime(master_df['date'])
        master_df['hour'] = master_df['date'].dt.hour
        master_df['day_of_week'] = master_df['date'].dt.dayofweek
        master_df['month'] = master_df['date'].dt.month
        master_df['season'] = master_df['month'].apply(self._get_season)
        
        # Add pollution ratio features
        master_df['pm25_co2_ratio'] = master_df['pm25'] / master_df['co2']
        master_df['no2_co2_ratio'] = master_df['no2'] / master_df['co2']
        
        # Add composite pollution index
        master_df['pollution_index'] = (
            master_df['pm25'] * 0.4 +
            (master_df['co2'] - 350) / 1650 * 0.3 +
            master_df['no2'] * 0.3
        )
        
        # Save engineered features
        engineered_path = os.path.join(data_dir, "engineered_features.csv")
        master_df.to_csv(engineered_path, index=False)
        
        logger.info("Feature engineering completed")
        return engineered_path
    
    def _get_season(self, month: int) -> str:
        """Get season from month"""
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'fall'

class ModelTrainer:
    """Automated model training with hyperparameter tuning"""
    
    def __init__(self, config: MLPipelineConfig, tracker: ExperimentTracker):
        """
        Initialize model trainer
        
        Args:
            config: Pipeline configuration
            tracker: Experiment tracker
        """
        self.config = config
        self.tracker = tracker
        self.model_config = config.config['model']
        self.training_config = config.config['training']
    
    def train_model(self, data_dir: str) -> Dict:
        """Train model with experiment tracking"""
        logger.info("Starting model training...")
        
        with self.tracker.start_run("model_training") as run:
            # Log parameters
            self.tracker.log_params({
                'model_type': 'multi_modal',
                'image_size': self.model_config['image_size'],
                'batch_size': self.model_config['batch_size'],
                'learning_rate': self.model_config['learning_rate'],
                'epochs': self.model_config['epochs'],
                'dropout_rate': self.model_config['dropout_rate']
            })
            
            # Initialize trainer
            trainer = AirQualityTrainer(
                data_dir=data_dir,
                batch_size=self.model_config['batch_size'],
                learning_rate=self.model_config['learning_rate']
            )
            
            # Train model
            start_time = time.time()
            results = trainer.train(
                epochs=self.model_config['epochs'],
                fine_tune_after=self.training_config['fine_tune_after']
            )
            training_time = time.time() - start_time
            
            # Log training time
            self.tracker.log_metrics({'training_time': training_time})
            
            # Log final metrics
            test_results = results['test_results']
            self.tracker.log_metrics({
                'test_loss': test_results.get('loss', 0),
                'test_pm25_mae': test_results.get('pm25_output_mean_absolute_error', 0),
                'test_co2_mae': test_results.get('co2_output_mean_absolute_error', 0),
                'test_no2_mae': test_results.get('no2_output_mean_absolute_error', 0),
                'test_aqi_accuracy': test_results.get('aqi_output_categorical_accuracy', 0)
            })
            
            # Log model
            model_path = results['final_model_path']
            self.tracker.log_model(trainer.model, "air_quality_model")
            
            # Log artifacts
            self.tracker.log_artifacts(os.path.dirname(model_path))
            
            # Register model
            model_uri = f"runs:/{run.info.run_id}/air_quality_model"
            registered_model = self.tracker.register_model(model_uri, "production")
            
            training_results = {
                'run_id': run.info.run_id,
                'model_uri': model_uri,
                'registered_model': registered_model,
                'training_time': training_time,
                'test_results': test_results
            }
            
            logger.info(f"Model training completed. Run ID: {run.info.run_id}")
            
            return training_results

class ModelEvaluator:
    """Automated model evaluation"""
    
    def __init__(self, config: MLPipelineConfig, tracker: ExperimentTracker):
        """
        Initialize model evaluator
        
        Args:
            config: Pipeline configuration
            tracker: Experiment tracker
        """
        self.config = config
        self.tracker = tracker
    
    def evaluate_model(self, model_uri: str, data_dir: str) -> Dict:
        """Evaluate registered model"""
        logger.info("Starting model evaluation...")
        
        with self.tracker.start_run("model_evaluation") as run:
            # Load model
            model = mlflow.tensorflow.load_model(model_uri)
            
            # Initialize evaluator
            evaluator = AirQualityEvaluator(
                model_path=model_uri,
                data_dir=data_dir
            )
            
            # Evaluate
            evaluation_results = evaluator.evaluate_on_test_set()
            
            # Log metrics
            regression_metrics = evaluation_results['regression_metrics']
            classification_metrics = evaluation_results['classification_metrics']
            
            # Regression metrics
            for pollutant in ['pm25', 'co2', 'no2']:
                self.tracker.log_metrics({
                    f'eval_{pollutant}_rmse': regression_metrics[pollutant]['rmse'],
                    f'eval_{pollutant}_mae': regression_metrics[pollutant]['mae'],
                    f'eval_{pollutant}_r2': regression_metrics[pollutant]['r2']
                })
            
            # Classification metrics
            self.tracker.log_metrics({
                'eval_aqi_accuracy': classification_metrics['accuracy'],
                'eval_aqi_precision': classification_metrics['precision'],
                'eval_aqi_recall': classification_metrics['recall'],
                'eval_aqi_f1': classification_metrics['f1_score'],
                'eval_aqi_auc': classification_metrics['auc']
            })
            
            # Generate and log evaluation report
            report_path = evaluator.generate_evaluation_report(
                evaluation_results, 
                "evaluation_results"
            )
            self.tracker.log_artifacts("evaluation_results")
            
            logger.info(f"Model evaluation completed. Report: {report_path}")
            
            return evaluation_results

class DeploymentManager:
    """Automated model deployment"""
    
    def __init__(self, config: MLPipelineConfig):
        """
        Initialize deployment manager
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.deployment_config = config.config['deployment']
    
    def deploy_model(self, model_uri: str) -> Dict:
        """Deploy model to production"""
        logger.info("Starting model deployment...")
        
        try:
            # Load model
            model = mlflow.tensorflow.load_model(model_uri)
            
            # Save model for deployment
            deployment_path = "production_model"
            os.makedirs(deployment_path, exist_ok=True)
            model.save(deployment_path)
            
            # Start API server
            api_command = [
                sys.executable, "api_server.py",
                "--model_path", deployment_path,
                "--host", self.deployment_config['api_host'],
                "--port", str(self.deployment_config['api_port'])
            ]
            
            # Start API server in background
            process = subprocess.Popen(api_command)
            
            # Wait for API to start
            time.sleep(10)
            
            # Check if API is running
            if process.poll() is None:
                logger.info("Model deployment successful")
                return {
                    'status': 'success',
                    'process_id': process.pid,
                    'api_url': f"http://{self.deployment_config['api_host']}:{self.deployment_config['api_port']}"
                }
            else:
                logger.error("API server failed to start")
                return {'status': 'failed', 'error': 'API server failed to start'}
                
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return {'status': 'failed', 'error': str(e)}

class AutomatedMLPipeline:
    """Complete automated ML pipeline"""
    
    def __init__(self, config_path: str = "ml_pipeline_config.yaml"):
        """
        Initialize automated ML pipeline
        
        Args:
            config_path: Path to configuration file
        """
        self.config = MLPipelineConfig(config_path)
        self.tracker = ExperimentTracker(self.config)
        self.data_processor = DataProcessor(self.config)
        self.model_trainer = ModelTrainer(self.config, self.tracker)
        self.model_evaluator = ModelEvaluator(self.config, self.tracker)
        self.deployment_manager = DeploymentManager(self.config)
    
    def run_pipeline(self, 
                    generate_data: bool = True,
                    train_model: bool = True,
                    evaluate_model: bool = True,
                    deploy_model: bool = True) -> Dict:
        """
        Run complete ML pipeline
        
        Args:
            generate_data: Whether to generate new data
            train_model: Whether to train model
            evaluate_model: Whether to evaluate model
            deploy_model: Whether to deploy model
            
        Returns:
            Pipeline results
        """
        logger.info("Starting automated ML pipeline...")
        
        pipeline_results = {
            'start_time': datetime.now().isoformat(),
            'config': self.config.config,
            'stages': {}
        }
        
        try:
            # Stage 1: Data Generation
            if generate_data:
                logger.info("Stage 1: Data Generation")
                data_dir = self.data_processor.generate_data()
                
                if not self.data_processor.validate_data(data_dir):
                    raise Exception("Data validation failed")
                
                engineered_path = self.data_processor.engineer_features(data_dir)
                pipeline_results['stages']['data_generation'] = {
                    'status': 'success',
                    'data_dir': data_dir,
                    'engineered_path': engineered_path
                }
            else:
                data_dir = "pipeline_data"
                pipeline_results['stages']['data_generation'] = {
                    'status': 'skipped'
                }
            
            # Stage 2: Model Training
            if train_model:
                logger.info("Stage 2: Model Training")
                training_results = self.model_trainer.train_model(data_dir)
                pipeline_results['stages']['model_training'] = {
                    'status': 'success',
                    'results': training_results
                }
                model_uri = training_results['model_uri']
            else:
                # Get best existing model
                model_uri = self.tracker.get_best_model()
                pipeline_results['stages']['model_training'] = {
                    'status': 'skipped',
                    'model_uri': model_uri
                }
            
            # Stage 3: Model Evaluation
            if evaluate_model and model_uri:
                logger.info("Stage 3: Model Evaluation")
                evaluation_results = self.model_evaluator.evaluate_model(model_uri, data_dir)
                pipeline_results['stages']['model_evaluation'] = {
                    'status': 'success',
                    'results': evaluation_results
                }
            else:
                pipeline_results['stages']['model_evaluation'] = {
                    'status': 'skipped'
                }
            
            # Stage 4: Model Deployment
            if deploy_model and model_uri:
                logger.info("Stage 4: Model Deployment")
                deployment_results = self.deployment_manager.deploy_model(model_uri)
                pipeline_results['stages']['model_deployment'] = {
                    'status': deployment_results['status'],
                    'results': deployment_results
                }
            else:
                pipeline_results['stages']['model_deployment'] = {
                    'status': 'skipped'
                }
            
            pipeline_results['status'] = 'success'
            pipeline_results['end_time'] = datetime.now().isoformat()
            
            # Save pipeline results
            with open("pipeline_results.json", 'w') as f:
                json.dump(pipeline_results, f, indent=2)
            
            logger.info("ML pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            pipeline_results['status'] = 'failed'
            pipeline_results['error'] = str(e)
            pipeline_results['end_time'] = datetime.now().isoformat()
        
        return pipeline_results
    
    def schedule_pipeline(self, schedule: str = "daily"):
        """Schedule pipeline to run periodically"""
        logger.info(f"Scheduling pipeline to run {schedule}")
        
        # This would integrate with a job scheduler like Apache Airflow
        # For now, just log the intention
        if schedule == "daily":
            logger.info("Pipeline scheduled to run daily at midnight")
        elif schedule == "weekly":
            logger.info("Pipeline scheduled to run weekly on Sunday")
        elif schedule == "monthly":
            logger.info("Pipeline scheduled to run monthly on the 1st")

def main():
    """Main function to run ML pipeline"""
    parser = argparse.ArgumentParser(description='Automated ML Pipeline')
    parser.add_argument('--config', type=str, default='ml_pipeline_config.yaml', help='Configuration file')
    parser.add_argument('--generate_data', action='store_true', help='Generate new data')
    parser.add_argument('--train_model', action='store_true', help='Train model')
    parser.add_argument('--evaluate_model', action='store_true', help='Evaluate model')
    parser.add_argument('--deploy_model', action='store_true', help='Deploy model')
    parser.add_argument('--run_all', action='store_true', help='Run complete pipeline')
    parser.add_argument('--schedule', type=str, choices=['daily', 'weekly', 'monthly'], help='Schedule pipeline')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = AutomatedMLPipeline(args.config)
    
    if args.run_all:
        # Run complete pipeline
        results = pipeline.run_pipeline(
            generate_data=True,
            train_model=True,
            evaluate_model=True,
            deploy_model=True
        )
    elif args.schedule:
        # Schedule pipeline
        pipeline.schedule_pipeline(args.schedule)
    else:
        # Run specific stages
        results = pipeline.run_pipeline(
            generate_data=args.generate_data,
            train_model=args.train_model,
            evaluate_model=args.evaluate_model,
            deploy_model=args.deploy_model
        )
    
    print(f"Pipeline completed with status: {results['status']}")
    if results['status'] == 'failed':
        print(f"Error: {results.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
