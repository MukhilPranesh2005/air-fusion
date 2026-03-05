"""
Evaluation Metrics Script for Multi-Modal Air Quality Prediction

This script implements comprehensive evaluation metrics including:
- Regression metrics (RMSE, MAE, R²) for pollutant prediction
- Classification metrics (Accuracy, F1, Precision, Recall) for AQI
- Statistical analysis and visualization
- Performance comparison and reporting

Author: Air Quality Prediction System
Date: 2026
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import label_binarize

import tensorflow as tf

from data_loader import AirQualityDataLoader
from model import MultiModalAirQualityModel

class AirQualityEvaluator:
    """
    Comprehensive evaluator for air quality prediction model
    """
    
    def __init__(self,
                 model_path: str,
                 data_dir: str = "data",
                 image_size: Tuple[int, int] = (224, 224),
                 window_size: int = 24,
                 batch_size: int = 32):
        """
        Initialize evaluator
        
        Args:
            model_path: Path to trained model
            data_dir: Directory containing dataset
            image_size: Input image size
            window_size: Time window size
            batch_size: Batch size for evaluation
        """
        self.model_path = model_path
        self.data_dir = data_dir
        self.image_size = image_size
        self.window_size = window_size
        self.batch_size = batch_size
        
        # Load model and data loader
        self._load_model()
        self._load_data_loader()
        
        # AQI categories
        self.aqi_categories = ['Good', 'Moderate', 'Unhealthy for Sensitive', 
                              'Unhealthy', 'Very Unhealthy', 'Hazardous']
    
    def _load_model(self):
        """Load trained model"""
        self.model = MultiModalAirQualityModel(
            image_size=self.image_size,
            window_size=self.window_size
        )
        self.model.load_model(self.model_path)
        print(f"Model loaded from {self.model_path}")
    
    def _load_data_loader(self):
        """Load data loader"""
        self.data_loader = AirQualityDataLoader(
            data_dir=self.data_dir,
            image_size=self.image_size,
            window_size=self.window_size,
            batch_size=self.batch_size,
            shuffle=False
        )
    
    def evaluate_on_test_set(self) -> Dict:
        """
        Comprehensive evaluation on test set
        
        Returns:
            Dictionary with all evaluation metrics
        """
        print("Evaluating model on test set...")
        
        # Create test dataset
        test_dataset = self.data_loader.create_dataset('test', 0, 0.15)
        
        # Collect predictions and ground truth
        all_predictions = []
        all_ground_truth = []
        
        print("Making predictions on test set...")
        for batch_inputs, batch_outputs in test_dataset:
            # Make predictions
            predictions = self.model.model.predict(batch_inputs, verbose=0)
            
            # Extract predictions
            batch_preds = {
                'pm25': predictions[0].flatten(),
                'co2': predictions[1].flatten(),
                'no2': predictions[2].flatten(),
                'aqi': predictions[3]
            }
            
            # Extract ground truth
            batch_truth = {
                'pm25': batch_outputs['pm25_output'].numpy(),
                'co2': batch_outputs['co2_output'].numpy(),
                'no2': batch_outputs['no2_output'].numpy(),
                'aqi': batch_outputs['aqi_output'].numpy()
            }
            
            all_predictions.append(batch_preds)
            all_ground_truth.append(batch_truth)
        
        # Concatenate all batches
        predictions = {
            key: np.concatenate([batch[key] for batch in all_predictions])
            for key in all_predictions[0].keys()
        }
        
        ground_truth = {
            key: np.concatenate([batch[key] for batch in all_ground_truth])
            for key in all_ground_truth[0].keys()
        }
        
        # Inverse transform to original scale
        predictions_original = self._inverse_transform_predictions(predictions)
        ground_truth_original = self._inverse_transform_ground_truth(ground_truth)
        
        # Calculate metrics
        regression_metrics = self._calculate_regression_metrics(predictions_original, ground_truth_original)
        classification_metrics = self._calculate_classification_metrics(predictions, ground_truth)
        
        # Combine results
        evaluation_results = {
            'timestamp': datetime.now().isoformat(),
            'dataset_size': len(predictions_original['pm25']),
            'regression_metrics': regression_metrics,
            'classification_metrics': classification_metrics,
            'predictions': predictions_original,
            'ground_truth': ground_truth_original
        }
        
        return evaluation_results
    
    def _inverse_transform_predictions(self, predictions: Dict) -> Dict:
        """Inverse transform predictions to original scale"""
        # Create dummy predictions for inverse transform
        dummy_predictions = np.array([
            predictions['pm25'],
            predictions['co2'],
            predictions['no2']
        ]).T
        
        # Inverse transform
        original_scale = self.data_loader.label_scaler.inverse_transform(dummy_predictions)
        
        return {
            'pm25': original_scale[:, 0],
            'co2': original_scale[:, 1],
            'no2': original_scale[:, 2],
            'aqi': predictions['aqi']
        }
    
    def _inverse_transform_ground_truth(self, ground_truth: Dict) -> Dict:
        """Inverse transform ground truth to original scale"""
        # Create dummy ground truth for inverse transform
        dummy_ground_truth = np.array([
            ground_truth['pm25'],
            ground_truth['co2'],
            ground_truth['no2']
        ]).T
        
        # Inverse transform
        original_scale = self.data_loader.label_scaler.inverse_transform(dummy_ground_truth)
        
        return {
            'pm25': original_scale[:, 0],
            'co2': original_scale[:, 1],
            'no2': original_scale[:, 2],
            'aqi': ground_truth['aqi']
        }
    
    def _calculate_regression_metrics(self, predictions: Dict, ground_truth: Dict) -> Dict:
        """Calculate regression metrics for pollutant predictions"""
        pollutants = ['pm25', 'co2', 'no2']
        metrics = {}
        
        for pollutant in pollutants:
            pred = predictions[pollutant]
            true = ground_truth[pollutant]
            
            # Calculate metrics
            mse = mean_squared_error(true, pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(true, pred)
            r2 = r2_score(true, pred)
            
            # Calculate percentage error
            mape = np.mean(np.abs((true - pred) / true)) * 100
            
            metrics[pollutant] = {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2),
                'mape': float(mape),
                'mean_true': float(np.mean(true)),
                'std_true': float(np.std(true)),
                'mean_pred': float(np.mean(pred)),
                'std_pred': float(np.std(pred))
            }
        
        return metrics
    
    def _calculate_classification_metrics(self, predictions: Dict, ground_truth: Dict) -> Dict:
        """Calculate classification metrics for AQI prediction"""
        # Convert probabilities to class predictions
        pred_classes = np.argmax(predictions['aqi'], axis=1)
        true_classes = np.argmax(ground_truth['aqi'], axis=1)
        
        # Calculate basic metrics
        accuracy = accuracy_score(true_classes, pred_classes)
        precision = precision_score(true_classes, pred_classes, average='weighted', zero_division=0)
        recall = recall_score(true_classes, pred_classes, average='weighted', zero_division=0)
        f1 = f1_score(true_classes, pred_classes, average='weighted', zero_division=0)
        
        # Calculate per-class metrics
        class_report = classification_report(
            true_classes, pred_classes, 
            target_names=self.aqi_categories,
            output_dict=True,
            zero_division=0
        )
        
        # Calculate confusion matrix
        cm = confusion_matrix(true_classes, pred_classes)
        
        # Calculate AUC (one-vs-rest)
        try:
            y_true_bin = label_binarize(true_classes, classes=range(len(self.aqi_categories)))
            auc = roc_auc_score(y_true_bin, predictions['aqi'], average='weighted', multi_class='ovr')
        except:
            auc = 0.0
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc': float(auc),
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'class_distribution': {
                'true': dict(zip(self.aqi_categories, np.bincount(true_classes, minlength=len(self.aqi_categories)))),
                'predicted': dict(zip(self.aqi_categories, np.bincount(pred_classes, minlength=len(self.aqi_categories))))
            }
        }
        
        return metrics
    
    def generate_evaluation_report(self, evaluation_results: Dict, output_dir: str = "evaluation_results") -> str:
        """
        Generate comprehensive evaluation report
        
        Args:
            evaluation_results: Results from evaluation
            output_dir: Directory to save report
            
        Returns:
            Path to generated report
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = f"{output_dir}/evaluation_report_{timestamp}.md"
        
        # Generate report content
        report_content = self._generate_report_content(evaluation_results)
        
        # Save report
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        # Save raw results as JSON
        json_path = f"{output_dir}/evaluation_results_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        print(f"Evaluation report saved to {report_path}")
        print(f"Raw results saved to {json_path}")
        
        return report_path
    
    def _generate_report_content(self, results: Dict) -> str:
        """Generate markdown report content"""
        content = f"""# Air Quality Prediction Model Evaluation Report

**Generated:** {results['timestamp']}  
**Dataset Size:** {results['dataset_size']} samples  
**Model:** {self.model_path}

## Executive Summary

This report presents a comprehensive evaluation of the multi-modal air quality prediction model. The model was evaluated on {results['dataset_size']} test samples using both regression and classification metrics.

## Regression Metrics (Pollutant Prediction)

### PM2.5 Prediction
- **RMSE:** {results['regression_metrics']['pm25']['rmse']:.2f} μg/m³
- **MAE:** {results['regression_metrics']['pm25']['mae']:.2f} μg/m³
- **R²:** {results['regression_metrics']['pm25']['r2']:.3f}
- **MAPE:** {results['regression_metrics']['pm25']['mape']:.1f}%
- **Mean True Value:** {results['regression_metrics']['pm25']['mean_true']:.1f} μg/m³

### CO₂ Prediction
- **RMSE:** {results['regression_metrics']['co2']['rmse']:.2f} ppm
- **MAE:** {results['regression_metrics']['co2']['mae']:.2f} ppm
- **R²:** {results['regression_metrics']['co2']['r2']:.3f}
- **MAPE:** {results['regression_metrics']['co2']['mape']:.1f}%
- **Mean True Value:** {results['regression_metrics']['co2']['mean_true']:.1f} ppm

### NO₂ Prediction
- **RMSE:** {results['regression_metrics']['no2']['rmse']:.2f} ppb
- **MAE:** {results['regression_metrics']['no2']['mae']:.2f} ppb
- **R²:** {results['regression_metrics']['no2']['r2']:.3f}
- **MAPE:** {results['regression_metrics']['no2']['mape']:.1f}%
- **Mean True Value:** {results['regression_metrics']['no2']['mean_true']:.1f} ppb

## Classification Metrics (AQI Prediction)

### Overall Performance
- **Accuracy:** {results['classification_metrics']['accuracy']:.3f} ({results['classification_metrics']['accuracy']*100:.1f}%)
- **Precision (Weighted):** {results['classification_metrics']['precision']:.3f}
- **Recall (Weighted):** {results['classification_metrics']['recall']:.3f}
- **F1-Score (Weighted):** {results['classification_metrics']['f1_score']:.3f}
- **AUC (Weighted):** {results['classification_metrics']['auc']:.3f}

### Per-Class Performance
"""
        
        # Add per-class metrics
        class_report = results['classification_metrics']['classification_report']
        for class_name in self.aqi_categories:
            if class_name in class_report:
                metrics = class_report[class_name]
                content += f"""
#### {class_name}
- **Precision:** {metrics['precision']:.3f}
- **Recall:** {metrics['recall']:.3f}
- **F1-Score:** {metrics['f1-score']:.3f}
- **Support:** {metrics['support']}
"""
        
        content += """
## Performance Analysis

### Strengths
"""
        
        # Identify strengths
        strengths = []
        for pollutant in ['pm25', 'co2', 'no2']:
            r2 = results['regression_metrics'][pollutant]['r2']
            if r2 > 0.8:
                strengths.append(f"Excellent {pollutant.upper()} prediction (R² = {r2:.3f})")
            elif r2 > 0.6:
                strengths.append(f"Good {pollutant.upper()} prediction (R² = {r2:.3f})")
        
        accuracy = results['classification_metrics']['accuracy']
        if accuracy > 0.8:
            strengths.append(f"High AQI classification accuracy ({accuracy:.1%})")
        elif accuracy > 0.7:
            strengths.append(f"Good AQI classification accuracy ({accuracy:.1%})")
        
        if strengths:
            for strength in strengths:
                content += f"- {strength}\n"
        else:
            content += "- Model performance requires improvement\n"
        
        content += """
### Areas for Improvement
"""
        
        # Identify areas for improvement
        improvements = []
        for pollutant in ['pm25', 'co2', 'no2']:
            r2 = results['regression_metrics'][pollutant]['r2']
            if r2 < 0.5:
                improvements.append(f"Poor {pollutant.upper()} prediction (R² = {r2:.3f})")
        
        if accuracy < 0.7:
            improvements.append(f"Low AQI classification accuracy ({accuracy:.1%})")
        
        if improvements:
            for improvement in improvements:
                content += f"- {improvement}\n"
        else:
            content += "- No major issues identified\n"
        
        content += """
## Recommendations

Based on the evaluation results, the following recommendations are made:

1. **Model Architecture:** Consider adjusting the fusion mechanism if certain modalities are underperforming
2. **Data Quality:** Investigate outliers and data quality issues for poorly performing pollutants
3. **Training Strategy:** Implement class balancing if AQI classification shows bias
4. **Feature Engineering:** Explore additional temporal features or image augmentations
5. **Ensemble Methods:** Consider ensemble approaches for improved robustness

## Conclusion

"""
        
        # Overall assessment
        avg_r2 = np.mean([results['regression_metrics'][p]['r2'] for p in ['pm25', 'co2', 'no2']])
        accuracy = results['classification_metrics']['accuracy']
        
        if avg_r2 > 0.8 and accuracy > 0.8:
            assessment = "excellent"
        elif avg_r2 > 0.6 and accuracy > 0.7:
            assessment = "good"
        elif avg_r2 > 0.4 and accuracy > 0.6:
            assessment = "moderate"
        else:
            assessment = "needs improvement"
        
        content += f"""The model demonstrates {assessment} performance overall. With an average R² of {avg_r2:.3f} for regression tasks and {accuracy:.1%} accuracy for AQI classification, the system shows promise for real-world air quality prediction applications.

---
*Report generated by Air Quality Prediction System*
"""
        
        return content
    
    def plot_evaluation_results(self, evaluation_results: Dict, output_dir: str = "evaluation_results"):
        """
        Create visualization plots for evaluation results
        
        Args:
            evaluation_results: Results from evaluation
            output_dir: Directory to save plots
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Evaluation Results', fontsize=16, fontweight='bold')
        
        # Regression plots
        pollutants = ['pm25', 'co2', 'no2']
        pollutant_names = ['PM2.5 (μg/m³)', 'CO₂ (ppm)', 'NO₂ (ppb)']
        
        for i, (pollutant, name) in enumerate(zip(pollutants, pollutant_names)):
            ax = axes[0, i]
            
            true = evaluation_results['ground_truth'][pollutant]
            pred = evaluation_results['predictions'][pollutant]
            
            # Scatter plot
            ax.scatter(true, pred, alpha=0.6, s=20)
            
            # Perfect prediction line
            min_val, max_val = min(true.min(), pred.min()), max(true.max(), pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            # Metrics
            r2 = evaluation_results['regression_metrics'][pollutant]['r2']
            rmse = evaluation_results['regression_metrics'][pollutant]['rmse']
            
            ax.set_xlabel(f'True {name}')
            ax.set_ylabel(f'Predicted {name}')
            ax.set_title(f'{name} Prediction\nR² = {r2:.3f}, RMSE = {rmse:.2f}')
            ax.grid(True, alpha=0.3)
        
        # Classification plots
        # Confusion Matrix
        cm = np.array(evaluation_results['classification_metrics']['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.aqi_categories, yticklabels=self.aqi_categories,
                   ax=axes[1, 0])
        axes[1, 0].set_title('Confusion Matrix')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('True')
        
        # Class distribution
        class_dist = evaluation_results['classification_metrics']['class_distribution']
        categories = list(class_dist['true'].keys())
        true_counts = list(class_dist['true'].values())
        pred_counts = list(class_dist['predicted'].values())
        
        x = np.arange(len(categories))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, true_counts, width, label='True', alpha=0.8)
        axes[1, 1].bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.8)
        axes[1, 1].set_xlabel('AQI Category')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Class Distribution')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(categories, rotation=45)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Per-class metrics
        class_report = evaluation_results['classification_metrics']['classification_report']
        metrics_data = []
        for category in self.aqi_categories:
            if category in class_report:
                metrics_data.append([
                    class_report[category]['precision'],
                    class_report[category]['recall'],
                    class_report[category]['f1-score']
                ])
        
        if metrics_data:
            metrics_array = np.array(metrics_data).T
            im = axes[1, 2].imshow(metrics_array, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
            axes[1, 2].set_xticks(range(len(self.aqi_categories)))
            axes[1, 2].set_xticklabels(self.aqi_categories, rotation=45)
            axes[1, 2].set_yticks(range(3))
            axes[1, 2].set_yticklabels(['Precision', 'Recall', 'F1-Score'])
            axes[1, 2].set_title('Per-Class Classification Metrics')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=axes[1, 2])
            cbar.set_label('Score')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = f"{output_dir}/evaluation_plots_{timestamp}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"Evaluation plots saved to {plot_path}")

def main():
    """Main function for evaluation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Air Quality Prediction Model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = AirQualityEvaluator(
        model_path=args.model_path,
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )
    
    # Run evaluation
    print("Starting comprehensive evaluation...")
    results = evaluator.evaluate_on_test_set()
    
    # Generate report
    report_path = evaluator.generate_evaluation_report(results, args.output_dir)
    
    # Create plots
    evaluator.plot_evaluation_results(results, args.output_dir)
    
    # Print summary
    print("\n=== Evaluation Summary ===")
    print(f"Test samples: {results['dataset_size']}")
    
    # Regression summary
    print("\nRegression Metrics:")
    for pollutant in ['pm25', 'co2', 'no2']:
        r2 = results['regression_metrics'][pollutant]['r2']
        rmse = results['regression_metrics'][pollutant]['rmse']
        print(f"  {pollutant.upper()}: R² = {r2:.3f}, RMSE = {rmse:.2f}")
    
    # Classification summary
    print(f"\nClassification Metrics:")
    accuracy = results['classification_metrics']['accuracy']
    f1 = results['classification_metrics']['f1_score']
    print(f"  Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"  F1-Score: {f1:.3f}")
    
    print(f"\nDetailed report saved to: {report_path}")

if __name__ == "__main__":
    main()
