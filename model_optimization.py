"""
Model Optimization and Quantization for Deployment

This module provides advanced model optimization techniques including:
- Model quantization (INT8, FP16)
- Pruning and sparsity optimization
- Knowledge distillation
- TensorRT optimization
- Mobile deployment optimization

Author: Air Quality Prediction System
Date: 2026
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_model_optimization as tfmot
from typing import Dict, List, Tuple, Optional, Union
import json
import time
import tempfile
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelOptimizer:
    """Advanced model optimization for deployment"""
    
    def __init__(self, 
                 model_path: str,
                 output_dir: str = "optimized_models"):
        """
        Initialize model optimizer
        
        Args:
            model_path: Path to trained model
            output_dir: Directory to save optimized models
        """
        self.model_path = model_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load original model
        self.original_model = keras.models.load_model(model_path)
        self.optimization_history = {}
        
        logger.info(f"Loaded model from {model_path}")
        logger.info(f"Original model summary:")
        self.original_model.summary()
    
    def get_model_size(self, model: keras.Model) -> Dict[str, float]:
        """Get model size information"""
        # Calculate model parameters
        total_params = model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        
        # Save model temporarily to get file size
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
            model.save(tmp_file.name)
            file_size_mb = os.path.getsize(tmp_file.name) / (1024 * 1024)
            os.unlink(tmp_file.name)
        
        return {
            'total_parameters': int(total_params),
            'trainable_parameters': int(trainable_params),
            'file_size_mb': file_size_mb
        }
    
    def benchmark_model(self, 
                      model: keras.Model,
                      input_shapes: Dict[str, Tuple],
                      num_runs: int = 100) -> Dict[str, float]:
        """
        Benchmark model performance
        
        Args:
            model: Model to benchmark
            input_shapes: Dictionary of input shapes
            num_runs: Number of inference runs
            
        Returns:
            Performance metrics
        """
        # Create dummy inputs
        dummy_inputs = {}
        for input_name, shape in input_shapes.items():
            dummy_inputs[input_name] = np.random.random(shape).astype(np.float32)
        
        # Warm up
        for _ in range(10):
            _ = model.predict(dummy_inputs, verbose=0)
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_runs):
            _ = model.predict(dummy_inputs, verbose=0)
        end_time = time.time()
        
        avg_inference_time = (end_time - start_time) / num_runs
        throughput = num_runs / (end_time - start_time)
        
        return {
            'avg_inference_time_ms': avg_inference_time * 1000,
            'throughput_fps': throughput,
            'total_time_s': end_time - start_time
        }
    
    def apply_post_training_quantization(self, 
                                      quantization_type: str = 'dynamic') -> keras.Model:
        """
        Apply post-training quantization
        
        Args:
            quantization_type: 'dynamic', 'full', or 'float16'
            
        Returns:
            Quantized model
        """
        logger.info(f"Applying {quantization_type} quantization...")
        
        if quantization_type == 'dynamic':
            # Dynamic range quantization
            converter = tf.lite.TFLiteConverter.from_keras_model(self.original_model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
        elif quantization_type == 'full':
            # Full integer quantization (requires representative dataset)
            converter = tf.lite.TFLiteConverter.from_keras_model(self.original_model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = self._representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            
        elif quantization_type == 'float16':
            # Float16 quantization
            converter = tf.lite.TFLiteConverter.from_keras_model(self.original_model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        
        else:
            raise ValueError(f"Unsupported quantization type: {quantization_type}")
        
        # Convert model
        quantized_model = converter.convert()
        
        # Save quantized model
        quantized_path = os.path.join(self.output_dir, f"model_{quantization_type}_quantized.tflite")
        with open(quantized_path, 'wb') as f:
            f.write(quantized_model)
        
        logger.info(f"Quantized model saved to {quantized_path}")
        
        # Store optimization info
        self.optimization_history[quantization_type] = {
            'type': 'quantization',
            'method': quantization_type,
            'path': quantized_path,
            'timestamp': time.time()
        }
        
        return quantized_model
    
    def _representative_dataset(self):
        """Generate representative dataset for quantization"""
        # This should be replaced with actual representative data
        for _ in range(100):
            # Generate dummy data with same shapes as training data
            yield {
                'image': np.random.random((1, 224, 224, 3)).astype(np.float32),
                'environmental': np.random.random((1, 24, 6)).astype(np.float32),
                'biosensing': np.random.random((1, 24, 3)).astype(np.float32)
            }
    
    def apply_pruning(self, 
                    pruning_schedule: tfmot.sparsity.keras.ConstantSparsity,
                    fine_tune_epochs: int = 5) -> keras.Model:
        """
        Apply model pruning
        
        Args:
            pruning_schedule: Pruning schedule
            fine_tune_epochs: Number of epochs for fine-tuning
            
        Returns:
            Pruned model
        """
        logger.info("Applying model pruning...")
        
        # Apply pruning to the model
        def apply_pruning_to_layer(layer):
            if isinstance(layer, (layers.Conv2D, layers.Dense)):
                return tfmot.sparsity.keras.prune_low_magnitude(layer, pruning_schedule=pruning_schedule)
            return layer
        
        pruned_model = keras.models.clone_model(
            self.original_model,
            clone_function=apply_pruning_to_layer
        )
        
        # Compile pruned model
        pruned_model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        # Fine-tune pruned model
        logger.info(f"Fine-tuning pruned model for {fine_tune_epochs} epochs...")
        
        # Create dummy data for fine-tuning
        dummy_inputs = {
            'image': np.random.random((32, 224, 224, 3)),
            'environmental': np.random.random((32, 24, 6)),
            'biosensing': np.random.random((32, 24, 3))
        }
        
        dummy_outputs = {
            'pm25_output': np.random.random((32, 1)),
            'co2_output': np.random.random((32, 1)),
            'no2_output': np.random.random((32, 1)),
            'aqi_output': np.random.random((32, 6))
        }
        
        # Add pruning callbacks
        callbacks = [
            tfmot.sparsity.keras.UpdatePruningStep(),
            tfmot.sparsity.keras.PruningSummaries(log_dir=os.path.join(self.output_dir, 'pruning_logs'))
        ]
        
        # Fine-tune
        pruned_model.fit(
            dummy_inputs, dummy_outputs,
            epochs=fine_tune_epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Strip pruning wrappers
        final_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
        
        # Save pruned model
        pruned_path = os.path.join(self.output_dir, "model_pruned.h5")
        final_model.save(pruned_path)
        
        logger.info(f"Pruned model saved to {pruned_path}")
        
        # Store optimization info
        self.optimization_history['pruning'] = {
            'type': 'pruning',
            'method': 'constant_sparsity',
            'path': pruned_path,
            'timestamp': time.time()
        }
        
        return final_model
    
    def apply_knowledge_distillation(self,
                                  teacher_model: keras.Model,
                                  student_model: Optional[keras.Model] = None,
                                  distillation_epochs: int = 10,
                                  temperature: float = 3.0,
                                  alpha: float = 0.7) -> keras.Model:
        """
        Apply knowledge distillation
        
        Args:
            teacher_model: Teacher model (original model)
            student_model: Student model (smaller model)
            distillation_epochs: Number of training epochs
            temperature: Distillation temperature
            alpha: Weight for student loss vs distillation loss
            
        Returns:
            Distilled student model
        """
        logger.info("Applying knowledge distillation...")
        
        # Create student model if not provided
        if student_model is None:
            student_model = self._create_student_model()
        
        # Define distillation loss
        class DistillationLoss(keras.losses.Loss):
            def __init__(self, temperature=3.0, alpha=0.7):
                super().__init__()
                self.temperature = temperature
                self.alpha = alpha
            
            def call(self, y_true, y_pred):
                # y_true contains both true labels and teacher predictions
                true_labels = y_true[:4]  # First 4 outputs are true labels
                teacher_preds = y_true[4:]  # Last 4 outputs are teacher predictions
                
                # Student loss
                student_loss = 0
                for i in range(4):
                    student_loss += keras.losses.mse(true_labels[i], y_pred[i])
                
                # Distillation loss
                distillation_loss = 0
                for i in range(4):
                    # Soft targets from teacher
                    teacher_soft = tf.nn.softmax(teacher_preds[i] / self.temperature)
                    student_soft = tf.nn.softmax(y_pred[i] / self.temperature)
                    distillation_loss += keras.losses.kullback_leibler_divergence(
                        teacher_soft, student_soft
                    ) * (self.temperature ** 2)
                
                return self.alpha * student_loss + (1 - self.alpha) * distillation_loss
        
        # Compile student model with distillation loss
        student_model.compile(
            optimizer='adam',
            loss=DistillationLoss(temperature=temperature, alpha=alpha),
            metrics=['mae']
        )
        
        # Prepare training data
        # Generate dummy data with teacher predictions
        dummy_inputs = {
            'image': np.random.random((32, 224, 224, 3)),
            'environmental': np.random.random((32, 24, 6)),
            'biosensing': np.random.random((32, 24, 3))
        }
        
        # Get teacher predictions
        teacher_preds = teacher_model.predict(dummy_inputs, verbose=0)
        
        # Prepare training targets (true labels + teacher predictions)
        dummy_outputs = {
            'pm25_output': np.random.random((32, 1)),
            'co2_output': np.random.random((32, 1)),
            'no2_output': np.random.random((32, 1)),
            'aqi_output': np.random.random((32, 6)),
            'teacher_pm25': teacher_preds[0],
            'teacher_co2': teacher_preds[1],
            'teacher_no2': teacher_preds[2],
            'teacher_aqi': teacher_preds[3]
        }
        
        # Train student model
        logger.info(f"Training student model for {distillation_epochs} epochs...")
        student_model.fit(
            dummy_inputs, dummy_outputs,
            epochs=distillation_epochs,
            batch_size=16,
            verbose=1
        )
        
        # Save student model
        student_path = os.path.join(self.output_dir, "model_distilled_student.h5")
        student_model.save(student_path)
        
        logger.info(f"Distilled student model saved to {student_path}")
        
        # Store optimization info
        self.optimization_history['distillation'] = {
            'type': 'distillation',
            'method': 'knowledge_distillation',
            'path': student_path,
            'timestamp': time.time',
            'temperature': temperature,
            'alpha': alpha
        }
        
        return student_model
    
    def _create_student_model(self) -> keras.Model:
        """Create a smaller student model for distillation"""
        # Define inputs
        image_input = layers.Input(shape=(224, 224, 3), name='image')
        env_input = layers.Input(shape=(24, 6), name='environmental')
        bio_input = layers.Input(shape=(24, 3), name='biosensing')
        
        # Simplified CNN branch
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(image_input)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.GlobalAveragePooling2D()(x)
        cnn_features = layers.Dense(64, activation='relu')(x)
        
        # Simplified temporal branches
        env_x = layers.Conv1D(32, 3, activation='relu', padding='same')(env_input)
        env_x = layers.GlobalAveragePooling1D()(env_x)
        env_features = layers.Dense(32, activation='relu')(env_x)
        
        bio_x = layers.Conv1D(16, 3, activation='relu', padding='same')(bio_input)
        bio_x = layers.GlobalAveragePooling1D()(bio_x)
        bio_features = layers.Dense(16, activation='relu')(bio_x)
        
        # Fusion
        fused = layers.Concatenate()([cnn_features, env_features, bio_features])
        fused = layers.Dense(128, activation='relu')(fused)
        fused = layers.Dropout(0.3)(fused)
        
        # Outputs
        pm25_output = layers.Dense(1, name='pm25_output')(fused)
        co2_output = layers.Dense(1, name='co2_output')(fused)
        no2_output = layers.Dense(1, name='no2_output')(fused)
        aqi_output = layers.Dense(6, activation='softmax', name='aqi_output')(fused)
        
        # Create model
        student_model = keras.Model(
            inputs=[image_input, env_input, bio_input],
            outputs=[pm25_output, co2_output, no2_output, aqi_output],
            name='student_model'
        )
        
        return student_model
    
    def apply_tensorrt_optimization(self, 
                                 precision_mode: str = 'FP16') -> str:
        """
        Apply TensorRT optimization (requires TensorRT installation)
        
        Args:
            precision_mode: 'FP32', 'FP16', or 'INT8'
            
        Returns:
            Path to optimized model
        """
        logger.info(f"Applying TensorRT optimization with {precision_mode} precision...")
        
        try:
            # Convert to TensorFlow SavedModel format
            saved_model_dir = os.path.join(self.output_dir, "saved_model")
            self.original_model.save(saved_model_dir)
            
            # Apply TensorRT optimization
            trt_path = os.path.join(self.output_dir, f"tensorrt_model_{precision_mode}")
            
            # Convert parameters
            conversion_params = tf.experimental.tensorrt.ConversionParams(
                precision_mode=precision_mode,
                maximum_cached_engines=100
            )
            
            # Convert model
            converter = tf.experimental.tensorrt.Converter(
                saved_model_dir,
                conversion_params=conversion_params
            )
            
            converter.convert()
            converter.save(trt_path)
            
            logger.info(f"TensorRT optimized model saved to {trt_path}")
            
            # Store optimization info
            self.optimization_history['tensorrt'] = {
                'type': 'tensorrt',
                'method': precision_mode,
                'path': trt_path,
                'timestamp': time.time()
            }
            
            return trt_path
            
        except Exception as e:
            logger.error(f"TensorRT optimization failed: {e}")
            logger.info("Make sure TensorRT is properly installed")
            return None
    
    def compare_optimizations(self, input_shapes: Dict[str, Tuple]) -> pd.DataFrame:
        """
        Compare all optimization methods
        
        Args:
            input_shapes: Dictionary of input shapes
            
        Returns:
            Comparison DataFrame
        """
        logger.info("Comparing optimization methods...")
        
        results = []
        
        # Benchmark original model
        original_size = self.get_model_size(self.original_model)
        original_perf = self.benchmark_model(self.original_model, input_shapes)
        
        results.append({
            'model': 'original',
            'method': 'none',
            'parameters': original_size['total_parameters'],
            'size_mb': original_size['file_size_mb'],
            'inference_time_ms': original_perf['avg_inference_time_ms'],
            'throughput_fps': original_perf['throughput_fps']
        })
        
        # Test quantized models
        for quant_type in ['dynamic', 'full', 'float16']:
            try:
                quantized_model = self.apply_post_training_quantization(quant_type)
                
                # Load quantized model for benchmarking
                interpreter = tf.lite.Interpreter(model_content=quantized_model)
                interpreter.allocate_tensors()
                
                # Benchmark TFLite model
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                
                # Prepare inputs
                inputs = {}
                for input_name, shape in input_shapes.items():
                    inputs[input_name] = np.random.random(shape).astype(np.float32)
                
                # Benchmark
                start_time = time.time()
                for _ in range(100):
                    # Set inputs
                    for i, (input_name, shape) in enumerate(input_shapes.items()):
                        interpreter.set_tensor(input_details[i]['index'], inputs[input_name])
                    
                    # Run inference
                    interpreter.invoke()
                    
                    # Get outputs
                    for output_detail in output_details:
                        interpreter.get_tensor(output_detail['index'])
                
                end_time = time.time()
                avg_time = (end_time - start_time) / 100
                
                results.append({
                    'model': 'quantized',
                    'method': quant_type,
                    'parameters': original_size['total_parameters'],  # Same parameters
                    'size_mb': len(quantized_model) / (1024 * 1024),
                    'inference_time_ms': avg_time * 1000,
                    'throughput_fps': 100 / (end_time - start_time)
                })
                
            except Exception as e:
                logger.error(f"Failed to benchmark {quant_type} quantization: {e}")
        
        # Test pruning
        try:
            pruned_model = self.apply_pruning(
                tfmot.sparsity.keras.ConstantSparsity(0.5, begin_step=0, frequency=100)
            )
            
            pruned_size = self.get_model_size(pruned_model)
            pruned_perf = self.benchmark_model(pruned_model, input_shapes)
            
            results.append({
                'model': 'pruned',
                'method': 'constant_sparsity_0.5',
                'parameters': pruned_size['total_parameters'],
                'size_mb': pruned_size['file_size_mb'],
                'inference_time_ms': pruned_perf['avg_inference_time_ms'],
                'throughput_fps': pruned_perf['throughput_fps']
            })
            
        except Exception as e:
            logger.error(f"Failed to benchmark pruning: {e}")
        
        # Test knowledge distillation
        try:
            student_model = self.apply_knowledge_distillation(self.original_model)
            
            student_size = self.get_model_size(student_model)
            student_perf = self.benchmark_model(student_model, input_shapes)
            
            results.append({
                'model': 'distilled',
                'method': 'knowledge_distillation',
                'parameters': student_size['total_parameters'],
                'size_mb': student_size['file_size_mb'],
                'inference_time_ms': student_perf['avg_inference_time_ms'],
                'throughput_fps': student_perf['throughput_fps']
            })
            
        except Exception as e:
            logger.error(f"Failed to benchmark knowledge distillation: {e}")
        
        # Create DataFrame
        import pandas as pd
        df = pd.DataFrame(results)
        
        # Calculate improvements
        baseline_inference = df[df['model'] == 'original']['inference_time_ms'].iloc[0]
        baseline_size = df[df['model'] == 'original']['size_mb'].iloc[0]
        
        df['speedup'] = baseline_inference / df['inference_time_ms']
        df['size_reduction'] = 1 - (df['size_mb'] / baseline_size)
        
        # Save comparison
        comparison_path = os.path.join(self.output_dir, "optimization_comparison.csv")
        df.to_csv(comparison_path, index=False)
        
        logger.info(f"Optimization comparison saved to {comparison_path}")
        
        return df
    
    def generate_optimization_report(self) -> str:
        """Generate comprehensive optimization report"""
        report_path = os.path.join(self.output_dir, "optimization_report.md")
        
        report_content = f"""# Model Optimization Report

**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Original Model:** {self.model_path}

## Optimization Methods Applied

"""
        
        for method, info in self.optimization_history.items():
            report_content += f"""
### {method.upper()}
- **Type:** {info['type']}
- **Method:** {info.get('method', 'N/A')}
- **Path:** {info['path']}
- **Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(info['timestamp']))}

"""
        
        report_content += """
## Recommendations

Based on the optimization results, here are the recommendations for deployment:

1. **For Mobile/Edge Devices:** Use dynamic quantization for best balance of size and performance
2. **For Server Deployment:** Consider TensorRT optimization with FP16 precision
3. **For Memory-Constrained Environments:** Apply pruning with 50% sparsity
4. **For Fast Inference:** Use knowledge distillation to create smaller student models

## Deployment Guidelines

### Mobile Deployment
- Use TensorFlow Lite with dynamic quantization
- Target model size: < 20MB
- Target inference time: < 100ms

### Edge Device Deployment
- Use TensorRT optimization
- Target model size: < 50MB
- Target inference time: < 50ms

### Cloud Deployment
- Use original or pruned model
- Consider model ensemble for better accuracy
- Implement batch inference for throughput

---
*Report generated by Air Quality Prediction System*
"""
        
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Optimization report saved to {report_path}")
        return report_path

def main():
    """Main function to test model optimization"""
    print("Testing Model Optimization...")
    
    # This would require a trained model
    # For demonstration, we'll create a simple model
    model_path = "models/test_model.h5"
    
    # Create a simple test model if it doesn't exist
    if not os.path.exists(model_path):
        print("Creating test model for optimization demonstration...")
        
        # Define inputs
        image_input = layers.Input(shape=(224, 224, 3), name='image')
        env_input = layers.Input(shape=(24, 6), name='environmental')
        bio_input = layers.Input(shape=(24, 3), name='biosensing')
        
        # Simple processing
        x = layers.Conv2D(32, (3, 3), activation='relu')(image_input)
        x = layers.GlobalAveragePooling2D()(x)
        
        y = layers.Conv1D(16, 3, activation='relu')(env_input)
        y = layers.GlobalAveragePooling1D()(y)
        
        z = layers.Conv1D(8, 3, activation='relu')(bio_input)
        z = layers.GlobalAveragePooling1D()(z)
        
        # Fusion
        fused = layers.Concatenate()([x, y, z])
        fused = layers.Dense(64, activation='relu')(fused)
        
        # Outputs
        pm25_output = layers.Dense(1, name='pm25_output')(fused)
        co2_output = layers.Dense(1, name='co2_output')(fused)
        no2_output = layers.Dense(1, name='no2_output')(fused)
        aqi_output = layers.Dense(6, activation='softmax', name='aqi_output')(fused)
        
        # Create model
        model = keras.Model(
            inputs=[image_input, env_input, bio_input],
            outputs=[pm25_output, co2_output, no2_output, aqi_output]
        )
        
        model.compile(optimizer='adam', loss='mse')
        
        # Save model
        os.makedirs("models", exist_ok=True)
        model.save(model_path)
        print(f"Test model saved to {model_path}")
    
    # Initialize optimizer
    optimizer = ModelOptimizer(model_path, "optimized_models")
    
    # Test quantization
    print("\n=== Testing Quantization ===")
    try:
        quantized_model = optimizer.apply_post_training_quantization('dynamic')
        print("Dynamic quantization completed successfully")
    except Exception as e:
        print(f"Quantization failed: {e}")
    
    # Test pruning
    print("\n=== Testing Pruning ===")
    try:
        pruned_model = optimizer.apply_pruning(
            tfmot.sparsity.keras.ConstantSparsity(0.3, begin_step=0, frequency=100),
            fine_tune_epochs=2
        )
        print("Pruning completed successfully")
    except Exception as e:
        print(f"Pruning failed: {e}")
    
    # Generate report
    print("\n=== Generating Optimization Report ===")
    report_path = optimizer.generate_optimization_report()
    print(f"Optimization report saved to {report_path}")
    
    print("\nModel optimization test completed!")

if __name__ == "__main__":
    main()
