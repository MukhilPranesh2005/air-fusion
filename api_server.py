"""
REST API Server for Air Quality Prediction Model

This module provides a production-ready REST API for model serving including:
- FastAPI-based web service
- Async request handling
- Input validation and preprocessing
- Rate limiting and caching
- Health monitoring and metrics
- Swagger documentation

Author: Air Quality Prediction System
Date: 2026
"""

import os
import io
import base64
import json
import asyncio
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from PIL import Image
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field, validator
import redis
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import psutil

# Import our modules
from inference import AirQualityInference
from data_loader import AirQualityDataLoader

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
prediction_counter = Counter('air_quality_predictions_total', 'Total predictions', ['model_version'])
prediction_duration = Histogram('air_quality_prediction_duration_seconds', 'Prediction duration')
active_connections = Gauge('air_quality_active_connections', 'Active connections')
model_load_time = Gauge('air_quality_model_load_time_seconds', 'Model load time')

# Pydantic models for request/response
class SensorData(BaseModel):
    """Sensor data model"""
    temperature: List[float] = Field(..., description="Temperature readings in Celsius")
    humidity: List[float] = Field(..., description="Humidity readings in percentage")
    pressure: List[float] = Field(..., description="Pressure readings in hPa")
    pm25: List[float] = Field(..., description="PM2.5 readings in μg/m³")
    co2: List[float] = Field(..., description="CO₂ readings in ppm")
    no2: List[float] = Field(..., description="NO₂ readings in ppb")
    
    @validator('temperature', 'humidity', 'pressure', 'pm25', 'co2', 'no2')
    def validate_length(cls, v):
        if len(v) != 24:
            raise ValueError(f"Expected 24 readings, got {len(v)}")
        return v

class BioData(BaseModel):
    """Biosensor data model"""
    heart_rate: List[float] = Field(..., description="Heart rate readings in bpm")
    spo2: List[float] = Field(..., description="SpO₂ readings in percentage")
    skin_temperature: List[float] = Field(..., description="Skin temperature readings in Celsius")
    
    @validator('heart_rate', 'spo2', 'skin_temperature')
    def validate_length(cls, v):
        if len(v) != 24:
            raise ValueError(f"Expected 24 readings, got {len(v)}")
        return v

class PredictionRequest(BaseModel):
    """Prediction request model"""
    image_base64: str = Field(..., description="Base64 encoded image")
    environmental_data: SensorData
    biosensor_data: BioData
    
    @validator('image_base64')
    def validate_image(cls, v):
        try:
            # Decode base64 to validate
            image_data = base64.b64decode(v)
            img = Image.open(io.BytesIO(image_data))
            if img.size != (224, 224):
                raise ValueError("Image must be 224x224 pixels")
            return v
        except Exception as e:
            raise ValueError(f"Invalid image data: {e}")

class PredictionResponse(BaseModel):
    """Prediction response model"""
    timestamp: str
    predictions: Dict[str, Dict[str, Union[float, str]]]
    aqi: Dict[str, Union[int, str, float, Dict]]
    risk_assessment: Dict[str, Union[str, List[str]]]
    processing_time_ms: float
    model_version: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    model_loaded: bool
    redis_connected: bool
    memory_usage_mb: float
    cpu_usage_percent: float
    uptime_seconds: float

class AirQualityAPI:
    """Air Quality Prediction API"""
    
    def __init__(self, 
                 model_path: str,
                 data_dir: str = "data",
                 redis_url: str = "redis://localhost:6379",
                 enable_caching: bool = True,
                 rate_limit: int = 100):
        """
        Initialize API server
        
        Args:
            model_path: Path to trained model
            data_dir: Data directory for preprocessing
            redis_url: Redis URL for caching
            enable_caching: Enable request caching
            rate_limit: Rate limit per minute
        """
        self.model_path = model_path
        self.data_dir = data_dir
        self.enable_caching = enable_caching
        self.rate_limit = rate_limit
        
        # Initialize model
        self.inference_engine = None
        self.model_loaded = False
        self.start_time = time.time()
        
        # Initialize Redis if caching enabled
        self.redis_client = None
        if enable_caching:
            try:
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()
                logger.info("Redis connection established")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")
                self.enable_caching = False
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title="Air Quality Prediction API",
            description="Multi-modal air quality prediction using deep learning",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Setup middleware
        self._setup_middleware()
        
        # Setup routes
        self._setup_routes()
        
        # Load model
        self._load_model()
    
    def _setup_middleware(self):
        """Setup FastAPI middleware"""
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Gzip middleware
        self.app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # Trusted host middleware
        self.app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["localhost", "127.0.0.1", "*"]  # Configure for production
        )
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.on_event("startup")
        async def startup_event():
            logger.info("Air Quality Prediction API started")
            active_connections.set(0)
        
        @self.app.on_event("shutdown")
        async def shutdown_event():
            logger.info("Air Quality Prediction API shutting down")
        
        @self.app.middleware("http")
        async def add_process_time_header(request, call_next):
            start_time = time.time()
            response = await call_next(request)
            process_time = time.time() - start_time
            response.headers["X-Process-Time"] = str(process_time)
            return response
        
        @self.app.get("/", response_class=JSONResponse)
        async def root():
            return {
                "message": "Air Quality Prediction API",
                "version": "1.0.0",
                "docs": "/docs",
                "health": "/health"
            }
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint"""
            memory_usage = psutil.virtual_memory().used / (1024 * 1024)  # MB
            cpu_usage = psutil.cpu_percent()
            uptime = time.time() - self.start_time
            
            return HealthResponse(
                status="healthy" if self.model_loaded else "unhealthy",
                timestamp=datetime.now().isoformat(),
                model_loaded=self.model_loaded,
                redis_connected=self.redis_client is not None,
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_usage,
                uptime_seconds=uptime
            )
        
        @self.app.get("/metrics")
        async def metrics():
            """Prometheus metrics endpoint"""
            return generate_latest()
        
        @self.app.post("/predict", response_model=PredictionResponse)
        async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
            """Make air quality prediction"""
            if not self.model_loaded:
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            start_time = time.time()
            
            # Check cache
            cache_key = self._generate_cache_key(request)
            if self.enable_caching and self.redis_client:
                cached_result = self.redis_client.get(cache_key)
                if cached_result:
                    logger.info("Cache hit")
                    return json.loads(cached_result)
            
            try:
                # Process request
                result = await self._process_prediction_request(request)
                
                # Add to cache
                if self.enable_caching and self.redis_client:
                    background_tasks.add_task(
                        self._cache_result, cache_key, json.dumps(result)
                    )
                
                # Update metrics
                processing_time = (time.time() - start_time) * 1000
                prediction_duration.observe(processing_time / 1000)
                prediction_counter.labels(model_version="1.0.0").inc()
                
                result['processing_time_ms'] = processing_time
                
                return result
                
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/predict/upload")
        async def predict_with_upload(
            image: UploadFile = File(...),
            environmental_csv: UploadFile = File(...),
            biosensor_csv: UploadFile = File(...)
        ):
            """Make prediction with file uploads"""
            if not self.model_loaded:
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            try:
                # Read and validate files
                image_data = await image.read()
                env_data = await environmental_csv.read()
                bio_data = await biosensor_csv.read()
                
                # Process files
                result = await self._process_file_prediction(
                    image_data, env_data, bio_data
                )
                
                return result
                
            except Exception as e:
                logger.error(f"File prediction error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/predict/batch")
        async def predict_batch(requests: List[PredictionRequest]):
            """Batch prediction endpoint"""
            if not self.model_loaded:
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            results = []
            for request in requests:
                try:
                    result = await self._process_prediction_request(request)
                    results.append(result)
                except Exception as e:
                    results.append({"error": str(e)})
            
            return {"results": results}
    
    def _load_model(self):
        """Load the inference model"""
        try:
            start_time = time.time()
            self.inference_engine = AirQualityInference(
                model_path=self.model_path,
                data_dir=self.data_dir
            )
            self.model_loaded = True
            
            load_time = time.time() - start_time
            model_load_time.set(load_time)
            
            logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model_loaded = False
    
    async def _process_prediction_request(self, request: PredictionRequest) -> Dict:
        """Process a single prediction request"""
        # Decode image
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_data))
        
        # Convert sensor data to DataFrames
        env_data = pd.DataFrame({
            'temperature': request.environmental_data.temperature,
            'humidity': request.environmental_data.humidity,
            'pressure': request.environmental_data.pressure,
            'pm25': request.environmental_data.pm25,
            'co2': request.environmental_data.co2,
            'no2': request.environmental_data.no2
        })
        
        bio_data = pd.DataFrame({
            'heart_rate': request.biosensor_data.heart_rate,
            'spo2': request.biosensor_data.spo2,
            'skin_temperature': request.biosensor_data.skin_temperature
        })
        
        # Save temporary files
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        
        timestamp = int(time.time())
        img_path = f"{temp_dir}/image_{timestamp}.png"
        env_path = f"{temp_dir}/env_{timestamp}.csv"
        bio_path = f"{temp_dir}/bio_{timestamp}.csv"
        
        image.save(img_path)
        env_data.to_csv(env_path, index=False)
        bio_data.to_csv(bio_path, index=False)
        
        try:
            # Make prediction
            result = self.inference_engine.predict_from_files(
                img_path, env_path, bio_path
            )
            
            # Format response
            return {
                "timestamp": result["timestamp"],
                "predictions": {
                    "pm25": {
                        "value": result["predictions"]["pm25"]["value"],
                        "unit": result["predictions"]["pm25"]["unit"],
                        "status": result["predictions"]["pm25"]["status"]
                    },
                    "co2": {
                        "value": result["predictions"]["co2"]["value"],
                        "unit": result["predictions"]["co2"]["unit"],
                        "status": result["predictions"]["co2"]["status"]
                    },
                    "no2": {
                        "value": result["predictions"]["no2"]["value"],
                        "unit": result["predictions"]["no2"]["unit"],
                        "status": result["predictions"]["no2"]["status"]
                    }
                },
                "aqi": {
                    "value": result["aqi"]["value"],
                    "category": result["aqi"]["category"],
                    "confidence": result["aqi"]["confidence"],
                    "probabilities": result["aqi"]["probabilities"]
                },
                "risk_assessment": {
                    "level": result["risk_assessment"]["level"],
                    "description": result["risk_assessment"]["description"],
                    "recommendations": result["risk_assessment"]["recommendations"]
                },
                "model_version": "1.0.0"
            }
            
        finally:
            # Clean up temporary files
            for path in [img_path, env_path, bio_path]:
                if os.path.exists(path):
                    os.remove(path)
    
    async def _process_file_prediction(self, 
                                    image_data: bytes,
                                    env_data: bytes,
                                    bio_data: bytes) -> Dict:
        """Process prediction with uploaded files"""
        # Save temporary files
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        
        timestamp = int(time.time())
        img_path = f"{temp_dir}/upload_image_{timestamp}.png"
        env_path = f"{temp_dir}/upload_env_{timestamp}.csv"
        bio_path = f"{temp_dir}/upload_bio_{timestamp}.csv"
        
        # Save files
        with open(img_path, 'wb') as f:
            f.write(image_data)
        
        with open(env_path, 'wb') as f:
            f.write(env_data)
        
        with open(bio_path, 'wb') as f:
            f.write(bio_data)
        
        try:
            # Make prediction
            result = self.inference_engine.predict_from_files(
                img_path, env_path, bio_path
            )
            
            return {
                "timestamp": result["timestamp"],
                "predictions": result["predictions"],
                "aqi": result["aqi"],
                "risk_assessment": result["risk_assessment"],
                "model_version": "1.0.0"
            }
            
        finally:
            # Clean up
            for path in [img_path, env_path, bio_path]:
                if os.path.exists(path):
                    os.remove(path)
    
    def _generate_cache_key(self, request: PredictionRequest) -> str:
        """Generate cache key for request"""
        # Create a hash of the request data
        import hashlib
        
        # Create a string representation of the request
        request_str = json.dumps({
            "image_hash": hashlib.md5(request.image_base64.encode()).hexdigest(),
            "env_data": request.environmental_data.dict(),
            "bio_data": request.biosensor_data.dict()
        }, sort_keys=True)
        
        return f"air_quality_pred:{hashlib.md5(request_str.encode()).hexdigest()}"
    
    def _cache_result(self, cache_key: str, result: str):
        """Cache prediction result"""
        try:
            self.redis_client.setex(cache_key, 300, result)  # Cache for 5 minutes
        except Exception as e:
            logger.error(f"Failed to cache result: {e}")
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the API server"""
        logger.info(f"Starting Air Quality Prediction API on {host}:{port}")
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="info",
            access_log=True
        )

def main():
    """Main function to run the API server"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Air Quality Prediction API Server')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=8000, help='Port number')
    parser.add_argument('--redis_url', type=str, default='redis://localhost:6379', help='Redis URL')
    parser.add_argument('--no_cache', action='store_true', help='Disable caching')
    
    args = parser.parse_args()
    
    # Initialize API
    api = AirQualityAPI(
        model_path=args.model_path,
        data_dir=args.data_dir,
        redis_url=args.redis_url,
        enable_caching=not args.no_cache
    )
    
    # Run server
    api.run(host=args.host, port=args.port)

if __name__ == "__main__":
    main()
