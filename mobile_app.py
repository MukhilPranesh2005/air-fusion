"""
Mobile-Friendly Inference Interface

This module provides a mobile-friendly web interface for air quality prediction including:
- Progressive Web App (PWA) capabilities
- Responsive design for mobile devices
- Offline functionality
- Camera integration for image capture
- Touch-optimized interface
- Geolocation-based predictions

Author: Air Quality Prediction System
Date: 2026
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import base64
import io

# Web framework
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import requests
import numpy as np
import pandas as pd
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MobileAirQualityApp:
    """Mobile-friendly air quality prediction app"""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        """
        Initialize mobile app
        
        Args:
            api_url: API server URL
        """
        self.api_url = api_url
        self.setup_mobile_config()
    
    def setup_mobile_config(self):
        """Setup mobile-specific configuration"""
        # Streamlit mobile configuration
        st.set_page_config(
            page_title="Air Quality Mobile",
            page_icon="🌍",
            layout="centered",
            initial_sidebar_state="collapsed"
        )
        
        # Mobile-optimized CSS
        st.markdown("""
        <style>
        /* Mobile-specific styles */
        @media (max-width: 768px) {
            .stApp {
                margin: 0;
                padding: 0;
            }
            
            .mobile-header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 1rem;
                text-align: center;
                border-radius: 10px;
                margin-bottom: 1rem;
            }
            
            .mobile-card {
                background: white;
                border-radius: 15px;
                padding: 1.5rem;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin: 1rem 0;
            }
            
            .mobile-button {
                background: linear-gradient(45deg, #667eea, #764ba2);
                color: white;
                border: none;
                padding: 1rem 2rem;
                border-radius: 25px;
                font-size: 1.1rem;
                font-weight: bold;
                width: 100%;
                margin: 0.5rem 0;
                transition: all 0.3s ease;
            }
            
            .mobile-button:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
            }
            
            .mobile-input {
                width: 100%;
                padding: 1rem;
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                font-size: 1rem;
                margin: 0.5rem 0;
            }
            
            .mobile-metric {
                text-align: center;
                padding: 1rem;
                background: #f8f9fa;
                border-radius: 10px;
                margin: 0.5rem 0;
            }
            
            .aqi-good { background: #00e400; color: white; }
            .aqi-moderate { background: #ffff00; color: black; }
            .aqi-unhealthy-sensitive { background: #ff7e00; color: white; }
            .aqi-unhealthy { background: #ff0000; color: white; }
            .aqi-very-unhealthy { background: #8f3f97; color: white; }
            .aqi-hazardous { background: #7e0023; color: white; }
        }
        
        /* PWA styles */
        .pwa-install {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: #667eea;
            color: white;
            padding: 10px 15px;
            border-radius: 20px;
            z-index: 1000;
            cursor: pointer;
        }
        
        /* Hide streamlit default elements on mobile */
        @media (max-width: 768px) {
            .stDeployButton, .stHeader {
                display: none;
            }
        }
        </style>
        
        <!-- PWA Manifest -->
        <link rel="manifest" href="manifest.json">
        <meta name="theme-color" content="#667eea">
        <meta name="apple-mobile-web-app-capable" content="yes">
        <meta name="apple-mobile-web-app-status-bar-style" content="default">
        <meta name="apple-mobile-web-app-title" content="Air Quality">
        <link rel="apple-touch-icon" href="icon-192.png">
        """, unsafe_allow_html=True)
    
    def render_mobile_header(self):
        """Render mobile-optimized header"""
        st.markdown("""
        <div class="mobile-header">
            <h1>🌍 Air Quality Predictor</h1>
            <p>Real-time air quality monitoring and prediction</p>
        </div>
        """, unsafe_allow_html=True)
        
        # PWA install prompt
        if 'pwa_dismissed' not in st.session_state:
            st.markdown("""
            <div class="pwa-install" onclick="installPWA()">
                📱 Install App
            </div>
            <script>
            function installPWA() {
                if ('serviceWorker' in navigator) {
                    window.addEventListener('beforeinstallprompt', (e) => {
                        e.prompt();
                    });
                }
            }
            </script>
            """, unsafe_allow_html=True)
    
    def render_mobile_navigation(self):
        """Render mobile navigation menu"""
        # Simple navigation using radio buttons
        selected = st.radio(
            "Navigate",
            ["📸 Predict", "📊 Dashboard", "📍 Location", "⚙️ Settings"],
            horizontal=True,
            index=0
        )
        
        return selected
    
    def render_camera_input(self):
        """Render camera input for mobile"""
        st.markdown('<div class="mobile-card">', unsafe_allow_html=True)
        st.subheader("📸 Capture or Upload Image")

        # Camera input
        camera_image = st.camera_input("Take a photo", key="camera")

        # File upload
        uploaded_file = st.file_uploader(
            "Or upload an image",
            type=['jpg', 'jpeg', 'png'],
            key="file_upload"
        )

        image = None

        # If camera image is captured
        if camera_image is not None:
            image = Image.open(camera_image)

        # If file uploaded
        elif uploaded_file is not None:
            image = Image.open(uploaded_file)

        if image is not None:
            # Display image
            st.image(image, caption="Captured image", use_column_width=True)

            # Get width and height safely
            width, height = image.size

            # Resize image if larger than 224
            if width > 224 or height > 224:
                image = image.resize((224, 224), Image.Resampling.LANCZOS)
                st.info("Image resized to 224x224 for optimal performance")

            # Convert to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            st.session_state.captured_image = img_str
            st.session_state.image_available = True

        st.markdown('</div>', unsafe_allow_html=True)

        return image is not None
    
    def render_sensor_inputs(self):
        """Render mobile-optimized sensor inputs"""
        st.markdown('<div class="mobile-card">', unsafe_allow_html=True)
        st.subheader("📡 Sensor Data")
        
        # Quick input options
        input_method = st.radio(
            "Input Method",
            ["Quick Values", "Manual Entry", "Use Current Location"],
            horizontal=True
        )
        
        if input_method == "Quick Values":
            # Quick preset values
            col1, col2 = st.columns(2)
            
            with col1:
                air_quality = st.selectbox(
                    "Air Quality",
                    ["Good", "Moderate", "Poor", "Very Poor"],
                    index=1
                )
                
                if air_quality == "Good":
                    pm25, co2, no2 = 15, 400, 30
                elif air_quality == "Moderate":
                    pm25, co2, no2 = 35, 600, 60
                elif air_quality == "Poor":
                    pm25, co2, no2 = 75, 1200, 120
                else:
                    pm25, co2, no2 = 150, 2000, 200
            
            with col2:
                st.info(f"PM2.5: {pm25} μg/m³\nCO₂: {co2} ppm\nNO₂: {no2} ppb")
        
        elif input_method == "Use Current Location":
            # Get location-based data
            if st.button("📍 Get Location Data"):
                location_data = self.get_location_based_data()
                if location_data:
                    pm25, co2, no2 = location_data
                    st.success(f"Location data loaded:\nPM2.5: {pm25}\nCO₂: {co2}\nNO₂: {no2}")
                else:
                    st.error("Could not get location data")
                    pm25, co2, no2 = 35, 600, 60  # Default values
        else:
            # Manual entry
            col1, col2 = st.columns(2)
            
            with col1:
                pm25 = st.slider("PM2.5 (μg/m³)", 0, 150, 35)
                co2 = st.slider("CO₂ (ppm)", 350, 2000, 600)
            
            with col2:
                no2 = st.slider("NO₂ (ppb)", 10, 200, 60)
        
        # Biosensor inputs
        st.subheader("💓 Health Data")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            heart_rate = st.slider("Heart Rate (bpm)", 60, 100, 75)
        with col2:
            spo2 = st.slider("SpO₂ (%)", 95, 100, 98)
        with col3:
            skin_temp = st.slider("Skin Temp (°C)", 32, 37, 34)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Store sensor data
        sensor_data = {
            'pm25': [pm25] * 24,
            'co2': [co2] * 24,
            'no2': [no2] * 24,
            'heart_rate': [heart_rate] * 24,
            'spo2': [spo2] * 24,
            'skin_temperature': [skin_temp] * 24
        }
        
        return sensor_data
    
    def get_location_based_data(self) -> Optional[tuple]:
        """Get air quality data based on location"""
        try:
            # This would integrate with a real air quality API
            # For now, return mock data based on time
            hour = datetime.now().hour
            
            if 6 <= hour <= 10:  # Morning rush hour
                return (50, 800, 80)
            elif 11 <= hour <= 15:  # Midday
                return (35, 600, 60)
            elif 16 <= hour <= 19:  # Evening rush hour
                return (60, 900, 90)
            else:  # Night
                return (25, 450, 40)
                
        except Exception as e:
            logger.error(f"Error getting location data: {e}")
            return None
    
    def make_prediction(self, image_data: str, sensor_data: Dict) -> Optional[Dict]:
        """Make prediction using API"""
        try:
            # Prepare request
            request_data = {
                "image_base64": image_data,
                "environmental_data": {
                    "temperature": [22] * 24,  # Default values
                    "humidity": [50] * 24,
                    "pressure": [1013] * 24,
                    "pm25": sensor_data['pm25'],
                    "co2": sensor_data['co2'],
                    "no2": sensor_data['no2']
                },
                "biosensor_data": {
                    "heart_rate": sensor_data['heart_rate'],
                    "spo2": sensor_data['spo2'],
                    "skin_temperature": sensor_data['skin_temperature']
                }
            }
            
            # Make API call
            response = requests.post(
                f"{self.api_url}/predict",
                json=request_data,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Prediction failed: {response.text}")
                return None
                
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            return None
    
    def render_prediction_results(self, results: Dict):
        """Render mobile-optimized prediction results"""
        st.markdown('<div class="mobile-card">', unsafe_allow_html=True)
        st.subheader("🎯 Prediction Results")
        
        # AQI Result
        aqi_data = results['aqi']
        aqi_value = aqi_data['value']
        aqi_category = aqi_data['category']
        aqi_class = self.get_aqi_class(aqi_category)
        
        st.markdown(f"""
        <div class="mobile-metric aqi-{aqi_class}">
            <h2>AQI: {aqi_value}</h2>
            <h3>{aqi_category}</h3>
            <p>Confidence: {aqi_data['confidence']:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Pollutant values
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="mobile-metric">
                <h4>PM2.5</h4>
                <p>{results['predictions']['pm25']['value']:.1f} μg/m³</p>
                <small>{results['predictions']['pm25']['status']}</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="mobile-metric">
                <h4>CO₂</h4>
                <p>{results['predictions']['co2']['value']:.0f} ppm</p>
                <small>{results['predictions']['co2']['status']}</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="mobile-metric">
                <h4>NO₂</h4>
                <p>{results['predictions']['no2']['value']:.1f} ppb</p>
                <small>{results['predictions']['no2']['status']}</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Risk assessment
        risk_data = results['risk_assessment']
        st.subheader("⚠️ Health Recommendations")
        
        risk_class = self.get_risk_class(risk_data['level'])
        
        st.markdown(f"""
        <div class="mobile-card" style="background: {risk_class['color']}; color: white;">
            <h4>{risk_data['level']}</h4>
            <p>{risk_data['description']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Recommendations
        for i, rec in enumerate(risk_data['recommendations'], 1):
            st.markdown(f"""
            <div class="mobile-card">
                <p><strong>{i}.</strong> {rec}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Action buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📤 Share Results", key="share"):
                self.share_results(results)
        
        with col2:
            if st.button("💾 Save Results", key="save"):
                self.save_results(results)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def get_aqi_class(self, category: str) -> str:
        """Get AQI class for styling"""
        class_map = {
            'Good': 'good',
            'Moderate': 'moderate',
            'Unhealthy for Sensitive': 'unhealthy-sensitive',
            'Unhealthy': 'unhealthy',
            'Very Unhealthy': 'very-unhealthy',
            'Hazardous': 'hazardous'
        }
        return class_map.get(category, 'moderate')
    
    def get_risk_class(self, level: str) -> Dict:
        """Get risk class for styling"""
        risk_map = {
            'Good': {'color': '#00e400', 'text': 'Low Risk'},
            'Moderate': {'color': '#ffff00', 'text': 'Moderate Risk'},
            'Unhealthy for Sensitive': {'color': '#ff7e00', 'text': 'High Risk for Sensitive Groups'},
            'Unhealthy': {'color': '#ff0000', 'text': 'High Risk'},
            'Very Unhealthy': {'color': '#8f3f97', 'text': 'Very High Risk'},
            'Hazardous': {'color': '#7e0023', 'text': 'Hazardous'}
        }
        return risk_map.get(level, {'color': '#ffff00', 'text': 'Moderate Risk'})
    
    def share_results(self, results: Dict):
        """Share prediction results"""
        # Create shareable text
        aqi = results['aqi']['value']
        category = results['aqi']['category']
        
        share_text = f"Air Quality: AQI {aqi} ({category})\n"
        share_text += f"PM2.5: {results['predictions']['pm25']['value']:.1f} μg/m³\n"
        share_text += f"CO₂: {results['predictions']['co2']['value']:.0f} ppm\n"
        share_text += f"NO₂: {results['predictions']['no2']['value']:.1f} ppb"
        
        # Copy to clipboard (JavaScript)
        st.markdown(f"""
        <script>
        navigator.clipboard.writeText(`{share_text}`);
        alert('Results copied to clipboard!');
        </script>
        """, unsafe_allow_html=True)
        
        st.success("Results copied to clipboard!")
    
    def save_results(self, results: Dict):
        """Save prediction results"""
        # Add timestamp
        results['timestamp'] = datetime.now().isoformat()
        
        # Save to session state
        if 'saved_results' not in st.session_state:
            st.session_state.saved_results = []
        
        st.session_state.saved_results.append(results)
        
        st.success("Results saved!")
    
    def render_dashboard(self):
        """Render mobile dashboard"""
        st.markdown('<div class="mobile-card">', unsafe_allow_html=True)
        st.subheader("📊 Air Quality Dashboard")
        
        # Mock historical data for mobile
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(hours=24),
            end=datetime.now(),
            freq='H'
        )
        
        # Create charts
        fig = go.Figure()
        
        # AQI trend
        aqi_values = np.random.normal(75, 20, len(timestamps))
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=aqi_values,
            mode='lines+markers',
            name='AQI',
            line=dict(color='purple', width=3)
        ))
        
        fig.update_layout(
            title="24-Hour AQI Trend",
            xaxis_title="Time",
            yaxis_title="AQI",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Current conditions
        col1, col2 = st.columns(2)
        
        with col1:
            current_aqi = int(np.random.normal(75, 20))
            aqi_category = self.get_aqi_category_from_value(current_aqi)
            aqi_class = self.get_aqi_class(aqi_category)
            
            st.markdown(f"""
            <div class="mobile-metric aqi-{aqi_class}">
                <h3>Current AQI</h3>
                <h2>{current_aqi}</h2>
                <p>{aqi_category}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Quick stats
            st.markdown("""
            <div class="mobile-metric">
                <h4>Today's Average</h4>
                <h2>72</h2>
                <p>Moderate</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def get_aqi_category_from_value(self, aqi: int) -> str:
        """Get AQI category from value"""
        if aqi <= 50:
            return 'Good'
        elif aqi <= 100:
            return 'Moderate'
        elif aqi <= 150:
            return 'Unhealthy for Sensitive'
        elif aqi <= 200:
            return 'Unhealthy'
        elif aqi <= 300:
            return 'Very Unhealthy'
        else:
            return 'Hazardous'
    
    def render_settings(self):
        """Render mobile settings"""
        st.markdown('<div class="mobile-card">', unsafe_allow_html=True)
        st.subheader("⚙️ Settings")
        
        # API settings
        st.text_input("API URL", value=self.api_url, key="api_url")
        
        # Notification settings
        st.subheader("🔔 Notifications")
        notifications = st.checkbox("Enable Notifications", value=True)
        
        if notifications:
            alert_threshold = st.slider("AQI Alert Threshold", 0, 300, 100)
            st.info(f"You'll be alerted when AQI exceeds {alert_threshold}")
        
        # Data settings
        st.subheader("💾 Data")
        if st.button("Clear Saved Data"):
            if 'saved_results' in st.session_state:
                del st.session_state.saved_results
            st.success("Saved data cleared")
        
        # About
        st.subheader("ℹ️ About")
        st.markdown("""
        **Air Quality Predictor v1.0**
        
        A mobile-friendly application for real-time air quality monitoring and prediction using advanced deep learning models.
        
        Features:
        - Real-time air quality prediction
        - Camera integration
        - Location-based data
        - Health recommendations
        - Offline support
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def run(self):
        """Run the mobile app"""
        # Render header
        self.render_mobile_header()
        
        # Navigation
        selected = self.render_mobile_navigation()
        
        if selected == "📸 Predict":
            # Prediction interface
            image_ready = self.render_camera_input()
            
            if image_ready or st.session_state.get('image_available', False):
                sensor_data = self.render_sensor_inputs()
                
                # Predict button
                if st.button("🔮 Make Prediction", key="predict", type="primary"):
                    with st.spinner("Analyzing..."):
                        image_data = st.session_state.get('captured_image', '')
                        if image_data:
                            results = self.make_prediction(image_data, sensor_data)
                            if results:
                                st.session_state.prediction_results = results
                                self.render_prediction_results(results)
                        else:
                            st.error("No image available")
                
                # Show previous results
                if 'prediction_results' in st.session_state:
                    st.markdown("---")
                    self.render_prediction_results(st.session_state.prediction_results)
        
        elif selected == "📊 Dashboard":
            self.render_dashboard()
        
        elif selected == "📍 Location":
            st.markdown('<div class="mobile-card">', unsafe_allow_html=True)
            st.subheader("📍 Location-Based Monitoring")
            
            # Location input
            location = st.text_input("Enter your location", placeholder="City, State or ZIP code")
            
            if st.button("🔍 Get Air Quality"):
                if location:
                    # This would integrate with a real API
                    st.success(f"Loading air quality data for {location}...")
                    # Mock data
                    st.json({
                        "location": location,
                        "aqi": 85,
                        "category": "Moderate",
                        "pm25": 35.2,
                        "co2": 650,
                        "no2": 65
                    })
                else:
                    st.error("Please enter a location")
            
            # Current location button
            if st.button("📍 Use Current Location"):
                st.info("Getting your current location...")
                # This would use browser geolocation
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        elif selected == "⚙️ Settings":
            self.render_settings()

def main():
    """Main function to run mobile app"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Mobile Air Quality App')
    parser.add_argument('--api_url', type=str, default='http://localhost:8000', help='API URL')
    
    args = parser.parse_args()
    
    # Initialize and run app
    app = MobileAirQualityApp(api_url=args.api_url)
    app.run()

if __name__ == "__main__":
    main()
