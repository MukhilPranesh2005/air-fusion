"""
Advanced Visualization Dashboard with Real-time Monitoring

This module provides a comprehensive web-based dashboard including:
- Real-time air quality monitoring
- Interactive data visualization
- Model performance metrics
- Historical trend analysis
- Alert system and notifications

Author: Air Quality Prediction System
Date: 2026
"""

import os
import json
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

# Web framework and visualization
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
from PIL import Image
import io
import base64

# Data processing
from sklearn.preprocessing import StandardScaler
import redis

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AirQualityDashboard:
    """Advanced air quality monitoring dashboard"""
    
    def __init__(self, 
                 api_url: str = "http://localhost:8000",
                 redis_url: str = "redis://localhost:6379",
                 refresh_interval: int = 30):
        """
        Initialize dashboard
        
        Args:
            api_url: API server URL
            redis_url: Redis URL for real-time data
            refresh_interval: Data refresh interval in seconds
        """
        self.api_url = api_url
        self.redis_url = redis_url
        self.refresh_interval = refresh_interval
        
        # Initialize Redis connection
        self.redis_client = None
        try:
            self.redis_client = redis.from_url(redis_url)
            self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
        
        # Initialize session state
        if 'data_history' not in st.session_state:
            st.session_state.data_history = []
        if 'alerts' not in st.session_state:
            st.session_state.alerts = []
        if 'last_update' not in st.session_state:
            st.session_state.last_update = None
    
    def setup_page_config(self):
        """Setup Streamlit page configuration"""
        st.set_page_config(
            page_title="Air Quality Prediction Dashboard",
            page_icon="🌍",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .alert-high {
            background-color: #ffcccc;
            border-left: 5px solid #ff0000;
            padding: 1rem;
            margin: 1rem 0;
        }
        .alert-medium {
            background-color: #fff3cd;
            border-left: 5px solid #ffc107;
            padding: 1rem;
            margin: 1rem 0;
        }
        .alert-low {
            background-color: #d4edda;
            border-left: 5px solid #28a745;
            padding: 1rem;
            margin: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def render_header(self):
        """Render dashboard header"""
        st.markdown('<div class="main-header">🌍 Air Quality Prediction Dashboard</div>', 
                    unsafe_allow_html=True)
        
        # Status indicators
        col1, col2, col3 = st.columns(3)
        
        with col1:
            api_status = self.check_api_health()
            status_color = "🟢" if api_status else "🔴"
            st.metric("API Status", f"{status_color} {'Online' if api_status else 'Offline'}")
        
        with col2:
            redis_status = self.redis_client is not None
            redis_color = "🟢" if redis_status else "🔴"
            st.metric("Redis Status", f"{redis_color} {'Connected' if redis_status else 'Disconnected'}")
        
        with col3:
            if st.session_state.last_update:
                time_diff = datetime.now() - st.session_state.last_update
                st.metric("Last Update", f"{time_diff.seconds}s ago")
            else:
                st.metric("Last Update", "Never")
    
    def check_api_health(self) -> bool:
        """Check API health status"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def render_sidebar(self):
        """Render sidebar controls"""
        st.sidebar.title("Dashboard Controls")
        
        # Refresh controls
        st.sidebar.subheader("Data Controls")
        auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
        if auto_refresh:
            st.sidebar.info(f"Refreshing every {self.refresh_interval} seconds")
        
        # Manual refresh
        if st.sidebar.button("Refresh Now"):
            st.session_state.last_update = datetime.now()
            st.rerun()
        
        # Data source selection
        st.sidebar.subheader("Data Source")
        data_source = st.sidebar.selectbox(
            "Select Data Source",
            ["Live API", "Historical Data", "Simulation"],
            index=0
        )
        
        # Time range selection
        st.sidebar.subheader("Time Range")
        time_range = st.sidebar.selectbox(
            "Select Time Range",
            ["Last Hour", "Last 6 Hours", "Last 24 Hours", "Last Week"],
            index=2
        )
        
        # Alert settings
        st.sidebar.subheader("Alert Settings")
        pm25_threshold = st.sidebar.slider("PM2.5 Alert Threshold (μg/m³)", 0, 150, 35)
        aqi_threshold = st.sidebar.slider("AQI Alert Threshold", 0, 300, 100)
        
        return data_source, time_range, pm25_threshold, aqi_threshold
    
    def render_main_metrics(self):
        """Render main metrics cards"""
        st.subheader("Current Air Quality Metrics")
        
        # Get current data
        current_data = self.get_current_data()
        
        if current_data:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                pm25 = current_data.get('pm25', 0)
                pm25_status = self.get_pollutant_status(pm25, 'pm25')
                st.metric(
                    "PM2.5", 
                    f"{pm25:.1f} μg/m³",
                    delta=pm25_status
                )
            
            with col2:
                co2 = current_data.get('co2', 0)
                co2_status = self.get_pollutant_status(co2, 'co2')
                st.metric(
                    "CO₂", 
                    f"{co2:.0f} ppm",
                    delta=co2_status
                )
            
            with col3:
                no2 = current_data.get('no2', 0)
                no2_status = self.get_pollutant_status(no2, 'no2')
                st.metric(
                    "NO₂", 
                    f"{no2:.1f} ppb",
                    delta=no2_status
                )
            
            with col4:
                aqi = current_data.get('aqi', 0)
                aqi_category = self.get_aqi_category(aqi)
                aqi_color = self.get_aqi_color(aqi_category)
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: {aqi_color};">AQI: {aqi}</h3>
                    <p>{aqi_category}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No current data available")
    
    def render_real_time_charts(self):
        """Render real-time charts"""
        st.subheader("Real-time Monitoring")
        
        # Get historical data
        historical_data = self.get_historical_data()
        
        if historical_data:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("PM2.5 Concentration", "CO₂ Levels", 
                             "NO₂ Levels", "AQI Trend"),
                vertical_spacing=0.1
            )
            
            # PM2.5 chart
            fig.add_trace(
                go.Scatter(
                    x=historical_data['timestamp'],
                    y=historical_data['pm25'],
                    mode='lines+markers',
                    name='PM2.5',
                    line=dict(color='red')
                ),
                row=1, col=1
            )
            
            # CO₂ chart
            fig.add_trace(
                go.Scatter(
                    x=historical_data['timestamp'],
                    y=historical_data['co2'],
                    mode='lines+markers',
                    name='CO₂',
                    line=dict(color='blue')
                ),
                row=1, col=2
            )
            
            # NO₂ chart
            fig.add_trace(
                go.Scatter(
                    x=historical_data['timestamp'],
                    y=historical_data['no2'],
                    mode='lines+markers',
                    name='NO₂',
                    line=dict(color='green')
                ),
                row=2, col=1
            )
            
            # AQI chart
            fig.add_trace(
                go.Scatter(
                    x=historical_data['timestamp'],
                    y=historical_data['aqi'],
                    mode='lines+markers',
                    name='AQI',
                    line=dict(color='purple')
                ),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                height=600,
                showlegend=False,
                title_text="Real-time Air Quality Trends"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No historical data available")
    
    def render_prediction_interface(self):
        """Render prediction interface"""
        st.subheader("Make Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Upload Image")
            uploaded_file = st.file_uploader(
                "Choose an image...",
                type=['jpg', 'jpeg', 'png']
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Convert to base64
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                st.session_state.uploaded_image = img_str
        
        with col2:
            st.write("Sensor Data Input")
            
            # Environmental sensors
            st.write("**Environmental Sensors**")
            temp = st.slider("Temperature (°C)", -10, 40, 22)
            humidity = st.slider("Humidity (%)", 20, 90, 50)
            pressure = st.slider("Pressure (hPa)", 980, 1030, 1013)
            pm25_input = st.slider("PM2.5 (μg/m³)", 0, 150, 25)
            co2_input = st.slider("CO₂ (ppm)", 350, 2000, 450)
            no2_input = st.slider("NO₂ (ppb)", 10, 200, 40)
            
            # Biosensors
            st.write("**Biosensors**")
            heart_rate = st.slider("Heart Rate (bpm)", 60, 100, 75)
            spo2 = st.slider("SpO₂ (%)", 95, 100, 98)
            skin_temp = st.slider("Skin Temperature (°C)", 32, 37, 34)
        
        # Prediction button
        if st.button("Make Prediction"):
            if 'uploaded_image' in st.session_state:
                with st.spinner("Making prediction..."):
                    try:
                        # Prepare request
                        request_data = {
                            "image_base64": st.session_state.uploaded_image,
                            "environmental_data": {
                                "temperature": [temp] * 24,
                                "humidity": [humidity] * 24,
                                "pressure": [pressure] * 24,
                                "pm25": [pm25_input] * 24,
                                "co2": [co2_input] * 24,
                                "no2": [no2_input] * 24
                            },
                            "biosensor_data": {
                                "heart_rate": [heart_rate] * 24,
                                "spo2": [spo2] * 24,
                                "skin_temperature": [skin_temp] * 24
                            }
                        }
                        
                        # Make API call
                        response = requests.post(
                            f"{self.api_url}/predict",
                            json=request_data,
                            timeout=30
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            self.render_prediction_result(result)
                        else:
                            st.error(f"Prediction failed: {response.text}")
                    
                    except Exception as e:
                        st.error(f"Error making prediction: {e}")
            else:
                st.error("Please upload an image first")
    
    def render_prediction_result(self, result: Dict):
        """Render prediction result"""
        st.subheader("Prediction Results")
        
        # Main metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "PM2.5",
                f"{result['predictions']['pm25']['value']:.1f} {result['predictions']['pm25']['unit']}",
                result['predictions']['pm25']['status']
            )
        
        with col2:
            st.metric(
                "CO₂",
                f"{result['predictions']['co2']['value']:.0f} {result['predictions']['co2']['unit']}",
                result['predictions']['co2']['status']
            )
        
        with col3:
            st.metric(
                "NO₂",
                f"{result['predictions']['no2']['value']:.1f} {result['predictions']['no2']['unit']}",
                result['predictions']['no2']['status']
            )
        
        # AQI result
        aqi_data = result['aqi']
        aqi_category = aqi_data['category']
        aqi_color = self.get_aqi_color(aqi_category)
        
        st.markdown(f"""
        <div style="background-color: {aqi_color}; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;">
            <h2 style="color: white;">AQI: {aqi_data['value']} - {aqi_category}</h2>
            <p style="color: white;">Confidence: {aqi_data['confidence']:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Risk assessment
        risk_data = result['risk_assessment']
        st.subheader("Risk Assessment")
        
        st.markdown(f"""
        <div class="alert-{self.get_alert_level(risk_data['level'])}">
            <h4>Risk Level: {risk_data['level']}</h4>
            <p>{risk_data['description']}</p>
            <h5>Recommendations:</h5>
            <ul>
        """, unsafe_allow_html=True)
        
        for rec in risk_data['recommendations']:
            st.markdown(f"<li>{rec}</li>", unsafe_allow_html=True)
        
        st.markdown("</ul></div>", unsafe_allow_html=True)
        
        # AQI probabilities
        st.subheader("AQI Classification Probabilities")
        
        probabilities = aqi_data['probabilities']
        categories = list(probabilities.keys())
        values = list(probabilities.values())
        
        fig = px.bar(
            x=categories,
            y=values,
            title="AQI Category Probabilities",
            labels={'x': 'AQI Category', 'y': 'Probability'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def render_alerts_panel(self):
        """Render alerts panel"""
        st.subheader("Alerts & Notifications")
        
        # Check for alerts
        current_data = self.get_current_data()
        if current_data:
            alerts = self.check_alerts(current_data)
            
            if alerts:
                for alert in alerts:
                    alert_level = self.get_alert_level(alert['severity'])
                    st.markdown(f"""
                    <div class="alert-{alert_level}">
                        <h4>{alert['title']}</h4>
                        <p>{alert['message']}</p>
                        <small>{alert['timestamp']}</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("No active alerts")
        else:
            st.info("No data available for alert checking")
    
    def get_current_data(self) -> Optional[Dict]:
        """Get current air quality data"""
        if self.redis_client:
            try:
                data = self.redis_client.get('current_air_quality')
                if data:
                    return json.loads(data)
            except Exception as e:
                logger.error(f"Error getting current data: {e}")
        
        # Fallback to API call
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                # Return mock data for demonstration
                return {
                    'pm25': 25.5,
                    'co2': 450.0,
                    'no2': 40.2,
                    'aqi': 75,
                    'timestamp': datetime.now().isoformat()
                }
        except:
            pass
        
        return None
    
    def get_historical_data(self) -> pd.DataFrame:
        """Get historical data"""
        # Generate mock historical data for demonstration
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(hours=24),
            end=datetime.now(),
            freq='H'
        )
        
        data = {
            'timestamp': timestamps,
            'pm25': np.random.normal(30, 10, len(timestamps)),
            'co2': np.random.normal(450, 50, len(timestamps)),
            'no2': np.random.normal(40, 10, len(timestamps)),
            'aqi': np.random.normal(75, 20, len(timestamps))
        }
        
        return pd.DataFrame(data)
    
    def get_pollutant_status(self, value: float, pollutant: str) -> str:
        """Get pollutant status"""
        thresholds = {
            'pm25': [(12, 'Good'), (35, 'Moderate'), (55, 'Unhealthy')],
            'co2': [(400, 'Good'), (1000, 'Moderate'), (2000, 'Unhealthy')],
            'no2': [(53, 'Good'), (100, 'Moderate'), (360, 'Unhealthy')]
        }
        
        for threshold, status in thresholds.get(pollutant, []):
            if value <= threshold:
                return status
        return 'Hazardous'
    
    def get_aqi_category(self, aqi: float) -> str:
        """Get AQI category"""
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
    
    def get_aqi_color(self, category: str) -> str:
        """Get AQI category color"""
        colors = {
            'Good': '#00e400',
            'Moderate': '#ffff00',
            'Unhealthy for Sensitive': '#ff7e00',
            'Unhealthy': '#ff0000',
            'Very Unhealthy': '#8f3f97',
            'Hazardous': '#7e0023'
        }
        return colors.get(category, '#808080')
    
    def get_alert_level(self, severity: str) -> str:
        """Get alert level"""
        if severity in ['Hazardous', 'Very Unhealthy']:
            return 'high'
        elif severity in ['Unhealthy', 'Unhealthy for Sensitive']:
            return 'medium'
        else:
            return 'low'
    
    def check_alerts(self, data: Dict) -> List[Dict]:
        """Check for alerts based on current data"""
        alerts = []
        
        # PM2.5 alert
        if data.get('pm25', 0) > 35:
            alerts.append({
                'title': 'High PM2.5 Levels',
                'message': f"PM2.5 concentration is {data['pm25']:.1f} μg/m³",
                'severity': self.get_aqi_category(self.calculate_aqi_from_pm25(data['pm25'])),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        
        # AQI alert
        if data.get('aqi', 0) > 100:
            alerts.append({
                'title': 'High AQI Levels',
                'message': f"AQI is {data['aqi']} - {self.get_aqi_category(data['aqi'])}",
                'severity': self.get_aqi_category(data['aqi']),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        
        return alerts
    
    def calculate_aqi_from_pm25(self, pm25: float) -> int:
        """Calculate AQI from PM2.5"""
        breakpoints = [
            (0, 12, 0, 50),
            (12.1, 35.4, 51, 100),
            (35.5, 55.4, 101, 150),
            (55.5, 150.4, 151, 200),
            (150.5, 250.4, 201, 300),
            (250.5, 500.4, 301, 500)
        ]
        
        for bp_low, bp_high, aqi_low, aqi_high in breakpoints:
            if bp_low <= pm25 <= bp_high:
                return int(((aqi_high - aqi_low) / (bp_high - bp_low)) * (pm25 - bp_low) + aqi_low)
        
        return 500
    
    def run(self):
        """Run the dashboard"""
        self.setup_page_config()
        
        # Auto refresh
        if st.session_state.get('auto_refresh', True):
            time.sleep(self.refresh_interval)
            st.rerun()
        
        # Render components
        self.render_header()
        
        data_source, time_range, pm25_threshold, aqi_threshold = self.render_sidebar()
        
        # Main content
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Real-time", "Prediction", "Alerts"])
        
        with tab1:
            self.render_main_metrics()
        
        with tab2:
            self.render_real_time_charts()
        
        with tab3:
            self.render_prediction_interface()
        
        with tab4:
            self.render_alerts_panel()

def main():
    """Main function to run the dashboard"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Air Quality Dashboard')
    parser.add_argument('--api_url', type=str, default='http://localhost:8000', help='API URL')
    parser.add_argument('--redis_url', type=str, default='redis://localhost:6379', help='Redis URL')
    parser.add_argument('--refresh_interval', type=int, default=30, help='Refresh interval (seconds)')
    
    args = parser.parse_args()
    
    # Initialize and run dashboard
    dashboard = AirQualityDashboard(
        api_url=args.api_url,
        redis_url=args.redis_url,
        refresh_interval=args.refresh_interval
    )
    
    dashboard.run()

if __name__ == "__main__":
    main()
