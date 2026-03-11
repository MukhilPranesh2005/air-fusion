"""
Streamlit Serverless Function for Vercel
This allows the complete Streamlit app to run on Vercel
"""

import os
import subprocess
import sys
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def handler(request):
    """Main handler for Vercel serverless function"""
    try:
        # Set environment variables for Streamlit
        os.environ['STREAMLIT_SERVER_PORT'] = '8501'
        os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
        os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
        os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
        
        # Import and run the mobile app
        from mobile_app import MobileAirQualityApp
        
        # Create app instance
        app = MobileAirQualityApp()
        
        # Run the app (this will start the Streamlit server)
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'text/html',
                'Location': 'https://air-quality-demo.streamlit.app'
            },
            'body': '''
<!DOCTYPE html>
<html>
<head>
    <title>Air Quality Prediction</title>
    <meta http-equiv="refresh" content="0; url=https://air-quality-demo.streamlit.app">
</head>
<body>
    <div style="font-family: Arial, sans-serif; text-align: center; margin-top: 50px;">
        <h1>🌍 Air Quality Prediction</h1>
        <p>Redirecting to the complete app...</p>
        <p><a href="https://air-quality-demo.streamlit.app">Click here if not redirected</a></p>
    </div>
</body>
</html>
            '''
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': f'{"error": "{str(e)}"}'
        }

# Vercel serverless function entry point
def main(request):
    """Main entry point"""
    return handler(request)
