"""
Netlify Deployment Script for Air Quality App

This script helps deploy the Streamlit app to Netlify using Streamlit Cloud
as the backend and Netlify for the frontend landing page.
"""

import os
import subprocess
from pathlib import Path

def create_netlify_site():
    """Create Netlify site configuration"""
    
    # Create public directory structure
    public_dir = Path("public")
    public_dir.mkdir(exist_ok=True)
    
    # Create placeholder icons (you should replace with actual icons)
    create_placeholder_icons()
    
    print("✅ Netlify site structure created")
    print("📁 Files ready for deployment:")
    
    for file in public_dir.rglob("*"):
        if file.is_file():
            print(f"   📄 {file.relative_to(public_dir)}")

def create_placeholder_icons():
    """Create placeholder icon files"""
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    
    # Create simple placeholder icons
    for size in [192, 512]:
        img = Image.new('RGBA', (size, size), (102, 126, 234, 255))
        draw = ImageDraw.Draw(img)
        
        # Draw simple earth icon
        center = size // 2
        radius = size // 3
        
        # Draw circle
        draw.ellipse([center-radius, center-radius, center+radius, center+radius], 
                    fill=(34, 197, 94, 255))
        
        # Draw continents (simplified)
        draw.ellipse([center-radius//2, center-radius//2, center+radius//3, center+radius//3], 
                    fill=(139, 69, 19, 255))
        
        img.save(f'public/icon-{size}.png')
        print(f"   🎨 Created icon-{size}.png")

def deploy_to_streamlit_cloud():
    """Deploy Streamlit app to Streamlit Cloud"""
    
    print("🚀 Deploying to Streamlit Cloud...")
    
    # Create requirements.txt for Streamlit Cloud
    with open("requirements_streamlit.txt", "w") as f:
        f.write("""streamlit==1.55.0
plotly==6.6.0
requests==2.32.5
streamlit-option-menu==0.4.0
Pillow==12.1.0
numpy==2.4.2
pandas==2.3.3
""")
    
    print("📋 Created requirements_streamlit.txt")
    print("🌐 To deploy to Streamlit Cloud:")
    print("   1. Go to https://share.streamlit.io/")
    print("   2. Connect your GitHub repository")
    print("   3. Use 'mobile_app.py' as main file")
    print("   4. Use 'requirements_streamlit.txt' as requirements file")

def deploy_to_netlify():
    """Deploy frontend to Netlify"""
    
    print("🌐 Deploying frontend to Netlify...")
    
    # Instructions for manual Netlify deployment
    print("📋 Manual Netlify Deployment Steps:")
    print("   1. Go to https://app.netlify.com/drop")
    print("   2. Drag and drop the 'public' folder")
    print("   3. Your site will be live instantly!")
    print()
    print("🔗 Or use Netlify CLI:")
    print("   1. Install: npm install -g netlify-cli")
    print("   2. Login: netlify login")
    print("   3. Deploy: netlify deploy --prod --dir=public")

def create_github_actions():
    """Create GitHub Actions for automatic deployment"""
    
    workflow = """name: Deploy to Netlify

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '18'
    
    - name: Deploy to Netlify
      uses: netlify/actions/cli@master
      with:
        args: deploy --prod --dir=public
      env:
        NETLIFY_AUTH_TOKEN: ${{ secrets.NETLIFY_AUTH_TOKEN }}
        NETLIFY_SITE_ID: ${{ secrets.NETLIFY_SITE_ID }}
"""
    
    os.makedirs(".github/workflows", exist_ok=True)
    with open(".github/workflows/deploy-netlify.yml", "w") as f:
        f.write(workflow)
    
    print("✅ GitHub Actions workflow created")
    print("🔐 Set these secrets in GitHub:")
    print("   - NETLIFY_AUTH_TOKEN")
    print("   - NETLIFY_SITE_ID")

def main():
    """Main deployment function"""
    
    print("🌍 Air Quality App - Netlify Deployment")
    print("=" * 50)
    
    # Create site structure
    create_netlify_site()
    
    # Deployment options
    print("\n🚀 Deployment Options:")
    print("1. Streamlit Cloud (for the app)")
    print("2. Netlify (for the landing page)")
    print("3. Both (recommended)")
    print("4. GitHub Actions setup")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == "1":
        deploy_to_streamlit_cloud()
    elif choice == "2":
        deploy_to_netlify()
    elif choice == "3":
        deploy_to_streamlit_cloud()
        print("\n" + "="*50)
        deploy_to_netlify()
    elif choice == "4":
        create_github_actions()
    else:
        print("❌ Invalid choice")
    
    print("\n✅ Deployment setup complete!")
    print("📚 For detailed guide, check README_ADVANCED.md")

if __name__ == "__main__":
    main()
