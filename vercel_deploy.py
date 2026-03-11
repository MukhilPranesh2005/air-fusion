"""
Simple Vercel Deployment Script
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed")
            return True
        else:
            print(f"❌ {description} failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main deployment function"""
    print("🚀 Vercel Deployment for Air Quality App")
    print("=" * 50)
    
    # Check if Vercel CLI is installed
    if not run_command("vercel --version", "Checking Vercel CLI"):
        print("\n📦 Installing Vercel CLI...")
        run_command("npm install -g vercel", "Installing Vercel CLI")
    
    # Add files to git
    run_command("git add .", "Adding files to Git")
    
    # Commit changes
    run_command('git commit -m "Add Vercel deployment files"', "Committing changes")
    
    # Deploy to Vercel
    print("\n🌍 Deploying to Vercel...")
    print("📋 Follow the prompts:")
    print("   1. Log in to Vercel")
    print("   2. Select or create project")
    print("   3. Confirm deployment settings")
    print("   4. Wait for deployment URL")
    
    # Run Vercel deployment
    subprocess.run("vercel --prod", shell=True)

if __name__ == "__main__":
    main()
