#!/usr/bin/env python3
"""
Startup script for the Streamlit UI
"""

import os
import sys
import subprocess
import webbrowser
import time
import requests

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("Checking dependencies...")
    
    required_packages = [
        'streamlit',
        'fastapi',
        'uvicorn',
        'langchain',
        'openai',
        'pandas',
        'sqlparse',
        'requests'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        print("Please install missing packages with: pip install -r requirements.txt")
        return False
    
    print("All dependencies are installed")
    return True

def check_api_server():
    """Check if the API server is running"""
    print("Checking API server...")
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("API server is running")
            return True
        else:
            print("API server responded with error")
            return False
    except:
        print("API server is not running")
        print("   Please start the API server first with: python api.py")
        return False

def start_streamlit():
    """Start the Streamlit application"""
    print("Starting Streamlit application...")
    
    try:
        # Start Streamlit
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Open browser
        print("Opening browser...")
        webbrowser.open("http://localhost:8501")
        
        print("Streamlit application started successfully!")
        print("Streamlit UI available at: http://localhost:8501")
        print("Press Ctrl+C to stop the application")
        
        # Wait for user to stop
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\nStopping Streamlit...")
            process.terminate()
            process.wait()
            print("Streamlit stopped")
        
    except Exception as e:
        print(f"Failed to start Streamlit: {e}")
        return False
    
    return True

def main():
    """Main startup function"""
    print("AI-Powered Internal Knowledge Assistant - Streamlit UI")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check API server
    if not check_api_server():
        print("\nTo start the API server:")
        print("   1. Open a new terminal")
        print("   2. Run: python api.py")
        print("   3. Then run this script again")
        sys.exit(1)
    
    print("\nSystem ready!")
    print("Starting Streamlit application...\n")
    
    # Start Streamlit
    start_streamlit()

if __name__ == "__main__":
    main()

