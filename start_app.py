#!/usr/bin/env python3
"""
Startup script for the AI-Powered Internal Knowledge Assistant
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("Checking dependencies...")
    
    required_packages = [
        'fastapi',
        'uvicorn',
        'langchain',
        'openai',
        'pandas',
        'sqlparse'
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

def check_database():
    """Check if database exists and is populated"""
    print("Checking database...")
    
    if not os.path.exists("company.db"):
        print("Database not found. Creating and populating database...")
        try:
            from database import main
            main()
            print("Database created and populated successfully")
        except Exception as e:
            print(f"Failed to create database: {e}")
            return False
    else:
        print("Database exists")
    
    return True

def check_openai_key():
    """Check if OpenAI API key is set"""
    print("Checking OpenAI API key...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("WARNING: OPENAI_API_KEY not set")
        print("   LLM features will not work without an API key")
        print("   Set it with: export OPENAI_API_KEY=your_key_here")
        print("   Or create a .env file with: OPENAI_API_KEY=your_key_here")
        return False
    
    print("OpenAI API key is set")
    return True

def start_server():
    """Start the FastAPI server"""
    print("Starting FastAPI server...")
    
    try:
        # Start the server
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "api:app", 
            "--host", "0.0.0.0", 
            "--port", "8000",
            "--reload"
        ])
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Open browser
        print("Opening browser...")
        webbrowser.open("http://localhost:8000")
        
        print("Server started successfully!")
        print("Chat interface available at: http://localhost:8000")
        print("API documentation available at: http://localhost:8000/docs")
        print("Press Ctrl+C to stop the server")
        
        # Wait for user to stop
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\nStopping server...")
            process.terminate()
            process.wait()
            print("Server stopped")
        
    except Exception as e:
        print(f"Failed to start server: {e}")
        return False
    
    return True

def main():
    """Main startup function"""
    print("AI-Powered Internal Knowledge Assistant")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check database
    if not check_database():
        sys.exit(1)
    
    # Check OpenAI key (warning only)
    check_openai_key()
    
    print("\nSystem ready!")
    print("Starting the application...\n")
    
    # Start server
    start_server()

if __name__ == "__main__":
    main()
