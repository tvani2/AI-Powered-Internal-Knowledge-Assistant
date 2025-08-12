#!/usr/bin/env python3
"""
Simple test script for the API
"""

import requests
import json

def test_api():
    """Test the API with a simple query"""
    
    # Test the health endpoint first
    try:
        print("Testing health endpoint...")
        response = requests.get("http://localhost:8000/health")
        print(f"Health status: {response.status_code}")
        if response.status_code == 200:
            health_data = response.json()
            print(f"Health data: {json.dumps(health_data, indent=2)}")
    except Exception as e:
        print(f"Health check failed: {e}")
        return
    
    # Test the chat endpoint
    try:
        print("\nTesting chat endpoint...")
        query = "What are the remote work policies?"
        
        response = requests.post(
            "http://localhost:8000/chat",
            json={"query": query}
        )
        
        print(f"Chat status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Query type: {data.get('query_type')}")
            print(f"Confidence: {data.get('confidence')}")
            print(f"Processing time: {data.get('processing_time')}s")
            print(f"Response: {data.get('response', '')[:500]}...")
        else:
            print(f"Error response: {response.text}")
            
    except Exception as e:
        print(f"Chat test failed: {e}")

if __name__ == "__main__":
    test_api()
