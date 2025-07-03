#!/usr/bin/env python3
"""
Test script for namespace-based presentation API
"""

import requests
import json
import sys

def test_namespace_presentation_api():
    """Test the new namespace-based presentation endpoint"""
    
    # Test data
    test_data = {
        "user_id": "123",
        "doc_id": "test-uuid-456",
        "prompt": "Create a presentation about machine learning techniques",
        "title": "ML Techniques Overview",
        "theme": "academic_professional",
        "slide_count": 10,
        "audience_type": "technical"
    }
    
    # API endpoint - Updated to use port 8001 for your FastAPI server
    url = "http://localhost:8001/api/v1/mcp/presentations/generate-from-namespace"
    
    print("🧪 Testing Namespace-Based Presentation API")
    print("=" * 60)
    print(f"URL: {url}")
    print(f"Data: {json.dumps(test_data, indent=2)}")
    print("=" * 60)
    
    try:
        # Make the request
        response = requests.post(url, json=test_data, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print("Response Body:")
        print(json.dumps(response.json(), indent=2))
        
        if response.status_code == 200:
            print("✅ API call successful!")
            return True
        else:
            print("❌ API call failed!")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode failed: {e}")
        print(f"Raw response: {response.text}")
        return False

def test_mcp_direct_call():
    """Test the MCP tool directly"""
    
    test_data = {
        "tool": "create_presentation_from_namespace",
        "arguments": {
            "namespace": "user_123_doc_test-uuid-789",
            "user_prompt": "Create a presentation about AI research methods",
            "title": "AI Research Methods",
            "slide_count": 8
        }
    }
    
    url = "http://localhost:3001/mcp/call"
    
    print("\n🧪 Testing MCP Direct Call")
    print("=" * 60)
    print(f"URL: {url}")
    print(f"Data: {json.dumps(test_data, indent=2)}")
    print("=" * 60)
    
    try:
        response = requests.post(url, json=test_data, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        print("Response Body:")
        print(json.dumps(response.json(), indent=2))
        
        if response.status_code == 200:
            print("✅ MCP call successful!")
            return True
        else:
            print("❌ MCP call failed!")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False

def test_fastapi_health():
    """Test FastAPI server health"""
    
    url = "http://localhost:8001/api/v1/mcp/health"
    
    print("\n🧪 Testing FastAPI Health Check")
    print("=" * 60)
    print(f"URL: {url}")
    print("=" * 60)
    
    try:
        response = requests.get(url, timeout=10)
        
        print(f"Status Code: {response.status_code}")
        print("Response Body:")
        print(json.dumps(response.json(), indent=2))
        
        if response.status_code == 200:
            print("✅ FastAPI health check successful!")
            return True
        else:
            print("❌ FastAPI health check failed!")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting API Tests")
    print("=" * 60)
    
    # Test 1: FastAPI health check
    health_success = test_fastapi_health()
    
    # Test 2: FastAPI integration endpoint
    success1 = test_namespace_presentation_api()
    
    # Test 3: Direct MCP call
    success2 = test_mcp_direct_call()
    
    print("\n" + "=" * 60)
    print("📊 Test Results")
    print("=" * 60)
    print(f"FastAPI Health Check: {'✅ PASS' if health_success else '❌ FAIL'}")
    print(f"FastAPI Integration: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"MCP Direct Call: {'✅ PASS' if success2 else '❌ FAIL'}")
    
    if success1 and success2:
        print("\n🎉 All tests passed! The namespace-based presentation API is working correctly.")
        print("\n📋 Summary:")
        print("- MCP Server (port 3001): ✅ Running and responding")
        print("- FastAPI Server (port 8001): ✅ Integrated with MCP")
        print("- Namespace-based PPT API: ✅ Working")
        print("\n🔧 Usage:")
        print("POST http://localhost:8001/api/v1/mcp/presentations/generate-from-namespace")
        print("Body: {\"user_id\": \"123\", \"doc_id\": \"uuid\", \"prompt\": \"your prompt\"}")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        sys.exit(1) 