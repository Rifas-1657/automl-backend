#!/usr/bin/env python3
"""
Simple test to verify the AutoML system is working
"""

import requests
import json

def test_system():
    print("Testing AutoML System...")
    
    # Test 1: Check if backend is running
    try:
        response = requests.get("http://localhost:8000/docs")
        if response.status_code == 200:
            print("✓ Backend server is running")
        else:
            print("✗ Backend server not responding")
            return False
    except Exception as e:
        print(f"✗ Cannot connect to backend: {e}")
        return False
    
    # Test 2: Test user signup
    signup_data = {
        "email": "test@example.com",
        "username": "testuser123",
        "password": "testpass123"
    }
    
    try:
        response = requests.post("http://localhost:8000/api/signup", json=signup_data)
        if response.status_code == 200:
            print("✓ User signup successful")
            user_data = response.json()
            user_id = user_data["id"]
        elif response.status_code == 400 and "already registered" in response.text:
            print("✓ User already exists")
            user_id = 1
        else:
            print(f"✗ Signup failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Signup error: {e}")
        return False
    
    # Test 3: Test login
    login_data = {
        "username": "testuser123",
        "password": "testpass123"
    }
    
    try:
        response = requests.post("http://localhost:8000/api/login", data=login_data)
        if response.status_code == 200:
            print("✓ User login successful")
            token_data = response.json()
            token = token_data["access_token"]
        else:
            print(f"✗ Login failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Login error: {e}")
        return False
    
    # Test 4: Test dataset upload
    test_csv = """age,income,score
25,45000,75
30,55000,82
35,65000,88"""
    
    with open("temp_test.csv", "w") as f:
        f.write(test_csv)
    
    try:
        headers = {"Authorization": f"Bearer {token}"}
        with open("temp_test.csv", "rb") as f:
            files = {"file": ("test.csv", f, "text/csv")}
            response = requests.post("http://localhost:8000/api/upload", files=files, headers=headers)
        
        if response.status_code == 200:
            print("✓ Dataset upload successful")
            upload_data = response.json()
            dataset_id = upload_data["dataset"]["id"]
        else:
            print(f"✗ Upload failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"✗ Upload error: {e}")
        return False
    finally:
        import os
        if os.path.exists("temp_test.csv"):
            os.remove("temp_test.csv")
    
    # Test 5: Test dataset analysis
    try:
        response = requests.post(f"http://localhost:8000/api/analyze/{dataset_id}", headers=headers)
        if response.status_code == 200:
            print("✓ Dataset analysis successful")
            analysis_data = response.json()
            print(f"   Task Type: {analysis_data['task_type']}")
            print(f"   Target: {analysis_data['target']}")
        else:
            print(f"✗ Analysis failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Analysis error: {e}")
        return False
    
    print("\nALL TESTS PASSED!")
    print("✓ Backend server is running")
    print("✓ Authentication is working")
    print("✓ Dataset upload is working")
    print("✓ Dataset analysis is working")
    print("\nYour AutoML system is ready!")
    print("Frontend: http://localhost:5173")
    print("Backend: http://localhost:8000")
    
    return True

if __name__ == "__main__":
    test_system()
