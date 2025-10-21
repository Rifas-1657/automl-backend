#!/usr/bin/env python3
import requests
import json
import random

BASE_URL = "http://127.0.0.1:8000/api"

def test_auth_flow():
    # Generate random user
    user_id = random.randint(1000, 9999)
    email = f"user{user_id}@test.com"
    username = f"user{user_id}"
    password = "Passw0rd!123456"
    
    print(f"Testing with user: {username}")
    
    # 1. Test signup
    print("\n1. Testing signup...")
    signup_data = {
        "email": email,
        "username": username,
        "password": password
    }
    
    try:
        response = requests.post(f"{BASE_URL}/signup", json=signup_data)
        print(f"Signup status: {response.status_code}")
        if response.status_code == 200:
            print("SUCCESS: Signup successful")
        else:
            print(f"FAILED: Signup failed: {response.text}")
            return
    except Exception as e:
        print(f"ERROR: Signup error: {e}")
        return
    
    # 2. Test login
    print("\n2. Testing login...")
    login_data = {
        "username": username,
        "password": password
    }
    
    try:
        response = requests.post(f"{BASE_URL}/login", data=login_data)
        print(f"Login status: {response.status_code}")
        if response.status_code == 200:
            token = response.json()["access_token"]
            print("SUCCESS: Login successful")
            print(f"Token: {token[:50]}...")
        else:
            print(f"FAILED: Login failed: {response.text}")
            return
    except Exception as e:
        print(f"ERROR: Login error: {e}")
        return
    
    # 3. Test token endpoint
    print("\n3. Testing token endpoint...")
    try:
        response = requests.post(f"{BASE_URL}/token", data=login_data)
        print(f"Token endpoint status: {response.status_code}")
        if response.status_code == 200:
            print("SUCCESS: Token endpoint successful")
        else:
            print(f"FAILED: Token endpoint failed: {response.text}")
    except Exception as e:
        print(f"ERROR: Token endpoint error: {e}")
    
    # 4. Test protected endpoints
    headers = {"Authorization": f"Bearer {token}"}
    
    print("\n4. Testing protected endpoints...")
    
    # Test /api/users/me
    try:
        response = requests.get(f"{BASE_URL}/users/me", headers=headers)
        print(f"Users/me status: {response.status_code}")
        if response.status_code == 200:
            print("SUCCESS: Users/me successful")
            print(f"User info: {response.json()}")
        else:
            print(f"FAILED: Users/me failed: {response.text}")
    except Exception as e:
        print(f"ERROR: Users/me error: {e}")
    
    # Test /api/datasets
    try:
        response = requests.get(f"{BASE_URL}/datasets", headers=headers)
        print(f"Datasets status: {response.status_code}")
        if response.status_code == 200:
            print("SUCCESS: Datasets successful")
            print(f"Datasets: {response.json()}")
        else:
            print(f"FAILED: Datasets failed: {response.text}")
    except Exception as e:
        print(f"ERROR: Datasets error: {e}")
    
    # Test /api/history
    try:
        response = requests.get(f"{BASE_URL}/history", headers=headers)
        print(f"History status: {response.status_code}")
        if response.status_code == 200:
            print("SUCCESS: History successful")
            print(f"History: {response.json()}")
        else:
            print(f"FAILED: History failed: {response.text}")
    except Exception as e:
        print(f"ERROR: History error: {e}")
    
    # Test /api/account
    try:
        response = requests.get(f"{BASE_URL}/account", headers=headers)
        print(f"Account status: {response.status_code}")
        if response.status_code == 200:
            print("SUCCESS: Account successful")
            print(f"Account: {response.json()}")
        else:
            print(f"FAILED: Account failed: {response.text}")
    except Exception as e:
        print(f"ERROR: Account error: {e}")
    
    # Test /api/test-auth
    try:
        response = requests.get(f"{BASE_URL}/test-auth", headers=headers)
        print(f"Test-auth status: {response.status_code}")
        if response.status_code == 200:
            print("SUCCESS: Test-auth successful")
            print(f"Test-auth: {response.json()}")
        else:
            print(f"FAILED: Test-auth failed: {response.text}")
    except Exception as e:
        print(f"ERROR: Test-auth error: {e}")

if __name__ == "__main__":
    test_auth_flow()