#!/usr/bin/env python3
"""
Comprehensive system test for AutoML application.
Tests authentication, dataset upload, analysis, training, and predictions.
"""

import requests
import json
import time
import os
from typing import Dict, Any

# Configuration
BASE_URL = "http://localhost:8000"
API_BASE = f"{BASE_URL}/api"

class AutoMLTester:
    def __init__(self):
        self.session = requests.Session()
        self.token = None
        self.user_id = None
        self.dataset_id = None
        self.model_id = None
        
    def test_auth(self) -> bool:
        """Test user authentication (signup and login)"""
        print("🔐 Testing Authentication...")
        
        # Test signup
        signup_data = {
            "email": "test@automl.com",
            "username": "testuser",
            "password": "testpass123"
        }
        
        try:
            response = self.session.post(f"{API_BASE}/signup", json=signup_data)
            if response.status_code == 200:
                print("✅ Signup successful")
                user_data = response.json()
                self.user_id = user_data["id"]
            elif response.status_code == 400 and "already registered" in response.text:
                print("✅ User already exists, proceeding with login")
            else:
                print(f"❌ Signup failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"❌ Signup error: {e}")
            return False
        
        # Test login
        login_data = {
            "username": "testuser",
            "password": "testpass123"
        }
        
        try:
            response = self.session.post(f"{API_BASE}/login", data=login_data)
            if response.status_code == 200:
                print("✅ Login successful")
                token_data = response.json()
                self.token = token_data["access_token"]
                self.session.headers.update({"Authorization": f"Bearer {self.token}"})
                return True
            else:
                print(f"❌ Login failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"❌ Login error: {e}")
            return False
    
    def test_dataset_upload(self) -> bool:
        """Test dataset upload functionality"""
        print("\n📤 Testing Dataset Upload...")
        
        # Create test CSV file
        test_csv_content = """age,income,education,experience,score
25,45000,Bachelor,2,75
30,55000,Master,5,82
35,65000,PhD,8,88
28,48000,Bachelor,3,78
32,58000,Master,6,85
40,75000,PhD,12,92
26,46000,Bachelor,2,76
33,62000,Master,7,87
29,52000,Bachelor,4,80
37,70000,PhD,10,90
31,56000,Master,5,83
27,47000,Bachelor,3,77
34,63000,Master,8,86
38,72000,PhD,11,91
24,44000,Bachelor,1,74
36,68000,PhD,9,89
39,73000,PhD,13,93
41,78000,PhD,15,94
42,80000,PhD,16,95
43,82000,PhD,17,96"""
        
        # Write test file
        with open("test_data.csv", "w") as f:
            f.write(test_csv_content)
        
        try:
            with open("test_data.csv", "rb") as f:
                files = {"file": ("test_data.csv", f, "text/csv")}
                response = self.session.post(f"{API_BASE}/upload", files=files)
            
            if response.status_code == 200:
                print("✅ Dataset upload successful")
                upload_data = response.json()
                self.dataset_id = upload_data["dataset"]["id"]
                print(f"📊 Dataset ID: {self.dataset_id}")
                return True
            else:
                print(f"❌ Upload failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"❌ Upload error: {e}")
            return False
        finally:
            # Clean up test file
            if os.path.exists("test_data.csv"):
                os.remove("test_data.csv")
    
    def test_dataset_analysis(self) -> bool:
        """Test dataset analysis functionality"""
        print("\n🔍 Testing Dataset Analysis...")
        
        if not self.dataset_id:
            print("❌ No dataset ID available for analysis")
            return False
        
        try:
            response = self.session.post(f"{API_BASE}/analyze/{self.dataset_id}")
            if response.status_code == 200:
                print("✅ Dataset analysis successful")
                analysis_data = response.json()
                print(f"📈 Task Type: {analysis_data['task_type']}")
                print(f"🎯 Target: {analysis_data['target']}")
                print(f"💡 Suggestions: {', '.join(analysis_data['suggestions'][:3])}")
                return True
            else:
                print(f"❌ Analysis failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"❌ Analysis error: {e}")
            return False
    
    def test_model_training(self) -> bool:
        """Test model training functionality"""
        print("\n🤖 Testing Model Training...")
        
        if not self.dataset_id:
            print("❌ No dataset ID available for training")
            return False
        
        training_data = {
            "dataset_id": self.dataset_id,
            "algorithm": "Random Forest",
            "task_type": "regression"
        }
        
        try:
            response = self.session.post(f"{API_BASE}/train", json=training_data)
            if response.status_code == 200:
                print("✅ Model training successful")
                training_result = response.json()
                self.model_id = training_result["model_id"]
                metrics = training_result["metrics"]
                print(f"📊 Model ID: {self.model_id}")
                print(f"🎯 R² Score: {metrics.get('r2_score', 'N/A')}")
                print(f"📉 MSE: {metrics.get('mse', 'N/A')}")
                return True
            else:
                print(f"❌ Training failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"❌ Training error: {e}")
            return False
    
    def test_predictions(self) -> bool:
        """Test prediction functionality"""
        print("\n🎯 Testing Predictions...")
        
        if not self.dataset_id:
            print("❌ No dataset ID available for predictions")
            return False
        
        prediction_data = {
            "dataset_id": self.dataset_id,
            "input_features": {
                "age": 30,
                "income": 55000,
                "education": "Master",
                "experience": 5
            }
        }
        
        try:
            response = self.session.post(f"{API_BASE}/predict", json=prediction_data)
            if response.status_code == 200:
                print("✅ Prediction successful")
                prediction_result = response.json()
                print(f"🔮 Prediction: {prediction_result['prediction']}")
                print(f"🤖 Algorithm: {prediction_result['algorithm_used']}")
                if prediction_result.get('confidence'):
                    print(f"📊 Confidence: {prediction_result['confidence']:.2%}")
                return True
            else:
                print(f"❌ Prediction failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return False
    
    def test_visualizations(self) -> bool:
        """Test visualization generation"""
        print("\n📊 Testing Visualizations...")
        
        if not self.dataset_id:
            print("❌ No dataset ID available for visualizations")
            return False
        
        try:
            response = self.session.get(f"{API_BASE}/visualizations/{self.dataset_id}")
            if response.status_code == 200:
                print("✅ Visualization generation successful")
                viz_data = response.json()
                print(f"📈 Generated {len(viz_data['plot_urls'])} visualizations")
                return True
            else:
                print(f"❌ Visualization failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"❌ Visualization error: {e}")
            return False
    
    def test_dataset_listing(self) -> bool:
        """Test dataset listing functionality"""
        print("\n📋 Testing Dataset Listing...")
        
        try:
            response = self.session.get(f"{API_BASE}/datasets")
            if response.status_code == 200:
                print("✅ Dataset listing successful")
                datasets = response.json()
                print(f"📊 Found {len(datasets)} datasets")
                return True
            else:
                print(f"❌ Dataset listing failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"❌ Dataset listing error: {e}")
            return False
    
    def run_all_tests(self) -> bool:
        """Run all tests and return overall success"""
        print("🚀 Starting AutoML System Tests")
        print("=" * 50)
        
        tests = [
            self.test_auth,
            self.test_dataset_upload,
            self.test_dataset_listing,
            self.test_dataset_analysis,
            self.test_model_training,
            self.test_predictions,
            self.test_visualizations
        ]
        
        results = []
        for test in tests:
            try:
                result = test()
                results.append(result)
                if result:
                    print("✅ PASSED")
                else:
                    print("❌ FAILED")
            except Exception as e:
                print(f"❌ TEST ERROR: {e}")
                results.append(False)
        
        print("\n" + "=" * 50)
        passed = sum(results)
        total = len(results)
        print(f"🎯 Test Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 ALL TESTS PASSED! System is working correctly.")
        else:
            print("⚠️ Some tests failed. Check the logs above.")
        
        return passed == total

def main():
    """Main test runner"""
    print("🔧 AutoML System Test Suite")
    print("Make sure the backend server is running on http://localhost:8000")
    print("Press Enter to continue or Ctrl+C to cancel...")
    
    try:
        input()
    except KeyboardInterrupt:
        print("\n❌ Tests cancelled by user")
        return
    
    tester = AutoMLTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n🎉 System verification complete! Your AutoML application is ready to use.")
        print("\n🌐 Frontend: http://localhost:5173")
        print("🔧 Backend API: http://localhost:8000")
        print("📚 API Docs: http://localhost:8000/docs")
    else:
        print("\n⚠️ Some tests failed. Please check the backend server and try again.")

if __name__ == "__main__":
    main()
