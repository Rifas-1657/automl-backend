#!/usr/bin/env python3
"""
Install all ML and visualization dependencies for the AutoML project.
Run this script to install all required packages.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    print("🚀 INSTALLING ML AND VISUALIZATION DEPENDENCIES")
    print("=" * 60)
    
    # Check if we're in a virtual environment
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Warning: You might not be in a virtual environment.")
        print("Consider activating your venv first:")
        print("  Windows: .\\venv\\Scripts\\activate")
        print("  Linux/Mac: source venv/bin/activate")
        print()
    
    # Core ML dependencies
    core_packages = [
        "pandas==2.2.2",
        "scikit-learn==1.5.2",
        "numpy==2.1.1",
        "joblib==1.4.2",
        "matplotlib==3.9.2",
        "seaborn==0.13.2",
        "plotly==5.17.0"
    ]
    
    # Advanced ML packages
    advanced_packages = [
        "xgboost==2.0.3",
        "lightgbm==4.1.0",
        "catboost==1.2.2",
        "imbalanced-learn==0.12.0"
    ]
    
    # Feature engineering packages
    feature_packages = [
        "feature-engine==1.6.2",
        "category-encoders==2.6.3"
    ]
    
    # Model selection packages
    model_packages = [
        "optuna==3.5.0",
        "mlxtend==0.23.0"
    ]
    
    # Utility packages
    utility_packages = [
        "orjson==3.9.10",
        "great-expectations==0.18.5"
    ]
    
    all_packages = core_packages + advanced_packages + feature_packages + model_packages + utility_packages
    
    success_count = 0
    failed_packages = []
    
    for package in all_packages:
        if run_command(f"pip install {package}", f"Installing {package}"):
            success_count += 1
        else:
            failed_packages.append(package)
    
    print("\n" + "=" * 60)
    print(f"📊 INSTALLATION SUMMARY")
    print(f"✅ Successfully installed: {success_count}/{len(all_packages)} packages")
    
    if failed_packages:
        print(f"❌ Failed packages: {len(failed_packages)}")
        for package in failed_packages:
            print(f"   - {package}")
        
        print("\n🛠️  TROUBLESHOOTING:")
        print("1. Try installing failed packages individually:")
        for package in failed_packages:
            print(f"   pip install {package}")
        print("2. Update pip: pip install --upgrade pip")
        print("3. Check your Python version (3.8+ recommended)")
        print("4. Try installing without version constraints:")
        for package in failed_packages:
            name = package.split("==")[0]
            print(f"   pip install {name}")
    else:
        print("🎉 ALL PACKAGES INSTALLED SUCCESSFULLY!")
        print("\nYour AutoML application is ready to use!")
        print("Features now available:")
        print("✅ Dataset analysis and preprocessing")
        print("✅ Algorithm recommendations")
        print("✅ Model training and evaluation")
        print("✅ Predictions on new data")
        print("✅ Interactive visualizations")
        print("✅ Feature importance analysis")
    
    print("\n📋 Next steps:")
    print("1. Restart your FastAPI server")
    print("2. Test the new ML endpoints")
    print("3. Upload a dataset and try the analysis features")

if __name__ == "__main__":
    main()
