#!/usr/bin/env python3
"""
Script to fix bcrypt and passlib compatibility issues.
Run this to update your virtual environment with compatible versions.
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
    print("🚨 FIXING BCRYPT AND PASSLIB COMPATIBILITY ISSUES")
    print("=" * 60)
    
    # Check if we're in a virtual environment
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Warning: You might not be in a virtual environment.")
        print("Consider activating your venv first:")
        print("  Windows: .\\venv\\Scripts\\activate")
        print("  Linux/Mac: source venv/bin/activate")
        print()
    
    # Uninstall problematic packages
    commands = [
        ("pip uninstall -y bcrypt passlib", "Uninstalling problematic bcrypt/passlib versions"),
        ("pip install bcrypt==4.1.2", "Installing compatible bcrypt version"),
        ("pip install 'passlib[bcrypt]==1.7.4'", "Installing compatible passlib version"),
        ("pip install --upgrade pip", "Upgrading pip"),
        ("pip install -r requirements.txt", "Installing all requirements"),
    ]
    
    success_count = 0
    for command, description in commands:
        if run_command(command, description):
            success_count += 1
        else:
            print(f"\n❌ Failed at: {description}")
            print("You may need to run this manually or check your environment.")
            break
    
    print("\n" + "=" * 60)
    if success_count == len(commands):
        print("🎉 ALL DEPENDENCIES FIXED SUCCESSFULLY!")
        print("\nYour FastAPI application should now work without bcrypt errors.")
        print("Try signing up a new user to test the fix.")
    else:
        print(f"⚠️  {success_count}/{len(commands)} commands completed successfully.")
        print("Some issues may remain. Check the error messages above.")
    
    print("\n📋 Next steps:")
    print("1. Restart your FastAPI server")
    print("2. Test user signup functionality")
    print("3. If issues persist, check the terminal output for specific errors")

if __name__ == "__main__":
    main()
