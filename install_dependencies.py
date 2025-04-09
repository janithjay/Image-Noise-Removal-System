"""
Script to install all required dependencies for the Image Noise Removal System.
"""

import subprocess
import sys

def install_dependencies():
    """
    Install all required dependencies from requirements.txt.
    """
    print("Installing dependencies from requirements.txt...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("All dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        sys.exit(1)

if __name__ == "__main__":
    install_dependencies() 