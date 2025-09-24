#!/usr/bin/env python3
"""
Startup script for AI-Powered Nutrition System Backend
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

def setup_environment():
    """Setup the backend environment"""
    print("🔧 Setting up AI-Powered Nutrition System Backend...")
    
    # Change to backend directory
    backend_dir = Path(__file__).parent / "backend"
    os.chdir(backend_dir)
    
    # Check if virtual environment exists
    venv_path = Path("venv")
    if not venv_path.exists():
        print("📦 Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
    
    # Determine activation script based on OS
    if os.name == 'nt':  # Windows
        activate_script = venv_path / "Scripts" / "activate.bat"
        pip_path = venv_path / "Scripts" / "pip.exe"
        python_path = venv_path / "Scripts" / "python.exe"
    else:  # Unix/Linux/MacOS
        activate_script = venv_path / "bin" / "activate"
        pip_path = venv_path / "bin" / "pip"
        python_path = venv_path / "bin" / "python"
    
    # Install requirements
    requirements_file = Path("../requirements.txt")
    if requirements_file.exists():
        print("📚 Installing Python dependencies...")
        subprocess.run([str(pip_path), "install", "-r", str(requirements_file)], check=True)
    else:
        print("⚠️  Requirements file not found. Installing basic dependencies...")
        basic_deps = [
            "fastapi==0.103.0",
            "uvicorn==0.23.2",
            "tensorflow==2.13.0",
            "pandas==2.0.3",
            "numpy==1.24.3",
            "scikit-learn==1.3.0",
            "pydantic==2.3.0"
        ]
        for dep in basic_deps:
            subprocess.run([str(pip_path), "install", dep], check=True)
    
    return python_path

def start_server(python_path):
    """Start the FastAPI server"""
    print("🚀 Starting AI-Powered Nutrition System Backend...")
    print("📊 Loading AI models (LSTM + Transformer + Fusion)...")
    print("🌐 Server will be available at: http://localhost:8000")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔬 Interactive API: http://localhost:8000/redoc")
    print("\n" + "="*60)
    print("🧠 AI MODELS OVERVIEW:")
    print("  • LSTM Model: Time-series health data analysis")
    print("  • Transformer Model: Genomic SNP analysis") 
    print("  • Fusion Model: Multi-modal recommendation engine")
    print("="*60 + "\n")
    
    try:
        # Start the FastAPI server
        subprocess.run([
            str(python_path), "-m", "uvicorn", 
            "main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return False
    
    return True

def main():
    """Main startup function"""
    try:
        python_path = setup_environment()
        start_server(python_path)
    except subprocess.CalledProcessError as e:
        print(f"❌ Setup failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Setup interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
