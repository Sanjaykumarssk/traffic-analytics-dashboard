"""
One-Click Server Startup Script
Automatically configures and starts the Traffic Dashboard server.
"""

import os
import sys
import subprocess
import time
import socket
from pathlib import Path

def check_port(port):
    """Check if a port is available."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', port))
    sock.close()
    return result != 0

def find_free_port(start_port=8000, max_attempts=10):
    """Find a free port starting from start_port."""
    for i in range(max_attempts):
        port = start_port + i
        if check_port(port):
            return port
    return None

def check_dependencies():
    """Check if required packages are installed and install automatically."""
    required = ['fastapi', 'uvicorn', 'sqlalchemy', 'requests', 'pandas', 'python-dotenv']
    missing = []
    
    for package in required:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print("📦 Installing missing dependencies automatically...")
        try:
            # Upgrade pip first
            subprocess.check_call(
                [sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            # Install all requirements
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
            print("✅ Dependencies installed successfully!")
        except subprocess.CalledProcessError:
            print("⚠️ Some dependencies may have failed. Continuing anyway...")
            # Continue anyway - dependencies might already be installed
    else:
        print("✅ All dependencies found")
    return True

def setup_env_file():
    """Ensure .env file exists with API key."""
    env_path = Path('.env')
    api_key = 'lDfiIMS9bisNZ02h56YGCXvHA5qkGRVP'
    
    if env_path.exists():
        # Check if API key is already in .env
        with open(env_path, 'r') as f:
            content = f.read()
            if 'TOMTOM_KEY' in content:
                print("✅ .env file found with API key")
                return True
        
        # Add API key to existing .env
        with open(env_path, 'a') as f:
            f.write(f'\nTOMTOM_KEY={api_key}\n')
        print("✅ API key added to .env file")
    else:
        # Create new .env file
        env_content = f"""# TomTom API Configuration
TOMTOM_KEY={api_key}

# Database Configuration
DB_PATH=traffic_analysis.db

# ML Model Configuration
MODEL_PATH=rf_model.pkl

# Output Configuration
OUTPUT_CSV=traffic_results.csv
ROUTES_CSV=routes.csv

# Power BI Configuration (Optional)
POWERBI_PUSH_ENABLED=false
POWERBI_PUSH_URL=
"""
        with open(env_path, 'w') as f:
            f.write(env_content)
        print("✅ .env file created with API key")
    
    return True

def initialize_database():
    """Initialize the database."""
    try:
        from db import init_db
        init_db()
        print("✅ Database initialized")
        return True
    except Exception as e:
        print(f"⚠️ Database initialization warning: {e}")
        return True  # Continue anyway

def check_ml_model():
    """Check if ML model exists, optionally train it."""
    model_path = Path('rf_model.pkl')
    csv_path = Path('traffic_results.csv')
    
    if model_path.exists():
        file_size = model_path.stat().st_size / (1024 * 1024)  # Size in MB
        print(f"✅ ML model found (rf_model.pkl - {file_size:.2f} MB)")
        print("   The model will be loaded when the server starts.")
        return True
    
    print("ℹ️ ML model file (rf_model.pkl) not found")
    
    if csv_path.exists():
        print("📊 Training data found (traffic_results.csv)")
        print("   Attempting to train ML model...")
        # Check if model_train.py exists
        if Path('model_train.py').exists():
            try:
                subprocess.check_call([sys.executable, 'model_train.py'], 
                                    stdout=subprocess.DEVNULL, 
                                    stderr=subprocess.PIPE)
                if model_path.exists():
                    print("✅ ML model trained successfully (rf_model.pkl created)")
                else:
                    print("⚠️ ML model training completed but file not created")
            except subprocess.CalledProcessError as e:
                print("⚠️ ML model training skipped (insufficient data or error)")
                print(f"   Note: The application will work without the ML model")
        else:
            print("⚠️ model_train.py not found - cannot train model")
            print("   Note: The application will work without the ML model")
    else:
        print("ℹ️ No training data (traffic_results.csv) found")
        print("   Note: The application will work without the ML model")
        print("   ML predictions will be disabled until a model is available.")
    
    return True

def start_server(port=8000):
    """Start the FastAPI server."""
    # Check if port is available
    if not check_port(port):
        print(f"⚠️ Port {port} is already in use. Finding alternative port...")
        port = find_free_port(port)
        if port is None:
            print("❌ Could not find an available port. Please free up port 8000 or close other applications.")
            return False
        print(f"✅ Using port {port} instead")
    
    print("\n" + "="*60)
    print("🚀 Starting Traffic Dashboard Server")
    print("="*60)
    print(f"\n📍 Server URL: http://localhost:{port}")
    print(f"📚 API Docs: http://localhost:{port}/docs")
    print(f"❤️ Health Check: http://localhost:{port}/health")
    print("\n💡 Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    try:
        # Start uvicorn server
        cmd = [
            sys.executable, '-m', 'uvicorn',
            'app:app',
            '--reload',
            '--host', '0.0.0.0',
            '--port', str(port)
        ]
        
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
        return True
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        return False

def main():
    """Main startup function."""
    print("="*60)
    print("🚦 Traffic Dashboard - Automatic Server Setup")
    print("="*60)
    print()
    print("📋 Startup Steps:")
    print("   [1/6] Check dependencies")
    print("   [2/6] Create directories")
    print("   [3/6] Setup environment (.env file)")
    print("   [4/6] Initialize database")
    print("   [5/6] Check ML model (rf_model.pkl)")
    print("   [6/6] Start server")
    print()
    print("="*60)
    print()
    
    # Create logs directory
    Path('logs').mkdir(exist_ok=True)
    
    # Step 1: Check dependencies
    print("[1/6] Checking dependencies...")
    check_dependencies()
    print()
    
    # Step 2: Create directories
    print("[2/6] Creating directories...")
    for directory in ['logs', 'static', 'templates', 'visuals']:
        Path(directory).mkdir(exist_ok=True)
    print("✅ Directories ready")
    print()
    
    # Step 3: Setup .env file
    print("[3/6] Setting up environment...")
    setup_env_file()
    print()
    
    # Step 4: Initialize database
    print("[4/6] Initializing database...")
    initialize_database()
    print()
    
    # Step 5: Check ML model
    print("[5/6] Checking ML model (rf_model.pkl)...")
    check_ml_model()
    print()
    
    # Step 6: Start server
    print("[6/6] Starting server...")
    start_server()

if __name__ == '__main__':
    main()

