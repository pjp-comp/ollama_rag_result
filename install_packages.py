import subprocess
import sys

# List of required packages
required_packages = [
    "gradio",
    "langchain_community",
    "ollama",
    "chromadb",
    "pdfplumber --no-deps"
]

# Function to install missing packages
def install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")

# Check and install missing packages
for package in required_packages:
    try:
        __import__(package)
        print(f"✅ {package} is already installed")
    except ImportError:
        print(f"🔍 Installing {package}...")
        install_package(package)
