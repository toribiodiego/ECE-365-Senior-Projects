#!/bin/bash
set -e

# Download aligned dataset
echo "Downloading aligned dataset..."
gdown 1NzL8oKWeO6qFCpP-CWZOwRJtBabrVGul -O align.tar.gz
mkdir -p align
tar -xzf align.tar.gz -C align
rm align.tar.gz

# Download LFW pairs
echo "Downloading LFW pairs..."
gdown 1o1GhF-0H-qQIzD7uIfCqjhEMRNfCKAB- -O lfw_test_pair.txt

# Create a virtual environment (using Python 3.10)
echo "Creating virtual environment..."
python3.10 -m venv venv

# Activate virtual environment and install dependencies
echo "Activating virtual environment and installing dependencies..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete."
