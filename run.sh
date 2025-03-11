#!/bin/bash
set -e

echo "Running setup..."
./setup.sh

echo "Installing requirements..."
pip3 install -r requirements.txt

echo "Running main script..."
python3 main.py
