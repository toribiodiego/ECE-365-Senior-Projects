#!/bin/bash
set -e

echo "Downloading aligned dataset..."
gdown 1NzL8oKWeO6qFCpP-CWZOwRJtBabrVGul -O align.tar.gz
mkdir -p align
tar -xzf align.tar.gz -C align
rm align.tar.gz

echo "Downloading LFW pairs..."
gdown 1o1GhF-0H-qQIzD7uIfCqjhEMRNfCKAB- -O lfw_test_pair.txt

# If using ResNetFace (not needed for EdgeFace), download the model:
# echo "Downloading model weights..."
# gdown 1vXV3NT35loFR984zwahi7LSlpDM3-U00 -O resnet18_110.pth

echo "Setup complete."
