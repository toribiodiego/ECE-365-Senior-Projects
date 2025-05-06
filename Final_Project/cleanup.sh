#!/bin/bash
set -e

echo "Cleaning up project directory for Git..."

# Remove Python bytecode files
echo "Removing Python bytecode..."
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name "*.pyc" -delete
find . -name "*.pyo" -delete
find . -name "*.pyd" -delete

# Clean temporary files in aligned_faces but keep the directory
echo "Cleaning aligned_faces directory..."
rm -f aligned_faces/*
touch aligned_faces/.gitkeep

# Clean the embeddings cache but keep the directory
echo "Cleaning embeddings_cache directory..."
rm -f embeddings_cache/*
touch embeddings_cache/.gitkeep

# Remove any log files
echo "Removing log files..."
find . -name "*.log" -delete

# Remove any temporary files
echo "Removing temporary files..."
find . -name "*~" -delete
find . -name ".DS_Store" -delete
find . -name "Thumbs.db" -delete

echo "Cleanup complete. Ready for git commit!" 