#!/bin/bash
# Script to activate the virtual environment and load environment variables

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root directory
cd "$PROJECT_ROOT"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
  echo "Virtual environment not found. Run setup.sh first."
  exit 1
fi

# Create .env file from example if it doesn't exist
ENV_FILE="$SCRIPT_DIR/.env"
ENV_EXAMPLE="$SCRIPT_DIR/env.example"

if [ ! -f "$ENV_FILE" ] && [ -f "$ENV_EXAMPLE" ]; then
  echo "Creating .env file from example..."
  cp "$ENV_EXAMPLE" "$ENV_FILE"
  echo "Please edit $ENV_FILE to set your configuration values."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate || source venv/Scripts/activate

# Load environment variables if .env exists
if [ -f "$ENV_FILE" ]; then
  echo "Loading environment variables from $ENV_FILE..."
  source "$ENV_FILE"
else
  echo "Warning: No .env file found. Some features may not work correctly."
fi

# Create necessary directories
mkdir -p embeddings_cache
mkdir -p aligned_faces
mkdir -p models

echo "Environment activated. Run 'deactivate' to exit the virtual environment."
echo "Run './run.sh' to start the application." 