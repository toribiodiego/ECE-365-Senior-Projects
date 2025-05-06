#!/bin/bash
set -e

# Directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default application to run
APP_TO_RUN="app.py"
APP_ARGS=""

# Process command line args
while getopts "eahsp:" opt; do
  case $opt in
    e) APP_TO_RUN="main.py" ;;      # Run evaluation mode
    a) APP_TO_RUN="app.py" ;;       # Run app (default)
    s) APP_ARGS="$APP_ARGS --share" ;; # Enable sharing
    p) APP_ARGS="$APP_ARGS --port $OPTARG" ;; # Custom port
    h) 
      echo "Usage: $0 [-e|-a|-s|-p PORT|-h]"
      echo "  -e  Run evaluation script (main.py)"
      echo "  -a  Run GUI application (app.py) - default"
      echo "  -s  Enable sharing (creates public URL)"
      echo "  -p  Specify port number (default: 7860)"
      echo "  -h  Show this help"
      exit 0
      ;;
    \?) 
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

# Check if virtual environment exists
if [ ! -d "venv" ]; then
  echo "Virtual environment not found. Setting up now..."
  bash setup.sh
  if [ $? -ne 0 ]; then
    echo "Failed to set up environment. Please run setup.sh manually."
    exit 1
  fi
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate || source venv/Scripts/activate

# Create necessary directories
mkdir -p embeddings_cache
mkdir -p aligned_faces
mkdir -p models

# Run the selected application
echo "Running $APP_TO_RUN $APP_ARGS..."
python3 "$APP_TO_RUN" $APP_ARGS

# Deactivate virtual environment
deactivate
