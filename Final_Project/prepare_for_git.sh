#!/bin/bash
set -e

echo "===== Preparing project for Git repository ====="

# Step 1: Run the cleanup script
echo -e "\n----- Cleaning up project files -----"
./cleanup.sh

# Step 2: Run the test script to ensure everything is working
echo -e "\n----- Running application tests -----"
python3 test_app.py

# Verify if the tests were successful
if [ $? -ne 0 ]; then
  echo -e "\n❌ Tests failed. Please fix the issues before continuing."
  exit 1
fi

# Step 3: Initialize git repository if it doesn't exist
if [ ! -d ".git" ]; then
  echo -e "\n----- Initializing Git repository -----"
  git init
  echo "Git repository initialized."
else
  echo -e "\n----- Git repository already exists -----"
fi

# Step 4: Show status of project files
echo -e "\n----- Current Git status -----"
cd ..  # Move up to the parent directory if needed
git status

echo -e "\n===== Project is ready for Git ====="
echo "You can now use the following commands to commit and push:"
echo "git add ."
echo "git commit -m \"Initial commit of face recognition project\""
echo "git remote add origin <your-repository-url>"
echo "git push -u origin main" 