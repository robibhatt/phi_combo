#!/bin/bash
# Delete all __pycache__ directories in the repository

find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

echo "Cleaned all __pycache__ directories"
