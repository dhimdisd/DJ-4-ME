#!/bin/bash
set -euo pipefail

# Move top-level JSON files from ../downloads into ../downloads/metadata
mv ../downloads/*.json ../downloads/metadata/ 2>/dev/null || true

# Move JSON files from ../downloads/audio (including subdirectories)
find ../downloads/audio -type f -name "*.json" -exec mv -t ../downloads/metadata {} +

echo "All JSON files moved to ../downloads/metadata/"