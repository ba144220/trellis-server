#!/bin/bash

# Script to send a single image to the TRELLIS API endpoint

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

IMAGE_PATH="$PROJECT_DIR/assets/example_image/T.png"
OUTPUT_PATH="$PROJECT_DIR/outputs/T_api.glb"
API_URL="http://localhost:8000/convert"

echo "Sending image to API..."
echo "Image: $IMAGE_PATH"
echo "Output: $OUTPUT_PATH"

curl -X POST "$API_URL" \
  -F "image=@$IMAGE_PATH" \
  -F "seed=1" \
  -F "simplify=0.95" \
  -F "texture_size=1024" \
  -o "$OUTPUT_PATH"

if [ $? -eq 0 ]; then
    echo ""
    echo "Success! GLB file saved to: $OUTPUT_PATH"
else
    echo ""
    echo "Error: Request failed"
    exit 1
fi

