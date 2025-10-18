#!/bin/bash

# Script to compare baseline vs optimized endpoints

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

IMAGE_PATH="$PROJECT_DIR/assets/example_image/T.png"
OUTPUT_BASELINE="$PROJECT_DIR/outputs/T_baseline.glb"
OUTPUT_OPTIMIZED="$PROJECT_DIR/outputs/T_optimized.glb"

echo "=========================================="
echo "Testing BASELINE endpoint"
echo "=========================================="
echo "Image: $IMAGE_PATH"
echo "Output: $OUTPUT_BASELINE"
echo ""

curl -v -X POST "http://localhost:8000/convert/baseline" \
  -F "image=@$IMAGE_PATH" \
  -F "seed=1" \
  -F "simplify=0.95" \
  -F "texture_size=1024" \
  -o "$OUTPUT_BASELINE" 2>&1 | grep -E "(X-Total-Time|X-GPU-Time|X-PostProc-Time|X-Method)"

echo ""
echo "Waiting 3 seconds before next test..."
sleep 15

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Baseline completed!"
else
    echo ""
    echo "✗ Baseline failed"
fi



echo ""
echo "=========================================="
echo "Testing OPTIMIZED endpoint"
echo "=========================================="
echo "Image: $IMAGE_PATH"
echo "Output: $OUTPUT_OPTIMIZED"
echo ""

curl -v -X POST "http://localhost:8000/convert/optimized" \
  -F "image=@$IMAGE_PATH" \
  -F "seed=1" \
  -F "simplify=0.95" \
  -F "texture_size=1024" \
  -o "$OUTPUT_OPTIMIZED" 2>&1 | grep -E "(X-Total-Time|X-GPU-Time|X-PostProc-Time|X-Method)"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Optimized completed!"
else
    echo ""
    echo "✗ Optimized failed"
fi

echo ""
echo "=========================================="
echo "Comparison complete!"
echo "=========================================="

