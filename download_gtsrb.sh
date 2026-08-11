#!/bin/bash
# Download GTSRB pickle file from VeriX repository

set -e

GTSRB_DIR="data/GTSRB"
GTSRB_PICKLE="$GTSRB_DIR/gtsrb.pickle"
GTSRB_URL="https://github.com/NeuralNetworkVerification/VeriX/raw/main/gtsrb.pickle"

echo "=========================================="
echo "GTSRB Dataset Download"
echo "=========================================="
echo ""

# Create directory
mkdir -p "$GTSRB_DIR"

# Check if already exists
if [ -f "$GTSRB_PICKLE" ]; then
    echo "✓ GTSRB pickle already exists: $GTSRB_PICKLE"
    echo ""
    echo "File info:"
    ls -lh "$GTSRB_PICKLE"
    echo ""
    echo "To re-download, delete the file first:"
    echo "  rm $GTSRB_PICKLE"
    exit 0
fi

echo "Downloading GTSRB pickle from VeriX repository..."
echo "Source: $GTSRB_URL"
echo "Target: $GTSRB_PICKLE"
echo ""

# Download with curl (fallback to wget if curl not available)
if command -v curl &> /dev/null; then
    curl -L -o "$GTSRB_PICKLE" "$GTSRB_URL"
elif command -v wget &> /dev/null; then
    wget -O "$GTSRB_PICKLE" "$GTSRB_URL"
else
    echo "❌ Error: Neither curl nor wget found."
    echo "Please install curl or wget, or download manually from:"
    echo "  $GTSRB_URL"
    exit 1
fi

# Verify download
if [ -f "$GTSRB_PICKLE" ]; then
    echo ""
    echo "✓ Download complete!"
    echo ""
    echo "File info:"
    ls -lh "$GTSRB_PICKLE"
    echo ""
    echo "GTSRB dataset info:"
    echo "  - 10-class subset of GTSRB (German Traffic Sign Recognition)"
    echo "  - From VeriX project: https://github.com/NeuralNetworkVerification/VeriX"
    echo "  - Test set: ~1000 images per class"
    echo "  - Format: pickle file with x_test, y_test"
    echo ""
    echo "You can now run experiments with GTSRB models:"
    echo "  ./run.sh single fc_10x2_gtsrb.pt"
    echo "  ./run.sh all"
else
    echo ""
    echo "❌ Download failed!"
    echo ""
    echo "Please download manually:"
    echo "  1. Go to: $GTSRB_URL"
    echo "  2. Save as: $GTSRB_PICKLE"
    exit 1
fi

echo "=========================================="
