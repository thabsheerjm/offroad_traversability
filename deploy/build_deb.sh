#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEB_DIR="$PROJECT_ROOT/deploy/deb"

echo "Building offroad-traversability v2.0..."

# 1. Copy C++ source
echo "Copying C++ source..."
rm -rf "$DEB_DIR/usr/local/lib/offroad/src"
mkdir -p "$DEB_DIR/usr/local/lib/offroad/src"
cp -r "$PROJECT_ROOT/backend" \
      "$DEB_DIR/usr/local/lib/offroad/src/"
cp "$PROJECT_ROOT/CMakeLists.txt" \
   "$DEB_DIR/usr/local/lib/offroad/src/"

# 2. Bundle ONNX Runtime
echo "Bundling ONNX Runtime..."
rm -rf "$DEB_DIR/usr/local/lib/offroad/onnxruntime"
cp -r "$PROJECT_ROOT/onnxruntime" \
      "$DEB_DIR/usr/local/lib/offroad/"

# 3. Copy Python bridge
echo "Copying bridge..."
cp "$PROJECT_ROOT/bridge/bridge.py" \
   "$DEB_DIR/usr/local/lib/offroad/bridge/bridge.py"
cp "$PROJECT_ROOT/bridge/requirements.txt" \
   "$DEB_DIR/usr/local/lib/offroad/bridge/requirements.txt"

# 4. Build React dashboard
echo "Building dashboard..."
cd "$PROJECT_ROOT/dashboard"
npm run build
rm -rf "$DEB_DIR/usr/share/offroad/dashboard"
mkdir -p "$DEB_DIR/usr/share/offroad/dashboard"
cp -r "$PROJECT_ROOT/dashboard/dist/." \
      "$DEB_DIR/usr/share/offroad/dashboard/"

# 5. Build .deb
echo "Building .deb package..."
cd "$PROJECT_ROOT"
dpkg-deb --build deploy/deb \
    "deploy/offroad-traversability_2.0_amd64.deb"

echo "Done: deploy/offroad-traversability_2.0_amd64.deb"
echo "Package size: $(du -sh deploy/offroad-traversability_2.0_amd64.deb | cut -f1)"
