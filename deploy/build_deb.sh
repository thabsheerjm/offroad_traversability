#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEB_DIR="$PROJECT_ROOT/deploy/deb"

echo "Building offroad-traversability v2.0..."

# 1. C++ binary
echo "Copying binary..."
cp "$PROJECT_ROOT/build/backend/offroad_segmentation" \
   "$DEB_DIR/usr/local/bin/offroad_segmentation"
chmod 755 "$DEB_DIR/usr/local/bin/offroad_segmentation"

# 2. Python bridge
echo "Copying bridge..."
cp "$PROJECT_ROOT/bridge/bridge.py" \
   "$DEB_DIR/usr/local/lib/offroad/bridge/bridge.py"
cp "$PROJECT_ROOT/bridge/requirements.txt" \
   "$DEB_DIR/usr/local/lib/offroad/bridge/requirements.txt"

# 3. React dashboard — build first
echo "Building dashboard..."
cd "$PROJECT_ROOT/dashboard"
npm run build
cp -r "$PROJECT_ROOT/dashboard/dist/." \
      "$DEB_DIR/usr/share/offroad/dashboard/"

# 4. Build the .deb
echo "Building .deb package..."
cd "$PROJECT_ROOT"
dpkg-deb --build deploy/deb \
    "deploy/offroad-traversability_2.0_amd64.deb"

echo "Done: deploy/offroad-traversability_2.0_amd64.deb"
