#!/bin/bash
set -e

# ============================================================
# WebArena-Lite-v2 Environment Setup Script
# Run from: ~/GUI-Libra/evaluation/WebArenaLiteV2
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ------------------------------------------------------------
# 1. Python environment
# ------------------------------------------------------------
echo "==> Setting up Python 3.12 virtual environment..."
uv venv -p 3.12
source .venv/bin/activate

echo "==> Installing Python dependencies..."
uv pip install -r requirements.txt

echo "==> Installing Playwright Chromium..."
uv run playwright install chromium

# ------------------------------------------------------------
# 2. Docker image archives
# ------------------------------------------------------------
echo "==> Creating launcher/images directory..."
mkdir -p ./launcher/images
cd ./launcher/images

echo "==> Downloading Shopping image..."
# wget -c http://metis.lti.cs.cmu.edu/webarena-images/shopping_final_0712.tar

echo "==> Downloading Shopping Admin image..."
# wget -c http://metis.lti.cs.cmu.edu/webarena-images/shopping_admin_final_0719.tar

echo "==> Downloading Reddit image..."
# wget -c http://metis.lti.cs.cmu.edu/webarena-images/postmill-populated-exposed-withimg.tar

echo "==> Downloading GitLab image..."
# wget -c http://metis.lti.cs.cmu.edu/webarena-images/gitlab-populated-final-port8023.tar

cd ..   # back to launcher/

# ------------------------------------------------------------
# 3. OpenStreetMap archives (downloaded into launcher/)
# ------------------------------------------------------------
echo "==> Downloading OpenStreetMap archives..."
# wget -c https://zenodo.org/records/12636845/files/openstreetmap-website-db.tar.gz
# wget -c https://zenodo.org/records/12636845/files/openstreetmap-website-web.tar.gz
# wget -c https://zenodo.org/records/12636845/files/openstreetmap-website.tar.gz

echo "==> Extracting openstreetmap-website.tar.gz..."
tar -xzf ./openstreetmap-website.tar.gz

cd "$SCRIPT_DIR"

# ------------------------------------------------------------
# 4. Patch launcher/00_vars.sh with ARCHIVES_LOCATION
# ------------------------------------------------------------
IMAGES_ABS_PATH="$SCRIPT_DIR/launcher/images"
VARS_FILE="$SCRIPT_DIR/launcher/00_vars.sh"

echo "==> Setting ARCHIVES_LOCATION in launcher/00_vars.sh..."
sed -i "s|export ARCHIVES_LOCATION=.*|export ARCHIVES_LOCATION=\"$IMAGES_ABS_PATH\"|" "$VARS_FILE"

echo ""
echo "==> Setup complete."
echo "    Review/edit launcher/00_vars.sh to confirm hostname and ports, then run:"
echo "      python launcher/start.py"
