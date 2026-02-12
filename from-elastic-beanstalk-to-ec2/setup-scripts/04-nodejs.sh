#!/bin/bash
# 04-nodejs.sh - Install Node.js and Yarn for asset compilation
# Required for the Rails asset pipeline (Webpacker/Sprockets).
set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
echo "[$SCRIPT_NAME] Starting Node.js setup..."

# ---------- Configuration ----------
NODE_VERSION="18"
YARN_VERSION="1.22.22"

# ---------- Idempotency Check ----------
if command -v node &>/dev/null; then
  INSTALLED_MAJOR=$(node -v | sed 's/v//' | cut -d. -f1)
  if [[ "$INSTALLED_MAJOR" == "$NODE_VERSION" ]]; then
    echo "[$SCRIPT_NAME] Node.js v${NODE_VERSION}.x is already installed. Skipping."
    exit 0
  fi
fi

# ---------- Install Node.js ----------
echo "[$SCRIPT_NAME] Installing Node.js v${NODE_VERSION}.x..."

# Use NodeSource repository
curl -fsSL "https://rpm.nodesource.com/setup_${NODE_VERSION}.x" | sudo bash -
sudo dnf install -y nodejs

# ---------- Install Yarn ----------
if ! command -v yarn &>/dev/null; then
  echo "[$SCRIPT_NAME] Installing Yarn v${YARN_VERSION}..."
  sudo npm install -g "yarn@${YARN_VERSION}"
fi

# ---------- Verify ----------
echo "[$SCRIPT_NAME] Node.js version: $(node -v)"
echo "[$SCRIPT_NAME] npm version: $(npm -v)"
echo "[$SCRIPT_NAME] Yarn version: $(yarn -v)"
echo "[$SCRIPT_NAME] Done."
