#!/bin/bash
# 06-pdf-tools.sh - Install PDF generation tools (wkhtmltopdf)
# Required for server-side PDF rendering of reports and exports.
set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
echo "[$SCRIPT_NAME] Starting PDF tools setup..."

# ---------- Configuration ----------
WKHTMLTOPDF_VERSION="0.12.6.1-3"
WKHTMLTOPDF_RPM="wkhtmltox-${WKHTMLTOPDF_VERSION}.almalinux9.x86_64.rpm"
WKHTMLTOPDF_URL="https://github.com/wkhtmltopdf/packaging/releases/download/${WKHTMLTOPDF_VERSION}/${WKHTMLTOPDF_RPM}"

# ---------- Idempotency Check ----------
if command -v wkhtmltopdf &>/dev/null; then
  echo "[$SCRIPT_NAME] wkhtmltopdf is already installed. Skipping."
  wkhtmltopdf --version
  exit 0
fi

# ---------- Install Dependencies ----------
echo "[$SCRIPT_NAME] Installing PDF rendering dependencies..."
sudo dnf install -y \
  fontconfig \
  libX11 \
  libXext \
  libXrender \
  libjpeg-turbo \
  xorg-x11-fonts-Type1 \
  xorg-x11-fonts-75dpi

# ---------- Install wkhtmltopdf ----------
echo "[$SCRIPT_NAME] Downloading wkhtmltopdf ${WKHTMLTOPDF_VERSION}..."
TEMP_DIR=$(mktemp -d)
curl -fsSL "$WKHTMLTOPDF_URL" -o "${TEMP_DIR}/${WKHTMLTOPDF_RPM}"
sudo rpm -ivh "${TEMP_DIR}/${WKHTMLTOPDF_RPM}"
rm -rf "$TEMP_DIR"

# ---------- Verify ----------
if ! command -v wkhtmltopdf &>/dev/null; then
  echo "[$SCRIPT_NAME] ERROR: wkhtmltopdf installation failed."
  exit 1
fi

echo "[$SCRIPT_NAME] wkhtmltopdf version: $(wkhtmltopdf --version)"
echo "[$SCRIPT_NAME] Done."
