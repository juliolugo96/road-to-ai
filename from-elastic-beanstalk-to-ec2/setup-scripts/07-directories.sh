#!/bin/bash
# 07-directories.sh - Create application directory layout
# Sets up the directory structure for atomic deployments using
# timestamped release directories and a symlinked current release.
set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
echo "[$SCRIPT_NAME] Starting directory layout setup..."

# ---------- Configuration ----------
APP_USER="webapp"
APP_GROUP="webapp"
APP_ROOT="/var/app"
RELEASES_DIR="${APP_ROOT}/releases"
SHARED_DIR="${APP_ROOT}/shared"
CURRENT_LINK="${APP_ROOT}/current"

SHARED_SUBDIRS=(
  "log"
  "tmp/pids"
  "tmp/cache"
  "tmp/sockets"
  "public/assets"
  "public/packs"
  "storage"
  "bundle"
)

# ---------- Create Application User ----------
if ! id "$APP_USER" &>/dev/null; then
  echo "[$SCRIPT_NAME] Creating application user: $APP_USER"
  sudo useradd --system --shell /sbin/nologin --home-dir "$APP_ROOT" "$APP_USER"
fi

# ---------- Create Directory Structure ----------
echo "[$SCRIPT_NAME] Creating directory structure..."
sudo mkdir -p "$RELEASES_DIR"

for subdir in "${SHARED_SUBDIRS[@]}"; do
  sudo mkdir -p "${SHARED_DIR}/${subdir}"
done

# ---------- Set Permissions ----------
sudo chown -R "${APP_USER}:${APP_GROUP}" "$APP_ROOT"
sudo chmod 755 "$APP_ROOT"

# ---------- Idempotency Check for Symlink ----------
if [[ -L "$CURRENT_LINK" ]]; then
  echo "[$SCRIPT_NAME] Current release symlink already exists: $(readlink "$CURRENT_LINK")"
else
  echo "[$SCRIPT_NAME] No current release symlink yet (will be created on first deploy)."
fi

# ---------- Verify ----------
echo "[$SCRIPT_NAME] Directory layout:"
echo "  ${APP_ROOT}/"
echo "  ├── releases/        (timestamped release directories)"
echo "  ├── shared/           (persistent files across releases)"
for subdir in "${SHARED_SUBDIRS[@]}"; do
  echo "  │   ├── ${subdir}/"
done
echo "  └── current -> ...   (symlink to active release)"
echo "[$SCRIPT_NAME] Done."
