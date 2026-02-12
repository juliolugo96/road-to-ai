#!/bin/bash
# rollback.sh - Rollback to the previous release
# Switches the current symlink to the previous release directory
# and restarts services. No new instance refresh needed.
set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
echo "[$SCRIPT_NAME] Starting rollback..."

# ---------- Configuration ----------
APP_ROOT="/var/app"
RELEASES_DIR="${APP_ROOT}/releases"
CURRENT_LINK="${APP_ROOT}/current"

# ---------- Find Previous Release ----------
CURRENT_RELEASE=$(basename "$(readlink "$CURRENT_LINK")")
echo "[$SCRIPT_NAME] Current release: ${CURRENT_RELEASE}"

PREVIOUS_RELEASE=$(ls -1t "$RELEASES_DIR" | grep -v "^${CURRENT_RELEASE}$" | head -1)

if [[ -z "$PREVIOUS_RELEASE" ]]; then
  echo "[$SCRIPT_NAME] ERROR: No previous release found to rollback to."
  exit 1
fi

echo "[$SCRIPT_NAME] Rolling back to: ${PREVIOUS_RELEASE}"

# ---------- Atomic Symlink Switch ----------
TEMP_LINK="${CURRENT_LINK}.rollback"
sudo ln -sfn "${RELEASES_DIR}/${PREVIOUS_RELEASE}" "$TEMP_LINK"
sudo mv -Tf "$TEMP_LINK" "$CURRENT_LINK"

echo "[$SCRIPT_NAME] Current release is now: $(readlink "$CURRENT_LINK")"

# ---------- Restart Services ----------
echo "[$SCRIPT_NAME] Restarting application services..."
sudo systemctl restart puma
sudo systemctl restart sidekiq

# ---------- Wait for Health ----------
echo "[$SCRIPT_NAME] Waiting for Puma to be ready..."
for i in $(seq 1 30); do
  if curl -sf http://localhost:3000/health &>/dev/null; then
    echo "[$SCRIPT_NAME] Application is healthy after rollback."
    break
  fi
  if [[ $i -eq 30 ]]; then
    echo "[$SCRIPT_NAME] WARNING: Application did not become healthy within 30 seconds."
  fi
  sleep 1
done

echo "[$SCRIPT_NAME] Rollback complete."
echo "[$SCRIPT_NAME] Done."
