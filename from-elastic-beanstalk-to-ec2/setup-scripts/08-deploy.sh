#!/bin/bash
# 08-deploy.sh - Deploy application from S3 artifact
# Downloads the deployment bundle from S3, extracts to a timestamped
# release directory, runs setup tasks, and performs an atomic symlink switch.
set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
echo "[$SCRIPT_NAME] Starting application deployment..."

# ---------- Configuration ----------
AWS_REGION="${AWS_REGION:-us-east-1}"
S3_BUCKET="${S3_BUCKET:-my-deployments}"
S3_KEY="${S3_KEY:-deployments/myapp/deploy.tar.gz}"
APP_ROOT="/var/app"
RELEASES_DIR="${APP_ROOT}/releases"
SHARED_DIR="${APP_ROOT}/shared"
CURRENT_LINK="${APP_ROOT}/current"
APP_USER="webapp"
APP_GROUP="webapp"
ENV_FILE="/etc/myapp/env"

RBENV_ROOT="/usr/local/rbenv"
RUBY_VERSION="2.6.6"
RUBY_BIN="${RBENV_ROOT}/versions/${RUBY_VERSION}/bin"
BUNDLE_BIN="${RUBY_BIN}/bundle"

RELEASE_NAME=$(date +%Y%m%d%H%M%S)
RELEASE_DIR="${RELEASES_DIR}/${RELEASE_NAME}"

# ---------- Download Artifact from S3 ----------
echo "[$SCRIPT_NAME] Downloading deployment artifact from s3://${S3_BUCKET}/${S3_KEY}..."
TEMP_DIR=$(mktemp -d)
aws s3 cp "s3://${S3_BUCKET}/${S3_KEY}" "${TEMP_DIR}/deploy.tar.gz" \
  --region "$AWS_REGION"

# ---------- Extract to Release Directory ----------
echo "[$SCRIPT_NAME] Extracting to ${RELEASE_DIR}..."
sudo mkdir -p "$RELEASE_DIR"
sudo tar xzf "${TEMP_DIR}/deploy.tar.gz" -C "$RELEASE_DIR"
rm -rf "$TEMP_DIR"

# ---------- Link Shared Directories ----------
echo "[$SCRIPT_NAME] Linking shared directories..."
SHARED_LINKS=(
  "log"
  "tmp"
  "public/assets"
  "public/packs"
  "storage"
  "vendor/bundle"
)

for link in "${SHARED_LINKS[@]}"; do
  target="${SHARED_DIR}/${link}"
  destination="${RELEASE_DIR}/${link}"

  # Remove existing directory/file in release
  sudo rm -rf "$destination"

  # Ensure parent directory exists
  sudo mkdir -p "$(dirname "$destination")"

  # Create symlink to shared
  sudo ln -sfn "$target" "$destination"
done

# ---------- Bundle Link ----------
sudo ln -sfn "${SHARED_DIR}/bundle" "${RELEASE_DIR}/vendor/bundle"

# ---------- Load Environment ----------
if [[ -f "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE"
  set +a
fi

# ---------- Install Dependencies ----------
echo "[$SCRIPT_NAME] Running bundle install..."
cd "$RELEASE_DIR"
sudo -u "$APP_USER" \
  PATH="${RUBY_BIN}:$PATH" \
  BUNDLE_PATH="${SHARED_DIR}/bundle" \
  BUNDLE_DEPLOYMENT=1 \
  BUNDLE_WITHOUT="development:test" \
  "$BUNDLE_BIN" install --quiet

# ---------- Database Migration ----------
echo "[$SCRIPT_NAME] Running database migrations..."
sudo -u "$APP_USER" \
  PATH="${RUBY_BIN}:$PATH" \
  RAILS_ENV=production \
  "$BUNDLE_BIN" exec rake db:migrate

# ---------- Asset Precompilation ----------
echo "[$SCRIPT_NAME] Precompiling assets..."
sudo -u "$APP_USER" \
  PATH="${RUBY_BIN}:$PATH" \
  RAILS_ENV=production \
  "$BUNDLE_BIN" exec rake assets:precompile

# ---------- Set Permissions ----------
sudo chown -R "${APP_USER}:${APP_GROUP}" "$RELEASE_DIR"

# ---------- Atomic Symlink Switch ----------
echo "[$SCRIPT_NAME] Switching current release..."
TEMP_LINK="${CURRENT_LINK}.new"
sudo ln -sfn "$RELEASE_DIR" "$TEMP_LINK"
sudo mv -Tf "$TEMP_LINK" "$CURRENT_LINK"

echo "[$SCRIPT_NAME] Current release: $(readlink "$CURRENT_LINK")"

# ---------- Restart Services ----------
echo "[$SCRIPT_NAME] Restarting application services..."
sudo systemctl restart puma
sudo systemctl restart sidekiq

# ---------- Cleanup Old Releases ----------
echo "[$SCRIPT_NAME] Cleaning up old releases (keeping last 5)..."
cd "$RELEASES_DIR"
ls -1dt */ | tail -n +6 | xargs -r sudo rm -rf

# ---------- Verify ----------
echo "[$SCRIPT_NAME] Deployment complete: release ${RELEASE_NAME}"
echo "[$SCRIPT_NAME] Done."
