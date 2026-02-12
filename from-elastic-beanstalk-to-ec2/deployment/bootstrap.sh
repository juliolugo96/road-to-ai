#!/bin/bash
# bootstrap.sh - EC2 User Data bootstrap script
# This is the entry point for new EC2 instances launched by the ASG.
# It downloads setup scripts from S3 and executes them in order.
# Keep this script minimal — real logic lives in the setup scripts.
set -euo pipefail

exec > >(tee /var/log/bootstrap.log) 2>&1

SCRIPT_NAME="bootstrap"
echo "[$SCRIPT_NAME] Instance bootstrap started at $(date -u)"
echo "[$SCRIPT_NAME] Instance ID: $(curl -sf http://169.254.169.254/latest/meta-data/instance-id)"

# ---------- Configuration ----------
AWS_REGION="${AWS_REGION:-us-east-1}"
S3_BUCKET="${S3_BUCKET:-my-deployments}"
SETUP_SCRIPTS_KEY="${SETUP_SCRIPTS_KEY:-deployments/myapp/setup-scripts.tar.gz}"
SETUP_DIR="/opt/setup-scripts"

# ---------- Download Setup Scripts ----------
echo "[$SCRIPT_NAME] Downloading setup scripts from S3..."
mkdir -p "$SETUP_DIR"

aws s3 cp "s3://${S3_BUCKET}/${SETUP_SCRIPTS_KEY}" /tmp/setup-scripts.tar.gz \
  --region "$AWS_REGION"

tar xzf /tmp/setup-scripts.tar.gz -C "$SETUP_DIR"
rm -f /tmp/setup-scripts.tar.gz

# ---------- Execute Setup Scripts in Order ----------
echo "[$SCRIPT_NAME] Running setup scripts..."

SCRIPTS=(
  "01-env.sh"
  "02-packages.sh"
  "03-redis.sh"
  "04-nodejs.sh"
  "05-ruby.sh"
  "06-pdf-tools.sh"
  "07-directories.sh"
  "08-deploy.sh"
  "09-monitoring.sh"
)

for script in "${SCRIPTS[@]}"; do
  SCRIPT_PATH="${SETUP_DIR}/${script}"

  if [[ ! -f "$SCRIPT_PATH" ]]; then
    echo "[$SCRIPT_NAME] ERROR: Script not found: ${SCRIPT_PATH}"
    exit 1
  fi

  chmod +x "$SCRIPT_PATH"
  echo "[$SCRIPT_NAME] ========================================="
  echo "[$SCRIPT_NAME] Running: ${script}"
  echo "[$SCRIPT_NAME] ========================================="

  if ! bash "$SCRIPT_PATH"; then
    echo "[$SCRIPT_NAME] ERROR: ${script} failed with exit code $?"
    exit 1
  fi

  echo "[$SCRIPT_NAME] Completed: ${script}"
done

# ---------- Install systemd Services ----------
echo "[$SCRIPT_NAME] Installing systemd service files..."
cp "${SETUP_DIR}/../systemd/"*.service /etc/systemd/system/ 2>/dev/null || true
systemctl daemon-reload
systemctl enable puma sidekiq

# ---------- Start Services ----------
echo "[$SCRIPT_NAME] Starting application services..."
systemctl start puma
systemctl start sidekiq

# ---------- Health Check ----------
echo "[$SCRIPT_NAME] Running health check..."
for i in $(seq 1 60); do
  if curl -sf http://localhost:3000/health &>/dev/null; then
    echo "[$SCRIPT_NAME] Application is healthy."
    break
  fi
  if [[ $i -eq 60 ]]; then
    echo "[$SCRIPT_NAME] ERROR: Application did not become healthy within 60 seconds."
    exit 1
  fi
  sleep 2
done

echo "[$SCRIPT_NAME] Bootstrap completed at $(date -u)"
echo "[$SCRIPT_NAME] Done."
