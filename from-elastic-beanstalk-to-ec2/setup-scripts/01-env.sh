#!/bin/bash
# 01-env.sh - Load environment variables from AWS Secrets Manager
# This script runs first to ensure all subsequent scripts have access
# to the application's configuration values.
set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
echo "[$SCRIPT_NAME] Starting environment setup..."

# ---------- Configuration ----------
AWS_REGION="${AWS_REGION:-us-east-1}"
SECRET_NAME="${SECRET_NAME:-production/myapp/env}"
ENV_FILE="/etc/myapp/env"

# ---------- Idempotency Check ----------
if [[ -f "$ENV_FILE" ]]; then
  echo "[$SCRIPT_NAME] Environment file already exists at $ENV_FILE. Skipping."
  exit 0
fi

# ---------- Fetch Secrets ----------
echo "[$SCRIPT_NAME] Fetching secrets from AWS Secrets Manager..."

SECRET_JSON=$(aws secretsmanager get-secret-value \
  --region "$AWS_REGION" \
  --secret-id "$SECRET_NAME" \
  --query 'SecretString' \
  --output text)

if [[ -z "$SECRET_JSON" ]]; then
  echo "[$SCRIPT_NAME] ERROR: Failed to retrieve secrets from $SECRET_NAME"
  exit 1
fi

# ---------- Write Environment File ----------
sudo mkdir -p "$(dirname "$ENV_FILE")"

# Parse JSON keys into KEY=VALUE format
echo "$SECRET_JSON" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for key, value in data.items():
    print(f'{key}={value}')
" | sudo tee "$ENV_FILE" > /dev/null

sudo chmod 600 "$ENV_FILE"
sudo chown root:root "$ENV_FILE"

# ---------- Verify ----------
KEY_COUNT=$(wc -l < "$ENV_FILE")
echo "[$SCRIPT_NAME] Wrote $KEY_COUNT environment variables to $ENV_FILE"
echo "[$SCRIPT_NAME] Done."
