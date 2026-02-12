#!/bin/bash
# build-and-upload.sh - Build deployment artifact and upload to S3
# Called by CI/CD pipeline after tests pass. Creates a tarball of the
# application code and uploads it to S3, decoupling deployments from
# CI artifact retention policies.
set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
echo "[$SCRIPT_NAME] Starting build and upload..."

# ---------- Configuration ----------
S3_BUCKET="${S3_BUCKET:?S3_BUCKET is required}"
S3_KEY="${S3_KEY:-deployments/myapp/deploy.tar.gz}"
AWS_REGION="${AWS_REGION:-us-east-1}"
APP_DIR="${APP_DIR:-.}"

# ---------- Build Artifact ----------
echo "[$SCRIPT_NAME] Creating deployment tarball..."

TEMP_DIR=$(mktemp -d)
ARTIFACT="${TEMP_DIR}/deploy.tar.gz"

tar czf "$ARTIFACT" \
  --exclude='.git' \
  --exclude='node_modules' \
  --exclude='tmp/*' \
  --exclude='log/*' \
  --exclude='spec' \
  --exclude='test' \
  --exclude='.ebextensions' \
  --exclude='.elasticbeanstalk' \
  -C "$APP_DIR" .

ARTIFACT_SIZE=$(du -h "$ARTIFACT" | cut -f1)
echo "[$SCRIPT_NAME] Artifact size: ${ARTIFACT_SIZE}"

# ---------- Upload to S3 ----------
echo "[$SCRIPT_NAME] Uploading to s3://${S3_BUCKET}/${S3_KEY}..."
aws s3 cp "$ARTIFACT" "s3://${S3_BUCKET}/${S3_KEY}" \
  --region "$AWS_REGION"

# Also upload a versioned copy for rollback
TIMESTAMP=$(date +%Y%m%d%H%M%S)
COMMIT_SHA="${GITHUB_SHA:-$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')}"
VERSIONED_KEY="${S3_KEY%.tar.gz}-${TIMESTAMP}-${COMMIT_SHA}.tar.gz"

aws s3 cp "$ARTIFACT" "s3://${S3_BUCKET}/${VERSIONED_KEY}" \
  --region "$AWS_REGION"

echo "[$SCRIPT_NAME] Versioned artifact: s3://${S3_BUCKET}/${VERSIONED_KEY}"

# ---------- Cleanup ----------
rm -rf "$TEMP_DIR"

echo "[$SCRIPT_NAME] Upload complete."
echo "[$SCRIPT_NAME] Done."
