#!/bin/bash
# trigger-deploy.sh - Trigger blue/green deployment via ASG instance refresh
# Initiates a rolling instance refresh on the Auto Scaling Group.
# New instances boot, install dependencies, deploy from S3, and pass
# health checks before old instances are terminated.
set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
echo "[$SCRIPT_NAME] Starting deployment trigger..."

# ---------- Configuration ----------
ASG_NAME="${ASG_NAME:?ASG_NAME is required}"
AWS_REGION="${AWS_REGION:-us-east-1}"
MIN_HEALTHY_PERCENT="${MIN_HEALTHY_PERCENT:-90}"
INSTANCE_WARMUP="${INSTANCE_WARMUP:-300}"

# ---------- Check Current Refresh Status ----------
echo "[$SCRIPT_NAME] Checking for active instance refreshes..."

ACTIVE_REFRESH=$(aws autoscaling describe-instance-refreshes \
  --auto-scaling-group-name "$ASG_NAME" \
  --region "$AWS_REGION" \
  --query 'InstanceRefreshes[?Status==`InProgress` || Status==`Pending`].[InstanceRefreshId]' \
  --output text)

if [[ -n "$ACTIVE_REFRESH" ]]; then
  echo "[$SCRIPT_NAME] ERROR: An instance refresh is already in progress: ${ACTIVE_REFRESH}"
  echo "[$SCRIPT_NAME] Wait for it to complete or cancel it before starting a new one."
  exit 1
fi

# ---------- Start Instance Refresh ----------
echo "[$SCRIPT_NAME] Starting instance refresh on ASG: ${ASG_NAME}"
echo "[$SCRIPT_NAME] Min healthy percentage: ${MIN_HEALTHY_PERCENT}%"
echo "[$SCRIPT_NAME] Instance warmup: ${INSTANCE_WARMUP}s"

REFRESH_ID=$(aws autoscaling start-instance-refresh \
  --auto-scaling-group-name "$ASG_NAME" \
  --region "$AWS_REGION" \
  --preferences "{
    \"MinHealthyPercentage\": ${MIN_HEALTHY_PERCENT},
    \"InstanceWarmup\": ${INSTANCE_WARMUP}
  }" \
  --query 'InstanceRefreshId' \
  --output text)

echo "[$SCRIPT_NAME] Instance refresh started: ${REFRESH_ID}"

# ---------- Monitor Progress ----------
echo "[$SCRIPT_NAME] Monitoring deployment progress..."

while true; do
  REFRESH_STATUS=$(aws autoscaling describe-instance-refreshes \
    --auto-scaling-group-name "$ASG_NAME" \
    --region "$AWS_REGION" \
    --instance-refresh-ids "$REFRESH_ID" \
    --query 'InstanceRefreshes[0].[Status,PercentageComplete]' \
    --output text)

  STATUS=$(echo "$REFRESH_STATUS" | awk '{print $1}')
  PERCENT=$(echo "$REFRESH_STATUS" | awk '{print $2}')

  echo "[$SCRIPT_NAME] Status: ${STATUS} | Progress: ${PERCENT}%"

  case "$STATUS" in
    Successful)
      echo "[$SCRIPT_NAME] Deployment completed successfully."
      break
      ;;
    Failed|Cancelled|RollbackSuccessful|RollbackFailed)
      echo "[$SCRIPT_NAME] ERROR: Deployment ended with status: ${STATUS}"
      exit 1
      ;;
    *)
      sleep 15
      ;;
  esac
done

# ---------- Verify Health ----------
echo "[$SCRIPT_NAME] Verifying instance health..."
HEALTHY_COUNT=$(aws autoscaling describe-auto-scaling-groups \
  --auto-scaling-group-names "$ASG_NAME" \
  --region "$AWS_REGION" \
  --query 'AutoScalingGroups[0].Instances[?HealthStatus==`Healthy`] | length(@)' \
  --output text)

TOTAL_COUNT=$(aws autoscaling describe-auto-scaling-groups \
  --auto-scaling-group-names "$ASG_NAME" \
  --region "$AWS_REGION" \
  --query 'AutoScalingGroups[0].Instances | length(@)' \
  --output text)

echo "[$SCRIPT_NAME] Healthy instances: ${HEALTHY_COUNT}/${TOTAL_COUNT}"
echo "[$SCRIPT_NAME] Done."
