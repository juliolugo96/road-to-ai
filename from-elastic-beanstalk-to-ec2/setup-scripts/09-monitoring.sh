#!/bin/bash
# 09-monitoring.sh - Set up Grafana, Loki, and Promtail monitoring stack
# Deploys the observability stack via Docker Compose for centralized
# log aggregation and real-time visibility.
set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
echo "[$SCRIPT_NAME] Starting monitoring setup..."

# ---------- Configuration ----------
MONITORING_DIR="/opt/monitoring"
COMPOSE_FILE="${MONITORING_DIR}/docker-compose.yml"

# ---------- Idempotency Check ----------
if docker compose -f "$COMPOSE_FILE" ps --status running 2>/dev/null | grep -q "grafana"; then
  echo "[$SCRIPT_NAME] Monitoring stack is already running. Skipping."
  exit 0
fi

# ---------- Create Monitoring Directory ----------
sudo mkdir -p "${MONITORING_DIR}"

# ---------- Copy Configuration Files ----------
# These files are expected to be in the deployment artifact under monitoring/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MONITORING_SRC="${SCRIPT_DIR}/../monitoring"

if [[ -d "$MONITORING_SRC" ]]; then
  echo "[$SCRIPT_NAME] Copying monitoring configuration..."
  sudo cp -r "${MONITORING_SRC}/"* "${MONITORING_DIR}/"
else
  echo "[$SCRIPT_NAME] ERROR: Monitoring config directory not found at ${MONITORING_SRC}"
  exit 1
fi

# ---------- Create Data Directories ----------
sudo mkdir -p "${MONITORING_DIR}/grafana-data"
sudo mkdir -p "${MONITORING_DIR}/loki-data"

# ---------- Start Stack ----------
echo "[$SCRIPT_NAME] Starting monitoring stack..."
cd "$MONITORING_DIR"
sudo docker compose up -d

# ---------- Wait for Services ----------
echo "[$SCRIPT_NAME] Waiting for Grafana to be ready..."
for i in $(seq 1 60); do
  if curl -sf http://localhost:3000/api/health &>/dev/null; then
    echo "[$SCRIPT_NAME] Grafana is ready."
    break
  fi
  if [[ $i -eq 60 ]]; then
    echo "[$SCRIPT_NAME] WARNING: Grafana did not become ready within 60 seconds."
  fi
  sleep 1
done

# ---------- Verify ----------
echo "[$SCRIPT_NAME] Running containers:"
docker compose -f "$COMPOSE_FILE" ps
echo "[$SCRIPT_NAME] Done."
