#!/bin/bash
# 03-redis.sh - Install and configure Redis via Docker
# Redis is used for caching and as the Sidekiq job queue backend.
set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
echo "[$SCRIPT_NAME] Starting Redis setup..."

# ---------- Configuration ----------
REDIS_CONTAINER_NAME="redis"
REDIS_PORT=6379
REDIS_DATA_DIR="/var/lib/redis-data"

# ---------- Idempotency Check ----------
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${REDIS_CONTAINER_NAME}$"; then
  echo "[$SCRIPT_NAME] Redis container is already running. Skipping."
  exit 0
fi

# ---------- Clean Up Stale Container ----------
if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -q "^${REDIS_CONTAINER_NAME}$"; then
  echo "[$SCRIPT_NAME] Removing stopped Redis container..."
  docker rm -f "$REDIS_CONTAINER_NAME"
fi

# ---------- Create Data Directory ----------
sudo mkdir -p "$REDIS_DATA_DIR"

# ---------- Start Redis ----------
echo "[$SCRIPT_NAME] Starting Redis container..."
docker run -d \
  --name "$REDIS_CONTAINER_NAME" \
  --restart unless-stopped \
  -p "127.0.0.1:${REDIS_PORT}:6379" \
  -v "${REDIS_DATA_DIR}:/data" \
  redis:7-alpine \
  redis-server --appendonly yes --maxmemory 128mb --maxmemory-policy allkeys-lru

# ---------- Wait for Ready ----------
echo "[$SCRIPT_NAME] Waiting for Redis to be ready..."
for i in $(seq 1 30); do
  if docker exec "$REDIS_CONTAINER_NAME" redis-cli ping 2>/dev/null | grep -q PONG; then
    echo "[$SCRIPT_NAME] Redis is ready."
    break
  fi
  if [[ $i -eq 30 ]]; then
    echo "[$SCRIPT_NAME] ERROR: Redis did not become ready within 30 seconds."
    exit 1
  fi
  sleep 1
done

# ---------- Verify ----------
echo "[$SCRIPT_NAME] Redis version: $(docker exec "$REDIS_CONTAINER_NAME" redis-cli INFO server | grep redis_version)"
echo "[$SCRIPT_NAME] Done."
