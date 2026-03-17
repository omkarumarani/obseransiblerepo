#!/bin/sh
# gitea/init.sh — runs inside the Gitea container as its entrypoint
# 1. Starts the real Gitea daemon in background
# 2. Waits for the HTTP API to become ready
# 3. Creates the admin user (silently skips if it already exists)
# 4. Foregrounds the Gitea process so Docker sees it as PID 1-equivalent

set -e

echo "[gitea-init] Starting Gitea daemon..."
/usr/bin/entrypoint &
GITEA_PID=$!

echo "[gitea-init] Waiting for Gitea API..."
for i in $(seq 1 60); do
    STATUS=$(wget -q -O- http://localhost:3000/api/v1/version 2>/dev/null && echo "ok" || true)
    if [ "$STATUS" = "ok" ] || wget -q --spider http://localhost:3000/api/v1/version 2>/dev/null; then
        echo "[gitea-init] Gitea API is ready"
        break
    fi
    sleep 2
done

echo "[gitea-init] Creating admin user (skipped if already exists)..."
su git -s /bin/sh -c "gitea admin user create \
    --admin \
    --username aiops \
    --password 'Aiops1234!' \
    --email 'aiops@local.test'" \
    2>/dev/null \
  && echo "[gitea-init] Admin user created." \
  || echo "[gitea-init] Admin user already exists — skipping."

echo "[gitea-init] Init complete. Gitea running as PID $GITEA_PID"
wait $GITEA_PID
