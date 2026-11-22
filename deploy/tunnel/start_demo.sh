#!/usr/bin/env bash
# Run the backend on this Mac and share it over a free tunnel, so the Vercel site can use the real model.
# Needs Cloudflare's cloudflared (free, no account). SSH tunnels such as localhost.run are throttled on their
# free tier and too slow for scans, so they are not used.
# Usage (from the repo root): deploy/tunnel/start_demo.sh      Stop everything with Ctrl+C.
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
port="${PORT:-8000}"

if lsof -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port $port is already in use. Stop the other backend first, or run with PORT=<another port>."
  exit 1
fi

# A fresh demo database and a random login secret each run, so no real accounts or scans are exposed.
export SKINSCAN_DB_PATH="${SKINSCAN_DB_PATH:-$HOME/.skinscan-demo.db}"
export SKINSCAN_AUTH_SECRET="${SKINSCAN_AUTH_SECRET:-$(python3 -c 'import secrets; print(secrets.token_hex(32))')}"
export BACKEND_PORT="$port"

cd "$root"
python3 backend/app.py &
backend=$!
trap 'kill "$backend" 2>/dev/null || true' EXIT INT TERM

for _ in $(seq 1 90); do
  curl -s -m 2 "http://localhost:$port/api/health" >/dev/null && break
  sleep 1
done
echo "Backend is up on http://localhost:$port"
echo

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "cloudflared is not installed. Download it from https://github.com/cloudflare/cloudflared/releases"
  echo "(or: brew install cloudflared) and run this script again."
  exit 1
fi
tunnel=(cloudflared tunnel --url "http://localhost:$port")

"${tunnel[@]}" 2>&1 | while IFS= read -r line; do
  echo "$line"
  url="$(printf '%s' "$line" | grep -oE 'https://[a-zA-Z0-9.-]+\.trycloudflare\.com' | head -1 || true)"
  if [ -n "$url" ]; then
    printf '\n>>> Set VITE_API_URL to this in Vercel, then redeploy: %s\n\n' "$url"
  fi
done
