#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-vibrato-db:smoke}"
PORT="${PORT:-18080}"
CONTAINER_NAME="${CONTAINER_NAME:-vibrato-smoke}"

tmp_root="$(mktemp -d)"
trap 'docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true; rm -rf "${tmp_root}"' EXIT

echo "[smoke] building ${IMAGE_TAG}"
docker build -t "${IMAGE_TAG}" .

echo "[smoke] starting container ${CONTAINER_NAME}"
docker run -d \
  --name "${CONTAINER_NAME}" \
  -p "${PORT}:8080" \
  -v "${tmp_root}/data:/var/lib/vibrato" \
  "${IMAGE_TAG}" \
  serve-v2 \
    --data-dir /var/lib/vibrato \
    --collection default \
    --dim 16 \
    --host 0.0.0.0 \
    --port 8080 \
    --public-health-metrics true \
    --checkpoint-interval-secs 3600 \
    --compaction-interval-secs 3600 >/dev/null

echo "[smoke] waiting for readiness"
for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:${PORT}/v2/health/ready" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

curl -fsS "http://127.0.0.1:${PORT}/v2/health/live" >/dev/null
curl -fsS "http://127.0.0.1:${PORT}/v2/health/ready" >/dev/null
curl -fsS "http://127.0.0.1:${PORT}/v2/metrics" >/dev/null

echo "[smoke] creating API key"
token="$(docker exec "${CONTAINER_NAME}" vibrato-db key-create --data-dir /var/lib/vibrato --name smoke --roles admin,query,ingest \
  | awk -F= '/^token=/{print $2}')"

if [[ -z "${token}" ]]; then
  echo "[smoke] failed to create token"
  exit 1
fi

echo "[smoke] ingest + query roundtrip"
curl -fsS -X POST "http://127.0.0.1:${PORT}/v2/vectors" \
  -H "Authorization: Bearer ${token}" \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
    "metadata": {"source_file":"smoke.wav","start_time_ms":0,"duration_ms":100,"bpm":120.0,"tags":["smoke"]},
    "idempotency_key":"smoke-1"
  }' >/dev/null

query_out="$(curl -fsS -X POST "http://127.0.0.1:${PORT}/v2/query" \
  -H "Authorization: Bearer ${token}" \
  -H "Content-Type: application/json" \
  -d '{"vector":[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"k":1,"ef":20,"include_metadata":true}' \
)"
if ! grep -q '"results"' <<<"${query_out}"; then
  echo "[smoke] query output missing results: ${query_out}"
  exit 1
fi

echo "[smoke] container smoke passed"
