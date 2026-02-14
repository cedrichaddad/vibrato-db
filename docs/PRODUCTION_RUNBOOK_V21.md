# Vibrato Production Runbook (v2.1)

This runbook is executable against the current `v2` server in this repository.

## 1. Bootstrap and Start

```bash
cargo run -- serve-v2 \
  --data-dir ./vibrato_data \
  --collection default \
  --dim 128 \
  --port 8080 \
  --host 0.0.0.0 \
  --checkpoint-interval-secs 30 \
  --compaction-interval-secs 180 \
  --orphan-ttl-hours 168 \
  --audio-colocated true \
  --public-health-metrics true \
  --bootstrap-admin-key true
```

If `--bootstrap-admin-key true` is used on first boot, capture `BOOTSTRAP_ADMIN_KEY=...` from stdout.

## 2. API Key Lifecycle

Create key:

```bash
cargo run -- key-create \
  --data-dir ./vibrato_data \
  --name prod-query \
  --roles query
```

Revoke key:

```bash
cargo run -- key-revoke \
  --data-dir ./vibrato_data \
  --key-id <vbk_id>
```

## 3. Checkpoint and Compaction

Manual checkpoint:

```bash
curl -X POST http://127.0.0.1:8080/v2/admin/checkpoint \
  -H "Authorization: Bearer <token>"
```

Manual compaction:

```bash
curl -X POST http://127.0.0.1:8080/v2/admin/compact \
  -H "Authorization: Bearer <token>"
```

Notes:
- L0+L0 compaction outputs L1 raw-f32 segments.
- L1+L1 compaction outputs L2 archive segments; when row count is sufficient these are PQ-encoded (`PQ_ENABLED`) and queried through the archive-tier rerank path.

Inspect engine state:

```bash
curl http://127.0.0.1:8080/v2/admin/stats \
  -H "Authorization: Bearer <token>"
```

## 4. Snapshot and Restore

Create snapshot:

```bash
cargo run -- snapshot-create \
  --data-dir ./vibrato_data \
  --collection default
```

Restore snapshot:

```bash
cargo run -- snapshot-restore \
  --data-dir ./vibrato_data \
  --snapshot-dir ./vibrato_data/snapshots/<snapshot_id>
```

Notes:
- Restore moves previous catalog/segments into `tmp/restore_backup_<ts>/`.
- Snapshot manifest checksums are validated before restore.

## 5. Point-in-Time Replay

Replay to a specific LSN:

```bash
cargo run -- replay-to-lsn \
  --data-dir ./vibrato_data \
  --collection default \
  --target-lsn 12345
```

Behavior:
- WAL rows newer than target LSN are deleted.
- Metadata rows without WAL backing are deleted.
- Active segments are marked `obsolete` to force clean rebuild from replayed WAL.

## 6. Recovery and Orphans

Startup recovery performs:
- orphan sweep (`segments/` files not in catalog are moved to `quarantine/`)
- quarantine TTL garbage collection
- in-flight job reconciliation (`checkpoint_jobs`, `compaction_jobs`)
- segment validation and state promotion/failure marking
- hot WAL tail/index rebuild

Readiness:
- `GET /v2/health/ready` is `503` when integrity blockers exist.
- Recovery report includes explicit blockers.

## 7. Health and Metrics Security

- With `--public-health-metrics true`, `/v2/health/*` and `/v2/metrics` are public.
- With `--public-health-metrics false`, health requires `query` role and metrics requires `admin` role.

## 8. SLO Triage Checklist

When latency regresses:
1. Inspect `/v2/admin/stats` for `wal_pending`, `checkpoint_jobs_inflight`, `compaction_jobs_inflight`.
2. Trigger manual checkpoint if WAL tail is large.
3. Verify background workers are running (`checkpoint_total` and `compaction_total` in `/v2/metrics`).
4. Check segment state drift (`active/obsolete/failed` counts).
5. If filter-heavy workload regressed, test with and without filter to isolate Layer-0 filtered path.
