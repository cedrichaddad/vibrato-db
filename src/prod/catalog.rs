use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Context, Result};
use rand::RngCore;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use vibrato_core::metadata::VectorMetadata;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Role {
    Query,
    Ingest,
    Admin,
}

impl Role {
    pub fn as_str(&self) -> &'static str {
        match self {
            Role::Query => "query",
            Role::Ingest => "ingest",
            Role::Admin => "admin",
        }
    }

    pub fn parse_csv(value: &str) -> Vec<Role> {
        value
            .split(',')
            .map(|s| s.trim().to_ascii_lowercase())
            .filter_map(|s| match s.as_str() {
                "query" => Some(Role::Query),
                "ingest" => Some(Role::Ingest),
                "admin" => Some(Role::Admin),
                _ => None,
            })
            .collect()
    }

    pub fn to_csv(roles: &[Role]) -> String {
        roles.iter().map(Role::as_str).collect::<Vec<_>>().join(",")
    }
}

#[derive(Debug, Clone)]
pub struct CollectionRecord {
    pub id: String,
    pub name: String,
    pub dim: usize,
}

#[derive(Debug, Clone)]
pub struct ApiKeyRecord {
    pub id: String,
    pub key_hash: String,
    pub roles: Vec<Role>,
    pub revoked: bool,
}

#[derive(Debug, Clone)]
pub struct ApiKeyCreateResult {
    pub id: String,
    pub token: String,
}

#[derive(Debug, Clone)]
pub struct WalIngestResult {
    pub vector_id: usize,
    pub created: bool,
    pub lsn: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct WalEntry {
    pub lsn: u64,
    pub vector_id: usize,
    pub vector: Vec<f32>,
    pub metadata: VectorMetadata,
}

#[derive(Debug, Clone)]
pub struct SegmentRecord {
    pub id: String,
    pub collection_id: String,
    pub level: i64,
    pub path: PathBuf,
    pub row_count: usize,
    pub vector_id_start: usize,
    pub vector_id_end: usize,
    pub created_lsn: u64,
    pub state: String,
}

#[derive(Debug, Clone)]
pub struct CheckpointJobRecord {
    pub id: String,
    pub collection_id: String,
    pub state: String,
    pub start_lsn: Option<u64>,
    pub end_lsn: Option<u64>,
    pub details: Value,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Debug, Clone)]
pub struct CompactionJobRecord {
    pub id: String,
    pub collection_id: String,
    pub state: String,
    pub details: Value,
    pub created_at: i64,
    pub updated_at: i64,
}

pub trait CatalogStore: Send + Sync {
    fn ensure_collection(&self, name: &str, dim: usize) -> Result<CollectionRecord>;
    fn get_collection(&self, name: &str) -> Result<Option<CollectionRecord>>;

    fn count_api_keys(&self) -> Result<usize>;
    fn create_api_key(
        &self,
        name: &str,
        roles: &[Role],
        pepper: &str,
    ) -> Result<ApiKeyCreateResult>;
    fn revoke_api_key(&self, id: &str) -> Result<()>;
    fn lookup_api_key(&self, id: &str) -> Result<Option<ApiKeyRecord>>;

    fn next_vector_id(&self, collection_id: &str) -> Result<usize>;
    fn ingest_wal(
        &self,
        collection_id: &str,
        vector_id: usize,
        vector: &[f32],
        metadata: &VectorMetadata,
        idempotency_key: Option<&str>,
    ) -> Result<WalIngestResult>;

    fn pending_wal(&self, collection_id: &str, limit: usize) -> Result<Vec<WalEntry>>;
    fn pending_wal_after_lsn(
        &self,
        collection_id: &str,
        lsn_exclusive: u64,
    ) -> Result<Vec<WalEntry>>;
    fn mark_wal_checkpointed(
        &self,
        collection_id: &str,
        start_lsn: u64,
        end_lsn: u64,
    ) -> Result<()>;
    fn count_wal_pending(&self, collection_id: &str) -> Result<usize>;

    fn insert_segment(&self, record: &SegmentRecord) -> Result<()>;
    fn update_segment_state(&self, segment_id: &str, state: &str) -> Result<()>;
    fn list_segments_by_state(
        &self,
        collection_id: &str,
        states: &[&str],
    ) -> Result<Vec<SegmentRecord>>;
    fn get_segment(&self, segment_id: &str) -> Result<Option<SegmentRecord>>;
    fn list_known_segment_paths(&self, collection_id: &str) -> Result<Vec<PathBuf>>;

    fn upsert_checkpoint_job(&self, job: &CheckpointJobRecord) -> Result<()>;
    fn update_checkpoint_job_state(&self, job_id: &str, state: &str, details: Value) -> Result<()>;
    fn checkpoint_mark_pending_activate(
        &self,
        collection_id: &str,
        segment_id: &str,
        job_id: &str,
        start_lsn: u64,
        end_lsn: u64,
        details: Value,
    ) -> Result<()>;
    fn checkpoint_activate(&self, segment_id: &str, job_id: &str, details: Value) -> Result<()>;
    fn list_checkpoint_jobs_by_state(
        &self,
        collection_id: &str,
        states: &[&str],
    ) -> Result<Vec<CheckpointJobRecord>>;

    fn upsert_compaction_job(&self, job: &CompactionJobRecord) -> Result<()>;
    fn update_compaction_job_state(&self, job_id: &str, state: &str, details: Value) -> Result<()>;
    fn compaction_mark_pending_activate(
        &self,
        output_segment_id: &str,
        job_id: &str,
        details: Value,
    ) -> Result<()>;
    fn compaction_activate(
        &self,
        output_segment_id: &str,
        input_segment_ids: &[String],
        job_id: &str,
        details: Value,
    ) -> Result<()>;
    fn list_compaction_jobs_by_state(
        &self,
        collection_id: &str,
        states: &[&str],
    ) -> Result<Vec<CompactionJobRecord>>;

    fn upsert_metadata(
        &self,
        collection_id: &str,
        vector_id: usize,
        metadata: &VectorMetadata,
    ) -> Result<()>;
    fn fetch_metadata(&self, ids: &[usize]) -> Result<HashMap<usize, VectorMetadata>>;
    fn fetch_all_metadata(&self, collection_id: &str) -> Result<Vec<(usize, VectorMetadata)>>;

    fn insert_orphan_file(
        &self,
        collection_id: &str,
        original_path: &Path,
        quarantine_path: &Path,
        reason: &str,
    ) -> Result<()>;
    fn quarantine_gc(&self, ttl_secs: i64) -> Result<Vec<PathBuf>>;

    fn total_vectors(&self, collection_id: &str) -> Result<usize>;
    fn audit_event(
        &self,
        request_id: &str,
        api_key_id: Option<&str>,
        endpoint: &str,
        action: &str,
        status_code: u16,
        latency_ms: f64,
        details: Value,
    ) -> Result<()>;
}

pub struct SqliteCatalog {
    path: PathBuf,
}

impl SqliteCatalog {
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let catalog = Self {
            path: path.to_path_buf(),
        };

        catalog.exec("PRAGMA journal_mode=WAL;")?;
        catalog.exec("PRAGMA synchronous=FULL;")?;
        catalog.exec("PRAGMA foreign_keys=ON;")?;
        catalog.exec("PRAGMA temp_store=MEMORY;")?;
        catalog.exec("PRAGMA busy_timeout=5000;")?;
        catalog.exec("PRAGMA mmap_size=268435456;")?;

        catalog.apply_migrations()?;
        Ok(catalog)
    }

    fn exec(&self, sql: &str) -> Result<()> {
        let output = Command::new("sqlite3")
            .arg(&self.path)
            .arg(sql)
            .output()
            .with_context(|| format!("running sqlite3 exec against {:?}", self.path))?;

        if !output.status.success() {
            return Err(anyhow!(
                "sqlite exec failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        Ok(())
    }

    fn query_json(&self, sql: &str) -> Result<Vec<Value>> {
        let output = Command::new("sqlite3")
            .arg("-json")
            .arg(&self.path)
            .arg(sql)
            .output()
            .with_context(|| format!("running sqlite3 query against {:?}", self.path))?;

        if !output.status.success() {
            return Err(anyhow!(
                "sqlite query failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }

        if output.stdout.is_empty() {
            return Ok(Vec::new());
        }

        let rows: Vec<Value> = serde_json::from_slice(&output.stdout).with_context(|| {
            format!(
                "parsing sqlite json output: {}",
                String::from_utf8_lossy(&output.stdout)
            )
        })?;
        Ok(rows)
    }

    fn apply_migrations(&self) -> Result<()> {
        let migration_sql = [
            "CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, applied_at INTEGER NOT NULL, checksum TEXT NOT NULL)",
            "CREATE TABLE IF NOT EXISTS collections (id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE, dim INTEGER NOT NULL, created_at INTEGER NOT NULL)",
            "CREATE TABLE IF NOT EXISTS segments (id TEXT PRIMARY KEY, collection_id TEXT NOT NULL, level INTEGER NOT NULL, path TEXT NOT NULL UNIQUE, row_count INTEGER NOT NULL, vector_id_start INTEGER NOT NULL, vector_id_end INTEGER NOT NULL, created_lsn INTEGER NOT NULL, state TEXT NOT NULL, created_at INTEGER NOT NULL)",
            "CREATE TABLE IF NOT EXISTS wal_entries (lsn INTEGER PRIMARY KEY AUTOINCREMENT, collection_id TEXT NOT NULL, vector_id INTEGER NOT NULL, vector_json TEXT NOT NULL, metadata_json TEXT NOT NULL, idempotency_key TEXT, checkpointed_at INTEGER, created_at INTEGER NOT NULL)",
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_wal_idempotency ON wal_entries(collection_id, idempotency_key) WHERE idempotency_key IS NOT NULL",
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_wal_vector_id ON wal_entries(collection_id, vector_id)",
            "CREATE TABLE IF NOT EXISTS vector_metadata (vector_id INTEGER PRIMARY KEY, collection_id TEXT NOT NULL, source_file TEXT NOT NULL, start_time_ms INTEGER NOT NULL, duration_ms INTEGER NOT NULL, bpm REAL NOT NULL, tags_json TEXT NOT NULL, created_at INTEGER NOT NULL)",
            "CREATE TABLE IF NOT EXISTS api_keys (id TEXT PRIMARY KEY, name TEXT NOT NULL, key_hash TEXT NOT NULL, roles TEXT NOT NULL, created_at INTEGER NOT NULL, revoked_at INTEGER)",
            "CREATE TABLE IF NOT EXISTS audit_events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER NOT NULL, request_id TEXT NOT NULL, api_key_id TEXT, endpoint TEXT NOT NULL, action TEXT NOT NULL, status_code INTEGER NOT NULL, latency_ms REAL NOT NULL, client_ip TEXT, details_json TEXT NOT NULL)",
            "CREATE TABLE IF NOT EXISTS checkpoint_jobs (id TEXT PRIMARY KEY, collection_id TEXT NOT NULL, state TEXT NOT NULL, start_lsn INTEGER, end_lsn INTEGER, details_json TEXT NOT NULL, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL)",
            "CREATE TABLE IF NOT EXISTS compaction_jobs (id TEXT PRIMARY KEY, collection_id TEXT NOT NULL, state TEXT NOT NULL, details_json TEXT NOT NULL, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL)",
            "CREATE TABLE IF NOT EXISTS orphan_files (id INTEGER PRIMARY KEY AUTOINCREMENT, collection_id TEXT NOT NULL, original_path TEXT NOT NULL, quarantine_path TEXT NOT NULL, reason TEXT NOT NULL, quarantined_at INTEGER NOT NULL, deleted_at INTEGER)",
        ];

        for sql in migration_sql {
            self.exec(sql)?;
        }
        self.exec(&format!(
            "INSERT OR REPLACE INTO schema_migrations(version, applied_at, checksum) VALUES (1, {}, '{}');",
            now_unix_ts(),
            sql_quote("v1")
        ))?;
        Ok(())
    }

    pub fn execute_sql(&self, sql: &str) -> Result<()> {
        self.exec(sql)
    }
}

impl CatalogStore for SqliteCatalog {
    fn ensure_collection(&self, name: &str, dim: usize) -> Result<CollectionRecord> {
        if let Some(c) = self.get_collection(name)? {
            if c.dim != dim {
                return Err(anyhow!(
                    "collection '{}' exists with dim {}, expected {}",
                    name,
                    c.dim,
                    dim
                ));
            }
            return Ok(c);
        }

        let id = random_id("col");
        self.exec(&format!(
            "INSERT INTO collections(id, name, dim, created_at) VALUES ('{}', '{}', {}, {});",
            sql_quote(&id),
            sql_quote(name),
            dim,
            now_unix_ts()
        ))?;

        Ok(CollectionRecord {
            id,
            name: name.to_string(),
            dim,
        })
    }

    fn get_collection(&self, name: &str) -> Result<Option<CollectionRecord>> {
        let rows = self.query_json(&format!(
            "SELECT id, name, dim FROM collections WHERE name='{}' LIMIT 1;",
            sql_quote(name)
        ))?;

        let Some(row) = rows.first() else {
            return Ok(None);
        };
        Ok(Some(CollectionRecord {
            id: row["id"].as_str().unwrap_or_default().to_string(),
            name: row["name"].as_str().unwrap_or_default().to_string(),
            dim: row["dim"].as_i64().unwrap_or_default() as usize,
        }))
    }

    fn count_api_keys(&self) -> Result<usize> {
        let rows =
            self.query_json("SELECT COUNT(*) AS n FROM api_keys WHERE revoked_at IS NULL;")?;
        Ok(rows
            .first()
            .and_then(|r| r["n"].as_i64())
            .unwrap_or_default() as usize)
    }

    fn create_api_key(
        &self,
        name: &str,
        roles: &[Role],
        pepper: &str,
    ) -> Result<ApiKeyCreateResult> {
        let id = random_id("vbk");
        let mut secret_bytes = [0u8; 24];
        rand::thread_rng().fill_bytes(&mut secret_bytes);
        let secret = hex_string(&secret_bytes);
        let token = format!("{}.{}", id, secret);
        let hash = hash_secret(pepper, &secret);

        self.exec(&format!(
            "INSERT INTO api_keys(id, name, key_hash, roles, created_at, revoked_at) VALUES ('{}', '{}', '{}', '{}', {}, NULL);",
            sql_quote(&id),
            sql_quote(name),
            sql_quote(&hash),
            sql_quote(&Role::to_csv(roles)),
            now_unix_ts()
        ))?;

        Ok(ApiKeyCreateResult { id, token })
    }

    fn revoke_api_key(&self, id: &str) -> Result<()> {
        self.exec(&format!(
            "UPDATE api_keys SET revoked_at = {} WHERE id = '{}';",
            now_unix_ts(),
            sql_quote(id)
        ))
    }

    fn lookup_api_key(&self, id: &str) -> Result<Option<ApiKeyRecord>> {
        let rows = self.query_json(&format!(
            "SELECT id, key_hash, roles, revoked_at FROM api_keys WHERE id='{}' LIMIT 1;",
            sql_quote(id)
        ))?;

        let Some(row) = rows.first() else {
            return Ok(None);
        };

        Ok(Some(ApiKeyRecord {
            id: row["id"].as_str().unwrap_or_default().to_string(),
            key_hash: row["key_hash"].as_str().unwrap_or_default().to_string(),
            roles: Role::parse_csv(row["roles"].as_str().unwrap_or_default()),
            revoked: !row["revoked_at"].is_null(),
        }))
    }

    fn next_vector_id(&self, collection_id: &str) -> Result<usize> {
        let rows = self.query_json(&format!(
            "SELECT COALESCE(MAX(vector_id), -1) AS max_id FROM vector_metadata WHERE collection_id='{}';",
            sql_quote(collection_id)
        ))?;
        let max_id = rows
            .first()
            .and_then(|r| r["max_id"].as_i64())
            .unwrap_or(-1);
        Ok((max_id + 1) as usize)
    }

    fn ingest_wal(
        &self,
        collection_id: &str,
        vector_id: usize,
        vector: &[f32],
        metadata: &VectorMetadata,
        idempotency_key: Option<&str>,
    ) -> Result<WalIngestResult> {
        if let Some(key) = idempotency_key {
            if !key.trim().is_empty() {
                let rows = self.query_json(&format!(
                    "SELECT vector_id FROM wal_entries WHERE collection_id='{}' AND idempotency_key='{}' LIMIT 1;",
                    sql_quote(collection_id),
                    sql_quote(key)
                ))?;
                if let Some(row) = rows.first() {
                    return Ok(WalIngestResult {
                        vector_id: row["vector_id"].as_i64().unwrap_or_default() as usize,
                        created: false,
                        lsn: None,
                    });
                }
            }
        }

        let vector_json = serde_json::to_string(vector)?;
        let metadata_json = serde_json::to_string(metadata)?;
        let tags_json = serde_json::to_string(&metadata.tags)?;

        self.exec(&format!(
            "BEGIN IMMEDIATE;
             INSERT INTO wal_entries(collection_id, vector_id, vector_json, metadata_json, idempotency_key, checkpointed_at, created_at)
             VALUES ('{}', {}, '{}', '{}', {}, NULL, {});
             INSERT OR REPLACE INTO vector_metadata(vector_id, collection_id, source_file, start_time_ms, duration_ms, bpm, tags_json, created_at)
             VALUES ({}, '{}', '{}', {}, {}, {}, '{}', {});
             COMMIT;",
            sql_quote(collection_id),
            vector_id,
            sql_quote(&vector_json),
            sql_quote(&metadata_json),
            idempotency_key
                .map(|k| format!("'{}'", sql_quote(k)))
                .unwrap_or_else(|| "NULL".to_string()),
            now_unix_ts(),
            vector_id,
            sql_quote(collection_id),
            sql_quote(&metadata.source_file),
            metadata.start_time_ms,
            metadata.duration_ms,
            metadata.bpm,
            sql_quote(&tags_json),
            now_unix_ts()
        ))?;

        let lsn_rows = self.query_json(&format!(
            "SELECT lsn FROM wal_entries WHERE collection_id='{}' AND vector_id={} ORDER BY lsn DESC LIMIT 1;",
            sql_quote(collection_id),
            vector_id
        ))?;
        let lsn = lsn_rows
            .first()
            .and_then(|r| {
                r["lsn"]
                    .as_i64()
                    .or_else(|| r["lsn"].as_str().and_then(|s| s.parse::<i64>().ok()))
            })
            .unwrap_or_default() as u64;

        Ok(WalIngestResult {
            vector_id,
            created: true,
            lsn: Some(lsn),
        })
    }

    fn pending_wal(&self, collection_id: &str, limit: usize) -> Result<Vec<WalEntry>> {
        self.query_wal(&format!(
            "SELECT lsn, vector_id, vector_json, metadata_json FROM wal_entries WHERE collection_id='{}' AND checkpointed_at IS NULL ORDER BY lsn ASC LIMIT {};",
            sql_quote(collection_id),
            limit
        ))
    }

    fn pending_wal_after_lsn(
        &self,
        collection_id: &str,
        lsn_exclusive: u64,
    ) -> Result<Vec<WalEntry>> {
        self.query_wal(&format!(
            "SELECT lsn, vector_id, vector_json, metadata_json FROM wal_entries WHERE collection_id='{}' AND checkpointed_at IS NULL AND lsn > {} ORDER BY lsn ASC;",
            sql_quote(collection_id),
            lsn_exclusive
        ))
    }

    fn mark_wal_checkpointed(
        &self,
        collection_id: &str,
        start_lsn: u64,
        end_lsn: u64,
    ) -> Result<()> {
        self.exec(&format!(
            "UPDATE wal_entries SET checkpointed_at = {} WHERE collection_id='{}' AND lsn >= {} AND lsn <= {};",
            now_unix_ts(),
            sql_quote(collection_id),
            start_lsn,
            end_lsn
        ))
    }

    fn count_wal_pending(&self, collection_id: &str) -> Result<usize> {
        let rows = self.query_json(&format!(
            "SELECT COUNT(*) AS n FROM wal_entries WHERE collection_id='{}' AND checkpointed_at IS NULL;",
            sql_quote(collection_id)
        ))?;
        Ok(rows
            .first()
            .and_then(|r| r["n"].as_i64())
            .unwrap_or_default() as usize)
    }

    fn insert_segment(&self, record: &SegmentRecord) -> Result<()> {
        self.exec(&format!(
            "INSERT OR REPLACE INTO segments(id, collection_id, level, path, row_count, vector_id_start, vector_id_end, created_lsn, state, created_at) VALUES ('{}', '{}', {}, '{}', {}, {}, {}, {}, '{}', {});",
            sql_quote(&record.id),
            sql_quote(&record.collection_id),
            record.level,
            sql_quote(&record.path.to_string_lossy()),
            record.row_count,
            record.vector_id_start,
            record.vector_id_end,
            record.created_lsn,
            sql_quote(&record.state),
            now_unix_ts(),
        ))
    }

    fn update_segment_state(&self, segment_id: &str, state: &str) -> Result<()> {
        self.exec(&format!(
            "UPDATE segments SET state='{}' WHERE id='{}';",
            sql_quote(state),
            sql_quote(segment_id)
        ))
    }

    fn list_segments_by_state(
        &self,
        collection_id: &str,
        states: &[&str],
    ) -> Result<Vec<SegmentRecord>> {
        if states.is_empty() {
            return Ok(Vec::new());
        }

        let in_clause = states
            .iter()
            .map(|s| format!("'{}'", sql_quote(s)))
            .collect::<Vec<_>>()
            .join(",");

        let rows = self.query_json(&format!(
            "SELECT id, collection_id, level, path, row_count, vector_id_start, vector_id_end, created_lsn, state FROM segments WHERE collection_id='{}' AND state IN ({}) ORDER BY level ASC, vector_id_start ASC;",
            sql_quote(collection_id),
            in_clause
        ))?;

        let mut out = Vec::new();
        for row in rows {
            out.push(SegmentRecord {
                id: row["id"].as_str().unwrap_or_default().to_string(),
                collection_id: row["collection_id"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string(),
                level: row["level"].as_i64().unwrap_or_default(),
                path: PathBuf::from(row["path"].as_str().unwrap_or_default()),
                row_count: row["row_count"].as_i64().unwrap_or_default() as usize,
                vector_id_start: row["vector_id_start"].as_i64().unwrap_or_default() as usize,
                vector_id_end: row["vector_id_end"].as_i64().unwrap_or_default() as usize,
                created_lsn: row["created_lsn"].as_i64().unwrap_or_default() as u64,
                state: row["state"].as_str().unwrap_or_default().to_string(),
            });
        }

        Ok(out)
    }

    fn list_known_segment_paths(&self, collection_id: &str) -> Result<Vec<PathBuf>> {
        let rows = self.query_json(&format!(
            "SELECT path FROM segments WHERE collection_id='{}';",
            sql_quote(collection_id)
        ))?;
        Ok(rows
            .into_iter()
            .filter_map(|r| r["path"].as_str().map(PathBuf::from))
            .collect())
    }

    fn get_segment(&self, segment_id: &str) -> Result<Option<SegmentRecord>> {
        let rows = self.query_json(&format!(
            "SELECT id, collection_id, level, path, row_count, vector_id_start, vector_id_end, created_lsn, state FROM segments WHERE id='{}' LIMIT 1;",
            sql_quote(segment_id)
        ))?;
        let Some(row) = rows.first() else {
            return Ok(None);
        };
        Ok(Some(SegmentRecord {
            id: row["id"].as_str().unwrap_or_default().to_string(),
            collection_id: row["collection_id"]
                .as_str()
                .unwrap_or_default()
                .to_string(),
            level: row["level"].as_i64().unwrap_or_default(),
            path: PathBuf::from(row["path"].as_str().unwrap_or_default()),
            row_count: row["row_count"].as_i64().unwrap_or_default() as usize,
            vector_id_start: row["vector_id_start"].as_i64().unwrap_or_default() as usize,
            vector_id_end: row["vector_id_end"].as_i64().unwrap_or_default() as usize,
            created_lsn: row["created_lsn"].as_i64().unwrap_or_default() as u64,
            state: row["state"].as_str().unwrap_or_default().to_string(),
        }))
    }

    fn upsert_checkpoint_job(&self, job: &CheckpointJobRecord) -> Result<()> {
        self.exec(&format!(
            "INSERT OR REPLACE INTO checkpoint_jobs(id, collection_id, state, start_lsn, end_lsn, details_json, created_at, updated_at)
             VALUES ('{}', '{}', '{}', {}, {}, '{}', {}, {});",
            sql_quote(&job.id),
            sql_quote(&job.collection_id),
            sql_quote(&job.state),
            job.start_lsn
                .map(|v| v.to_string())
                .unwrap_or_else(|| "NULL".to_string()),
            job.end_lsn
                .map(|v| v.to_string())
                .unwrap_or_else(|| "NULL".to_string()),
            sql_quote(&job.details.to_string()),
            job.created_at,
            job.updated_at,
        ))
    }

    fn update_checkpoint_job_state(&self, job_id: &str, state: &str, details: Value) -> Result<()> {
        self.exec(&format!(
            "UPDATE checkpoint_jobs SET state='{}', details_json='{}', updated_at={} WHERE id='{}';",
            sql_quote(state),
            sql_quote(&details.to_string()),
            now_unix_ts(),
            sql_quote(job_id)
        ))
    }

    fn checkpoint_mark_pending_activate(
        &self,
        collection_id: &str,
        segment_id: &str,
        job_id: &str,
        start_lsn: u64,
        end_lsn: u64,
        details: Value,
    ) -> Result<()> {
        let now = now_unix_ts();
        self.exec(&format!(
            "BEGIN IMMEDIATE;
             UPDATE segments SET state='pending_activate' WHERE id='{}';
             UPDATE wal_entries SET checkpointed_at = {} WHERE collection_id='{}' AND lsn >= {} AND lsn <= {};
             UPDATE checkpoint_jobs SET state='pending_activate', details_json='{}', updated_at={} WHERE id='{}';
             COMMIT;",
            sql_quote(segment_id),
            now,
            sql_quote(collection_id),
            start_lsn,
            end_lsn,
            sql_quote(&details.to_string()),
            now,
            sql_quote(job_id)
        ))
    }

    fn checkpoint_activate(&self, segment_id: &str, job_id: &str, details: Value) -> Result<()> {
        let now = now_unix_ts();
        self.exec(&format!(
            "BEGIN IMMEDIATE;
             UPDATE segments SET state='active' WHERE id='{}';
             UPDATE checkpoint_jobs SET state='completed', details_json='{}', updated_at={} WHERE id='{}';
             COMMIT;",
            sql_quote(segment_id),
            sql_quote(&details.to_string()),
            now,
            sql_quote(job_id)
        ))
    }

    fn list_checkpoint_jobs_by_state(
        &self,
        collection_id: &str,
        states: &[&str],
    ) -> Result<Vec<CheckpointJobRecord>> {
        if states.is_empty() {
            return Ok(Vec::new());
        }
        let in_clause = states
            .iter()
            .map(|s| format!("'{}'", sql_quote(s)))
            .collect::<Vec<_>>()
            .join(",");

        let rows = self.query_json(&format!(
            "SELECT id, collection_id, state, start_lsn, end_lsn, details_json, created_at, updated_at
             FROM checkpoint_jobs WHERE collection_id='{}' AND state IN ({}) ORDER BY updated_at ASC;",
            sql_quote(collection_id),
            in_clause
        ))?;

        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            let details = serde_json::from_str(row["details_json"].as_str().unwrap_or("{}"))
                .unwrap_or_else(|_| json!({}));
            out.push(CheckpointJobRecord {
                id: row["id"].as_str().unwrap_or_default().to_string(),
                collection_id: row["collection_id"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string(),
                state: row["state"].as_str().unwrap_or_default().to_string(),
                start_lsn: row["start_lsn"].as_i64().map(|v| v as u64),
                end_lsn: row["end_lsn"].as_i64().map(|v| v as u64),
                details,
                created_at: row["created_at"].as_i64().unwrap_or_default(),
                updated_at: row["updated_at"].as_i64().unwrap_or_default(),
            });
        }
        Ok(out)
    }

    fn upsert_compaction_job(&self, job: &CompactionJobRecord) -> Result<()> {
        self.exec(&format!(
            "INSERT OR REPLACE INTO compaction_jobs(id, collection_id, state, details_json, created_at, updated_at)
             VALUES ('{}', '{}', '{}', '{}', {}, {});",
            sql_quote(&job.id),
            sql_quote(&job.collection_id),
            sql_quote(&job.state),
            sql_quote(&job.details.to_string()),
            job.created_at,
            job.updated_at,
        ))
    }

    fn update_compaction_job_state(&self, job_id: &str, state: &str, details: Value) -> Result<()> {
        self.exec(&format!(
            "UPDATE compaction_jobs SET state='{}', details_json='{}', updated_at={} WHERE id='{}';",
            sql_quote(state),
            sql_quote(&details.to_string()),
            now_unix_ts(),
            sql_quote(job_id)
        ))
    }

    fn compaction_mark_pending_activate(
        &self,
        output_segment_id: &str,
        job_id: &str,
        details: Value,
    ) -> Result<()> {
        let now = now_unix_ts();
        self.exec(&format!(
            "BEGIN IMMEDIATE;
             UPDATE segments SET state='pending_activate' WHERE id='{}';
             UPDATE compaction_jobs SET state='pending_activate', details_json='{}', updated_at={} WHERE id='{}';
             COMMIT;",
            sql_quote(output_segment_id),
            sql_quote(&details.to_string()),
            now,
            sql_quote(job_id)
        ))
    }

    fn compaction_activate(
        &self,
        output_segment_id: &str,
        input_segment_ids: &[String],
        job_id: &str,
        details: Value,
    ) -> Result<()> {
        let now = now_unix_ts();
        let input_update = if input_segment_ids.is_empty() {
            String::new()
        } else {
            let in_clause = input_segment_ids
                .iter()
                .map(|id| format!("'{}'", sql_quote(id)))
                .collect::<Vec<_>>()
                .join(",");
            format!(
                "UPDATE segments SET state='obsolete' WHERE id IN ({});",
                in_clause
            )
        };

        self.exec(&format!(
            "BEGIN IMMEDIATE;
             UPDATE segments SET state='active' WHERE id='{}';
             {}
             UPDATE compaction_jobs SET state='completed', details_json='{}', updated_at={} WHERE id='{}';
             COMMIT;",
            sql_quote(output_segment_id),
            input_update,
            sql_quote(&details.to_string()),
            now,
            sql_quote(job_id)
        ))
    }

    fn list_compaction_jobs_by_state(
        &self,
        collection_id: &str,
        states: &[&str],
    ) -> Result<Vec<CompactionJobRecord>> {
        if states.is_empty() {
            return Ok(Vec::new());
        }
        let in_clause = states
            .iter()
            .map(|s| format!("'{}'", sql_quote(s)))
            .collect::<Vec<_>>()
            .join(",");
        let rows = self.query_json(&format!(
            "SELECT id, collection_id, state, details_json, created_at, updated_at
             FROM compaction_jobs WHERE collection_id='{}' AND state IN ({}) ORDER BY updated_at ASC;",
            sql_quote(collection_id),
            in_clause
        ))?;

        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            let details = serde_json::from_str(row["details_json"].as_str().unwrap_or("{}"))
                .unwrap_or_else(|_| json!({}));
            out.push(CompactionJobRecord {
                id: row["id"].as_str().unwrap_or_default().to_string(),
                collection_id: row["collection_id"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string(),
                state: row["state"].as_str().unwrap_or_default().to_string(),
                details,
                created_at: row["created_at"].as_i64().unwrap_or_default(),
                updated_at: row["updated_at"].as_i64().unwrap_or_default(),
            });
        }
        Ok(out)
    }

    fn upsert_metadata(
        &self,
        collection_id: &str,
        vector_id: usize,
        metadata: &VectorMetadata,
    ) -> Result<()> {
        self.exec(&format!(
            "INSERT OR REPLACE INTO vector_metadata(vector_id, collection_id, source_file, start_time_ms, duration_ms, bpm, tags_json, created_at) VALUES ({}, '{}', '{}', {}, {}, {}, '{}', {});",
            vector_id,
            sql_quote(collection_id),
            sql_quote(&metadata.source_file),
            metadata.start_time_ms,
            metadata.duration_ms,
            metadata.bpm,
            sql_quote(&serde_json::to_string(&metadata.tags)?),
            now_unix_ts(),
        ))
    }

    fn fetch_metadata(&self, ids: &[usize]) -> Result<HashMap<usize, VectorMetadata>> {
        if ids.is_empty() {
            return Ok(HashMap::new());
        }

        let mut unique_ids = ids.to_vec();
        unique_ids.sort_unstable();
        unique_ids.dedup();
        let in_clause = unique_ids
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(",");

        let rows = self.query_json(&format!(
            "SELECT vector_id, source_file, start_time_ms, duration_ms, bpm, tags_json FROM vector_metadata WHERE vector_id IN ({}) ORDER BY vector_id ASC;",
            in_clause
        ))?;

        let mut out = HashMap::with_capacity(rows.len());
        for row in rows {
            let id = row["vector_id"].as_i64().unwrap_or_default() as usize;
            out.insert(
                id,
                VectorMetadata {
                    source_file: row["source_file"].as_str().unwrap_or_default().to_string(),
                    start_time_ms: row["start_time_ms"].as_i64().unwrap_or_default() as u32,
                    duration_ms: row["duration_ms"].as_i64().unwrap_or_default() as u16,
                    bpm: row["bpm"].as_f64().unwrap_or_default() as f32,
                    tags: serde_json::from_str(row["tags_json"].as_str().unwrap_or("[]"))?,
                },
            );
        }
        Ok(out)
    }

    fn fetch_all_metadata(&self, collection_id: &str) -> Result<Vec<(usize, VectorMetadata)>> {
        let rows = self.query_json(&format!(
            "SELECT vector_id, source_file, start_time_ms, duration_ms, bpm, tags_json FROM vector_metadata WHERE collection_id='{}' ORDER BY vector_id ASC;",
            sql_quote(collection_id)
        ))?;

        let mut out = Vec::new();
        for row in rows {
            out.push((
                row["vector_id"].as_i64().unwrap_or_default() as usize,
                VectorMetadata {
                    source_file: row["source_file"].as_str().unwrap_or_default().to_string(),
                    start_time_ms: row["start_time_ms"].as_i64().unwrap_or_default() as u32,
                    duration_ms: row["duration_ms"].as_i64().unwrap_or_default() as u16,
                    bpm: row["bpm"].as_f64().unwrap_or_default() as f32,
                    tags: serde_json::from_str(row["tags_json"].as_str().unwrap_or("[]"))?,
                },
            ));
        }
        Ok(out)
    }

    fn insert_orphan_file(
        &self,
        collection_id: &str,
        original_path: &Path,
        quarantine_path: &Path,
        reason: &str,
    ) -> Result<()> {
        self.exec(&format!(
            "INSERT INTO orphan_files(collection_id, original_path, quarantine_path, reason, quarantined_at, deleted_at) VALUES ('{}', '{}', '{}', '{}', {}, NULL);",
            sql_quote(collection_id),
            sql_quote(&original_path.to_string_lossy()),
            sql_quote(&quarantine_path.to_string_lossy()),
            sql_quote(reason),
            now_unix_ts()
        ))
    }

    fn quarantine_gc(&self, ttl_secs: i64) -> Result<Vec<PathBuf>> {
        let cutoff = now_unix_ts() - ttl_secs;
        let rows = self.query_json(&format!(
            "SELECT id, quarantine_path FROM orphan_files WHERE deleted_at IS NULL AND quarantined_at < {};",
            cutoff
        ))?;

        let mut deleted = Vec::new();
        for row in rows {
            let id = row["id"].as_i64().unwrap_or_default();
            let path = PathBuf::from(row["quarantine_path"].as_str().unwrap_or_default());
            if path.exists() {
                let _ = std::fs::remove_file(&path);
            }
            self.exec(&format!(
                "UPDATE orphan_files SET deleted_at = {} WHERE id = {};",
                now_unix_ts(),
                id
            ))?;
            deleted.push(path);
        }
        Ok(deleted)
    }

    fn total_vectors(&self, collection_id: &str) -> Result<usize> {
        let rows = self.query_json(&format!(
            "SELECT COUNT(*) AS n FROM vector_metadata WHERE collection_id='{}';",
            sql_quote(collection_id)
        ))?;
        Ok(rows
            .first()
            .and_then(|r| r["n"].as_i64())
            .unwrap_or_default() as usize)
    }

    fn audit_event(
        &self,
        request_id: &str,
        api_key_id: Option<&str>,
        endpoint: &str,
        action: &str,
        status_code: u16,
        latency_ms: f64,
        details: Value,
    ) -> Result<()> {
        self.exec(&format!(
            "INSERT INTO audit_events(ts, request_id, api_key_id, endpoint, action, status_code, latency_ms, client_ip, details_json) VALUES ({}, '{}', {}, '{}', '{}', {}, {}, NULL, '{}');",
            now_unix_ts(),
            sql_quote(request_id),
            api_key_id
                .map(|v| format!("'{}'", sql_quote(v)))
                .unwrap_or_else(|| "NULL".to_string()),
            sql_quote(endpoint),
            sql_quote(action),
            status_code,
            latency_ms,
            sql_quote(&details.to_string())
        ))
    }
}

impl SqliteCatalog {
    fn query_wal(&self, sql: &str) -> Result<Vec<WalEntry>> {
        let rows = self.query_json(sql)?;
        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            let vector: Vec<f32> =
                serde_json::from_str(row["vector_json"].as_str().unwrap_or("[]"))?;
            let metadata: VectorMetadata =
                serde_json::from_str(row["metadata_json"].as_str().unwrap_or("{}"))?;
            out.push(WalEntry {
                lsn: row["lsn"].as_i64().unwrap_or_default() as u64,
                vector_id: row["vector_id"].as_i64().unwrap_or_default() as usize,
                vector,
                metadata,
            });
        }
        Ok(out)
    }
}

fn now_unix_ts() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

fn sql_quote(value: &str) -> String {
    value.replace('\'', "''")
}

fn random_id(prefix: &str) -> String {
    let mut bytes = [0u8; 12];
    rand::thread_rng().fill_bytes(&mut bytes);
    format!("{}_{}{}", prefix, now_unix_ts(), hex_string(&bytes))
}

fn hex_string(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push_str(&format!("{:02x}", b));
    }
    out
}

fn hash_secret(pepper: &str, secret: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(pepper.as_bytes());
    hasher.update(b":");
    hasher.update(secret.as_bytes());
    let digest = hasher.finalize();
    hex_string(&digest)
}

pub fn verify_token_hash(pepper: &str, secret: &str, expected_hash: &str) -> bool {
    // Backward-compatible verification for pre-v2.1 short hashes.
    if expected_hash.len() == 16 {
        let legacy = legacy_hash(pepper, secret);
        return constant_time_eq(legacy.as_bytes(), expected_hash.as_bytes());
    }
    let actual = hash_secret(pepper, secret);
    constant_time_eq(actual.as_bytes(), expected_hash.as_bytes())
}

pub fn parse_token(header_value: &str) -> Option<(String, String)> {
    let token = header_value.trim();
    let token = token.strip_prefix("Bearer ")?;
    let (id, secret) = token.split_once('.')?;
    if !id.starts_with("vbk_") || secret.is_empty() {
        return None;
    }
    Some((id.to_string(), secret.to_string()))
}

pub fn audit_error_payload(code: &'static str, message: impl Into<String>) -> Value {
    json!({"code": code, "message": message.into()})
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

fn legacy_hash(pepper: &str, secret: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    pepper.hash(&mut hasher);
    secret.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}
