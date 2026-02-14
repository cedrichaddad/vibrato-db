use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use rand::RngCore;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::catalog::{CatalogStore, SqliteCatalog};
use super::engine::ProductionConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotSegmentEntry {
    pub segment_id: String,
    pub source_path: String,
    pub snapshot_file: String,
    pub state: String,
    pub size_bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotManifest {
    pub snapshot_id: String,
    pub created_at: i64,
    pub collection: String,
    pub collection_id: String,
    pub catalog_file: String,
    pub catalog_sha256: String,
    pub segments: Vec<SnapshotSegmentEntry>,
}

#[derive(Debug, Clone)]
pub struct SnapshotResult {
    pub snapshot_id: String,
    pub snapshot_dir: PathBuf,
    pub segments: usize,
}

pub fn create_snapshot(
    config: &ProductionConfig,
    catalog: &SqliteCatalog,
    collection_name: &str,
) -> Result<SnapshotResult> {
    let collection = catalog
        .get_collection(collection_name)?
        .ok_or_else(|| anyhow!("collection '{}' not found", collection_name))?;

    let snapshot_id = make_snapshot_id();
    let snapshot_dir = config.snapshots_dir.join(&snapshot_id);
    let snapshot_segments_dir = snapshot_dir.join("segments");
    std::fs::create_dir_all(&snapshot_segments_dir)?;

    let catalog_dst = snapshot_dir.join("catalog.sqlite3");
    // Use SQLite-native snapshot to avoid torn copies under WAL mode.
    catalog.vacuum_into(&catalog_dst)?;

    let mut segments_out = Vec::new();
    let segments = catalog
        .list_segments_by_state(&collection.id, &["active", "pending_activate", "building"])?;
    for seg in segments {
        if !seg.path.exists() {
            continue;
        }
        let file_name = seg
            .path
            .file_name()
            .map(|v| v.to_string_lossy().to_string())
            .unwrap_or_else(|| format!("{}.vdb", seg.id));
        let dst = snapshot_segments_dir.join(&file_name);
        std::fs::copy(&seg.path, &dst)
            .with_context(|| format!("copying segment {:?} to {:?}", seg.path, dst))?;

        let size = std::fs::metadata(&dst)?.len();
        segments_out.push(SnapshotSegmentEntry {
            segment_id: seg.id,
            source_path: seg.path.to_string_lossy().to_string(),
            snapshot_file: file_name,
            state: seg.state,
            size_bytes: size,
            sha256: sha256_file(&dst)?,
        });
    }

    let manifest = SnapshotManifest {
        snapshot_id: snapshot_id.clone(),
        created_at: now_unix_ts(),
        collection: collection.name,
        collection_id: collection.id,
        catalog_file: "catalog.sqlite3".to_string(),
        catalog_sha256: sha256_file(&catalog_dst)?,
        segments: segments_out,
    };
    let manifest_path = snapshot_dir.join("manifest.json");
    let manifest_json = serde_json::to_string_pretty(&manifest)?;
    std::fs::write(&manifest_path, manifest_json.as_bytes())?;
    sync_file(&manifest_path)?;
    sync_dir(&snapshot_dir)?;

    Ok(SnapshotResult {
        snapshot_id,
        snapshot_dir,
        segments: manifest.segments.len(),
    })
}

pub fn restore_snapshot(config: &ProductionConfig, snapshot_dir: &Path) -> Result<()> {
    let manifest_path = snapshot_dir.join("manifest.json");
    let manifest: SnapshotManifest = serde_json::from_slice(&std::fs::read(&manifest_path)?)
        .with_context(|| format!("parsing snapshot manifest {:?}", manifest_path))?;

    let catalog_src = snapshot_dir.join(&manifest.catalog_file);
    if !catalog_src.exists() {
        return Err(anyhow!("snapshot catalog file missing: {:?}", catalog_src));
    }
    let catalog_hash = sha256_file(&catalog_src)?;
    if catalog_hash != manifest.catalog_sha256 {
        return Err(anyhow!(
            "catalog checksum mismatch for {:?}: expected {}, got {}",
            catalog_src,
            manifest.catalog_sha256,
            catalog_hash
        ));
    }

    let backup_root = config
        .tmp_dir
        .join(format!("restore_backup_{}", now_unix_ts()));
    std::fs::create_dir_all(backup_root.join("segments"))?;

    let catalog_dst = config.catalog_path();
    if catalog_dst.exists() {
        let backup_catalog = backup_root.join("catalog.sqlite3");
        std::fs::rename(&catalog_dst, &backup_catalog).with_context(|| {
            format!(
                "backing up catalog {:?} -> {:?}",
                catalog_dst, backup_catalog
            )
        })?;
    }

    for entry in std::fs::read_dir(&config.segments_dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let name = path
            .file_name()
            .map(|v| v.to_os_string())
            .unwrap_or_else(|| std::ffi::OsString::from("segment.vdb"));
        let backup = backup_root.join("segments").join(name);
        std::fs::rename(&path, &backup)?;
    }

    std::fs::copy(&catalog_src, &catalog_dst).with_context(|| {
        format!(
            "restoring catalog from snapshot {:?} -> {:?}",
            catalog_src, catalog_dst
        )
    })?;
    sync_file(&catalog_dst)?;

    for seg in &manifest.segments {
        let src = snapshot_dir.join("segments").join(&seg.snapshot_file);
        if !src.exists() {
            return Err(anyhow!("snapshot segment missing: {:?}", src));
        }
        let actual = sha256_file(&src)?;
        if actual != seg.sha256 {
            return Err(anyhow!(
                "segment checksum mismatch for {:?}: expected {}, got {}",
                src,
                seg.sha256,
                actual
            ));
        }
        let dst = config.segments_dir.join(&seg.snapshot_file);
        std::fs::copy(&src, &dst)?;
    }
    sync_dir(&config.segments_dir)?;
    Ok(())
}

pub fn replay_to_lsn(
    catalog: &SqliteCatalog,
    collection_name: &str,
    target_lsn: u64,
) -> Result<()> {
    let collection = catalog
        .get_collection(collection_name)?
        .ok_or_else(|| anyhow!("collection '{}' not found", collection_name))?;

    let collection_id = sql_quote(&collection.id);
    let sql = format!(
        "BEGIN IMMEDIATE;
         DELETE FROM wal_entries WHERE collection_id='{}' AND lsn > {};
         DELETE FROM vector_metadata WHERE collection_id='{}' AND vector_id NOT IN
           (SELECT vector_id FROM wal_entries WHERE collection_id='{}');
         UPDATE wal_entries SET checkpointed_at=NULL WHERE collection_id='{}';
         UPDATE segments SET state='obsolete' WHERE collection_id='{}' AND state='active';
         COMMIT;",
        collection_id, target_lsn, collection_id, collection_id, collection_id, collection_id
    );
    catalog.execute_sql(&sql)
}

fn now_unix_ts() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex_string(&hasher.finalize()))
}

fn hex_string(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push_str(&format!("{:02x}", b));
    }
    out
}

fn sync_file(path: &Path) -> Result<()> {
    File::open(path)?.sync_all()?;
    Ok(())
}

fn sync_dir(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        File::open(path)?.sync_all()?;
    }
    Ok(())
}

fn make_snapshot_id() -> String {
    let mut bytes = [0u8; 8];
    rand::thread_rng().fill_bytes(&mut bytes);
    let mut suffix = String::new();
    for b in bytes {
        suffix.push_str(&format!("{:02x}", b));
    }
    format!("snap_{}_{}", now_unix_ts(), suffix)
}

fn sql_quote(value: &str) -> String {
    value.replace('\'', "''")
}
