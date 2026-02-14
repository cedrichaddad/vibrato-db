use std::collections::HashSet;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use vibrato_core::format_v2::VdbHeaderV2;
use vibrato_core::pq::ProductQuantizer;
use vibrato_core::store::VectorStore;

use super::catalog::CatalogStore;
use super::engine::{ProductionConfig, ProductionState};

#[derive(Debug, Clone)]
pub struct RecoveryReport {
    pub quarantined_files: usize,
    pub gc_deleted_files: usize,
    pub active_segments: usize,
    pub report: String,
}

pub fn bootstrap_data_dirs(config: &ProductionConfig) -> Result<()> {
    std::fs::create_dir_all(&config.data_dir)?;
    std::fs::create_dir_all(&config.segments_dir)?;
    std::fs::create_dir_all(&config.quarantine_dir)?;
    std::fs::create_dir_all(&config.snapshots_dir)?;
    std::fs::create_dir_all(&config.tmp_dir)?;
    std::fs::create_dir_all(&config.logs_dir)?;
    Ok(())
}

pub fn recover_state(state: &Arc<ProductionState>) -> Result<RecoveryReport> {
    let mut quarantined_files = 0usize;

    let known: HashSet<PathBuf> = state
        .catalog
        .list_known_segment_paths(&state.collection.id)?
        .into_iter()
        .collect();

    for entry in std::fs::read_dir(&state.config.segments_dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        if path.extension().and_then(|x| x.to_str()) != Some("vdb") {
            continue;
        }
        if known.contains(&path) {
            continue;
        }

        let _ = state.catalog.enforce_quarantine_cap(
            state.config.quarantine_max_files,
            state.config.quarantine_max_bytes,
        )?;
        let usage = state.catalog.quarantine_usage()?;
        let incoming_bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        let can_admit = usage.files < state.config.quarantine_max_files
            && usage.bytes.saturating_add(incoming_bytes) <= state.config.quarantine_max_bytes;
        if !can_admit {
            if let Err(err) = std::fs::remove_file(&path) {
                tracing::warn!(
                    "quarantine_overflow_drop_failed path={:?} err={}",
                    path,
                    err
                );
            } else {
                tracing::warn!(
                    "quarantine_overflow_drop path={:?} size_bytes={} max_files={} max_bytes={}",
                    path,
                    incoming_bytes,
                    state.config.quarantine_max_files,
                    state.config.quarantine_max_bytes
                );
            }
            continue;
        }

        let quarantine_path = state.config.quarantine_dir.join(
            path.file_name()
                .unwrap_or_else(|| std::ffi::OsStr::new("orphan.vdb")),
        );
        std::fs::rename(&path, &quarantine_path).with_context(|| {
            format!("moving orphan segment {:?} to {:?}", path, quarantine_path)
        })?;
        state.catalog.insert_orphan_file(
            &state.collection.id,
            &path,
            &quarantine_path,
            "unregistered_segment",
        )?;
        quarantined_files += 1;
        let _ = state.catalog.enforce_quarantine_cap(
            state.config.quarantine_max_files,
            state.config.quarantine_max_bytes,
        )?;
    }

    reconcile_inflight_jobs(state)?;
    let blockers = reconcile_segment_states(state)?;

    let gc_deleted = state
        .catalog
        .quarantine_gc(state.config.orphan_ttl.as_secs() as i64)?
        .len();

    state
        .load_active_segments_from_catalog()
        .context("loading active segments")?;
    state
        .rebuild_hot_from_pending()
        .context("rebuilding hot WAL tail")?;
    state
        .rebuild_filter_index()
        .context("rebuilding filter index")?;
    let gc_obsolete_deleted = state
        .gc_obsolete_segment_files()
        .context("garbage collecting obsolete segment files")?;
    if gc_obsolete_deleted > 0 {
        state.metrics.obsolete_files_deleted_total.fetch_add(
            gc_obsolete_deleted as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
    }

    let active_segments = state.segments.load().len() + state.archive_segments.load().len();
    let report = if blockers.is_empty() {
        format!(
            "recovered: active_segments={}, quarantined={}, gc_deleted={}, obsolete_deleted={}",
            active_segments, quarantined_files, gc_deleted, gc_obsolete_deleted
        )
    } else {
        format!(
            "degraded: active_segments={}, quarantined={}, gc_deleted={}, obsolete_deleted={}, blockers={}",
            active_segments,
            quarantined_files,
            gc_deleted,
            gc_obsolete_deleted,
            blockers.join(" | ")
        )
    };
    state.set_ready(blockers.is_empty(), report.clone());

    Ok(RecoveryReport {
        quarantined_files,
        gc_deleted_files: gc_deleted,
        active_segments,
        report,
    })
}

pub fn migrate_existing_vdb_to_segment(
    state: &Arc<ProductionState>,
    input_vdb: &Path,
    level: i64,
) -> Result<String> {
    use rand::RngCore;
    let mut bytes = [0u8; 8];
    rand::thread_rng().fill_bytes(&mut bytes);
    let mut suffix = String::new();
    for b in bytes {
        suffix.push_str(&format!("{:02x}", b));
    }
    let segment_id = format!(
        "seg_{}_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        suffix
    );
    let out_path = state
        .config
        .segments_dir
        .join(format!("{}.vdb", segment_id));
    std::fs::copy(input_vdb, &out_path)
        .with_context(|| format!("copying {:?} to {:?}", input_vdb, out_path))?;

    let store = vibrato_core::store::VectorStore::open(&out_path)?;
    let metadata = crate::server::load_store_metadata(&store);

    for (idx, item) in metadata.iter().enumerate() {
        state
            .catalog
            .upsert_metadata(&state.collection.id, idx, item)?;
    }

    state
        .catalog
        .insert_segment(&super::catalog::SegmentRecord {
            id: segment_id.clone(),
            collection_id: state.collection.id.clone(),
            level,
            path: out_path,
            row_count: store.count,
            vector_id_start: 0,
            vector_id_end: store.count.saturating_sub(1),
            created_lsn: 0,
            state: "active".to_string(),
        })?;

    Ok(segment_id)
}

fn reconcile_inflight_jobs(state: &Arc<ProductionState>) -> Result<()> {
    let checkpoint_jobs = state
        .catalog
        .list_checkpoint_jobs_by_state(&state.collection.id, &["building", "pending_activate"])?;
    for job in checkpoint_jobs {
        reconcile_checkpoint_job(state, &job.id, &job.details, job.start_lsn, job.end_lsn)?;
    }

    let compaction_jobs = state
        .catalog
        .list_compaction_jobs_by_state(&state.collection.id, &["building", "pending_activate"])?;
    for job in compaction_jobs {
        reconcile_compaction_job(state, &job.id, &job.details)?;
    }

    Ok(())
}

fn reconcile_segment_states(state: &Arc<ProductionState>) -> Result<Vec<String>> {
    let mut blockers = Vec::new();

    let active = state
        .catalog
        .list_segments_by_state(&state.collection.id, &["active"])?;
    for seg in active {
        if let Err(err) = validate_segment_file(&seg.path) {
            let _ = state.catalog.update_segment_state(&seg.id, "failed");
            blockers.push(format!("active segment {} invalid: {}", seg.id, err));
        }
    }

    let pending = state.catalog.list_segments_by_state(
        &state.collection.id,
        &["building", "pending_activate", "compacting"],
    )?;
    for seg in pending {
        if seg.state == "compacting" {
            match validate_segment_file(&seg.path) {
                Ok(()) => {
                    let _ = state.catalog.update_segment_state(&seg.id, "active");
                }
                Err(err) => {
                    let _ = state.catalog.update_segment_state(&seg.id, "failed");
                    blockers.push(format!("compacting segment {} invalid: {}", seg.id, err));
                }
            }
            continue;
        }
        match validate_segment_file(&seg.path) {
            Ok(()) => {
                let _ = state.catalog.update_segment_state(&seg.id, "active");
            }
            Err(err) => {
                let _ = state.catalog.update_segment_state(&seg.id, "failed");
                blockers.push(format!("pending segment {} invalid: {}", seg.id, err));
            }
        }
    }

    Ok(blockers)
}

fn reconcile_checkpoint_job(
    state: &Arc<ProductionState>,
    job_id: &str,
    details: &serde_json::Value,
    start_lsn: Option<u64>,
    end_lsn: Option<u64>,
) -> Result<()> {
    let Some(segment_id) = details
        .get("segment_id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
    else {
        state.catalog.update_checkpoint_job_state(
            job_id,
            "failed",
            serde_json::json!({"phase":"failed","error":"missing segment_id in details"}),
        )?;
        return Ok(());
    };

    let Some(seg) = state.catalog.get_segment(&segment_id)? else {
        state.catalog.update_checkpoint_job_state(
            job_id,
            "failed",
            serde_json::json!({"phase":"failed","segment_id":segment_id,"error":"segment row missing"}),
        )?;
        return Ok(());
    };

    match validate_segment_file(&seg.path) {
        Ok(()) => {
            if seg.state != "active" {
                state.catalog.update_segment_state(&seg.id, "active")?;
            }
            if let (Some(start), Some(end)) = (start_lsn, end_lsn) {
                state
                    .catalog
                    .mark_wal_checkpointed(&state.collection.id, start, end)?;
            }
            state.catalog.update_checkpoint_job_state(
                job_id,
                "completed",
                serde_json::json!({"phase":"completed","segment_id":segment_id}),
            )?;
        }
        Err(err) => {
            let _ = state.catalog.update_segment_state(&seg.id, "failed");
            state.catalog.update_checkpoint_job_state(
                job_id,
                "failed",
                serde_json::json!({"phase":"failed","segment_id":segment_id,"error":err.to_string()}),
            )?;
        }
    }
    Ok(())
}

fn reconcile_compaction_job(
    state: &Arc<ProductionState>,
    job_id: &str,
    details: &serde_json::Value,
) -> Result<()> {
    let Some(output_segment) = details
        .get("output_segment")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
    else {
        state.catalog.update_compaction_job_state(
            job_id,
            "failed",
            serde_json::json!({"phase":"failed","error":"missing output_segment in details"}),
        )?;
        return Ok(());
    };

    let input_segments = details
        .get("input_segments")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    let Some(seg) = state.catalog.get_segment(&output_segment)? else {
        state.catalog.update_compaction_job_state(
            job_id,
            "failed",
            serde_json::json!({"phase":"failed","output_segment":output_segment,"error":"segment row missing"}),
        )?;
        return Ok(());
    };

    match validate_segment_file(&seg.path) {
        Ok(()) => {
            if seg.state != "active" {
                state.catalog.update_segment_state(&seg.id, "active")?;
            }
            for input in input_segments {
                if let Some(id) = input.as_str() {
                    let _ = state.catalog.update_segment_state(id, "obsolete");
                }
            }
            state.catalog.update_compaction_job_state(
                job_id,
                "completed",
                serde_json::json!({"phase":"completed","output_segment":output_segment}),
            )?;
        }
        Err(err) => {
            let _ = state.catalog.update_segment_state(&seg.id, "failed");
            state.catalog.update_compaction_job_state(
                job_id,
                "failed",
                serde_json::json!({"phase":"failed","output_segment":output_segment,"error":err.to_string()}),
            )?;
        }
    }
    Ok(())
}

fn validate_segment_file(path: &Path) -> Result<()> {
    if !path.exists() {
        anyhow::bail!("segment file {:?} missing", path);
    }

    let mut file = File::open(path)?;
    let mut header_bytes = [0u8; 64];
    Read::read_exact(&mut file, &mut header_bytes)?;
    let header = VdbHeaderV2::from_bytes(&header_bytes)?;
    if header.is_pq_enabled() {
        validate_archive_pq_segment(path, header)?;
    } else {
        let _ = VectorStore::open(path)?;
    }
    Ok(())
}

fn validate_archive_pq_segment(path: &Path, header: VdbHeaderV2) -> Result<()> {
    if header.pq_subspaces == 0 {
        anyhow::bail!("pq-enabled segment {:?} has zero subspaces", path);
    }
    if header.codebook_offset == 0 {
        anyhow::bail!("pq-enabled segment {:?} missing codebook offset", path);
    }

    let bytes = std::fs::read(path)?;
    let nsub = header.pq_subspaces as usize;
    let code_len = header.count as usize * nsub;
    let codes_start = header.vectors_offset as usize;
    let codes_end = codes_start + code_len;
    if codes_end > bytes.len() {
        anyhow::bail!(
            "pq codes out of bounds in {:?}: {}..{} > {}",
            path,
            codes_start,
            codes_end,
            bytes.len()
        );
    }

    let codebook_start = header.codebook_offset as usize;
    let codebook_end = if header.metadata_offset > 0 {
        header.metadata_offset as usize
    } else if header.graph_offset > 0 {
        header.graph_offset as usize
    } else {
        bytes.len()
    };
    if codebook_end <= codebook_start || codebook_end > bytes.len() {
        anyhow::bail!(
            "invalid pq codebook range in {:?}: {}..{}",
            path,
            codebook_start,
            codebook_end
        );
    }

    let _ = ProductQuantizer::from_codebook_bytes(
        header.dimensions as usize,
        nsub,
        &bytes[codebook_start..codebook_end],
    )?;
    Ok(())
}
