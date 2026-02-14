use std::path::PathBuf;

use tempfile::tempdir;
use vibrato_core::metadata::VectorMetadata;
use vibrato_db::prod::catalog::{CheckpointJobRecord, CompactionJobRecord, SegmentRecord};
use vibrato_db::prod::{bootstrap_data_dirs, CatalogStore, ProductionConfig, SqliteCatalog};

fn test_config(data_dir: PathBuf) -> ProductionConfig {
    let mut cfg = ProductionConfig::from_data_dir(data_dir, "default".to_string(), 2);
    cfg.public_health_metrics = true;
    cfg
}

#[test]
fn checkpoint_phase_transitions_update_segment_wal_and_job() {
    let dir = tempdir().expect("tempdir");
    let config = test_config(dir.path().join("catalog_protocol_chk"));
    bootstrap_data_dirs(&config).expect("bootstrap dirs");

    let catalog = SqliteCatalog::open(&config.catalog_path()).expect("open catalog");
    let collection = catalog
        .ensure_collection(&config.collection_name, config.dim)
        .expect("ensure collection");

    let mut lsn_start = 0u64;
    let mut lsn_end = 0u64;
    for i in 0..3usize {
        let meta = VectorMetadata {
            source_file: format!("sample-{}.wav", i),
            start_time_ms: (i * 10) as u32,
            duration_ms: 100,
            bpm: 120.0,
            tags: vec!["drums".to_string()],
        };
        let idempotency = format!("chk-{}", i);
        let wal = catalog
            .ingest_wal(
                &collection.id,
                i,
                &[i as f32, 1.0 - (i as f32)],
                &meta,
                Some(&idempotency),
            )
            .expect("insert wal");
        let lsn = wal.lsn.expect("lsn");
        if i == 0 {
            lsn_start = lsn;
        }
        lsn_end = lsn;
    }

    let segment_id = "seg_checkpoint_proto";
    let segment_path = config.segments_dir.join("seg_checkpoint_proto.vdb");
    catalog
        .insert_segment(&SegmentRecord {
            id: segment_id.to_string(),
            collection_id: collection.id.clone(),
            level: 0,
            path: segment_path,
            row_count: 3,
            vector_id_start: 0,
            vector_id_end: 2,
            created_lsn: lsn_end,
            state: "building".to_string(),
        })
        .expect("insert segment");

    let job_id = "chk_job_proto";
    catalog
        .upsert_checkpoint_job(&CheckpointJobRecord {
            id: job_id.to_string(),
            collection_id: collection.id.clone(),
            state: "building".to_string(),
            start_lsn: Some(lsn_start),
            end_lsn: Some(lsn_end),
            details: serde_json::json!({"phase":"start"}),
            created_at: 0,
            updated_at: 0,
        })
        .expect("upsert checkpoint job");

    catalog
        .checkpoint_mark_pending_activate(
            &collection.id,
            segment_id,
            job_id,
            lsn_start,
            lsn_end,
            serde_json::json!({"phase":"pending_activate"}),
        )
        .expect("checkpoint pending_activate transition");

    assert_eq!(
        catalog
            .list_segments_by_state(&collection.id, &["pending_activate"])
            .expect("list pending segments")
            .len(),
        1
    );
    assert_eq!(
        catalog
            .count_wal_pending(&collection.id)
            .expect("count wal pending"),
        0
    );
    assert_eq!(
        catalog
            .list_checkpoint_jobs_by_state(&collection.id, &["pending_activate"])
            .expect("list pending checkpoint jobs")
            .len(),
        1
    );

    catalog
        .checkpoint_activate(segment_id, job_id, serde_json::json!({"phase":"completed"}))
        .expect("checkpoint activate transition");

    assert_eq!(
        catalog
            .list_segments_by_state(&collection.id, &["active"])
            .expect("list active segments")
            .len(),
        1
    );
    assert_eq!(
        catalog
            .list_checkpoint_jobs_by_state(&collection.id, &["completed"])
            .expect("list completed checkpoint jobs")
            .len(),
        1
    );
}

#[test]
fn compaction_activation_is_atomic_for_output_and_inputs() {
    let dir = tempdir().expect("tempdir");
    let config = test_config(dir.path().join("catalog_protocol_cmp"));
    bootstrap_data_dirs(&config).expect("bootstrap dirs");

    let catalog = SqliteCatalog::open(&config.catalog_path()).expect("open catalog");
    let collection = catalog
        .ensure_collection(&config.collection_name, config.dim)
        .expect("ensure collection");

    let in_a = "seg_in_a";
    let in_b = "seg_in_b";
    let out = "seg_out";

    for seg_id in [in_a, in_b] {
        catalog
            .insert_segment(&SegmentRecord {
                id: seg_id.to_string(),
                collection_id: collection.id.clone(),
                level: 1,
                path: config.segments_dir.join(format!("{}.vdb", seg_id)),
                row_count: 10,
                vector_id_start: 0,
                vector_id_end: 9,
                created_lsn: 0,
                state: "active".to_string(),
            })
            .expect("insert input segment");
    }
    catalog
        .insert_segment(&SegmentRecord {
            id: out.to_string(),
            collection_id: collection.id.clone(),
            level: 2,
            path: config.segments_dir.join("seg_out.vdb"),
            row_count: 20,
            vector_id_start: 0,
            vector_id_end: 19,
            created_lsn: 0,
            state: "building".to_string(),
        })
        .expect("insert output segment");

    let job_id = "cmp_job_proto";
    catalog
        .upsert_compaction_job(&CompactionJobRecord {
            id: job_id.to_string(),
            collection_id: collection.id.clone(),
            state: "building".to_string(),
            details: serde_json::json!({"phase":"start"}),
            created_at: 0,
            updated_at: 0,
        })
        .expect("upsert compaction job");

    catalog
        .compaction_mark_pending_activate(
            out,
            job_id,
            serde_json::json!({"phase":"pending_activate"}),
        )
        .expect("mark pending activate");
    assert_eq!(
        catalog
            .list_segments_by_state(&collection.id, &["pending_activate"])
            .expect("list pending segments")
            .len(),
        1
    );
    assert_eq!(
        catalog
            .list_compaction_jobs_by_state(&collection.id, &["pending_activate"])
            .expect("list pending jobs")
            .len(),
        1
    );

    let inputs = vec![in_a.to_string(), in_b.to_string()];
    catalog
        .compaction_activate(
            out,
            &inputs,
            job_id,
            serde_json::json!({"phase":"completed"}),
        )
        .expect("compaction activate");

    let active = catalog
        .list_segments_by_state(&collection.id, &["active"])
        .expect("list active");
    assert!(
        active.iter().any(|s| s.id == out),
        "output segment should be active"
    );
    let obsolete = catalog
        .list_segments_by_state(&collection.id, &["obsolete"])
        .expect("list obsolete");
    assert_eq!(obsolete.len(), 2, "both input segments should be obsolete");
    assert_eq!(
        catalog
            .list_compaction_jobs_by_state(&collection.id, &["completed"])
            .expect("list completed jobs")
            .len(),
        1
    );
}
