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

#[test]
fn pending_wal_replay_is_lsn_ordered() {
    let dir = tempdir().expect("tempdir");
    let config = test_config(dir.path().join("catalog_protocol_lsn"));
    bootstrap_data_dirs(&config).expect("bootstrap dirs");

    let catalog = SqliteCatalog::open(&config.catalog_path()).expect("open catalog");
    let collection = catalog
        .ensure_collection(&config.collection_name, config.dim)
        .expect("ensure collection");

    for i in 0..8usize {
        let meta = VectorMetadata {
            source_file: format!("lsn-{}.wav", i),
            start_time_ms: i as u32,
            duration_ms: 100,
            bpm: 100.0 + i as f32,
            tags: vec!["order".to_string()],
        };
        catalog
            .ingest_wal(
                &collection.id,
                i,
                &[i as f32, 1.0 - (i as f32 / 10.0)],
                &meta,
                Some(&format!("lsn-key-{}", i)),
            )
            .expect("ingest wal");
    }

    let pending = catalog
        .pending_wal_after_lsn(&collection.id, 0)
        .expect("pending wal");
    assert!(pending.len() >= 8);
    for pair in pending.windows(2) {
        assert!(
            pair[0].lsn <= pair[1].lsn,
            "WAL replay order must be monotonic by lsn"
        );
    }
}

#[test]
fn ingest_transaction_rolls_back_on_constraint_failure() {
    let dir = tempdir().expect("tempdir");
    let config = test_config(dir.path().join("catalog_protocol_atomic"));
    bootstrap_data_dirs(&config).expect("bootstrap dirs");

    let catalog = SqliteCatalog::open(&config.catalog_path()).expect("open catalog");
    let collection = catalog
        .ensure_collection(&config.collection_name, config.dim)
        .expect("ensure collection");

    let first = VectorMetadata {
        source_file: "first.wav".to_string(),
        start_time_ms: 0,
        duration_ms: 100,
        bpm: 120.0,
        tags: vec!["drums".to_string()],
    };
    catalog
        .ingest_wal(&collection.id, 7, &[0.1, 0.2], &first, Some("atomic-1"))
        .expect("first ingest");

    let second = VectorMetadata {
        source_file: "second.wav".to_string(),
        start_time_ms: 10,
        duration_ms: 120,
        bpm: 121.0,
        tags: vec!["snare".to_string()],
    };
    let err = catalog
        .ingest_wal(&collection.id, 7, &[0.3, 0.4], &second, Some("atomic-2"))
        .expect_err("duplicate vector_id should fail");
    assert!(
        err.to_string().to_ascii_lowercase().contains("constraint"),
        "expected constraint error, got: {err}"
    );

    let wal = catalog
        .pending_wal_after_lsn(&collection.id, 0)
        .expect("pending wal");
    assert_eq!(wal.len(), 1, "failed ingest must not append WAL row");
    assert_eq!(wal[0].vector_id, 7);
    assert_eq!(wal[0].metadata.source_file, "first.wav");

    let metadata = catalog
        .fetch_all_metadata(&collection.id)
        .expect("fetch all metadata");
    assert_eq!(metadata.len(), 1, "failed ingest must not write metadata");
    assert_eq!(metadata[0].0, 7);
    assert_eq!(metadata[0].1.source_file, "first.wav");
}

#[test]
fn monotonic_id_counter_is_atomic_with_wal_and_metadata() {
    let dir = tempdir().expect("tempdir");
    let config = test_config(dir.path().join("catalog_protocol_counter"));
    bootstrap_data_dirs(&config).expect("bootstrap dirs");

    let catalog = SqliteCatalog::open(&config.catalog_path()).expect("open catalog");
    let collection = catalog
        .ensure_collection(&config.collection_name, config.dim)
        .expect("ensure collection");

    for i in 0..3usize {
        let meta = VectorMetadata {
            source_file: format!("counter-{i}.wav"),
            start_time_ms: i as u32,
            duration_ms: 100,
            bpm: 128.0,
            tags: vec!["drums".to_string()],
        };
        let res = catalog
            .ingest_wal_atomic(
                &collection.id,
                &[i as f32, 1.0 - (i as f32 / 10.0)],
                &meta,
                Some(&format!("counter-key-{i}")),
            )
            .expect("ingest_wal_atomic");
        assert!(res.created);
        assert_eq!(res.vector_id, i, "vector ids must be monotonic");
    }

    let dup = catalog
        .ingest_wal_atomic(
            &collection.id,
            &[0.2, 0.8],
            &VectorMetadata {
                source_file: "dup.wav".to_string(),
                start_time_ms: 0,
                duration_ms: 100,
                bpm: 128.0,
                tags: vec!["drums".to_string()],
            },
            Some("counter-key-1"),
        )
        .expect("idempotent duplicate lookup");
    assert!(
        !dup.created,
        "duplicate idempotency key must not create new row"
    );
    assert_eq!(dup.vector_id, 1);

    let next = catalog
        .next_vector_id(&collection.id)
        .expect("next vector id from counter");
    assert_eq!(next, 3, "counter should not regress or reuse ids");
}

#[test]
fn tag_dictionary_and_vector_tags_are_stable_across_restart() {
    let dir = tempdir().expect("tempdir");
    let config = test_config(dir.path().join("catalog_protocol_tags"));
    bootstrap_data_dirs(&config).expect("bootstrap dirs");

    let collection_id = {
        let catalog = SqliteCatalog::open(&config.catalog_path()).expect("open catalog");
        let collection = catalog
            .ensure_collection(&config.collection_name, config.dim)
            .expect("ensure collection");

        let meta_a = VectorMetadata {
            source_file: "a.wav".to_string(),
            start_time_ms: 0,
            duration_ms: 100,
            bpm: 120.0,
            tags: vec!["Drums".to_string(), "Snare".to_string()],
        };
        let meta_b = VectorMetadata {
            source_file: "b.wav".to_string(),
            start_time_ms: 10,
            duration_ms: 100,
            bpm: 121.0,
            tags: vec!["drums".to_string(), "Kick".to_string()],
        };
        catalog
            .ingest_wal_atomic(&collection.id, &[0.1, 0.9], &meta_a, Some("tags-a"))
            .expect("ingest a");
        catalog
            .ingest_wal_atomic(&collection.id, &[0.2, 0.8], &meta_b, Some("tags-b"))
            .expect("ingest b");

        collection.id
    };

    let reopened = SqliteCatalog::open(&config.catalog_path()).expect("reopen catalog");
    let dictionary = reopened
        .fetch_tag_dictionary(&collection_id)
        .expect("fetch tag dictionary");
    assert!(
        dictionary.contains_key("drums"),
        "normalized tag id for drums must persist"
    );
    assert!(
        dictionary.contains_key("snare"),
        "normalized tag id for snare must persist"
    );
    assert!(
        dictionary.contains_key("kick"),
        "normalized tag id for kick must persist"
    );

    let rows = reopened
        .fetch_filter_rows(&collection_id)
        .expect("fetch filter rows");
    assert_eq!(rows.len(), 2);
    assert!(
        rows.iter().all(|row| !row.tag_ids.is_empty()),
        "vector_tags join rows must be present for rebuild"
    );
}
