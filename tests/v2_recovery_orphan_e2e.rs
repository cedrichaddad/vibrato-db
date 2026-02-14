use std::sync::Arc;

use tempfile::tempdir;
use vibrato_core::format_v2::VdbWriterV2;
use vibrato_db::prod::{
    bootstrap_data_dirs, recover_state, CatalogStore, ProductionConfig, ProductionState,
    SqliteCatalog,
};

#[test]
fn recover_state_quarantines_unregistered_segment_files() {
    let dir = tempdir().expect("tempdir");
    let data_dir = dir.path().join("recovery_orphan_data");
    let config = ProductionConfig::from_data_dir(data_dir, "default".to_string(), 2);
    bootstrap_data_dirs(&config).expect("bootstrap dirs");

    let orphan_path = config.segments_dir.join("orphan_segment.vdb");
    let mut writer = VdbWriterV2::new_raw(&orphan_path, 2).expect("create orphan segment");
    writer
        .write_vector(&[0.1, 0.9])
        .expect("write orphan segment vector");
    writer.finish().expect("finish orphan segment");

    let catalog = Arc::new(SqliteCatalog::open(&config.catalog_path()).expect("open catalog"));
    let state = ProductionState::initialize(config.clone(), catalog).expect("initialize state");
    let report = recover_state(&state).expect("recover state");

    assert_eq!(
        report.quarantined_files, 1,
        "orphan file should be quarantined"
    );
    assert!(
        !orphan_path.exists(),
        "orphan file should no longer remain in segments dir"
    );
    let quarantined = config.quarantine_dir.join("orphan_segment.vdb");
    assert!(
        quarantined.exists(),
        "orphan file should be moved to quarantine"
    );
}

#[test]
fn recover_state_quarantine_cap_bounds_retained_orphans() {
    let dir = tempdir().expect("tempdir");
    let data_dir = dir.path().join("recovery_orphan_cap");
    let mut config = ProductionConfig::from_data_dir(data_dir, "default".to_string(), 2);
    config.quarantine_max_files = 2;
    config.quarantine_max_bytes = 8 * 1024;
    bootstrap_data_dirs(&config).expect("bootstrap dirs");

    for i in 0..6usize {
        let orphan_path = config.segments_dir.join(format!("orphan_{i}.vdb"));
        let mut writer = VdbWriterV2::new_raw(&orphan_path, 2).expect("create orphan");
        writer
            .write_vector(&[i as f32 / 10.0, 1.0 - (i as f32 / 10.0)])
            .expect("write orphan vector");
        writer.finish().expect("finish orphan segment");
    }

    let catalog = Arc::new(SqliteCatalog::open(&config.catalog_path()).expect("open catalog"));
    let state = ProductionState::initialize(config.clone(), catalog.clone()).expect("init state");
    let _report = recover_state(&state).expect("recover state");

    let usage = catalog.quarantine_usage().expect("quarantine usage");
    assert!(
        usage.files <= config.quarantine_max_files,
        "quarantine files should be capped (files={}, cap={})",
        usage.files,
        config.quarantine_max_files
    );
    assert!(
        usage.bytes <= config.quarantine_max_bytes,
        "quarantine bytes should be capped (bytes={}, cap={})",
        usage.bytes,
        config.quarantine_max_bytes
    );
}
