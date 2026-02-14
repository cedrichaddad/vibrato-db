use std::sync::Arc;

use tempfile::tempdir;
use vibrato_core::format_v2::VdbWriterV2;
use vibrato_db::prod::{
    bootstrap_data_dirs, recover_state, ProductionConfig, ProductionState, SqliteCatalog,
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
