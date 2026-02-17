use std::path::PathBuf;

use tempfile::tempdir;
use vibrato_core::metadata::VectorMetadata;
use vibrato_db::prod::{
    bootstrap_data_dirs, CatalogOptions, CatalogStore, ProductionConfig, SqliteCatalog,
};

fn test_config(data_dir: PathBuf) -> ProductionConfig {
    let mut cfg = ProductionConfig::from_data_dir(data_dir, "default".to_string(), 2);
    cfg.public_health_metrics = true;
    cfg
}

#[test]
fn catalog_read_timeout_bounds_slow_metadata_scan() {
    let dir = tempdir().expect("tempdir");
    let config = test_config(dir.path().join("catalog_timeout"));
    bootstrap_data_dirs(&config).expect("bootstrap dirs");

    let catalog = SqliteCatalog::open(&config.catalog_path()).expect("open catalog");
    let collection = catalog
        .ensure_collection(&config.collection_name, config.dim)
        .expect("ensure collection");

    for i in 0..20_000usize {
        let meta = VectorMetadata {
            source_file: format!("timeout-{i}.wav"),
            start_time_ms: i as u32,
            duration_ms: 100,
            bpm: 120.0,
            tags: vec!["drums".to_string()],
        };
        let _ = catalog
            .ingest_wal_atomic(
                &collection.id,
                &[i as f32 / 20_000.0, 1.0 - (i as f32 / 20_000.0)],
                &meta,
                Some(&format!("timeout-key-{i}")),
            )
            .expect("ingest");
    }

    let tight_timeout = SqliteCatalog::open_with_options(
        &config.catalog_path(),
        CatalogOptions {
            read_timeout_ms: 1,
            wal_autocheckpoint_pages: 1000,
        },
    )
    .expect("open catalog with tight timeout");

    let err = tight_timeout
        .fetch_all_metadata(&collection.id)
        .expect_err("metadata scan should exceed configured timeout");
    assert!(
        err.to_string()
            .to_ascii_lowercase()
            .contains("catalog_read_timeout"),
        "expected catalog_read_timeout error, got: {err}"
    );
    assert!(
        tight_timeout.read_timeout_total() >= 1,
        "catalog read timeout counter should increment"
    );
}
