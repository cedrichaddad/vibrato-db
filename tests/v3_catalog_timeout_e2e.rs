use std::path::PathBuf;

use tempfile::tempdir;
use vibrato_db::prod::{
    bootstrap_data_dirs, CatalogOptions, CatalogStore, IngestMetadataV3Input, ProductionConfig,
    SqliteCatalog,
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

    // Keep this test deterministic across debug/release:
    // release builds can decode 20k small rows fast enough to beat 5ms.
    let (row_count, read_timeout_ms, payload_len) = if cfg!(debug_assertions) {
        (20_000usize, 5u64, 0usize)
    } else {
        (80_000usize, 2u64, 64usize)
    };

    for i in 0..row_count {
        let meta = IngestMetadataV3Input {
            entity_id: i as u64,
            sequence_ts: i as u64,
            tags: vec!["drums".to_string()],
            payload: vec![0u8; payload_len],
        };
        let _ = catalog
            .ingest_wal_atomic(
                &collection.id,
                &[
                    i as f32 / row_count as f32,
                    1.0 - (i as f32 / row_count as f32),
                ],
                &meta,
                Some(&format!("timeout-key-{i}")),
            )
            .expect("ingest");
    }

    let tight_timeout = SqliteCatalog::open_with_options(
        &config.catalog_path(),
        CatalogOptions {
            // Extremely tight timeout to force deterministic timeout behavior in both
            // debug and optimized release binaries.
            read_timeout_ms,
            wal_autocheckpoint_pages: 1000,
            max_tag_registry_size: 500_000,
        },
    );

    match tight_timeout {
        Ok(catalog) => {
            let err = match catalog.fetch_all_metadata(&collection.id) {
                Ok(rows) => panic!(
                    "metadata scan should exceed configured timeout (rows_returned={})",
                    rows.len()
                ),
                Err(err) => err,
            };
            assert!(
                err.to_string()
                    .to_ascii_lowercase()
                    .contains("catalog_read_timeout"),
                "expected catalog_read_timeout error, got: {err}"
            );
            assert!(
                catalog.read_timeout_total() >= 1,
                "catalog read timeout counter should increment"
            );
        }
        Err(err) => {
            // On slower/contended hosts, the same tight timeout can trigger during
            // catalog bootstrap reads before `fetch_all_metadata` runs.
            assert!(
                err.to_string()
                    .to_ascii_lowercase()
                    .contains("catalog_read_timeout"),
                "expected catalog_read_timeout during open, got: {err}"
            );
        }
    }
}
