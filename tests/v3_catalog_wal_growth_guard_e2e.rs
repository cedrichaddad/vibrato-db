use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use tempfile::tempdir;
use vibrato_db::prod::{
    bootstrap_data_dirs, CatalogOptions, CatalogStore, IngestMetadataV3Input, ProductionConfig,
    SqliteCatalog,
};

fn test_config(data_dir: PathBuf) -> ProductionConfig {
    let mut cfg = ProductionConfig::from_data_dir(data_dir, "default".to_string(), 2);
    cfg.public_health_metrics = true;
    cfg.sqlite_wal_autocheckpoint_pages = 32;
    cfg
}

#[test]
fn sqlite_wal_growth_stays_bounded_under_timeout_prone_reads() {
    let dir = tempdir().expect("tempdir");
    let config = test_config(dir.path().join("catalog_wal_guard"));
    bootstrap_data_dirs(&config).expect("bootstrap dirs");

    let writer = Arc::new(
        SqliteCatalog::open_with_options(
            &config.catalog_path(),
            CatalogOptions {
                read_timeout_ms: 500,
                wal_autocheckpoint_pages: config.sqlite_wal_autocheckpoint_pages,
                max_tag_registry_size: 500_000,
            },
        )
        .expect("open writer"),
    );
    let collection = writer
        .ensure_collection(&config.collection_name, config.dim)
        .expect("ensure collection");

    // Open the low-timeout read handle before heavy preload, so startup migration work
    // never consumes the read timeout budget intended for the scan pressure phase.
    let read_catalog = Arc::new(
        SqliteCatalog::open_with_options(
            &config.catalog_path(),
            CatalogOptions {
                read_timeout_ms: 5,
                wal_autocheckpoint_pages: config.sqlite_wal_autocheckpoint_pages,
                max_tag_registry_size: 500_000,
            },
        )
        .expect("open read catalog"),
    );

    // Preload enough metadata so tight-timeout scans are still timeout-prone in read thread.
    const PRELOAD_ROWS: usize = 12_000;
    for i in 0..PRELOAD_ROWS {
        let meta = IngestMetadataV3Input {
            entity_id: i as u64,
            sequence_ts: i as u64,
            tags: vec!["drums".to_string(), "wal_guard".to_string()],
            payload: Vec::new(),
        };
        let _ = writer
            .ingest_wal_atomic(
                &collection.id,
                &[
                    i as f32 / PRELOAD_ROWS as f32,
                    1.0 - (i as f32 / PRELOAD_ROWS as f32),
                ],
                &meta,
                Some(&format!("preload-{i}")),
            )
            .expect("preload ingest");
    }
    let wal_bytes_before_pressure = writer.sqlite_wal_bytes();

    let stop = Arc::new(AtomicBool::new(false));
    let stop_reader = Arc::clone(&stop);
    let read_catalog_thread = Arc::clone(&read_catalog);
    let collection_id = collection.id.clone();
    let reader = thread::spawn(move || {
        while !stop_reader.load(Ordering::Relaxed) {
            let _ = read_catalog_thread.fetch_all_metadata(&collection_id);
            thread::sleep(Duration::from_millis(1));
        }
    });

    const WRITE_ROWS: usize = 1_200;
    for i in 0..WRITE_ROWS {
        let base = i as f32 / WRITE_ROWS as f32;
        let meta = IngestMetadataV3Input {
            entity_id: i as u64,
            sequence_ts: i as u64,
            tags: vec!["writes".to_string(), "wal_guard".to_string()],
            payload: Vec::new(),
        };
        let _ = writer
            .ingest_wal_atomic(
                &collection.id,
                &[base, 1.0 - base],
                &meta,
                Some(&format!("write-pressure-{i}")),
            )
            .expect("write-pressure ingest");
        if i % 1000 == 0 {
            thread::sleep(Duration::from_millis(2));
        }
    }

    stop.store(true, Ordering::Relaxed);
    reader.join().expect("join reader thread");

    // Guardrail: timeout-prone reads should not pin a long-lived reader and let WAL grow unbounded.
    let wal_bytes = writer.sqlite_wal_bytes();
    let wal_growth = wal_bytes.saturating_sub(wal_bytes_before_pressure);
    assert!(
        wal_growth <= 96 * 1024 * 1024,
        "sqlite WAL growth exceeded guardrail (growth={} bytes > 96MiB, before={}, after={})",
        wal_growth,
        wal_bytes_before_pressure,
        wal_bytes
    );

    assert!(
        read_catalog.read_timeout_total() > 0,
        "expected at least one catalog_read_timeout in guard test"
    );
}
