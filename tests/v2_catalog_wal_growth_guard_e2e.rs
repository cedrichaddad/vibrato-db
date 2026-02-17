use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use tempfile::tempdir;
use vibrato_core::metadata::VectorMetadata;
use vibrato_db::prod::{
    bootstrap_data_dirs, CatalogOptions, CatalogStore, ProductionConfig, SqliteCatalog,
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
            },
        )
        .expect("open writer"),
    );
    let collection = writer
        .ensure_collection(&config.collection_name, config.dim)
        .expect("ensure collection");

    // Preload enough metadata so 1ms scans reliably timeout in read thread.
    for i in 0..5000usize {
        let meta = VectorMetadata {
            source_file: format!("preload-{i}.wav"),
            start_time_ms: i as u32,
            duration_ms: 100,
            bpm: 120.0,
            tags: vec!["drums".to_string(), "wal_guard".to_string()],
        };
        let _ = writer
            .ingest_wal_atomic(
                &collection.id,
                &[i as f32 / 5000.0, 1.0 - (i as f32 / 5000.0)],
                &meta,
                Some(&format!("preload-{i}")),
            )
            .expect("preload ingest");
    }

    let read_catalog = Arc::new(
        SqliteCatalog::open_with_options(
            &config.catalog_path(),
            CatalogOptions {
                read_timeout_ms: 1,
                wal_autocheckpoint_pages: config.sqlite_wal_autocheckpoint_pages,
            },
        )
        .expect("open read catalog"),
    );

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

    for i in 0..15_000usize {
        let base = i as f32 / 15_000.0;
        let meta = VectorMetadata {
            source_file: format!("write-pressure-{i}.wav"),
            start_time_ms: i as u32,
            duration_ms: 80,
            bpm: 110.0 + ((i % 50) as f32),
            tags: vec!["writes".to_string(), "wal_guard".to_string()],
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
    assert!(
        wal_bytes <= 32 * 1024 * 1024,
        "sqlite WAL grew beyond guardrail ({} bytes > 32MiB)",
        wal_bytes
    );

    assert!(
        read_catalog.read_timeout_total() > 0,
        "expected at least one catalog_read_timeout in guard test"
    );
}
