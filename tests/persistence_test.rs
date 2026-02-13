use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use parking_lot::RwLock;
use std::sync::Arc;
use tempfile::tempdir;
use tower::ServiceExt; // for oneshot
use vibrato_core::metadata::VectorMetadata;
use vibrato_db::hnsw::HNSW;
use vibrato_db::server::{create_router, load_store_metadata, AppState};
use vibrato_db::store::VectorStore;

#[tokio::test]
async fn test_persistence_integrity() {
    // 1. Setup Environment
    let dir = tempdir().unwrap();
    let vdb_path = dir.path().join("index.vdb");
    // let idx_path = ... removed
    // ...
    // let _guard = TestDirGuard::new(dir.path()); // Removed per review

    // 2. Create Initial Empty .vdb
    {
        let dim = 2;
        let writer = vibrato_db::format_v2::VdbWriterV2::new_raw(&vdb_path, dim).unwrap();
        writer.finish().unwrap(); // Empty V2 file
    }

    // 3. Initialize Server State
    let store = Arc::new(VectorStore::open(&vdb_path).unwrap());
    let shared_store = Arc::new(arc_swap::ArcSwap::from(store.clone()));
    let dynamic_store = Arc::new(RwLock::new(Vec::<Vec<f32>>::new()));
    let dynamic_metadata = Arc::new(RwLock::new(Vec::<VectorMetadata>::new()));
    let persisted_metadata = Arc::new(arc_swap::ArcSwap::from(Arc::new(
        Vec::<VectorMetadata>::new(),
    )));

    // Empty HNSW
    // We need to match accessor signature
    let accessor_store = shared_store.clone();
    let accessor_dynamic = dynamic_store.clone();
    let accessor = move |id| {
        let store_guard = accessor_store.load();
        if id < store_guard.count {
            store_guard.get(id).to_vec()
        } else {
            let offset = id - store_guard.count;
            accessor_dynamic.read()[offset].clone()
        }
    };

    // Just mock inference (None for test)
    // let inference = ... removed

    let hnsw = HNSW::new(16, 100, accessor);
    let state = Arc::new(AppState {
        index: RwLock::new(hnsw),
        store: shared_store.clone(),
        dynamic_store: dynamic_store.clone(),
        persisted_metadata,
        dynamic_metadata: dynamic_metadata.clone(),
        inference: None,
        flush_mutex: RwLock::new(()),
        data_path: vdb_path.clone(),
    });

    let router = create_router(state.clone());

    // 4. Ingest Vector (Dynamic)
    // Vector: [0.1, 0.2]
    let ingest_body = serde_json::json!({
        "vector": [0.1, 0.2],
        "metadata": {
            "source_file": "demo/snare.wav",
            "bpm": 128.0,
            "tags": ["drums", "snare"]
        }
    });
    let req = Request::builder()
        .method("POST")
        .uri("/ingest")
        .header("Content-Type", "application/json")
        .body(Body::from(serde_json::to_string(&ingest_body).unwrap()))
        .unwrap();

    let resp = router.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);

    // Verify it is in dynamic store
    assert_eq!(state.dynamic_store.read().len(), 1);
    // Note: Store count is handled via shared pointer, which we haven't swapped yet.
    assert_eq!(state.store.load().count, 0);

    // 5. Flush
    // This will write to vdb_path
    let req = Request::builder()
        .method("POST")
        .uri("/flush")
        .body(Body::empty())
        .unwrap();

    let resp = router.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // 6. Verify Persistence
    // Dynamic store should be empty
    assert_eq!(state.dynamic_store.read().len(), 0);
    // Store should have 1 vector
    assert_eq!(state.store.load().count, 1);

    // Verify file content
    let stored_vec = state.store.load().get(0).to_vec();
    assert_eq!(stored_vec, vec![0.1, 0.2]);

    // Metadata should have been persisted and hot-swapped too.
    let persisted = state.persisted_metadata.load();
    assert_eq!(persisted.len(), 1);
    assert_eq!(persisted[0].source_file, "demo/snare.wav");
    assert!((persisted[0].bpm - 128.0).abs() < 1e-6);
    assert_eq!(
        persisted[0].tags,
        vec!["drums".to_string(), "snare".to_string()]
    );

    // 7. Restart / Reload HNSW (Simulate server restart)
    // Read header to find graph offset
    let mut file = std::fs::File::open(&vdb_path).unwrap();

    // Read first 1KB for header
    let mut buffer = vec![0u8; 1024];
    use std::io::Read;
    file.read(&mut buffer).unwrap();
    let header = vibrato_db::format_v2::VdbHeaderV2::from_bytes(&buffer).unwrap();

    // Seek to graph
    use std::io::Seek;
    file.seek(std::io::SeekFrom::Start(header.graph_offset))
        .unwrap();
    let mut reader = std::io::BufReader::new(file);

    // Load HNSW
    // We need a fresh accessor because safe reload usually implies fresh start
    // But for test we reuse existing stores logic or create new ones?
    // Let's create new ones to be sure.
    let new_store = Arc::new(VectorStore::open(&vdb_path).unwrap());
    let new_shared = Arc::new(arc_swap::ArcSwap::from(new_store.clone()));
    let new_dynamic = Arc::new(RwLock::new(Vec::<Vec<f32>>::new())); // Empty dynamically

    let new_accessor_store = new_shared.clone();
    let new_accessor_dynamic = new_dynamic.clone();
    let new_accessor = move |id| {
        let store_guard = new_accessor_store.load();
        if id < store_guard.count {
            store_guard.get(id).to_vec()
        } else {
            let offset = id - store_guard.count;
            new_accessor_dynamic.read()[offset].clone()
        }
    };

    let loaded_hnsw = HNSW::load_from_reader(&mut reader, new_accessor).unwrap();

    // 8. Verify Data
    let results = loaded_hnsw.search(&[0.1, 0.2], 1, 10);
    assert!(!results.is_empty());
    assert_eq!(results[0].0, 0); // ID 0 (was ingested first)

    let reloaded_metadata = load_store_metadata(&new_store);
    assert_eq!(reloaded_metadata.len(), 1);
    assert_eq!(reloaded_metadata[0].source_file, "demo/snare.wav");
    assert!((reloaded_metadata[0].bpm - 128.0).abs() < 1e-6);
    assert_eq!(
        reloaded_metadata[0].tags,
        vec!["drums".to_string(), "snare".to_string()]
    );
}
