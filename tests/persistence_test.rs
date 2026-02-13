use std::sync::Arc;
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use parking_lot::RwLock;
use tempfile::tempdir;
use tower::ServiceExt; // for oneshot
use vibrato_db::hnsw::HNSW;
use vibrato_db::server::{create_router, AppState};
use vibrato_db::store::VectorStore;
use vibrato_neural::inference::InferenceEngine;

#[tokio::test]
async fn test_persistence_flow() {
    // 1. Setup Environment
    let dir = tempdir().unwrap();
    let vdb_path = dir.path().join("index.vdb");
    // let idx_path = ... removed
    // ...
    let _guard = TestDirGuard::new(dir.path());

    // 2. Create Initial Empty .vdb
    {
        let dim = 2;
        let writer = vibrato_db::format_v2::VdbWriterV2::new_raw(&vdb_path, dim).unwrap();
        writer.finish().unwrap(); // Empty V2 file
    }

    // 3. Initialize Server State
    let store = Arc::new(VectorStore::open(&vdb_path).unwrap());
    let shared_store = Arc::new(RwLock::new(store.clone()));
    let dynamic_store = Arc::new(RwLock::new(Vec::<Vec<f32>>::new()));
    
    // Empty HNSW
    // We need to match accessor signature
    let accessor_store = shared_store.clone();
    let accessor_dynamic = dynamic_store.clone();
    let accessor = move |id| {
        let store = accessor_store.read();
        let store_ref = store.as_ref();
        if id < store_ref.count {
            store_ref.get(id).to_vec()
        } else {
            let offset = id - store_ref.count;
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
        inference: None,
        flush_mutex: RwLock::new(()),
    });

    let router = create_router(state.clone());

    // 4. Ingest Vector (Dynamic)
    // Vector: [0.1, 0.2]
    let ingest_body = serde_json::json!({
        "vector": [0.1, 0.2]
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
    assert_eq!(state.store.read().count, 0);

    // 5. Flush
    // This will write "index.vdb" in current directory (which is temp dir due to guard)
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
    assert_eq!(state.store.read().count, 1);
    
    // Verify file content
    let stored_vec = state.store.read().get(0).to_vec();
    assert_eq!(stored_vec, vec![0.1, 0.2]);

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
    file.seek(std::io::SeekFrom::Start(header.graph_offset)).unwrap();
    let mut reader = std::io::BufReader::new(file);
    
    // Load HNSW
    // We need a fresh accessor because safe reload usually implies fresh start
    // But for test we reuse existing stores logic or create new ones?
    // Let's create new ones to be sure.
    let new_store = Arc::new(VectorStore::open(&vdb_path).unwrap());
    let new_shared = Arc::new(RwLock::new(new_store.clone()));
    let new_dynamic = Arc::new(RwLock::new(Vec::<Vec<f32>>::new())); // Empty dynamically
    
    let new_accessor_store = new_shared.clone();
    let new_accessor_dynamic = new_dynamic.clone();
    let new_accessor = move |id| {
        let store = new_accessor_store.read();
        let store_ref = store.as_ref();
        if id < store_ref.count {
            store_ref.get(id).to_vec()
        } else {
            let offset = id - store_ref.count;
            new_accessor_dynamic.read()[offset].clone()
        }
    };

    let loaded_hnsw = HNSW::load_from_reader(&mut reader, new_accessor).unwrap();
    
    // 8. Verify Data
    let results = loaded_hnsw.search(&[0.1, 0.2], 1, 10);
    assert!(!results.is_empty());
    assert_eq!(results[0].0, 0); // ID 0 (was ingested first)
    // Note: ID 0 in new store is the vector we ingested.
    // Ingested ID was 0 because store was empty.
}

struct TestDirGuard {
    original: std::path::PathBuf,
}

impl TestDirGuard {
    fn new(path: &std::path::Path) -> Self {
        let original = std::env::current_dir().unwrap();
        std::env::set_current_dir(path).unwrap();
        Self { original }
    }
}

impl Drop for TestDirGuard {
    fn drop(&mut self) {
        let _ = std::env::set_current_dir(&self.original);
    }
}
