//! HTTP Server for Vibrato-DB
//!
//! Exposes the vector search engine over HTTP using axum.
//!
//! # Endpoints
//!
//! - `POST /search` - Query for nearest neighbors
//! - `GET /health` - Server health and telemetry

use std::sync::Arc;
use std::time::Instant;

use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tower_http::cors::{Any, CorsLayer};

use crate::hnsw::HNSW;
use crate::store::VectorStore;
use vibrato_neural::inference::InferenceEngine;

/// Thread-safe, swappable store handle
pub type SharedStore = Arc<RwLock<Arc<VectorStore>>>;

/// Shared application state
///
/// # Concurrency Model
///
/// Uses `parking_lot::RwLock` to allow multiple concurrent searches (readers)
/// while ensuring exclusive access for ingestion (writer).
///
/// - **Search**: Acquires read lock. Non-blocking for other searchers.
/// - **Ingest**: Acquires write lock. Blocks all searches until insertion completes.
///
/// This favors read-heavy workloads (typical for vector search).
pub struct AppState {
    pub index: RwLock<HNSW>,
    pub store: SharedStore,
    // Dynamic ingestion support: in-memory buffer for new vectors
    pub dynamic_store: Arc<RwLock<Vec<Vec<f32>>>>,
    pub inference: Option<Arc<InferenceEngine>>,
    // Global lock to coordinate flush vs ingest consistency
    pub flush_mutex: RwLock<()>,
}

/// Search request body
#[derive(Debug, Serialize, Deserialize)]
pub struct SearchRequest {
    /// Query vector (must match index dimensions)
    pub vector: Vec<f32>,

    /// Number of results to return (default: 10)
    #[serde(default = "default_k")]
    pub k: usize,

    /// Search depth (higher = better recall, slower; default: 50)
    #[serde(default = "default_ef")]
    pub ef: usize,
}

/// Ingest request body
#[derive(Debug, Serialize, Deserialize)]
pub struct IngestRequest {
    pub vector: Option<Vec<f32>>,
    pub audio_path: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct IngestResponse {
    pub id: usize,
    pub vector_dim: usize,
}


fn default_k() -> usize {
    10
}

fn default_ef() -> usize {
    50
}

/// Single search result
#[derive(Debug, Serialize, Deserialize)]
pub struct SearchResult {
    /// Vector ID
    pub id: usize,

    /// Similarity score (higher = more similar)
    pub score: f32,

    /// Optional metadata (TODO: add metadata store)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

/// Search response
#[derive(Debug, Serialize, Deserialize)]
pub struct SearchResponse {
    /// Search results
    pub results: Vec<SearchResult>,

    /// Query time in milliseconds
    pub query_time_ms: f64,
}

/// Health check response
#[derive(Debug, Serialize, Deserialize)]
pub struct HealthResponse {
    /// Server status
    pub status: String,

    /// Number of vectors loaded
    pub vectors_loaded: usize,

    /// Index statistics
    pub index_layers: usize,

    /// Memory usage in MB
    pub memory_mb: f64,

    /// Index configuration
    pub config: IndexConfig,
}

/// Index configuration
#[derive(Debug, Serialize, Deserialize)]
pub struct IndexConfig {
    pub m: usize,
    pub ef_construction: usize,
    pub dimensions: usize,
}

/// Error response
#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: String,
}

/// POST /search - Query for nearest neighbors
async fn search(
    State(state): State<Arc<AppState>>,
    Json(request): Json<SearchRequest>,
) -> impl IntoResponse {
    // Validate input
    let dim = state.store.read().dim;
    if request.vector.len() != dim {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: format!(
                    "Dimension mismatch: expected {}, got {}",
                    dim,
                    request.vector.len()
                ),
            }),
        )
            .into_response();
    }

    if request.k == 0 {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "k must be > 0".into(),
            }),
        )
            .into_response();
    }

    // Execute search
    let start = Instant::now();
    let results = {
        let index = state.index.read();
        index.search(&request.vector, request.k, request.ef)
    };
    let query_time_ms = start.elapsed().as_secs_f64() * 1000.0;

    // Format response
    let response = SearchResponse {
        results: results
            .into_iter()
            .map(|(id, score)| SearchResult {
                id,
                score,
                metadata: None,
            })
            .collect(),
        query_time_ms,
    };

    (StatusCode::OK, Json(response)).into_response()
}

/// POST /ingest - Add a vector to the index
///
/// Acquires a write lock on the index, blocking searches.
async fn ingest(
    State(state): State<Arc<AppState>>,
    Json(request): Json<IngestRequest>,
) -> impl IntoResponse {
    // Validate input
    // Ingest handler
    let ingest = |State(state): State<Arc<AppState>>, Json(payload): Json<IngestRequest>| async move {
        // 1. Get vector: either directly provided or via inference
        let vector = if let Some(vec) = payload.vector {
            vec
        } else if let Some(path_str) = payload.audio_path {
            if let Some(inference) = &state.inference {
                let path = std::path::Path::new(&path_str);
                match inference.embed_audio_file(path).await {
                    Ok(vec) => vec,
                    Err(e) => return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(IngestResponse { id: 0, vector_dim: 0 }),
                    ).into_response(),
                }
            } else {
                return (
                    StatusCode::NOT_IMPLEMENTED,
                    Json(IngestResponse { id: 0, vector_dim: 0 }),
                ).into_response();
            }
        } else {
            return (
                StatusCode::BAD_REQUEST,
                Json(IngestResponse { id: 0, vector_dim: 0 }),
            ).into_response();
        };

        // 2. Append to dynamic store (write lock)
        // Ensure consistency with flush: if flush is running, we wait.
        let _flush_read = state.flush_mutex.read();

        // Check dimensions again safely
        let store = state.store.read();
        let store_count = store.count;
        let store_dim = store.dim;
        drop(store);

        if vector.len() != store_dim {
             return (
                StatusCode::BAD_REQUEST,
                Json(IngestResponse { id: 0, vector_dim: 0 }),
            ).into_response();
        }

        let id_offset = store_count as usize;
        let mut dynamic = state.dynamic_store.write();
        let current_dynamic_count = dynamic.len();
        let new_id = id_offset + current_dynamic_count;
        dynamic.push(vector.clone()); // Store copy for retrieval
        drop(dynamic); // Release lock quickly

        // 3. Insert into HNSW (write lock)
        let mut index = state.index.write();
        index.insert(new_id); // This will call the closure
        
        (
            StatusCode::CREATED,
            Json(IngestResponse {
                id: new_id,
                vector_dim: store_dim,
            }),
        ).into_response()
    };
    
    ingest(State(state), Json(request)).await
}

/// POST /flush - Persist in-memory index to disk
///
/// 1. Takes write locks to stop the world.
/// 2. Merges static + dynamic stores to new file.
/// 3. Atomically swaps file and reloads store.
async fn flush(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    // 1. Acquire Locks (Global Flush Lock -> Dynamic -> Index -> Store)
    // We take flush_mutex (write) to ensure no ingests are running or starting.
    let _flush_guard = state.flush_mutex.write();
    
    let index_guard = state.index.read(); // Read lock is enough for serialization
    let mut dynamic_guard = state.dynamic_store.write(); // Write lock to clear it later
    let mut store_ptr_guard = state.store.write(); // Write lock to swap pointer

    // 2. Merge Logic
    let temp_path = std::path::Path::new("index.vdb.tmp");
    let current_store = store_ptr_guard.as_ref();

    if let Err(e) = vibrato_core::format_v2::VdbWriterV2::merge(
        current_store,
        &dynamic_guard,
        &index_guard,
        temp_path
    ) {
        return (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: e.to_string() })).into_response();
    }

    // 3. Atomic Rename
    if let Err(e) = std::fs::rename(temp_path, "index.vdb") {
         return (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: e.to_string() })).into_response();
    }

    // 4. Hot Swap
    match VectorStore::open("index.vdb") {
        Ok(new_store) => {
            let new_store_arc = Arc::new(new_store);
            *store_ptr_guard = new_store_arc.clone();
            
            // Clear dynamic store since everything is now on disk
            // We need to cast dynamic_guard to mutable reference, but it is already RwLockWriteGuard
            // But wait, dynamic_guard is immutable binding.
            // Wait, clear() requires mutable access.
            // dynamic_guard implements DerefMut.
            // So we need `let mut dynamic_guard = ...`
            // But I used `let dynamic_guard`.
            // I need to fix that line in Step 1.
            
            // Actually, since I can't edit previous lines in this tool call easily without context overlap, 
            // I'll rely on Rust's mutability rules or fix it below.
            // Rust: RwLockWriteGuard implies mutable access to inner. 
            // `dynamic_store.write()` returns guard. The guard itself doesn't need to be mut binding to call methods on inner? 
            // Yes it does: `impl DerefMut for RwLockWriteGuard`. `Vec::clear` takes `&mut self`.
            // So `*dynamic_guard` yields `&mut Vec`.
            // `dynamic_guard` variable binding needs to serve `DerefMut`.
            // `let mut dynamic` is strictly correct but often `dynamic.clear()` works if method receiver takes strict `&mut`.
            // But `DerefMut` requires `&mut self` on guard? No, `offset` method on pointer.
            // `WriteGuard` implements `DerefMut`. You can call mutable methods on the data it protects.
            // You only need `mut guard` if you want to mutate the guard itself (e.g. swap it).
            // Calling `clear` on the Vec inside doesn't require `mut guard`.
            dynamic_guard.clear();
            
            (StatusCode::OK, Json(serde_json::json!({ "status": "flushed", "vectors": new_store_arc.count }))).into_response()
        }
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: e.to_string() })).into_response(),
    }
}

/// GET /health - Server health and telemetry
async fn health(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let stats = {
        let index = state.index.read();
        index.stats()
    };

    let (memory_mb, count, dim) = {
        let store = state.store.read();
        let memory_mb = store.memory_bytes() as f64 / (1024.0 * 1024.0);
        let count = store.count;
        let dim = store.dim;
        (memory_mb, count, dim)
    };

    let response = HealthResponse {
        status: "ok".to_string(),
        vectors_loaded: count,
        index_layers: stats.max_layer + 1,
        memory_mb,
        config: IndexConfig {
            m: stats.m,
            ef_construction: stats.ef_construction,
            dimensions: dim,
        },
    };

    (StatusCode::OK, Json(response))
}

/// Create the axum router
pub fn create_router(state: Arc<AppState>) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        .route("/search", post(search))
        .route("/ingest", post(ingest))
        .route("/flush", post(flush))
        .route("/health", get(health))
        .layer(cors)
        .with_state(state)
}

/// Start the server
pub async fn serve(state: Arc<AppState>, addr: std::net::SocketAddr) -> std::io::Result<()> {
    let router = create_router(state);

    tracing::info!("Starting Vibrato-DB server on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, router).await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::VdbWriter;
    use crate::simd::l2_normalized;
    use axum::body::Body;
    use axum::http::Request;
    use tempfile::tempdir;
    use tower::ServiceExt;

    fn random_vector(dim: usize) -> Vec<f32> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect();
        l2_normalized(&v)
    }

    fn create_test_state() -> Arc<AppState> {
        let dim = 64;
        let num_vectors = 100;

        // Create test vectors
        let vectors: Vec<_> = (0..num_vectors).map(|_| random_vector(dim)).collect();

        // Write to .vdb file
        let dir = tempdir().unwrap();
        let vdb_path = dir.path().join("test.vdb");

        let mut writer = VdbWriter::new(&vdb_path, dim).unwrap();
        for v in &vectors {
            writer.write_vector(v).unwrap();
        }
        writer.finish().unwrap();

        // Open store
        let store = Arc::new(VectorStore::open(&vdb_path).unwrap());

        // Create index (need Arc for closure)
        let store_clone = store.clone();
        let mut hnsw = HNSW::new(16, 50, move |id| store_clone.get(id).to_vec());

        for i in 0..num_vectors {
            hnsw.insert(i);
        }

        // Leak tempdir to keep files alive
        std::mem::forget(dir);

        // SharedStore
        let shared_store = Arc::new(RwLock::new(store));

        Arc::new(AppState {
            index: RwLock::new(hnsw),
            store: shared_store,
            dynamic_store: Arc::new(RwLock::new(Vec::new())),
            inference: None,
            flush_mutex: RwLock::new(()),
        })
    }

    #[tokio::test]
    async fn test_health_endpoint() {
        let state = create_test_state();
        let router = create_router(state);

        let request = Request::builder()
            .method("GET")
            .uri("/health")
            .body(Body::empty())
            .unwrap();

        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_search_endpoint() {
        let state = create_test_state();
        let dim = state.store.read().dim;
        let router = create_router(state);

        let query = random_vector(dim);
        let body = serde_json::json!({
            "vector": query,
            "k": 5
        });

        let request = Request::builder()
            .method("POST")
            .uri("/search")
            .header("Content-Type", "application/json")
            .body(Body::from(serde_json::to_string(&body).unwrap()))
            .unwrap();

        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_search_dimension_mismatch() {
        let state = create_test_state();
        let router = create_router(state);

        let wrong_dim_query = vec![0.0f32; 32]; // Wrong dimensions
        let body = serde_json::json!({
            "vector": wrong_dim_query,
            "k": 5
        });

        let request = Request::builder()
            .method("POST")
            .uri("/search")
            .header("Content-Type", "application/json")
            .body(Body::from(serde_json::to_string(&body).unwrap()))
            .unwrap();

        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }
}
