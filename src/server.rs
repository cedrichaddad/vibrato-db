//! HTTP Server for Vibrato-DB
//!
//! Exposes the vector search engine over HTTP using axum.
//!
//! # Endpoints
//!
//! - `POST /search` - Query for nearest neighbors
//! - `GET /health` - Server health and telemetry

use std::fs::File;
use std::path::{Path, PathBuf};
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
use vibrato_core::format_v2::VdbWriterV2;
use vibrato_core::metadata::{MetadataReader, VectorMetadata};
use vibrato_neural::inference::InferenceEngine;

/// Thread-safe, swappable store handle
pub type SharedStore = Arc<arc_swap::ArcSwap<VectorStore>>;

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
    // Metadata for vectors already persisted in `store`
    pub persisted_metadata: Arc<arc_swap::ArcSwap<Vec<VectorMetadata>>>,
    // Metadata for vectors in `dynamic_store` (same ordering/length)
    pub dynamic_metadata: Arc<RwLock<Vec<VectorMetadata>>>,
    pub inference: Option<Arc<InferenceEngine>>,
    // Global lock to coordinate flush vs ingest consistency
    pub flush_mutex: RwLock<()>,
    // Single source-of-truth container path (same as --data)
    pub data_path: PathBuf,
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
    pub metadata: Option<IngestMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IngestMetadata {
    #[serde(default)]
    pub source_file: String,
    #[serde(default)]
    pub start_time_ms: u32,
    #[serde(default)]
    pub duration_ms: u16,
    #[serde(default)]
    pub bpm: f32,
    #[serde(default)]
    pub tags: Vec<String>,
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

    /// Optional metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<VectorMetadata>,
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
    let dim = state.store.load().dim;
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

    let persisted_metadata = state.persisted_metadata.load();
    let dynamic_metadata = state.dynamic_metadata.read();
    let persisted_count = persisted_metadata.len();

    let metadata_for_id = |id: usize| -> Option<VectorMetadata> {
        if id < persisted_count {
            let item = persisted_metadata[id].clone();
            if item.is_empty() {
                None
            } else {
                Some(item)
            }
        } else {
            let offset = id.saturating_sub(persisted_count);
            dynamic_metadata.get(offset).cloned().and_then(|item| {
                if item.is_empty() {
                    None
                } else {
                    Some(item)
                }
            })
        }
    };

    // Format response
    let response = SearchResponse {
        results: results
            .into_iter()
            .map(|(id, score)| SearchResult {
                id,
                score,
                metadata: metadata_for_id(id),
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
    let provided_audio_path = request.audio_path.clone();

    // 1. Get vector: either directly provided or via inference
    let vector = if let Some(vec) = request.vector {
        vec
    } else if let Some(path_str) = request.audio_path {
        if let Some(inference) = &state.inference {
            let path = std::path::Path::new(&path_str);
            match inference.embed_audio_file(path).await {
                Ok(vec) => vec,
                Err(e) => {
                    return (
                        StatusCode::UNPROCESSABLE_ENTITY,
                        Json(ErrorResponse {
                            error: format!("Audio embedding failed: {}", e),
                        }),
                    )
                        .into_response();
                }
            }
        } else {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(ErrorResponse {
                    error: "Inference engine unavailable. Run setup-models and restart.".into(),
                }),
            )
                .into_response();
        }
    } else {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Provide either `vector` or `audio_path`".into(),
            }),
        )
            .into_response();
    };

    // Build metadata with defaults so ID -> metadata mapping remains stable.
    let mut metadata = request
        .metadata
        .map(|m| VectorMetadata {
            source_file: m.source_file,
            start_time_ms: m.start_time_ms,
            duration_ms: m.duration_ms,
            bpm: m.bpm,
            tags: m
                .tags
                .into_iter()
                .filter(|t| !t.trim().is_empty())
                .collect(),
        })
        .unwrap_or_default();

    if metadata.source_file.is_empty() {
        if let Some(audio_path) = provided_audio_path {
            metadata.source_file = audio_path;
        }
    }

    // 2. Append to dynamic store (write lock)
    // Ensure consistency with flush: if flush is running, we wait.
    let _flush_read = state.flush_mutex.read();

    // Check dimensions again safely
    let store = state.store.load();
    let store_count = store.count;
    let store_dim = store.dim;
    drop(store);

    if vector.len() != store_dim {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: format!(
                    "Dimension mismatch: expected {}, got {}",
                    store_dim,
                    vector.len()
                ),
            }),
        )
            .into_response();
    }

    let id_offset = store_count as usize;
    let mut dynamic = state.dynamic_store.write();
    let mut dynamic_metadata = state.dynamic_metadata.write();
    let current_dynamic_count = dynamic.len();
    let new_id = id_offset + current_dynamic_count;
    dynamic.push(vector); // move, no extra clone
    dynamic_metadata.push(metadata);
    drop(dynamic_metadata);
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
    )
        .into_response()
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
    let mut dynamic_metadata_guard = state.dynamic_metadata.write();
    // No store lock needed for ArcSwap read, but we need the current one for merging
    let current_store = state.store.load_full();
    let current_metadata = state.persisted_metadata.load_full();

    // 2. Merge Logic
    let temp_path = state.data_path.with_extension("vdb.tmp");

    let mut merged_metadata = Vec::with_capacity(current_store.count + dynamic_guard.len());
    if current_metadata.len() == current_store.count {
        merged_metadata.extend(current_metadata.iter().cloned());
    } else {
        tracing::warn!(
            "Persisted metadata count mismatch (metadata={}, vectors={}), filling defaults",
            current_metadata.len(),
            current_store.count
        );
        merged_metadata.resize(current_store.count, VectorMetadata::default());
    }
    merged_metadata.extend(dynamic_metadata_guard.iter().cloned());

    if let Err(e) = VdbWriterV2::merge_with_metadata(
        &current_store,
        &dynamic_guard,
        &merged_metadata,
        &index_guard,
        &temp_path,
    ) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
            .into_response();
    }

    // 3. Atomic Rename
    if let Err(e) = std::fs::rename(&temp_path, &state.data_path) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
            .into_response();
    }
    if let Err(e) = sync_path_durable(&state.data_path) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("Flush durability sync failed: {}", e),
            }),
        )
            .into_response();
    }

    // 4. Hot Swap
    match VectorStore::open(&state.data_path) {
        Ok(new_store) => {
            let new_store_arc = Arc::new(new_store);
            state.store.store(new_store_arc.clone());
            state.persisted_metadata.store(Arc::new(merged_metadata));

            // Clear dynamic store since everything is now on disk
            dynamic_guard.clear();
            dynamic_metadata_guard.clear();

            (
                StatusCode::OK,
                Json(serde_json::json!({ "status": "flushed", "vectors": new_store_arc.count })),
            )
                .into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
            .into_response(),
    }
}

/// GET /health - Server health and telemetry
async fn health(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let stats = {
        let index = state.index.read();
        index.stats()
    };

    let (memory_mb, count, dim) = {
        let store = state.store.load();
        let memory_mb = store.memory_bytes() as f64 / (1024.0 * 1024.0);
        let count = store.count + state.dynamic_store.read().len();
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

pub fn load_store_metadata(store: &VectorStore) -> Vec<VectorMetadata> {
    let mut metadata = vec![VectorMetadata::default(); store.count];
    let Some(bytes) = store.metadata_bytes() else {
        return metadata;
    };

    let reader = match MetadataReader::new(bytes) {
        Ok(reader) => reader,
        Err(e) => {
            tracing::warn!("Failed to parse metadata section: {}", e);
            return metadata;
        }
    };

    if reader.count() != store.count {
        tracing::warn!(
            "Metadata count mismatch: metadata={}, vectors={}",
            reader.count(),
            store.count
        );
        return metadata;
    }

    for (i, slot) in metadata.iter_mut().enumerate() {
        if let Ok(item) = reader.get(i) {
            *slot = item;
        }
    }

    metadata
}

fn sync_path_durable(path: &Path) -> std::io::Result<()> {
    // Sync file contents + metadata.
    File::open(path)?.sync_all()?;

    // Ensure directory entry durability after rename on POSIX filesystems.
    #[cfg(unix)]
    {
        if let Some(parent) = path.parent() {
            File::open(parent)?.sync_all()?;
        }
    }

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

        // Leak tempdir to keep files alive
        std::mem::forget(dir);

        // Use ArcSwap for shared_store
        let shared_store = Arc::new(arc_swap::ArcSwap::from(store.clone()));
        let dynamic_store = Arc::new(RwLock::new(Vec::<Vec<f32>>::new()));
        let dynamic_metadata = Arc::new(RwLock::new(Vec::new()));
        let persisted_metadata = Arc::new(arc_swap::ArcSwap::from(Arc::new(
            vec![VectorMetadata::default(); store.count],
        )));

        let accessor_store = shared_store.clone();
        let accessor_dynamic = dynamic_store.clone();
        let mut hnsw = HNSW::new_with_accessor(16, 50, move |id, sink| {
            let store_guard = accessor_store.load();
            if id < store_guard.count {
                sink(store_guard.get(id));
            } else {
                let guard = accessor_dynamic.read();
                sink(&guard[id - store_guard.count]);
            }
        });
        for i in 0..num_vectors {
            hnsw.insert(i);
        }

        Arc::new(AppState {
            index: RwLock::new(hnsw),
            store: shared_store,
            dynamic_store,
            persisted_metadata,
            dynamic_metadata,
            inference: None,
            flush_mutex: RwLock::new(()),
            data_path: vdb_path, // Use the temporary path for data_path
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
        let dim = state.store.load().dim;
        let router = create_router(state);

        let query = vec![0.1; dim]; // Simple query
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
