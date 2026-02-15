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
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
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
    pub flush_status: Arc<RwLock<FlushStatus>>,
    pub flush_job_seq: AtomicU64,
    // Single source-of-truth container path (same as --data)
    pub data_path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlushStatus {
    pub state: String,
    pub job_id: u64,
    pub snapshot_vectors: usize,
    pub persisted_vectors: usize,
    pub dynamic_vectors: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
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

fn idle_flush_status(job_id: u64, persisted_vectors: usize, dynamic_vectors: usize) -> FlushStatus {
    FlushStatus {
        state: "idle".to_string(),
        job_id,
        snapshot_vectors: 0,
        persisted_vectors,
        dynamic_vectors,
        error: None,
    }
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
    let _flush_read = state.flush_mutex.read();
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
/// Fast path:
/// 1. Snapshot current dynamic buffers under lock.
/// 2. Return `202 Accepted` immediately.
///
/// Slow path runs in background:
/// 1. Build merged graph + V2 container from persisted store + snapshot.
/// 2. Atomically swap store on success.
/// 3. Trim flushed prefix from dynamic buffers.
async fn flush(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let (
        job_id,
        snapshot_vectors,
        snapshot_metadata,
        current_store,
        current_metadata,
        hnsw_m,
        hnsw_ef_construction,
    ) = {
        let _flush_guard = state.flush_mutex.write();
        let status = state.flush_status.read().clone();
        if status.state == "running" {
            return (
                StatusCode::CONFLICT,
                Json(serde_json::json!({
                    "status": "flush_in_progress",
                    "job_id": status.job_id
                })),
            )
                .into_response();
        }

        let snapshot_vectors = state.dynamic_store.read().clone();
        let snapshot_metadata = state.dynamic_metadata.read().clone();
        let current_store = state.store.load_full();
        let current_metadata = state.persisted_metadata.load_full();
        if snapshot_vectors.is_empty() {
            let persisted = current_store.count;
            let dynamic = 0usize;
            let current_job_id = state.flush_job_seq.load(AtomicOrdering::Relaxed);
            *state.flush_status.write() = idle_flush_status(current_job_id, persisted, dynamic);
            return (
                StatusCode::OK,
                Json(serde_json::json!({
                    "status": "idle",
                    "job_id": current_job_id,
                    "persisted_vectors": persisted,
                    "dynamic_vectors": dynamic
                })),
            )
                .into_response();
        }

        let stats = state.index.read().stats();
        let job_id = state.flush_job_seq.fetch_add(1, AtomicOrdering::SeqCst) + 1;
        *state.flush_status.write() = FlushStatus {
            state: "running".to_string(),
            job_id,
            snapshot_vectors: snapshot_vectors.len(),
            persisted_vectors: current_store.count,
            dynamic_vectors: snapshot_vectors.len(),
            error: None,
        };

        (
            job_id,
            snapshot_vectors,
            snapshot_metadata,
            current_store,
            current_metadata,
            stats.m,
            stats.ef_construction,
        )
    };

    let state_for_task = Arc::clone(&state);
    tokio::spawn(async move {
        run_flush_job(
            state_for_task,
            job_id,
            snapshot_vectors,
            snapshot_metadata,
            current_store,
            current_metadata,
            hnsw_m,
            hnsw_ef_construction,
        )
        .await;
    });

    (
        StatusCode::ACCEPTED,
        Json(serde_json::json!({
            "status": "started",
            "job_id": job_id
        })),
    )
        .into_response()
}

async fn flush_status(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let mut status = state.flush_status.read().clone();
    status.persisted_vectors = state.store.load().count;
    status.dynamic_vectors = state.dynamic_store.read().len();
    (StatusCode::OK, Json(status)).into_response()
}

async fn run_flush_job(
    state: Arc<AppState>,
    job_id: u64,
    snapshot_vectors: Vec<Vec<f32>>,
    snapshot_metadata: Vec<VectorMetadata>,
    current_store: Arc<VectorStore>,
    current_metadata: Arc<Vec<VectorMetadata>>,
    hnsw_m: usize,
    hnsw_ef_construction: usize,
) {
    let snapshot_len = snapshot_vectors.len();
    let data_path = state.data_path.clone();
    let temp_path = state
        .data_path
        .with_extension(format!("vdb.tmp.{}", job_id));

    let persist_result = tokio::task::spawn_blocking(
        move || -> anyhow::Result<(Arc<VectorStore>, Vec<VectorMetadata>)> {
            let snapshot_vectors = Arc::new(snapshot_vectors);
            let dim = current_store.dim;
            let fallback = vec![0.0f32; dim];
            let accessor_store = current_store.clone();
            let accessor_dynamic = snapshot_vectors.clone();
            let mut rebuilt =
                HNSW::new_with_accessor(hnsw_m, hnsw_ef_construction, move |id, sink| {
                    if id < accessor_store.count {
                        sink(accessor_store.get(id));
                    } else {
                        let offset = id - accessor_store.count;
                        if let Some(v) = accessor_dynamic.get(offset) {
                            sink(v);
                        } else {
                            sink(&fallback);
                        }
                    }
                });

            for id in 0..(current_store.count + snapshot_vectors.len()) {
                rebuilt.insert(id);
            }

            let mut merged_metadata =
                Vec::with_capacity(current_store.count + snapshot_vectors.len());
            if current_metadata.len() == current_store.count {
                merged_metadata.extend(current_metadata.iter().cloned());
            } else {
                merged_metadata.resize(current_store.count, VectorMetadata::default());
            }
            merged_metadata.extend(snapshot_metadata.into_iter());

            VdbWriterV2::merge_with_metadata(
                &current_store,
                snapshot_vectors.as_ref(),
                &merged_metadata,
                &rebuilt,
                &temp_path,
            )?;
            std::fs::rename(&temp_path, &data_path)?;
            sync_path_durable(&data_path)?;
            let new_store = Arc::new(VectorStore::open(&data_path)?);
            Ok((new_store, merged_metadata))
        },
    )
    .await;

    match persist_result {
        Ok(Ok((new_store, merged_metadata))) => {
            let _flush_guard = state.flush_mutex.write();
            state.store.store(new_store.clone());
            state.persisted_metadata.store(Arc::new(merged_metadata));

            let mut dynamic = state.dynamic_store.write();
            let mut dynamic_metadata = state.dynamic_metadata.write();
            let drain_n = snapshot_len.min(dynamic.len()).min(dynamic_metadata.len());
            if drain_n > 0 {
                dynamic.drain(0..drain_n);
                dynamic_metadata.drain(0..drain_n);
            }

            *state.flush_status.write() = FlushStatus {
                state: "completed".to_string(),
                job_id,
                snapshot_vectors: drain_n,
                persisted_vectors: new_store.count,
                dynamic_vectors: dynamic.len(),
                error: None,
            };
        }
        Ok(Err(err)) => {
            let persisted = state.store.load().count;
            let dynamic = state.dynamic_store.read().len();
            *state.flush_status.write() = FlushStatus {
                state: "failed".to_string(),
                job_id,
                snapshot_vectors: snapshot_len,
                persisted_vectors: persisted,
                dynamic_vectors: dynamic,
                error: Some(err.to_string()),
            };
        }
        Err(join_err) => {
            let persisted = state.store.load().count;
            let dynamic = state.dynamic_store.read().len();
            *state.flush_status.write() = FlushStatus {
                state: "failed".to_string(),
                job_id,
                snapshot_vectors: snapshot_len,
                persisted_vectors: persisted,
                dynamic_vectors: dynamic,
                error: Some(format!("flush task join error: {}", join_err)),
            };
        }
    }
}

/// GET /health - Server health and telemetry
async fn health(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let _flush_read = state.flush_mutex.read();
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
        .route("/flush/status", get(flush_status))
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
            flush_status: Arc::new(RwLock::new(idle_flush_status(0, store.count, 0))),
            flush_job_seq: AtomicU64::new(0),
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
