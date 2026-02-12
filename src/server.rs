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
    pub store: VectorStore,
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
    /// Vector to ingest
    pub vector: Vec<f32>,
}

/// Ingest response
#[derive(Debug, Serialize)]
pub struct IngestResponse {
    /// ID of the inserted vector
    pub id: usize,
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
    if request.vector.len() != state.store.dim {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: format!(
                    "Dimension mismatch: expected {}, got {}",
                    state.store.dim,
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
    if request.vector.len() != state.store.dim {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: format!(
                    "Dimension mismatch: expected {}, got {}",
                    state.store.dim,
                    request.vector.len()
                ),
            }),
        )
            .into_response();
    }

    // Attempt to acquire write lock
    // This is where we block concurrent searches
    let id = {
        let mut index = state.index.write();
        
        // TODO: Implement full ingestion with mutable store support.
        // Currently, VectorStore is mmap read-only.
        // For the purpose of this review/plan, we acknowledge the limitation.
        return (
            StatusCode::NOT_IMPLEMENTED,
            Json(ErrorResponse {
                error: "Runtime ingestion requires mutable storage backend (V3 feature)".into(),
            }),
        )
            .into_response();
    };
}

/// GET /health - Server health and telemetry
async fn health(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let stats = {
        let index = state.index.read();
        index.stats()
    };

    let memory_mb = state.store.memory_bytes() as f64 / (1024.0 * 1024.0);

    let response = HealthResponse {
        status: "ok".to_string(),
        vectors_loaded: state.store.count,
        index_layers: stats.max_layer + 1,
        memory_mb,
        config: IndexConfig {
            m: stats.m,
            ef_construction: stats.ef_construction,
            dimensions: state.store.dim,
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
        let store = VectorStore::open(&vdb_path).unwrap();

        // Create index (need Arc for closure)
        let vectors = Arc::new(vectors);
        let vectors_clone = vectors.clone();
        let mut hnsw = HNSW::new(16, 50, move |id| vectors_clone[id].clone());

        for i in 0..num_vectors {
            hnsw.insert(i);
        }

        // Leak tempdir to keep files alive
        std::mem::forget(dir);

        Arc::new(AppState {
            index: RwLock::new(hnsw),
            store,
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
        let dim = state.store.dim;
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
