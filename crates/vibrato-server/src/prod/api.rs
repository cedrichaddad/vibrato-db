use std::sync::Arc;
use std::time::Instant;

use axum::extract::{DefaultBodyLimit, Request, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{any, get, post};
use axum::{Json, Router};
use tokio::time::sleep;

use super::auth::{authorize, AuthError};
use super::catalog::{IngestMetadataV3Input, Role};
use super::engine::{IngestBackpressureDecision, ProductionState};
use super::model::{
    ApiResponse, ErrorBody, HealthResponseV2, IdentifyRequestV2, IngestBatchRequestV2,
    IngestBatchResponseV2, IngestMetadataV3, IngestRequestV2, IngestResponseV2, JobResponseV2,
    QueryRequestV2,
};
use super::snapshot::create_snapshot;

const MAX_HTTP_BODY_BYTES: usize = 64 * 1024 * 1024;

pub fn create_v3_router(state: Arc<ProductionState>) -> Router {
    let connection_state = state.clone();
    Router::new()
        .route("/v3/vectors", post(v2_ingest))
        .route("/v3/vectors/batch", post(v2_ingest_batch))
        .route("/v3/query", post(v2_query))
        .route("/v3/identify", post(v2_identify))
        .route("/v3/health/live", get(v2_health_live))
        .route("/v3/health/ready", get(v2_health_ready))
        .route("/v3/admin/checkpoint", post(v2_admin_checkpoint))
        .route("/v3/admin/compact", post(v2_admin_compact))
        .route("/v3/admin/snapshot", post(v2_admin_snapshot))
        .route("/v3/admin/stats", get(v2_admin_stats))
        .route("/v3/metrics", get(v2_metrics))
        .route("/v2/*path", any(v2_deprecated))
        .layer(DefaultBodyLimit::max(MAX_HTTP_BODY_BYTES))
        .layer(middleware::from_fn_with_state(
            connection_state,
            track_active_connections,
        ))
        .with_state(state)
}

async fn track_active_connections(
    State(state): State<Arc<ProductionState>>,
    request: Request,
    next: Next,
) -> Response {
    state.http_request_opened();
    let response = next.run(request).await;
    state.http_request_closed();
    response
}

async fn v2_ingest(
    State(state): State<Arc<ProductionState>>,
    headers: HeaderMap,
    Json(body): Json<IngestRequestV2>,
) -> Response {
    let request_id = request_id(&headers);
    let started = Instant::now();

    let auth = match authorize(
        state.catalog.as_ref(),
        &headers,
        Some(Role::Ingest),
        &state.config.api_pepper,
    ) {
        Ok(auth) => auth,
        Err(e) => {
            state
                .metrics
                .auth_failures_total
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return auth_error_response(&state, &request_id, "/v3/vectors", "ingest", started, e);
        }
    };

    let metadata = match ingest_metadata_to_input(&state, &body.metadata) {
        Ok(meta) => meta,
        Err(msg) => {
            state
                .metrics
                .tag_reject_invalid_total
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return error_response(
                StatusCode::BAD_REQUEST,
                &request_id,
                "invalid_tag",
                msg,
                None,
            );
        }
    };
    let idempotency_key = match validate_idempotency_key(&state, body.idempotency_key.clone()) {
        Ok(v) => v,
        Err(msg) => {
            return error_response(StatusCode::BAD_REQUEST, &request_id, "bad_request", msg, None);
        }
    };
    let vector = body.vector.clone();
    let incoming_bytes = ProductionState::estimate_ingest_entry_bytes(
        &vector,
        &metadata,
        idempotency_key.as_deref(),
    );
    match state.ingest_backpressure_decision(incoming_bytes) {
        IngestBackpressureDecision::Allow => {}
        IngestBackpressureDecision::Throttle {
            delay,
            projected_bytes,
            soft_limit_bytes,
            hard_limit_bytes,
        } => {
            state.record_ingest_soft_throttle(delay);
            tracing::debug!(
                "semantic_throttle endpoint=/v3/vectors delay_ms={} projected={} soft={} hard={}",
                delay.as_millis(),
                projected_bytes,
                soft_limit_bytes,
                hard_limit_bytes
            );
            sleep(delay).await;
        }
        IngestBackpressureDecision::Reject {
            projected_bytes,
            hard_limit_bytes,
        } => {
            state.record_ingest_hard_reject();
            let msg = format!(
                "resource exhausted: projected memory proxy {} exceeds hard limit {}",
                projected_bytes, hard_limit_bytes
            );
            let resp = error_response(
                StatusCode::TOO_MANY_REQUESTS,
                &request_id,
                "resource_exhausted",
                msg.clone(),
                Some(serde_json::json!({
                    "projected_bytes": projected_bytes,
                    "hard_limit_bytes": hard_limit_bytes
                })),
            );
            audit_best_effort(
                &state,
                &request_id,
                auth.as_deref(),
                "/v3/vectors",
                "ingest",
                StatusCode::TOO_MANY_REQUESTS.as_u16(),
                started.elapsed().as_secs_f64() * 1000.0,
                serde_json::json!({"error": msg, "code": "resource_exhausted"}),
            );
            return resp;
        }
    }

    let state_bg = state.clone();
    state
        .inflight_decode_bytes
        .fetch_add(incoming_bytes, std::sync::atomic::Ordering::Relaxed);
    let result = tokio::task::spawn_blocking(move || {
        state_bg.ingest_vector(&vector, &metadata, idempotency_key.as_deref())
    })
    .await;
    let result = match result {
        Ok(inner) => inner,
        Err(e) => Err(anyhow::anyhow!("ingest join error: {}", e)),
    };
    state
        .inflight_decode_bytes
        .fetch_sub(incoming_bytes, std::sync::atomic::Ordering::Relaxed);

    match result {
        Ok((id, created)) => {
            let payload = ApiResponse {
                data: IngestResponseV2 { id, created },
            };
            let resp = json_response(StatusCode::CREATED, &request_id, &payload);
            audit_best_effort(
                &state,
                &request_id,
                auth.as_deref(),
                "/v3/vectors",
                "ingest",
                201,
                started.elapsed().as_secs_f64() * 1000.0,
                serde_json::json!({"id": id, "created": created}),
            );
            resp
        }
        Err(e) => {
            let (status, code) = classify_ingest_error(&e);
            if code == "tag_registry_overflow" {
                state
                    .metrics
                    .tag_reject_overflow_total
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            let resp = error_response(status, &request_id, code, e.to_string(), None);
            audit_best_effort(
                &state,
                &request_id,
                auth.as_deref(),
                "/v3/vectors",
                "ingest",
                status.as_u16(),
                started.elapsed().as_secs_f64() * 1000.0,
                serde_json::json!({"error": e.to_string()}),
            );
            resp
        }
    }
}

async fn v2_ingest_batch(
    State(state): State<Arc<ProductionState>>,
    headers: HeaderMap,
    Json(body): Json<IngestBatchRequestV2>,
) -> Response {
    let request_id = request_id(&headers);
    let started = Instant::now();

    let auth = match authorize(
        state.catalog.as_ref(),
        &headers,
        Some(Role::Ingest),
        &state.config.api_pepper,
    ) {
        Ok(auth) => auth,
        Err(e) => {
            state
                .metrics
                .auth_failures_total
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return auth_error_response(
                &state,
                &request_id,
                "/v3/vectors/batch",
                "ingest_batch",
                started,
                e,
            );
        }
    };

    let batch_len = body.vectors.len();
    if batch_len == 0 {
        let payload = ApiResponse {
            data: IngestBatchResponseV2 { results: vec![] },
        };
        return json_response(StatusCode::CREATED, &request_id, &payload);
    }

    // Convert to engine format
    let entries = body
        .vectors
        .into_iter()
        .map(|req| -> std::result::Result<_, String> {
            let meta = ingest_metadata_to_input(&state, &req.metadata)?;
            let idempotency_key = validate_idempotency_key(&state, req.idempotency_key)?;
            Ok((req.vector, meta, idempotency_key))
        })
        .collect::<std::result::Result<Vec<_>, _>>();
    let entries = match entries {
        Ok(v) => v,
        Err(msg) => {
            state
                .metrics
                .tag_reject_invalid_total
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return error_response(
                StatusCode::BAD_REQUEST,
                &request_id,
                "invalid_tag",
                msg,
                None,
            );
        }
    };

    let incoming_bytes = ProductionState::estimate_ingest_batch_bytes(&entries);
    match state.ingest_backpressure_decision(incoming_bytes) {
        IngestBackpressureDecision::Allow => {}
        IngestBackpressureDecision::Throttle {
            delay,
            projected_bytes,
            soft_limit_bytes,
            hard_limit_bytes,
        } => {
            state.record_ingest_soft_throttle(delay);
            tracing::debug!(
                "semantic_throttle endpoint=/v3/vectors/batch delay_ms={} projected={} soft={} hard={}",
                delay.as_millis(),
                projected_bytes,
                soft_limit_bytes,
                hard_limit_bytes
            );
            sleep(delay).await;
        }
        IngestBackpressureDecision::Reject {
            projected_bytes,
            hard_limit_bytes,
        } => {
            state.record_ingest_hard_reject();
            let msg = format!(
                "resource exhausted: projected memory proxy {} exceeds hard limit {}",
                projected_bytes, hard_limit_bytes
            );
            let resp = error_response(
                StatusCode::TOO_MANY_REQUESTS,
                &request_id,
                "resource_exhausted",
                msg.clone(),
                Some(serde_json::json!({
                    "projected_bytes": projected_bytes,
                    "hard_limit_bytes": hard_limit_bytes
                })),
            );
            audit_best_effort(
                &state,
                &request_id,
                auth.as_deref(),
                "/v3/vectors/batch",
                "ingest_batch",
                StatusCode::TOO_MANY_REQUESTS.as_u16(),
                started.elapsed().as_secs_f64() * 1000.0,
                serde_json::json!({"error": msg, "code": "resource_exhausted"}),
            );
            return resp;
        }
    }

    let state_bg = state.clone();
    state
        .inflight_decode_bytes
        .fetch_add(incoming_bytes, std::sync::atomic::Ordering::Relaxed);
    let result = tokio::task::spawn_blocking(move || state_bg.ingest_batch_owned(entries)).await;
    let result = match result {
        Ok(inner) => inner,
        Err(e) => Err(anyhow::anyhow!("batch ingest join error: {}", e)),
    };
    state
        .inflight_decode_bytes
        .fetch_sub(incoming_bytes, std::sync::atomic::Ordering::Relaxed);

    match result {
        Ok(results) => {
            let payload = ApiResponse {
                data: IngestBatchResponseV2 {
                    results: results
                        .iter()
                        .map(|&(id, created)| IngestResponseV2 { id, created })
                        .collect(),
                },
            };
            let resp = json_response(StatusCode::CREATED, &request_id, &payload);
            audit_best_effort(
                &state,
                &request_id,
                auth.as_deref(),
                "/v3/vectors/batch",
                "ingest_batch",
                201,
                started.elapsed().as_secs_f64() * 1000.0,
                serde_json::json!({"count": batch_len}),
            );
            resp
        }
        Err(e) => {
            let (status, code) = classify_ingest_error(&e);
            if code == "tag_registry_overflow" {
                state
                    .metrics
                    .tag_reject_overflow_total
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            let resp = error_response(status, &request_id, code, e.to_string(), None);
            audit_best_effort(
                &state,
                &request_id,
                auth.as_deref(),
                "/v3/vectors/batch",
                "ingest_batch",
                status.as_u16(),
                started.elapsed().as_secs_f64() * 1000.0,
                serde_json::json!({"error": e.to_string()}),
            );
            resp
        }
    }
}

async fn v2_query(
    State(state): State<Arc<ProductionState>>,
    headers: HeaderMap,
    Json(body): Json<QueryRequestV2>,
) -> Response {
    let request_id = request_id(&headers);
    let started = Instant::now();

    let auth = match authorize(
        state.catalog.as_ref(),
        &headers,
        Some(Role::Query),
        &state.config.api_pepper,
    ) {
        Ok(auth) => auth,
        Err(e) => {
            state
                .metrics
                .auth_failures_total
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return auth_error_response(&state, &request_id, "/v3/query", "query", started, e);
        }
    };

    let state_bg = state.clone();
    let body_clone = body.clone();
    let result = tokio::task::spawn_blocking(move || state_bg.query(&body_clone)).await;
    let result = match result {
        Ok(inner) => inner,
        Err(e) => Err(anyhow::anyhow!("query join error: {}", e)),
    };
    match result {
        Ok(query) => {
            let payload = ApiResponse { data: query };
            let resp = json_response(StatusCode::OK, &request_id, &payload);
            audit_best_effort(
                &state,
                &request_id,
                auth.as_deref(),
                "/v3/query",
                "query",
                200,
                started.elapsed().as_secs_f64() * 1000.0,
                serde_json::json!({"k": body.k, "ef": body.ef}),
            );
            resp
        }
        Err(e) => {
            let (status, code) = classify_query_error(&e);
            let resp = error_response(status, &request_id, code, e.to_string(), None);
            audit_best_effort(
                &state,
                &request_id,
                auth.as_deref(),
                "/v3/query",
                "query",
                status.as_u16(),
                started.elapsed().as_secs_f64() * 1000.0,
                serde_json::json!({"error": e.to_string()}),
            );
            resp
        }
    }
}

async fn v2_identify(
    State(state): State<Arc<ProductionState>>,
    headers: HeaderMap,
    Json(body): Json<IdentifyRequestV2>,
) -> Response {
    let request_id = request_id(&headers);
    let started = Instant::now();

    let auth = match authorize(
        state.catalog.as_ref(),
        &headers,
        Some(Role::Query),
        &state.config.api_pepper,
    ) {
        Ok(auth) => auth,
        Err(e) => {
            state
                .metrics
                .auth_failures_total
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return auth_error_response(
                &state,
                &request_id,
                "/v3/identify",
                "identify",
                started,
                e,
            );
        }
    };

    let state_bg = state.clone();
    let body_clone = body.clone();
    let result = tokio::task::spawn_blocking(move || state_bg.identify(&body_clone)).await;
    let result = match result {
        Ok(inner) => inner,
        Err(e) => Err(anyhow::anyhow!("identify join error: {}", e)),
    };
    match result {
        Ok(result) => {
            let payload = ApiResponse { data: result };
            let resp = json_response(StatusCode::OK, &request_id, &payload);
            audit_best_effort(
                &state,
                &request_id,
                auth.as_deref(),
                "/v3/identify",
                "identify",
                200,
                started.elapsed().as_secs_f64() * 1000.0,
                serde_json::json!({"k": body.k, "ef": body.ef, "len": body.vectors.len()}),
            );
            resp
        }
        Err(e) => {
            let (status, code) = classify_identify_error(&e);
            let resp = error_response(status, &request_id, code, e.to_string(), None);
            audit_best_effort(
                &state,
                &request_id,
                auth.as_deref(),
                "/v3/identify",
                "identify",
                status.as_u16(),
                started.elapsed().as_secs_f64() * 1000.0,
                serde_json::json!({"error": e.to_string()}),
            );
            resp
        }
    }
}

async fn v2_admin_checkpoint(
    State(state): State<Arc<ProductionState>>,
    headers: HeaderMap,
) -> Response {
    let request_id = request_id(&headers);
    let started = Instant::now();

    let auth = match authorize(
        state.catalog.as_ref(),
        &headers,
        Some(Role::Admin),
        &state.config.api_pepper,
    ) {
        Ok(auth) => auth,
        Err(e) => {
            state
                .metrics
                .auth_failures_total
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return auth_error_response(
                &state,
                &request_id,
                "/v3/admin/checkpoint",
                "checkpoint",
                started,
                e,
            );
        }
    };

    let state_bg = state.clone();
    let result = tokio::task::spawn_blocking(move || state_bg.checkpoint_once()).await;
    let result = match result {
        Ok(inner) => inner,
        Err(e) => Err(anyhow::anyhow!("checkpoint join error: {}", e)),
    };
    match result {
        Ok(job) => {
            let payload = ApiResponse::<JobResponseV2> { data: job };
            let resp = json_response(StatusCode::OK, &request_id, &payload);
            audit_best_effort(
                &state,
                &request_id,
                auth.as_deref(),
                "/v3/admin/checkpoint",
                "checkpoint",
                200,
                started.elapsed().as_secs_f64() * 1000.0,
                serde_json::json!({}),
            );
            resp
        }
        Err(e) => {
            let resp = error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &request_id,
                "checkpoint_failed",
                e.to_string(),
                None,
            );
            audit_best_effort(
                &state,
                &request_id,
                auth.as_deref(),
                "/v3/admin/checkpoint",
                "checkpoint",
                500,
                started.elapsed().as_secs_f64() * 1000.0,
                serde_json::json!({"error": e.to_string()}),
            );
            resp
        }
    }
}

async fn v2_admin_compact(
    State(state): State<Arc<ProductionState>>,
    headers: HeaderMap,
) -> Response {
    let request_id = request_id(&headers);
    let started = Instant::now();

    let auth = match authorize(
        state.catalog.as_ref(),
        &headers,
        Some(Role::Admin),
        &state.config.api_pepper,
    ) {
        Ok(auth) => auth,
        Err(e) => {
            state
                .metrics
                .auth_failures_total
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return auth_error_response(
                &state,
                &request_id,
                "/v3/admin/compact",
                "compact",
                started,
                e,
            );
        }
    };

    let state_bg = state.clone();
    let result = tokio::task::spawn_blocking(move || state_bg.compact_once()).await;
    let result = match result {
        Ok(inner) => inner,
        Err(e) => Err(anyhow::anyhow!("compact join error: {}", e)),
    };
    match result {
        Ok(job) => {
            let payload = ApiResponse::<JobResponseV2> { data: job };
            let resp = json_response(StatusCode::OK, &request_id, &payload);
            audit_best_effort(
                &state,
                &request_id,
                auth.as_deref(),
                "/v3/admin/compact",
                "compact",
                200,
                started.elapsed().as_secs_f64() * 1000.0,
                serde_json::json!({}),
            );
            resp
        }
        Err(e) => {
            let resp = error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &request_id,
                "compact_failed",
                e.to_string(),
                None,
            );
            audit_best_effort(
                &state,
                &request_id,
                auth.as_deref(),
                "/v3/admin/compact",
                "compact",
                500,
                started.elapsed().as_secs_f64() * 1000.0,
                serde_json::json!({"error": e.to_string()}),
            );
            resp
        }
    }
}

async fn v2_admin_snapshot(
    State(state): State<Arc<ProductionState>>,
    headers: HeaderMap,
) -> Response {
    let request_id = request_id(&headers);
    let started = Instant::now();

    let auth = match authorize(
        state.catalog.as_ref(),
        &headers,
        Some(Role::Admin),
        &state.config.api_pepper,
    ) {
        Ok(auth) => auth,
        Err(e) => {
            state
                .metrics
                .auth_failures_total
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return auth_error_response(
                &state,
                &request_id,
                "/v3/admin/snapshot",
                "snapshot",
                started,
                e,
            );
        }
    };

    match create_snapshot(
        &state.config,
        state.catalog.as_ref(),
        &state.collection.name,
    ) {
        Ok(snapshot) => {
            let payload = ApiResponse::<JobResponseV2> {
                data: JobResponseV2 {
                    job_id: snapshot.snapshot_id.clone(),
                    state: "completed".to_string(),
                    details: Some(serde_json::json!({
                        "snapshot_id": snapshot.snapshot_id,
                        "snapshot_dir": snapshot.snapshot_dir,
                        "segments": snapshot.segments,
                    })),
                },
            };
            let resp = json_response(StatusCode::OK, &request_id, &payload);
            audit_best_effort(
                &state,
                &request_id,
                auth.as_deref(),
                "/v3/admin/snapshot",
                "snapshot",
                200,
                started.elapsed().as_secs_f64() * 1000.0,
                serde_json::json!({}),
            );
            resp
        }
        Err(e) => {
            let resp = error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &request_id,
                "snapshot_failed",
                e.to_string(),
                None,
            );
            audit_best_effort(
                &state,
                &request_id,
                auth.as_deref(),
                "/v3/admin/snapshot",
                "snapshot",
                500,
                started.elapsed().as_secs_f64() * 1000.0,
                serde_json::json!({"error": e.to_string()}),
            );
            resp
        }
    }
}

async fn v2_admin_stats(State(state): State<Arc<ProductionState>>, headers: HeaderMap) -> Response {
    let request_id = request_id(&headers);

    match authorize(
        state.catalog.as_ref(),
        &headers,
        Some(Role::Admin),
        &state.config.api_pepper,
    ) {
        Ok(_) => {
            let state_bg = state.clone();
            let result = tokio::task::spawn_blocking(move || state_bg.stats()).await;
            let result = match result {
                Ok(inner) => inner,
                Err(e) => Err(anyhow::anyhow!("stats join error: {}", e)),
            };
            match result {
                Ok(stats) => {
                    json_response(StatusCode::OK, &request_id, &ApiResponse { data: stats })
                }
                Err(e) => error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &request_id,
                    "stats_failed",
                    e.to_string(),
                    None,
                ),
            }
        }
        Err(_) => error_response(
            StatusCode::UNAUTHORIZED,
            &request_id,
            "unauthorized",
            "invalid api key".to_string(),
            None,
        ),
    }
}

async fn v2_health_live(State(state): State<Arc<ProductionState>>, headers: HeaderMap) -> Response {
    let request_id = request_id(&headers);
    let started = Instant::now();
    if !state.config.public_health_metrics {
        if let Err(e) = authorize(
            state.catalog.as_ref(),
            &headers,
            Some(Role::Query),
            &state.config.api_pepper,
        ) {
            state
                .metrics
                .auth_failures_total
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return auth_error_response(
                &state,
                &request_id,
                "/v3/health/live",
                "health_live",
                started,
                e,
            );
        }
    }
    let live = state.live_status();
    let payload = ApiResponse {
        data: HealthResponseV2 {
            status: "ok".to_string(),
            ready: live,
            report: "live".to_string(),
        },
    };
    json_response(StatusCode::OK, &request_id, &payload)
}

async fn v2_health_ready(
    State(state): State<Arc<ProductionState>>,
    headers: HeaderMap,
) -> Response {
    let request_id = request_id(&headers);
    let started = Instant::now();
    if !state.config.public_health_metrics {
        if let Err(e) = authorize(
            state.catalog.as_ref(),
            &headers,
            Some(Role::Query),
            &state.config.api_pepper,
        ) {
            state
                .metrics
                .auth_failures_total
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return auth_error_response(
                &state,
                &request_id,
                "/v3/health/ready",
                "health_ready",
                started,
                e,
            );
        }
    }
    let live = state.live_status();
    let ready = state.ready.load(std::sync::atomic::Ordering::SeqCst) && live;
    let report = if live {
        state.recovery_report.read().clone()
    } else {
        "ingest writer unavailable".to_string()
    };
    let payload = ApiResponse {
        data: HealthResponseV2 {
            status: if ready { "ok" } else { "degraded" }.to_string(),
            ready,
            report,
        },
    };
    let status = if ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    json_response(status, &request_id, &payload)
}

async fn v2_metrics(State(state): State<Arc<ProductionState>>, headers: HeaderMap) -> Response {
    let request_id = request_id(&headers);
    let started = Instant::now();
    if !state.config.public_health_metrics {
        if let Err(e) = authorize(
            state.catalog.as_ref(),
            &headers,
            Some(Role::Admin),
            &state.config.api_pepper,
        ) {
            state
                .metrics
                .auth_failures_total
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return auth_error_response(&state, &request_id, "/v3/metrics", "metrics", started, e);
        }
    }
    match state.render_metrics() {
        Ok(body) => {
            let mut resp = (StatusCode::OK, body).into_response();
            resp.headers_mut().insert(
                "content-type",
                HeaderValue::from_static("text/plain; version=0.0.4"),
            );
            set_request_id(&mut resp, &request_id);
            resp
        }
        Err(e) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            &request_id,
            "metrics_failed",
            e.to_string(),
            None,
        ),
    }
}

async fn v2_deprecated(headers: HeaderMap) -> Response {
    let request_id = request_id(&headers);
    error_response(
        StatusCode::GONE,
        &request_id,
        "v2_deprecated",
        "v2 API is removed; migrate to /v3 endpoints".to_string(),
        None,
    )
}

fn auth_error_response(
    state: &ProductionState,
    request_id: &str,
    endpoint: &str,
    action: &str,
    started: Instant,
    err: AuthError,
) -> Response {
    let (status, code, msg) = match err {
        AuthError::Missing => (
            StatusCode::UNAUTHORIZED,
            "missing_authorization",
            "missing bearer token".to_string(),
        ),
        AuthError::Invalid => (
            StatusCode::UNAUTHORIZED,
            "invalid_authorization",
            "invalid api key".to_string(),
        ),
        AuthError::Forbidden => (
            StatusCode::FORBIDDEN,
            "forbidden",
            "insufficient role".to_string(),
        ),
        AuthError::Internal => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "auth_internal",
            "auth backend failure".to_string(),
        ),
    };

    audit_best_effort(
        state,
        request_id,
        None,
        endpoint,
        action,
        status.as_u16(),
        started.elapsed().as_secs_f64() * 1000.0,
        serde_json::json!({"error": msg}),
    );

    error_response(status, request_id, code, msg, None)
}

fn audit_best_effort(
    state: &ProductionState,
    request_id: &str,
    api_key_id: Option<&str>,
    endpoint: &str,
    action: &str,
    status_code: u16,
    latency_ms: f64,
    details: serde_json::Value,
) {
    state.audit_event_best_effort(
        request_id,
        api_key_id,
        endpoint,
        action,
        status_code,
        latency_ms,
        details,
    );
}

fn request_id(headers: &HeaderMap) -> String {
    headers
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .filter(|v| !v.trim().is_empty())
        .map(|s| s.to_string())
        .unwrap_or_else(new_request_id)
}

fn set_request_id(resp: &mut Response, request_id: &str) {
    if let Ok(hv) = HeaderValue::from_str(request_id) {
        resp.headers_mut().insert("x-request-id", hv);
    }
}

fn json_response<T: serde::Serialize>(
    status: StatusCode,
    request_id: &str,
    payload: &T,
) -> Response {
    let mut resp = (status, Json(payload)).into_response();
    set_request_id(&mut resp, request_id);
    resp
}

fn error_response(
    status: StatusCode,
    request_id: &str,
    code: &'static str,
    message: String,
    details: Option<serde_json::Value>,
) -> Response {
    let payload = ErrorBody {
        code,
        message,
        details,
    };
    json_response(status, request_id, &payload)
}

fn ingest_metadata_to_input(
    state: &ProductionState,
    metadata: &IngestMetadataV3,
) -> std::result::Result<IngestMetadataV3Input, String> {
    let tags = metadata.normalized_tags();
    if tags.len() > state.config.max_tags_per_vector {
        return Err(format!(
            "too many tags: limit={} actual={}",
            state.config.max_tags_per_vector,
            tags.len()
        ));
    }
    for tag in &tags {
        if tag.len() > state.config.max_tag_len {
            return Err(format!(
                "tag too long: max_tag_len={} tag='{}'",
                state.config.max_tag_len, tag
            ));
        }
        if !is_valid_tag(tag) {
            return Err(format!("invalid tag format: '{}'", tag));
        }
    }
    let payload = metadata.decode_payload()?;
    Ok(IngestMetadataV3Input {
        entity_id: metadata.entity_id,
        sequence_ts: metadata.sequence_ts,
        tags,
        payload,
    })
}

fn is_valid_tag(tag: &str) -> bool {
    tag.chars()
        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.' | '/' | ':'))
}

fn validate_idempotency_key(
    state: &ProductionState,
    key: Option<String>,
) -> std::result::Result<Option<String>, String> {
    if let Some(k) = key {
        if k.len() > state.config.max_idempotency_key_len {
            return Err(format!(
                "idempotency key too long: max={} actual={}",
                state.config.max_idempotency_key_len,
                k.len()
            ));
        }
        Ok(Some(k))
    } else {
        Ok(None)
    }
}

fn classify_ingest_error(err: &anyhow::Error) -> (StatusCode, &'static str) {
    let msg = err.to_string();
    let lower = msg.to_ascii_lowercase();
    if lower.contains("tag_registry_overflow") {
        return (StatusCode::TOO_MANY_REQUESTS, "tag_registry_overflow");
    }
    if lower.contains("catalog_read_timeout") {
        return (StatusCode::SERVICE_UNAVAILABLE, "catalog_read_timeout");
    }
    if lower.contains("dimension mismatch") {
        return (StatusCode::BAD_REQUEST, "bad_request");
    }
    if lower.contains("idempotency key too long") {
        return (StatusCode::BAD_REQUEST, "bad_request");
    }
    if lower.contains("resource exhausted") || lower.contains("resource_exhausted") {
        return (StatusCode::TOO_MANY_REQUESTS, "resource_exhausted");
    }
    if lower.contains("join error")
        || lower.contains("panic")
        || lower.contains("data integrity fault")
        || lower.contains("internal")
    {
        return (StatusCode::INTERNAL_SERVER_ERROR, "internal_error");
    }
    if lower.contains("unique constraint failed") || lower.contains("constraint failed") {
        return (StatusCode::CONFLICT, "conflict");
    }
    if lower.contains("sqlite query failed") || lower.contains("sqlite exec failed") {
        return (StatusCode::INTERNAL_SERVER_ERROR, "storage_error");
    }
    (StatusCode::INTERNAL_SERVER_ERROR, "internal_error")
}

fn classify_query_error(err: &anyhow::Error) -> (StatusCode, &'static str) {
    let msg = err.to_string();
    let lower = msg.to_ascii_lowercase();
    if lower.contains("catalog_read_timeout") {
        return (StatusCode::SERVICE_UNAVAILABLE, "catalog_read_timeout");
    }
    if lower.contains("dimension mismatch") {
        return (StatusCode::BAD_REQUEST, "bad_request");
    }
    if lower.contains("join error")
        || lower.contains("panic")
        || lower.contains("data integrity fault")
        || lower.contains("internal")
    {
        return (StatusCode::INTERNAL_SERVER_ERROR, "internal_error");
    }
    if lower.contains("sqlite query failed") || lower.contains("sqlite exec failed") {
        return (StatusCode::INTERNAL_SERVER_ERROR, "storage_error");
    }
    (StatusCode::INTERNAL_SERVER_ERROR, "internal_error")
}

fn classify_identify_error(err: &anyhow::Error) -> (StatusCode, &'static str) {
    classify_query_error(err)
}

fn new_request_id() -> String {
    use rand::RngCore;
    let mut bytes = [0u8; 8];
    rand::thread_rng().fill_bytes(&mut bytes);
    let mut suffix = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        suffix.push_str(&format!("{:02x}", b));
    }
    format!(
        "req_{}_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        suffix
    )
}
