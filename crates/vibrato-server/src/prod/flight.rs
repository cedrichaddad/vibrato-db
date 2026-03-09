use std::collections::HashMap;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::atomic::Ordering as AtomicOrdering;
use std::sync::Arc;
use std::time::Instant;

use super::catalog::{parse_token, verify_token_hash, CatalogStore, IngestMetadataV3Input, Role};
use super::engine::{IngestBackpressureDecision, ProductionState};
use anyhow::{anyhow, Result};
use arrow_array::{
    types::{
        ArrowDictionaryKeyType, Int16Type, Int32Type, Int64Type, Int8Type, UInt16Type, UInt32Type,
        UInt64Type, UInt8Type,
    },
    Array, BinaryArray, DictionaryArray, FixedSizeListArray, Float32Array, ListArray, RecordBatch,
    StringArray, UInt64Array,
};
use arrow_buffer::ArrowNativeType;
use arrow_flight::decode::FlightRecordBatchStream;
use arrow_flight::flight_service_server::{FlightService, FlightServiceServer};
use arrow_flight::{
    Action, ActionType, Criteria, Empty, FlightData, FlightDescriptor, FlightInfo,
    HandshakeRequest, HandshakeResponse, PollInfo, PutResult, SchemaResult, Ticket,
};
use arrow_ipc::writer::StreamWriter;
use arrow_schema::DataType;
use futures::{stream, Stream, StreamExt, TryStreamExt};
use tokio::sync::oneshot;
use tonic::{Request, Response, Status};

type TonicStream<T> = Pin<Box<dyn Stream<Item = std::result::Result<T, Status>> + Send + 'static>>;
const FLIGHT_DECODE_CHUNK_ROWS: usize = 10_000;
const FLIGHT_MAX_DECODING_MESSAGE_BYTES: usize = 256 * 1024 * 1024;
const FLIGHT_MAX_ENCODING_MESSAGE_BYTES: usize = 64 * 1024 * 1024;

struct ConnectionGaugeGuard {
    state: Arc<ProductionState>,
}

impl ConnectionGaugeGuard {
    fn new(state: Arc<ProductionState>) -> Self {
        state.flight_stream_opened();
        Self { state }
    }
}

impl Drop for ConnectionGaugeGuard {
    fn drop(&mut self) {
        self.state.flight_stream_closed();
    }
}

pub async fn start_flight_server(state: Arc<ProductionState>, addr: SocketAddr) -> Result<()> {
    let service = FlightServiceServer::new(VibratoFlightService { state })
        .max_decoding_message_size(FLIGHT_MAX_DECODING_MESSAGE_BYTES)
        .max_encoding_message_size(FLIGHT_MAX_ENCODING_MESSAGE_BYTES);
    tonic::transport::Server::builder()
        .add_service(service)
        .serve(addr)
        .await
        .map_err(|e| anyhow!("arrow flight server failed on {addr}: {e}"))
}

struct VibratoFlightService {
    state: Arc<ProductionState>,
}

fn authorize_flight_ingest(
    state: &ProductionState,
    metadata: &tonic::metadata::MetadataMap,
) -> std::result::Result<String, Status> {
    let auth = metadata
        .get("authorization")
        .ok_or_else(|| Status::unauthenticated("missing authorization metadata"))?;
    let auth = auth
        .to_str()
        .map_err(|_| Status::unauthenticated("invalid authorization metadata"))?;
    let (key_id, secret) =
        parse_token(auth).ok_or_else(|| Status::unauthenticated("invalid bearer token"))?;
    let key = state
        .catalog
        .lookup_api_key(&key_id)
        .map_err(|e| Status::internal(format!("api key lookup failed: {e}")))?
        .ok_or_else(|| Status::unauthenticated("unknown api key"))?;
    if key.revoked {
        return Err(Status::permission_denied("api key revoked"));
    }
    if !verify_token_hash(&state.config.api_pepper, &secret, &key.key_hash) {
        return Err(Status::unauthenticated("invalid api key secret"));
    }
    let has_role = key
        .roles
        .iter()
        .any(|r| r == &Role::Admin || r == &Role::Ingest);
    if !has_role {
        return Err(Status::permission_denied("api key missing ingest role"));
    }
    Ok(key.id)
}

fn as_string_array<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> std::result::Result<Option<&'a StringArray>, Status> {
    let Some(idx) = batch.schema().index_of(name).ok() else {
        return Ok(None);
    };
    batch
        .column(idx)
        .as_any()
        .downcast_ref::<StringArray>()
        .map(Some)
        .ok_or_else(|| Status::invalid_argument(format!("column '{name}' must be Utf8")))
}

fn as_u64_array<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> std::result::Result<Option<&'a UInt64Array>, Status> {
    let Some(idx) = batch.schema().index_of(name).ok() else {
        return Ok(None);
    };
    batch
        .column(idx)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .map(Some)
        .ok_or_else(|| Status::invalid_argument(format!("column '{name}' must be UInt64")))
}

fn as_binary_array<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> std::result::Result<Option<&'a BinaryArray>, Status> {
    let Some(idx) = batch.schema().index_of(name).ok() else {
        return Ok(None);
    };
    batch
        .column(idx)
        .as_any()
        .downcast_ref::<BinaryArray>()
        .map(Some)
        .ok_or_else(|| Status::invalid_argument(format!("column '{name}' must be Binary")))
}

fn as_list_utf8_array<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> std::result::Result<Option<&'a ListArray>, Status> {
    let Some(idx) = batch.schema().index_of(name).ok() else {
        return Ok(None);
    };
    batch
        .column(idx)
        .as_any()
        .downcast_ref::<ListArray>()
        .map(Some)
        .ok_or_else(|| Status::invalid_argument(format!("column '{name}' must be List<Utf8>")))
}

struct FlightBatchColumns<'a> {
    dim: usize,
    num_rows: usize,
    vectors: &'a FixedSizeListArray,
    vector_values: &'a [f32],
    entity_id: &'a UInt64Array,
    sequence_ts: &'a UInt64Array,
    payload: &'a BinaryArray,
    tags: Option<&'a ListArray>,
    idempotency_key: Option<&'a StringArray>,
}

struct FlightBatchIngestResult {
    results: Vec<(usize, bool)>,
    decode_us: u64,
    commit_us: u64,
}

fn elapsed_micros_u64(duration: std::time::Duration) -> u64 {
    duration.as_micros().min(u64::MAX as u128) as u64
}

fn serialize_batch_to_ipc(batch: &RecordBatch) -> std::result::Result<Vec<u8>, Status> {
    let mut out = Vec::new();
    let mut writer = StreamWriter::try_new(&mut out, &batch.schema())
        .map_err(|e| Status::internal(format!("ipc writer init failed: {e}")))?;
    writer
        .write(batch)
        .map_err(|e| Status::internal(format!("ipc writer write failed: {e}")))?;
    writer
        .finish()
        .map_err(|e| Status::internal(format!("ipc writer finish failed: {e}")))?;
    drop(writer);
    Ok(out)
}

fn parse_flight_batch_columns<'a>(
    batch: &'a RecordBatch,
    dim: usize,
) -> std::result::Result<FlightBatchColumns<'a>, Status> {
    let vector_idx = batch
        .schema()
        .index_of("vector")
        .map_err(|_| Status::invalid_argument("missing required 'vector' column"))?;
    let vectors = batch
        .column(vector_idx)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .ok_or_else(|| {
            Status::invalid_argument("column 'vector' must be FixedSizeList<Float32>")
        })?;
    if vectors.value_length() as usize != dim {
        return Err(Status::invalid_argument(format!(
            "vector dimension mismatch: expected {dim}, got {}",
            vectors.value_length()
        )));
    }
    let vector_values = vectors
        .values()
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| {
            Status::invalid_argument("column 'vector' values must be Float32 for Flight ingest")
        })?;

    let tags = as_list_utf8_array(batch, "tags")?;
    let entity_id = as_u64_array(batch, "entity_id")?
        .ok_or_else(|| Status::invalid_argument("missing required 'entity_id' column"))?;
    let sequence_ts = as_u64_array(batch, "sequence_ts")?
        .ok_or_else(|| Status::invalid_argument("missing required 'sequence_ts' column"))?;
    let payload = as_binary_array(batch, "payload")?
        .ok_or_else(|| Status::invalid_argument("missing required 'payload' column"))?;

    Ok(FlightBatchColumns {
        dim,
        num_rows: batch.num_rows(),
        vectors,
        vector_values: vector_values.values(),
        entity_id,
        sequence_ts,
        payload,
        tags,
        idempotency_key: as_string_array(batch, "idempotency_key")?,
    })
}

fn extract_dict_tags<
    K: ArrowDictionaryKeyType,
    F: FnMut(&str) -> std::result::Result<(), Status>,
>(
    dict: &DictionaryArray<K>,
    start: usize,
    end: usize,
    visit: &mut F,
) -> std::result::Result<(), Status> {
    let values = dict
        .values()
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| Status::invalid_argument("tags dictionary values must be Utf8"))?;
    let keys = dict.keys();
    for idx in start..end {
        if keys.is_null(idx) {
            continue;
        }
        let value_idx = keys.value(idx).as_usize();
        if value_idx >= values.len() || values.is_null(value_idx) {
            continue;
        }
        visit(values.value(value_idx))?;
    }
    Ok(())
}

fn visit_row_tags<F: FnMut(&str) -> std::result::Result<(), Status>>(
    list: &ListArray,
    row: usize,
    visit: &mut F,
) -> std::result::Result<(), Status> {
    if list.is_null(row) {
        return Ok(());
    }
    let offsets = list.value_offsets();
    let start = offsets[row] as usize;
    let end = offsets[row + 1] as usize;
    let values = list.values();
    if let Some(strings) = values.as_any().downcast_ref::<StringArray>() {
        for idx in start..end {
            if !strings.is_null(idx) {
                visit(strings.value(idx))?;
            }
        }
        return Ok(());
    }

    match values.data_type() {
        DataType::Dictionary(_, _) => {
            if let Some(dict) = values.as_any().downcast_ref::<DictionaryArray<Int8Type>>() {
                return extract_dict_tags(dict, start, end, visit);
            }
            if let Some(dict) = values.as_any().downcast_ref::<DictionaryArray<Int16Type>>() {
                return extract_dict_tags(dict, start, end, visit);
            }
            if let Some(dict) = values.as_any().downcast_ref::<DictionaryArray<Int32Type>>() {
                return extract_dict_tags(dict, start, end, visit);
            }
            if let Some(dict) = values.as_any().downcast_ref::<DictionaryArray<Int64Type>>() {
                return extract_dict_tags(dict, start, end, visit);
            }
            if let Some(dict) = values.as_any().downcast_ref::<DictionaryArray<UInt8Type>>() {
                return extract_dict_tags(dict, start, end, visit);
            }
            if let Some(dict) = values
                .as_any()
                .downcast_ref::<DictionaryArray<UInt16Type>>()
            {
                return extract_dict_tags(dict, start, end, visit);
            }
            if let Some(dict) = values
                .as_any()
                .downcast_ref::<DictionaryArray<UInt32Type>>()
            {
                return extract_dict_tags(dict, start, end, visit);
            }
            if let Some(dict) = values
                .as_any()
                .downcast_ref::<DictionaryArray<UInt64Type>>()
            {
                return extract_dict_tags(dict, start, end, visit);
            }
            Err(Status::invalid_argument(
                "unsupported dictionary key type for tags List<Dictionary<*, Utf8>>",
            ))
        }
        _ => Err(Status::invalid_argument(
            "tags column must be List<Utf8> or List<Dictionary<*, Utf8>>",
        )),
    }
}

#[derive(Clone)]
enum TagCacheEntry {
    Empty,
    Normalized(Arc<str>),
}

#[derive(Default)]
struct TagNormalizationCache {
    by_raw: HashMap<String, TagCacheEntry>,
}

#[inline]
fn is_valid_tag(tag: &str) -> bool {
    tag.chars()
        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.' | '/' | ':'))
}

fn normalize_tag_cached(
    cache: &mut TagNormalizationCache,
    raw: &str,
    max_tag_len: usize,
) -> std::result::Result<Option<Arc<str>>, Status> {
    if let Some(existing) = cache.by_raw.get(raw) {
        return Ok(match existing {
            TagCacheEntry::Empty => None,
            TagCacheEntry::Normalized(tag) => Some(Arc::clone(tag)),
        });
    }

    let normalized = raw.trim().to_ascii_lowercase();
    if normalized.is_empty() {
        cache.by_raw.insert(raw.to_string(), TagCacheEntry::Empty);
        return Ok(None);
    }
    if normalized.len() > max_tag_len {
        return Err(Status::invalid_argument(format!(
            "tag too long: max_tag_len={} tag='{}'",
            max_tag_len, normalized
        )));
    }
    if !is_valid_tag(&normalized) {
        return Err(Status::invalid_argument(format!(
            "invalid tag format: '{}'",
            normalized
        )));
    }
    let normalized = Arc::<str>::from(normalized);
    cache.by_raw.insert(
        raw.to_string(),
        TagCacheEntry::Normalized(Arc::clone(&normalized)),
    );
    Ok(Some(normalized))
}

fn decode_row_tags(
    list: &ListArray,
    row: usize,
    max_tags: usize,
    max_tag_len: usize,
    tag_cache: &mut TagNormalizationCache,
) -> std::result::Result<Vec<String>, Status> {
    let mut unique = Vec::<Arc<str>>::new();
    visit_row_tags(list, row, &mut |raw| {
        if let Some(tag) = normalize_tag_cached(tag_cache, raw, max_tag_len)? {
            if !unique
                .iter()
                .any(|existing| existing.as_ref() == tag.as_ref())
            {
                unique.push(tag);
                if unique.len() > max_tags {
                    return Err(Status::invalid_argument(format!(
                        "too many tags: limit={} actual={}",
                        max_tags,
                        unique.len()
                    )));
                }
            }
        }
        Ok(())
    })?;
    unique.sort_unstable_by(|a, b| a.as_ref().cmp(b.as_ref()));
    Ok(unique.into_iter().map(|tag| tag.to_string()).collect())
}

fn decode_entry_at_row(
    cols: &FlightBatchColumns<'_>,
    row: usize,
    max_tags: usize,
    max_tag_len: usize,
    max_idempotency_key_len: usize,
    tag_cache: &mut TagNormalizationCache,
) -> std::result::Result<(Vec<f32>, IngestMetadataV3Input, Option<String>), Status> {
    if cols.vectors.is_null(row) {
        return Err(Status::invalid_argument("vector row must not be null"));
    }

    let start = row * cols.dim;
    let vector = cols.vector_values[start..start + cols.dim].to_vec();

    if cols.entity_id.is_null(row) || cols.sequence_ts.is_null(row) || cols.payload.is_null(row) {
        return Err(Status::invalid_argument(
            "entity_id, sequence_ts, and payload must be non-null",
        ));
    }
    let entity_id = cols.entity_id.value(row);
    let sequence_ts = cols.sequence_ts.value(row);
    let payload = cols.payload.value(row).to_vec();
    let tags = if let Some(col) = cols.tags {
        decode_row_tags(col, row, max_tags, max_tag_len, tag_cache)?
    } else {
        Vec::new()
    };

    let idempotency = cols.idempotency_key.and_then(|col| {
        if col.is_null(row) {
            None
        } else {
            Some(col.value(row).to_string())
        }
    });
    if let Some(key) = idempotency.as_ref() {
        if key.len() > max_idempotency_key_len {
            return Err(Status::invalid_argument(format!(
                "idempotency key too long: max={} actual={}",
                max_idempotency_key_len,
                key.len()
            )));
        }
    }

    Ok((
        vector,
        IngestMetadataV3Input {
            entity_id,
            sequence_ts,
            tags,
            payload,
        },
        idempotency,
    ))
}

#[cfg(test)]
fn extract_batch_entries(
    batch: &RecordBatch,
    dim: usize,
) -> std::result::Result<Vec<(Vec<f32>, IngestMetadataV3Input, Option<String>)>, Status> {
    let cols = parse_flight_batch_columns(batch, dim)?;
    let mut entries = Vec::with_capacity(cols.num_rows);
    let mut tag_cache = TagNormalizationCache::default();
    for row in 0..cols.num_rows {
        entries.push(decode_entry_at_row(&cols, row, 64, 64, 64, &mut tag_cache)?);
    }

    Ok(entries)
}

fn ingest_flight_batch_streaming(
    state: &ProductionState,
    batch: RecordBatch,
    dim: usize,
) -> std::result::Result<FlightBatchIngestResult, Status> {
    let cols = parse_flight_batch_columns(&batch, dim)?;
    let mut results = Vec::with_capacity(cols.num_rows);
    let chunk_capacity = FLIGHT_DECODE_CHUNK_ROWS.min(cols.num_rows.max(1));
    let mut decode_us = 0u64;
    let mut commit_us = 0u64;
    let mut row = 0usize;

    // Pipeline: decode chunk N+1 while chunk N commits via ingest_batch_owned.
    // The channel depth of 1 keeps memory bounded while still overlapping work.
    let mut pending_commit: Option<(
        std::sync::mpsc::Receiver<Result<Vec<(usize, bool)>>>,
        std::time::Instant,
        usize, // row count for error messages
    )> = None;

    while row < cols.num_rows {
        let end = (row + FLIGHT_DECODE_CHUNK_ROWS).min(cols.num_rows);
        let decode_started = Instant::now();
        let mut tag_cache = TagNormalizationCache::default();
        let mut chunk = Vec::with_capacity(chunk_capacity);
        for idx in row..end {
            chunk.push(decode_entry_at_row(
                &cols,
                idx,
                state.config.max_tags_per_vector,
                state.config.max_tag_len,
                state.config.max_idempotency_key_len,
                &mut tag_cache,
            )?);
        }
        let decode_elapsed = decode_started.elapsed();
        decode_us = decode_us.saturating_add(elapsed_micros_u64(decode_elapsed));
        let decode_slow = decode_elapsed.as_millis() as u64 > state.config.flight_decode_warn_ms;
        if decode_slow {
            state
                .metrics
                .flight_decode_chunk_warn_total
                .fetch_add(1, AtomicOrdering::Relaxed);
            tracing::warn!(
                "flight_decode_chunk_slow ms={} rows={} threshold_ms={}",
                decode_elapsed.as_millis(),
                end.saturating_sub(row),
                state.config.flight_decode_warn_ms
            );
            std::thread::yield_now();
        }

        // Collect the previous commit result before starting the next one.
        if let Some((rx, commit_started, _prev_count)) = pending_commit.take() {
            let mut batch_results = rx
                .recv()
                .map_err(|_| Status::internal("pipeline commit channel dropped"))?
                .map_err(map_ingest_error)?;
            commit_us = commit_us.saturating_add(elapsed_micros_u64(commit_started.elapsed()));
            results.append(&mut batch_results);
        }

        // Dispatch this chunk's commit without blocking decode of the next chunk.
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        let commit_started = Instant::now();
        let _ = tx.send(state.ingest_batch_owned(chunk));
        pending_commit = Some((rx, commit_started, end - row));

        row = end;
    }

    // Drain the final pending commit.
    if let Some((rx, commit_started, _count)) = pending_commit.take() {
        let mut batch_results = rx
            .recv()
            .map_err(|_| Status::internal("pipeline commit channel dropped"))?
            .map_err(map_ingest_error)?;
        commit_us = commit_us.saturating_add(elapsed_micros_u64(commit_started.elapsed()));
        results.append(&mut batch_results);
    }

    // Persist the original Arrow payload once per Flight batch for WAL bridge/recovery.
    let wal_started = Instant::now();
    let ipc_blob = serialize_batch_to_ipc(&batch)?;
    state
        .catalog
        .ingest_wal_ipc_batch(&state.collection.id, cols.num_rows, &ipc_blob)
        .map_err(|e| Status::internal(format!("ipc wal append failed: {e}")))?;
    commit_us = commit_us.saturating_add(elapsed_micros_u64(wal_started.elapsed()));

    Ok(FlightBatchIngestResult {
        results,
        decode_us,
        commit_us,
    })
}

fn estimate_batch_ingest_bytes(
    batch: &RecordBatch,
    dim: usize,
) -> std::result::Result<u64, Status> {
    let vector_idx = batch
        .schema()
        .index_of("vector")
        .map_err(|_| Status::invalid_argument("missing required 'vector' column"))?;
    let vectors = batch
        .column(vector_idx)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .ok_or_else(|| {
            Status::invalid_argument("column 'vector' must be FixedSizeList<Float32>")
        })?;
    if vectors.value_length() as usize != dim {
        return Err(Status::invalid_argument(format!(
            "vector dimension mismatch: expected {dim}, got {}",
            vectors.value_length()
        )));
    }

    // O(columns) memory proxy using Arrow's own accounting for array buffer sizes.
    // This avoids O(rows) scans on the async path while still reflecting payload scale.
    let payload_bytes = batch.columns().iter().fold(0u64, |acc, col| {
        acc.saturating_add(col.get_array_memory_size() as u64)
    });
    let decode_overhead = (batch.num_rows() as u64).saturating_mul(32);
    Ok(payload_bytes.saturating_add(decode_overhead))
}

fn map_ingest_error(err: anyhow::Error) -> Status {
    let msg = err.to_string();
    if msg.contains("dimension mismatch") {
        return Status::invalid_argument(msg);
    }
    if msg.contains("idempotency key too long") {
        return Status::invalid_argument(msg);
    }
    if msg.contains("tag_registry_overflow") {
        return Status::resource_exhausted(msg);
    }
    if msg.contains("resource exhausted") {
        return Status::resource_exhausted(msg);
    }
    Status::internal(msg)
}

#[tonic::async_trait]
impl FlightService for VibratoFlightService {
    type HandshakeStream = TonicStream<HandshakeResponse>;
    type ListFlightsStream = TonicStream<FlightInfo>;
    type DoGetStream = TonicStream<FlightData>;
    type DoPutStream = TonicStream<PutResult>;
    type DoExchangeStream = TonicStream<FlightData>;
    type DoActionStream = TonicStream<arrow_flight::Result>;
    type ListActionsStream = TonicStream<ActionType>;

    async fn handshake(
        &self,
        _request: Request<tonic::Streaming<HandshakeRequest>>,
    ) -> std::result::Result<Response<Self::HandshakeStream>, Status> {
        Err(Status::unimplemented("handshake is not implemented"))
    }

    async fn list_flights(
        &self,
        _request: Request<Criteria>,
    ) -> std::result::Result<Response<Self::ListFlightsStream>, Status> {
        Err(Status::unimplemented("list_flights is not implemented"))
    }

    async fn get_flight_info(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> std::result::Result<Response<FlightInfo>, Status> {
        Err(Status::unimplemented("get_flight_info is not implemented"))
    }

    async fn poll_flight_info(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> std::result::Result<Response<PollInfo>, Status> {
        Err(Status::unimplemented("poll_flight_info is not implemented"))
    }

    async fn get_schema(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> std::result::Result<Response<SchemaResult>, Status> {
        Err(Status::unimplemented("get_schema is not implemented"))
    }

    async fn do_get(
        &self,
        _request: Request<Ticket>,
    ) -> std::result::Result<Response<Self::DoGetStream>, Status> {
        Err(Status::unimplemented("do_get is not implemented"))
    }

    async fn do_put(
        &self,
        request: Request<tonic::Streaming<FlightData>>,
    ) -> std::result::Result<Response<Self::DoPutStream>, Status> {
        let _connection_guard = ConnectionGaugeGuard::new(self.state.clone());
        let api_key_id = authorize_flight_ingest(&self.state, request.metadata())?;

        let mut batch_stream =
            FlightRecordBatchStream::new_from_flight_data(request.into_inner().map_err(Into::into));

        let mut accepted = 0usize;
        let mut created = 0usize;
        while let Some(batch_result) = batch_stream.next().await {
            let batch = batch_result
                .map_err(|e| Status::invalid_argument(format!("invalid flight payload: {e}")))?;
            if batch.num_rows() == 0 {
                continue;
            }
            let row_count = batch.num_rows() as u64;
            let incoming_bytes = estimate_batch_ingest_bytes(&batch, self.state.collection.dim)?;
            match self.state.ingest_backpressure_decision(incoming_bytes) {
                IngestBackpressureDecision::Reject { .. } => {
                    self.state.record_ingest_hard_reject();
                    return Err(Status::resource_exhausted(
                        "memory budget exceeded for flight ingest",
                    ));
                }
                IngestBackpressureDecision::Throttle { delay, .. } => {
                    self.state.record_ingest_soft_throttle(delay);
                    tokio::time::sleep(delay).await;
                }
                IngestBackpressureDecision::Allow => {}
            }

            self.state
                .inflight_decode_bytes
                .fetch_add(incoming_bytes, AtomicOrdering::Relaxed);
            let state_bg = self.state.clone();
            let dim = self.state.collection.dim;
            let (result_tx, result_rx) = oneshot::channel();
            self.state.flight_decode_pool.spawn_fifo(move || {
                let _ = result_tx.send(ingest_flight_batch_streaming(&state_bg, batch, dim));
            });
            let ingest_result = result_rx.await;
            self.state
                .inflight_decode_bytes
                .fetch_sub(incoming_bytes, AtomicOrdering::Relaxed);

            let ingest_result = ingest_result
                .map_err(|_| Status::internal("flight ingest worker dropped before returning"))?;
            let ingest_result = match ingest_result {
                Ok(v) => v,
                Err(status) => {
                    if status.code() == tonic::Code::InvalidArgument {
                        self.state
                            .metrics
                            .tag_reject_invalid_total
                            .fetch_add(1, AtomicOrdering::Relaxed);
                    } else if status.code() == tonic::Code::ResourceExhausted
                        && status.message().contains("tag_registry_overflow")
                    {
                        self.state
                            .metrics
                            .tag_reject_overflow_total
                            .fetch_add(1, AtomicOrdering::Relaxed);
                    }
                    return Err(status);
                }
            };
            accepted += ingest_result.results.len();
            created += ingest_result
                .results
                .iter()
                .filter(|(_, was_created)| *was_created)
                .count();
            self.state
                .metrics
                .flight_ingest_batches_total
                .fetch_add(1, AtomicOrdering::Relaxed);
            self.state
                .metrics
                .flight_ingest_rows_total
                .fetch_add(row_count, AtomicOrdering::Relaxed);
            self.state
                .metrics
                .flight_decode_us_total
                .fetch_add(ingest_result.decode_us, AtomicOrdering::Relaxed);
            self.state
                .metrics
                .flight_commit_us_total
                .fetch_add(ingest_result.commit_us, AtomicOrdering::Relaxed);
            tokio::task::yield_now().await;
        }

        let ack_started = Instant::now();
        let metadata = serde_json::json!({
            "collection_id": self.state.collection.id,
            "accepted": accepted,
            "created": created,
            "api_key_id": api_key_id,
        })
        .to_string()
        .into_bytes();
        self.state.metrics.flight_ack_us_total.fetch_add(
            elapsed_micros_u64(ack_started.elapsed()),
            AtomicOrdering::Relaxed,
        );

        let out_stream = stream::once(async move {
            Ok(PutResult {
                app_metadata: metadata.into(),
            })
        });
        Ok(Response::new(Box::pin(out_stream)))
    }

    async fn do_exchange(
        &self,
        _request: Request<tonic::Streaming<FlightData>>,
    ) -> std::result::Result<Response<Self::DoExchangeStream>, Status> {
        Err(Status::unimplemented("do_exchange is not implemented"))
    }

    async fn do_action(
        &self,
        _request: Request<Action>,
    ) -> std::result::Result<Response<Self::DoActionStream>, Status> {
        Err(Status::unimplemented("do_action is not implemented"))
    }

    async fn list_actions(
        &self,
        _request: Request<Empty>,
    ) -> std::result::Result<Response<Self::ListActionsStream>, Status> {
        Err(Status::unimplemented("list_actions is not implemented"))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::builder::{ListBuilder, StringBuilder};
    use arrow_array::types::Float32Type;
    use arrow_array::{BinaryArray, FixedSizeListArray, RecordBatch, StringArray, UInt64Array};
    use arrow_schema::{DataType, Field, Schema};

    use super::{
        estimate_batch_ingest_bytes, extract_batch_entries, FLIGHT_MAX_DECODING_MESSAGE_BYTES,
        FLIGHT_MAX_ENCODING_MESSAGE_BYTES,
    };

    fn make_batch() -> RecordBatch {
        let vectors = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
            vec![
                Some(vec![Some(1.0f32), Some(0.0), Some(0.0), Some(0.0)]),
                Some(vec![Some(0.0f32), Some(1.0), Some(0.0), Some(0.0)]),
            ],
            4,
        );
        let entity_id = UInt64Array::from(vec![Some(42), Some(99)]);
        let sequence_ts = UInt64Array::from(vec![Some(10), Some(20)]);
        let payload = BinaryArray::from(vec![Some(b"abc".as_slice()), Some(&[][..])]);
        let mut tags_builder = ListBuilder::new(StringBuilder::new());
        tags_builder.values().append_value("x");
        tags_builder.values().append_value("y");
        tags_builder.append(true);
        tags_builder.append(false);
        let tags = tags_builder.finish();
        let idempotency_key = StringArray::from(vec![Some("k1"), None]);

        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "vector",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 4),
                false,
            ),
            Field::new("entity_id", DataType::UInt64, true),
            Field::new("sequence_ts", DataType::UInt64, true),
            Field::new("payload", DataType::Binary, true),
            Field::new(
                "tags",
                DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                true,
            ),
            Field::new("idempotency_key", DataType::Utf8, true),
        ]));

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(vectors),
                Arc::new(entity_id),
                Arc::new(sequence_ts),
                Arc::new(payload),
                Arc::new(tags),
                Arc::new(idempotency_key),
            ],
        )
        .expect("record batch")
    }

    fn make_batch_with_idempotency(idempotency: Option<&str>) -> RecordBatch {
        let vectors = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
            vec![Some(vec![Some(1.0f32), Some(0.0), Some(0.0), Some(0.0)])],
            4,
        );
        let entity_id = UInt64Array::from(vec![Some(42)]);
        let sequence_ts = UInt64Array::from(vec![Some(10)]);
        let payload = BinaryArray::from(vec![Some(b"abc".as_slice())]);
        let mut tags_builder = ListBuilder::new(StringBuilder::new());
        tags_builder.append(false);
        let tags = tags_builder.finish();
        let idempotency_key = StringArray::from(vec![idempotency]);

        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "vector",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 4),
                false,
            ),
            Field::new("entity_id", DataType::UInt64, true),
            Field::new("sequence_ts", DataType::UInt64, true),
            Field::new("payload", DataType::Binary, true),
            Field::new(
                "tags",
                DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                true,
            ),
            Field::new("idempotency_key", DataType::Utf8, true),
        ]));

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(vectors),
                Arc::new(entity_id),
                Arc::new(sequence_ts),
                Arc::new(payload),
                Arc::new(tags),
                Arc::new(idempotency_key),
            ],
        )
        .expect("record batch")
    }

    #[test]
    fn extract_batch_entries_parses_rows() {
        let batch = make_batch();
        let entries = extract_batch_entries(&batch, 4).expect("extract");
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].0, vec![1.0, 0.0, 0.0, 0.0]);
        assert_eq!(entries[0].1.entity_id, 42);
        assert_eq!(entries[0].1.sequence_ts, 10);
        assert_eq!(entries[0].1.payload, b"abc".to_vec());
        assert_eq!(entries[0].1.tags, vec!["x".to_string(), "y".to_string()]);
        assert_eq!(entries[0].2.as_deref(), Some("k1"));
        assert_eq!(entries[1].2, None);
    }

    #[test]
    fn extract_batch_entries_rejects_dim_mismatch() {
        let batch = make_batch();
        let err = extract_batch_entries(&batch, 8).expect_err("dim mismatch");
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
    }

    #[test]
    fn extract_batch_entries_rejects_oversized_idempotency_key() {
        let long_key = "x".repeat(65);
        let batch = make_batch_with_idempotency(Some(&long_key));
        let err = extract_batch_entries(&batch, 4).expect_err("oversized idempotency key");
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
        assert!(
            err.message().contains("idempotency key too long"),
            "unexpected error: {}",
            err.message()
        );
    }

    #[test]
    fn estimate_batch_ingest_bytes_accounts_for_metadata_payload() {
        let batch = make_batch();
        let raw_vector_bytes = (batch.num_rows() as u64) * 4 * std::mem::size_of::<f32>() as u64;
        let estimated = estimate_batch_ingest_bytes(&batch, 4).expect("estimate bytes");
        assert!(
            estimated > raw_vector_bytes,
            "expected metadata-aware estimate to exceed raw vector payload"
        );
    }

    #[test]
    fn flight_message_size_limits_match_contract() {
        assert_eq!(FLIGHT_MAX_DECODING_MESSAGE_BYTES, 256 * 1024 * 1024);
        assert_eq!(FLIGHT_MAX_ENCODING_MESSAGE_BYTES, 64 * 1024 * 1024);
    }
}
