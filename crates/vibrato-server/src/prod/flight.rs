use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::atomic::Ordering as AtomicOrdering;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use arrow_array::{
    Array, FixedSizeListArray, Float32Array, ListArray, RecordBatch, StringArray, UInt16Array,
    UInt32Array,
};
use arrow_flight::decode::FlightRecordBatchStream;
use arrow_flight::flight_service_server::{FlightService, FlightServiceServer};
use arrow_flight::{
    Action, ActionType, Criteria, Empty, FlightData, FlightDescriptor, FlightInfo,
    HandshakeRequest, HandshakeResponse, PollInfo, PutResult, SchemaResult, Ticket,
};
use futures::{stream, Stream, StreamExt, TryStreamExt};
use tonic::{Request, Response, Status};
use vibrato_core::metadata::VectorMetadata;

use super::catalog::{parse_token, verify_token_hash, CatalogStore, Role};
use super::engine::{IngestBackpressureDecision, ProductionState};
use super::model::IngestMetadata;

type TonicStream<T> = Pin<Box<dyn Stream<Item = std::result::Result<T, Status>> + Send + 'static>>;

pub async fn start_flight_server(state: Arc<ProductionState>, addr: SocketAddr) -> Result<()> {
    let service = VibratoFlightService { state };
    tonic::transport::Server::builder()
        .add_service(FlightServiceServer::new(service))
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

fn as_u32_array<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> std::result::Result<Option<&'a UInt32Array>, Status> {
    let Some(idx) = batch.schema().index_of(name).ok() else {
        return Ok(None);
    };
    batch
        .column(idx)
        .as_any()
        .downcast_ref::<UInt32Array>()
        .map(Some)
        .ok_or_else(|| Status::invalid_argument(format!("column '{name}' must be UInt32")))
}

fn as_u16_array<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> std::result::Result<Option<&'a UInt16Array>, Status> {
    let Some(idx) = batch.schema().index_of(name).ok() else {
        return Ok(None);
    };
    batch
        .column(idx)
        .as_any()
        .downcast_ref::<UInt16Array>()
        .map(Some)
        .ok_or_else(|| Status::invalid_argument(format!("column '{name}' must be UInt16")))
}

fn as_f32_array<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> std::result::Result<Option<&'a Float32Array>, Status> {
    let Some(idx) = batch.schema().index_of(name).ok() else {
        return Ok(None);
    };
    batch
        .column(idx)
        .as_any()
        .downcast_ref::<Float32Array>()
        .map(Some)
        .ok_or_else(|| Status::invalid_argument(format!("column '{name}' must be Float32")))
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

fn extract_batch_entries(
    batch: &RecordBatch,
    dim: usize,
) -> std::result::Result<Vec<(Vec<f32>, VectorMetadata, Option<String>)>, Status> {
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
    let vector_values_slice = vector_values.values();

    let source_file = as_string_array(batch, "source_file")?;
    let start_time_ms = as_u32_array(batch, "start_time_ms")?;
    let duration_ms = as_u16_array(batch, "duration_ms")?;
    let bpm = as_f32_array(batch, "bpm")?;
    let tags = as_list_utf8_array(batch, "tags")?;
    let metadata_json = as_string_array(batch, "metadata_json")?;
    let idempotency_key = as_string_array(batch, "idempotency_key")?;

    let mut entries = Vec::with_capacity(batch.num_rows());
    for row in 0..batch.num_rows() {
        if vectors.is_null(row) {
            return Err(Status::invalid_argument("vector row must not be null"));
        }
        let start = row * dim;
        let vector = vector_values_slice[start..start + dim].to_vec();

        let mut metadata = if let Some(col) = metadata_json {
            if !col.is_null(row) {
                serde_json::from_str::<IngestMetadata>(col.value(row)).map_err(|e| {
                    Status::invalid_argument(format!("invalid metadata_json at row {row}: {e}"))
                })?
            } else {
                IngestMetadata::default()
            }
        } else {
            IngestMetadata::default()
        };

        if let Some(col) = source_file {
            if !col.is_null(row) {
                metadata.source_file = col.value(row).to_string();
            }
        }
        if let Some(col) = start_time_ms {
            if !col.is_null(row) {
                metadata.start_time_ms = col.value(row);
            }
        }
        if let Some(col) = duration_ms {
            if !col.is_null(row) {
                metadata.duration_ms = col.value(row);
            }
        }
        if let Some(col) = bpm {
            if !col.is_null(row) {
                metadata.bpm = col.value(row);
            }
        }
        if let Some(col) = tags {
            if !col.is_null(row) {
                let tag_values = col.value(row);
                let tag_strings = tag_values
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or_else(|| {
                        Status::invalid_argument("tags column must be List<Utf8> for Flight ingest")
                    })?;
                let mut parsed = Vec::with_capacity(tag_strings.len());
                for i in 0..tag_strings.len() {
                    if !tag_strings.is_null(i) {
                        parsed.push(tag_strings.value(i).to_string());
                    }
                }
                metadata.tags = parsed;
            }
        }

        let idempotency = idempotency_key.and_then(|col| {
            if col.is_null(row) {
                None
            } else {
                Some(col.value(row).to_string())
            }
        });

        entries.push((vector, metadata.into(), idempotency));
    }

    Ok(entries)
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
            let ingest_result = tokio::task::spawn_blocking(
                move || -> std::result::Result<Vec<(usize, bool)>, Status> {
                    let entries = extract_batch_entries(&batch, dim)?;
                    state_bg
                        .ingest_batch_owned(entries)
                        .map_err(map_ingest_error)
                },
            )
            .await;
            self.state
                .inflight_decode_bytes
                .fetch_sub(incoming_bytes, AtomicOrdering::Relaxed);

            let ingest_result = ingest_result
                .map_err(|e| Status::internal(format!("flight ingest task join error: {e}")))??;
            accepted += ingest_result.len();
            created += ingest_result
                .iter()
                .filter(|(_, was_created)| *was_created)
                .count();
        }

        let metadata = serde_json::json!({
            "collection_id": self.state.collection.id,
            "accepted": accepted,
            "created": created,
            "api_key_id": api_key_id,
        })
        .to_string()
        .into_bytes();

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

    use arrow_array::types::Float32Type;
    use arrow_array::{FixedSizeListArray, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};

    use super::{estimate_batch_ingest_bytes, extract_batch_entries};

    fn make_batch() -> RecordBatch {
        let vectors = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
            vec![
                Some(vec![Some(1.0f32), Some(0.0), Some(0.0), Some(0.0)]),
                Some(vec![Some(0.0f32), Some(1.0), Some(0.0), Some(0.0)]),
            ],
            4,
        );
        let metadata_json = StringArray::from(vec![
            Some(
                r#"{"source_file":"a.wav","start_time_ms":10,"duration_ms":5,"bpm":120.0,"tags":["x","y"]}"#,
            ),
            None,
        ]);
        let idempotency_key = StringArray::from(vec![Some("k1"), None]);

        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "vector",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 4),
                false,
            ),
            Field::new("metadata_json", DataType::Utf8, true),
            Field::new("idempotency_key", DataType::Utf8, true),
        ]));

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(vectors),
                Arc::new(metadata_json),
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
        assert_eq!(entries[0].1.source_file, "a.wav");
        assert_eq!(entries[0].1.start_time_ms, 10);
        assert_eq!(entries[0].1.duration_ms, 5);
        assert_eq!(entries[0].1.bpm, 120.0);
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
    fn estimate_batch_ingest_bytes_accounts_for_metadata_payload() {
        let batch = make_batch();
        let raw_vector_bytes = (batch.num_rows() as u64) * 4 * std::mem::size_of::<f32>() as u64;
        let estimated = estimate_batch_ingest_bytes(&batch, 4).expect("estimate bytes");
        assert!(
            estimated > raw_vector_bytes,
            "expected metadata-aware estimate to exceed raw vector payload"
        );
    }
}
