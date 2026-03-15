use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine;
use serde::{Deserialize, Serialize};
use vibrato_core::metadata::VectorMetadataV3;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorBody {
    pub code: &'static str,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiResponse<T> {
    pub data: T,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    pub request_id: String,
    pub api_key_id: Option<String>,
    pub endpoint: String,
    pub action: String,
    pub status_code: u16,
    pub latency_ms: f64,
    pub details: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestRequestV3 {
    pub vector: Vec<f32>,
    #[serde(default)]
    pub metadata: IngestMetadataV3,
    #[serde(default)]
    pub idempotency_key: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IngestMetadataV3 {
    #[serde(default)]
    pub entity_id: u64,
    #[serde(default)]
    pub sequence_ts: u64,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub payload_base64: String,
}

impl IngestMetadataV3 {
    pub fn normalized_tags(&self) -> Vec<String> {
        let mut tags = self.tags.clone();
        for t in &mut tags {
            *t = t.trim().to_ascii_lowercase();
        }
        tags.retain(|t| !t.is_empty());
        tags.sort();
        tags.dedup();
        tags
    }

    pub fn decode_payload(&self) -> std::result::Result<Vec<u8>, String> {
        if self.payload_base64.trim().is_empty() {
            return Ok(Vec::new());
        }
        BASE64_STANDARD
            .decode(self.payload_base64.trim())
            .map_err(|e| format!("invalid payload_base64: {e}"))
    }
}

pub fn encode_payload_base64(payload: &[u8]) -> String {
    if payload.is_empty() {
        String::new()
    } else {
        BASE64_STANDARD.encode(payload)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestResponseV3 {
    pub id: usize,
    pub created: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestBatchRequestV3 {
    pub vectors: Vec<IngestRequestV3>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestBatchResponseV3 {
    pub results: Vec<IngestResponseV3>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryRequestV3 {
    pub vector: Vec<f32>,
    #[serde(default = "default_k")]
    pub k: usize,
    #[serde(default = "default_ef")]
    pub ef: usize,
    #[serde(default = "default_true")]
    pub include_metadata: bool,
    #[serde(default)]
    pub filter: Option<QueryFilter>,
    #[serde(default = "default_search_tier")]
    pub search_tier: SearchTier,
}

fn default_true() -> bool {
    true
}

fn default_k() -> usize {
    10
}

fn default_ef() -> usize {
    50
}

fn default_search_tier() -> SearchTier {
    SearchTier::Active
}

fn default_identify_k() -> usize {
    5
}

fn default_identify_ef() -> usize {
    100
}

fn default_identify_search_tier() -> SearchTier {
    SearchTier::All
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchTier {
    Active,
    All,
    Archive,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QueryFilterV3 {
    #[serde(default)]
    pub tags_any: Vec<String>,
    #[serde(default)]
    pub tags_all: Vec<String>,
}

impl QueryFilterV3 {
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.tags_any.is_empty() && self.tags_all.is_empty()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResultV3 {
    pub id: usize,
    pub score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<MetadataEnvelopeV3>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResponseV3 {
    pub results: Vec<QueryResultV3>,
    pub query_time_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentifyRequestV3 {
    pub vectors: Vec<Vec<f32>>,
    #[serde(default = "default_identify_k")]
    pub k: usize,
    #[serde(default = "default_identify_ef")]
    pub ef: usize,
    #[serde(default = "default_true")]
    pub include_metadata: bool,
    #[serde(default)]
    pub future_steps: usize,
    #[serde(default)]
    pub include_sequence_metadata: bool,
    #[serde(default = "default_identify_search_tier")]
    pub search_tier: SearchTier,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentifySequenceSpanV3 {
    pub start_id: usize,
    pub end_id: usize,
    pub length: usize,
    pub start_timestamp_ms: u64,
    pub duration_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entity_id: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentifyResultV3 {
    pub id: usize,
    pub start_timestamp_ms: u64,
    pub duration_ms: u64,
    pub score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<MetadataEnvelopeV3>,
    pub matched_sequence: IdentifySequenceSpanV3,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub future_sequence: Option<IdentifySequenceSpanV3>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub matched_sequence_metadata: Option<Vec<MetadataEnvelopeV3>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub future_sequence_metadata: Option<Vec<MetadataEnvelopeV3>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentifyResponseV3 {
    pub results: Vec<IdentifyResultV3>,
    pub query_time_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataEnvelopeV3 {
    pub entity_id: u64,
    pub sequence_ts: u64,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub payload_base64: String,
}

impl MetadataEnvelopeV3 {
    pub fn from_internal(metadata: &VectorMetadataV3, tags: Vec<String>) -> Self {
        Self {
            entity_id: metadata.entity_id,
            sequence_ts: metadata.sequence_ts,
            tags,
            payload_base64: encode_payload_base64(&metadata.payload),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatsResponseV2 {
    pub ready: bool,
    pub live: bool,
    pub active_segments: usize,
    pub obsolete_segments: usize,
    pub failed_segments: usize,
    pub hot_vectors: usize,
    pub wal_pending: usize,
    pub total_vectors: usize,
    pub checkpoint_jobs_inflight: usize,
    pub compaction_jobs_inflight: usize,
    pub sqlite_wal_bytes: u64,
    pub catalog_read_timeout_total: u64,
    pub quarantine_files: usize,
    pub quarantine_bytes: u64,
    pub quarantine_evictions_total: u64,
    pub ingest_queue_bytes: u64,
    pub memory_proxy_bytes: u64,
    pub memory_budget_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobResponseV2 {
    pub job_id: String,
    pub state: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponseV2 {
    pub status: String,
    pub ready: bool,
    pub report: String,
}

// Temporary compatibility aliases while internal call-sites migrate.
pub type IngestRequestV2 = IngestRequestV3;
pub type IngestMetadata = IngestMetadataV3;
pub type IngestResponseV2 = IngestResponseV3;
pub type IngestBatchRequestV2 = IngestBatchRequestV3;
pub type IngestBatchResponseV2 = IngestBatchResponseV3;
pub type QueryRequestV2 = QueryRequestV3;
pub type QueryFilter = QueryFilterV3;
pub type QueryResultV2 = QueryResultV3;
pub type QueryResponseV2 = QueryResponseV3;
pub type IdentifyRequestV2 = IdentifyRequestV3;
pub type IdentifyResultV2 = IdentifyResultV3;
pub type IdentifyResponseV2 = IdentifyResponseV3;
