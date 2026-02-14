use serde::{Deserialize, Serialize};
use vibrato_core::metadata::VectorMetadata;

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
pub struct IngestRequestV2 {
    pub vector: Vec<f32>,
    #[serde(default)]
    pub metadata: IngestMetadata,
    #[serde(default)]
    pub idempotency_key: Option<String>,
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

impl From<IngestMetadata> for VectorMetadata {
    fn from(value: IngestMetadata) -> Self {
        let mut tags = value.tags;
        for t in &mut tags {
            *t = t.trim().to_ascii_lowercase();
        }
        tags.retain(|t| !t.is_empty());
        tags.sort();
        tags.dedup();

        VectorMetadata {
            source_file: value.source_file,
            start_time_ms: value.start_time_ms,
            duration_ms: value.duration_ms,
            bpm: value.bpm,
            tags,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestResponseV2 {
    pub id: usize,
    pub created: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryRequestV2 {
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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchTier {
    Active,
    All,
    Archive,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QueryFilter {
    #[serde(default)]
    pub tags_any: Vec<String>,
    #[serde(default)]
    pub tags_all: Vec<String>,
    #[serde(default)]
    pub bpm_gte: Option<f32>,
    #[serde(default)]
    pub bpm_lte: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResultV2 {
    pub id: usize,
    pub score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<VectorMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResponseV2 {
    pub results: Vec<QueryResultV2>,
    pub query_time_ms: f64,
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
