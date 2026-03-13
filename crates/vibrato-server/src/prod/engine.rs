use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::hash::{BuildHasher, BuildHasherDefault, Hasher};
use std::io::{BufReader, Read, Seek};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering as AtomicOrdering};
use std::sync::mpsc::{sync_channel, Receiver, RecvTimeoutError, SyncSender};
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use arc_swap::ArcSwap;
use crossbeam_queue::SegQueue;
use arrow_array::builder::{
    BinaryBuilder, FixedSizeListBuilder, Float32Builder, ListBuilder, StringBuilder, UInt32Builder,
    UInt64Builder,
};
use arrow_array::RecordBatch;
use arrow_ipc::writer::StreamWriter;
use arrow_schema::{DataType, Field, Schema};
use parking_lot::RwLock;
use rayon::prelude::*;
use rayon::ThreadPool;
use scc::HashMap as SccHashMap;
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;
use vibrato_core::format_v2::{VdbHeaderV2, VdbWriterV2};
use vibrato_core::hnsw::HNSW;
use vibrato_core::metadata::{MetadataBuilder, VectorMetadata, VectorMetadataV3};
use vibrato_core::pq::ProductQuantizer;
use vibrato_core::simd::{dot_product, dot_product_scores, l2_distance_squared};
use vibrato_core::store::VectorStore;
use vibrato_core::training::{train_pq, TrainingConfig};

use super::catalog::{
    CatalogStore, CheckpointJobRecord, CollectionRecord, CompactionJobRecord,
    IngestMetadataV3Input, SegmentRecord, SqliteCatalog, WalEntry, WalIngestResult,
};
use super::filter::BitmapSet;
use super::model::{
    AuditEvent, IdentifyRequestV2, IdentifyResponseV2, IdentifyResultV2, JobResponseV2,
    MetadataEnvelopeV3, QueryRequestV2, QueryResponseV2, QueryResultV2, SearchTier,
    StatsResponseV2,
};

#[derive(Default)]
pub struct FastIdHasher(u64);

impl Hasher for FastIdHasher {
    #[inline(always)]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline(always)]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = self.0.rotate_left(11) ^ (b as u64);
        }
        self.0 = self.0.wrapping_mul(0x517cc1b727220a95);
    }

    #[inline(always)]
    fn write_usize(&mut self, i: usize) {
        self.0 = (i as u64).wrapping_mul(0x517cc1b727220a95);
    }
}

pub type FastIdMap<V> = HashMap<usize, V, BuildHasherDefault<FastIdHasher>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorMadviseMode {
    Normal,
    Random,
}

#[derive(Debug, Clone, Copy)]
pub enum IngestBackpressureDecision {
    Allow,
    Throttle {
        delay: Duration,
        projected_bytes: u64,
        soft_limit_bytes: u64,
        hard_limit_bytes: u64,
    },
    Reject {
        projected_bytes: u64,
        hard_limit_bytes: u64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointTrigger {
    Admin,
    Background,
}

#[derive(Debug)]
struct IngestWriteJob {
    entries: Vec<(Vec<f32>, IngestMetadataV3Input, Option<String>)>,
    estimated_bytes: u64,
    result_tx: SyncSender<Result<IngestWriteOutcome>>,
}

#[derive(Debug)]
struct IngestWriteOutcome {
    wal_results: Vec<WalIngestResult>,
    entries: Vec<(Vec<f32>, IngestMetadataV3Input, Option<String>)>,
}

#[derive(Debug)]
struct AcceptedWalBatchRow {
    vector_id: u64,
    vector: Vec<f32>,
    metadata: VectorMetadataV3,
    idempotency_key: Option<String>,
}

#[derive(Debug, Clone)]
pub struct UnindexedChunkRow {
    pub vector_id: u64,
    pub metadata: VectorMetadataV3,
}

#[derive(Debug)]
pub struct UnindexedChunk {
    pub vector_id_start: u64,
    pub vector_id_end: u64,
    pub vectors: Vec<f32>,
    pub rows: Vec<UnindexedChunkRow>,
    pub tag_allow_list: HashMap<u32, BitmapSet>,
    pub approx_bytes: u64,
    recycle_pool: Weak<SegQueue<Vec<f32>>>,
    unindexed_bytes: Weak<AtomicU64>,
}

#[derive(Debug, Clone)]
struct ActiveIdentifyRun {
    start_id: usize,
    vectors: Vec<f32>,
}

impl ActiveIdentifyRun {
    #[inline]
    fn len(&self, dim: usize) -> usize {
        self.vectors.len() / dim
    }

    #[inline]
    fn end_id_exclusive(&self, dim: usize) -> usize {
        self.start_id.saturating_add(self.len(dim))
    }
}

#[derive(Debug, Default)]
struct ActiveIdentifyArena {
    runs: Vec<Arc<ActiveIdentifyRun>>,
    min_vector_id: usize,
    max_vector_id_exclusive: usize,
    total_vectors: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct IdentifyScoreEntry {
    id: usize,
    primary_score: f32,
    known_sum: f32,
    known_mask: u8,
}

thread_local! {
    static IDENTIFY_CANDIDATE_SCRATCH: RefCell<Vec<IdentifyScoreEntry>> = RefCell::new(Vec::new());
    static IDENTIFY_TILE_SCORE_SCRATCH: RefCell<Vec<f32>> = RefCell::new(Vec::new());
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
struct AnchorPlan {
    primary_offset: usize,
    secondary_offset: Option<usize>,
    max_delta_sq: f32,
    max_norm_sq: f32,
    low_information: bool,
}

trait IdentifyRunView {
    fn run_count(&self) -> usize;
    fn run_start_id(&self, run_idx: usize) -> usize;
    fn run_vectors(&self, run_idx: usize) -> &[f32];

    #[inline(always)]
    fn run_len(&self, run_idx: usize, dim: usize) -> usize {
        self.run_vectors(run_idx).len() / dim
    }

    #[inline(always)]
    fn run_end_id_exclusive(&self, run_idx: usize, dim: usize) -> usize {
        self.run_start_id(run_idx)
            .saturating_add(self.run_len(run_idx, dim))
    }

    #[inline(always)]
    fn vector_at(&self, run_idx: usize, dim: usize, local_idx: usize) -> Option<&[f32]> {
        let vectors = self.run_vectors(run_idx);
        let start = local_idx.saturating_mul(dim);
        let end = start.saturating_add(dim);
        (end <= vectors.len()).then_some(&vectors[start..end])
    }
}

impl IdentifyRunView for [Arc<ActiveIdentifyRun>] {
    #[inline(always)]
    fn run_count(&self) -> usize {
        self.len()
    }

    #[inline(always)]
    fn run_start_id(&self, run_idx: usize) -> usize {
        self[run_idx].start_id
    }

    #[inline(always)]
    fn run_vectors(&self, run_idx: usize) -> &[f32] {
        self[run_idx].vectors.as_slice()
    }
}

impl IdentifyRunView for [Arc<UnindexedChunk>] {
    #[inline(always)]
    fn run_count(&self) -> usize {
        self.len()
    }

    #[inline(always)]
    fn run_start_id(&self, run_idx: usize) -> usize {
        usize::try_from(self[run_idx].vector_id_start).unwrap_or_else(|_| {
            panic!(
                "data integrity fault: unindexed chunk start id overflow id={}",
                self[run_idx].vector_id_start
            )
        })
    }

    #[inline(always)]
    fn run_vectors(&self, run_idx: usize) -> &[f32] {
        self[run_idx].vectors.as_slice()
    }
}

pub struct UnindexedState {
    pub chunks: ArcSwap<Vec<Arc<UnindexedChunk>>>,
    pub index_queue: SegQueue<Arc<UnindexedChunk>>,
    pub vector_pool: Arc<SegQueue<Vec<f32>>>,
}

impl UnindexedState {
    pub fn new() -> Self {
        Self {
            chunks: ArcSwap::from_pointee(Vec::new()),
            index_queue: SegQueue::new(),
            vector_pool: Arc::new(SegQueue::new()),
        }
    }
}

impl Drop for UnindexedChunk {
    fn drop(&mut self) {
        if let Some(unindexed_bytes) = self.unindexed_bytes.upgrade() {
            atomic_saturating_sub(unindexed_bytes.as_ref(), self.approx_bytes);
        }
        if let Some(recycle_pool) = self.recycle_pool.upgrade() {
            let mut recycled = std::mem::take(&mut self.vectors);
            recycled.clear();
            recycle_pool.push(recycled);
        }
    }
}

#[derive(Debug)]
struct BackgroundIoThrottle {
    bytes_per_sec: u64,
    next_available: std::sync::Mutex<Instant>,
}

impl BackgroundIoThrottle {
    fn new(bytes_per_sec: u64) -> Self {
        Self {
            bytes_per_sec,
            next_available: std::sync::Mutex::new(Instant::now()),
        }
    }

    fn consume(&self, bytes: u64) {
        if bytes == 0 || self.bytes_per_sec == 0 {
            return;
        }
        let reserve = Duration::from_secs_f64(bytes as f64 / self.bytes_per_sec as f64);
        let now = Instant::now();
        let sleep_for = {
            let mut guard = self
                .next_available
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let start = if *guard > now { *guard } else { now };
            *guard = start + reserve;
            start.saturating_duration_since(now)
        };
        if !sleep_for.is_zero() {
            std::thread::sleep(sleep_for);
        }
    }
}

#[derive(Debug, Clone)]
pub struct ProductionConfig {
    pub data_dir: PathBuf,
    pub segments_dir: PathBuf,
    pub quarantine_dir: PathBuf,
    pub snapshots_dir: PathBuf,
    pub tmp_dir: PathBuf,
    pub logs_dir: PathBuf,
    pub collection_name: String,
    pub dim: usize,
    pub hnsw_m: usize,
    pub hnsw_ef_construction: usize,
    pub checkpoint_interval: Duration,
    pub compaction_interval: Duration,
    pub admin_checkpoint_cooldown_secs: u64,
    pub admin_compaction_cooldown_secs: u64,
    pub orphan_ttl: Duration,
    pub audio_colocated: bool,
    pub public_health_metrics: bool,
    pub catalog_read_timeout_ms: u64,
    pub sqlite_wal_autocheckpoint_pages: u32,
    pub quarantine_max_files: usize,
    pub quarantine_max_bytes: u64,
    pub background_io_mb_per_sec: u64,
    pub hot_index_shards: usize,
    pub ingest_queue_capacity: usize,
    pub memory_budget_bytes: u64,
    /// Query rayon pool size (0 = auto).
    pub query_pool_threads: usize,
    /// Flight decode rayon pool size (0 = auto).
    pub flight_decode_pool_threads: usize,
    /// Minimum shard count before allow-set union parallelization is enabled.
    pub filter_parallel_min_shards: usize,
    pub max_tag_registry_size: usize,
    pub max_tags_per_vector: usize,
    pub max_tag_len: usize,
    pub max_idempotency_key_len: usize,
    pub unindexed_memory_limit_bytes: u64,
    pub flight_decode_warn_ms: u64,
    pub vector_madvise_mode: VectorMadviseMode,
    pub api_pepper: String,
}

#[derive(Default)]
pub struct Metrics {
    pub query_total: AtomicU64,
    pub identify_total: AtomicU64,
    pub ingest_total: AtomicU64,
    pub flight_ingest_batches_total: AtomicU64,
    pub flight_ingest_rows_total: AtomicU64,
    pub flight_decode_us_total: AtomicU64,
    pub flight_commit_us_total: AtomicU64,
    pub flight_ack_us_total: AtomicU64,
    pub auth_failures_total: AtomicU64,
    pub audit_failures_total: AtomicU64,
    pub checkpoint_total: AtomicU64,
    pub compaction_total: AtomicU64,
    pub obsolete_files_deleted_total: AtomicU64,
    pub ingest_backpressure_soft_total: AtomicU64,
    pub ingest_backpressure_hard_total: AtomicU64,
    pub ingest_semantic_throttle_ms_total: AtomicU64,
    pub query_latency_count: AtomicU64,
    pub query_latency_seconds_sum: AtomicU64,
    pub query_latency_seconds_le_10us: AtomicU64,
    pub query_latency_seconds_le_25us: AtomicU64,
    pub query_latency_seconds_le_50us: AtomicU64,
    pub query_latency_seconds_le_100us: AtomicU64,
    pub query_latency_seconds_le_500us: AtomicU64,
    pub query_latency_seconds_le_1000us: AtomicU64,
    pub metadata_cache_hits_total: AtomicU64,
    pub metadata_cache_misses_total: AtomicU64,
    pub filter_allow_cache_hits_total: AtomicU64,
    pub filter_allow_cache_misses_total: AtomicU64,
    pub active_connections: AtomicU64,
    pub active_http_requests: AtomicU64,
    pub active_flight_streams: AtomicU64,
    pub tag_reject_overflow_total: AtomicU64,
    pub tag_reject_invalid_total: AtomicU64,
    pub flight_decode_chunk_warn_total: AtomicU64,
}

pub struct SegmentHandle {
    pub record: SegmentRecord,
    pub store: Arc<VectorStore>,
    pub index: Arc<RwLock<HNSW>>,
    pub filter_index: HashMap<u32, BitmapSet>,
}

pub struct ArchivePqSegment {
    pub record: SegmentRecord,
    pub pq: ProductQuantizer,
    pub codes: Arc<Vec<u8>>,
    pub num_subspaces: usize,
    pub filter_index: HashMap<u32, BitmapSet>,
}

pub struct HotShard {
    pub vectors: FastIdMap<Arc<Vec<f32>>>,
    pub dense_vectors: Vec<Arc<Vec<f32>>>,
    pub index: HNSW,
    pub filter_index: HashMap<u32, BitmapSet>,
}

#[derive(Clone, Default)]
struct ResolvedTagFilter {
    tags_all_ids: Vec<u32>,
    tags_any_ids: Vec<u32>,
    impossible: bool,
}

fn build_local_allow_set(
    local_index: &HashMap<u32, BitmapSet>,
    resolved: &ResolvedTagFilter,
) -> Option<BitmapSet> {
    if resolved.impossible {
        return Some(BitmapSet::default());
    }

    let mut allow: Option<BitmapSet> = None;

    if !resolved.tags_all_ids.is_empty() {
        for tag_id in &resolved.tags_all_ids {
            let bm = local_index.get(tag_id).cloned().unwrap_or_default();
            allow = Some(match allow {
                Some(curr) => curr.intersect(&bm),
                None => bm,
            });
        }
    }

    if !resolved.tags_any_ids.is_empty() {
        let mut any_union = BitmapSet::default();
        for tag_id in &resolved.tags_any_ids {
            if let Some(bm) = local_index.get(tag_id) {
                any_union.union_with(bm);
            }
        }
        allow = Some(match allow {
            Some(curr) => curr.intersect(&any_union),
            None => any_union,
        });
    }

    allow
}

pub struct ProductionState {
    pub config: ProductionConfig,
    pub catalog: Arc<SqliteCatalog>,
    pub collection: CollectionRecord,
    pub live: AtomicBool,
    pub ready: AtomicBool,
    pub recovery_report: RwLock<String>,

    pub hot_shards: Vec<RwLock<HotShard>>,
    pub shard_mask: usize,
    pub metadata_cache: Arc<SccHashMap<usize, VectorMetadataV3>>,
    pub segments: ArcSwap<Vec<Arc<SegmentHandle>>>,
    pub archive_segments: ArcSwap<Vec<Arc<ArchivePqSegment>>>,
    pub retired_segments: std::sync::Mutex<HashMap<String, Weak<SegmentHandle>>>,

    pub metrics: Metrics,
    wal_ipc_schema: Arc<Schema>,
    pub unindexed: UnindexedState,
    ingest_writer_tx: SyncSender<IngestWriteJob>,
    ingest_writer_handle: std::sync::Mutex<Option<std::thread::JoinHandle<()>>>,
    ingest_writer_failed: AtomicBool,
    pub ingest_queue_bytes: AtomicU64,
    pub unindexed_bytes: Arc<AtomicU64>,
    pub hot_vectors_bytes: AtomicU64,
    pub active_identify_bytes: AtomicU64,
    pub hnsw_graph_bytes: AtomicU64,
    pub metadata_cache_bytes: AtomicU64,
    active_identify_arena: ArcSwap<ActiveIdentifyArena>,
    pub hot_min_id: AtomicUsize,
    pub hot_max_id_exclusive: AtomicUsize,
    pub inflight_decode_bytes: AtomicU64,
    pub admin_ops_lock: std::sync::Mutex<()>,
    pub checkpoint_lock: std::sync::Mutex<()>,
    pub compaction_lock: std::sync::Mutex<()>,
    last_checkpoint_started_unix: AtomicU64,
    last_compaction_started_unix: AtomicU64,
    background_cancel: CancellationToken,
    pub background_worker_handles: std::sync::Mutex<Vec<std::thread::JoinHandle<()>>>,
    background_io_throttle: Option<Arc<BackgroundIoThrottle>>,
    pub flight_decode_pool: Arc<ThreadPool>,
    pub query_pool: Arc<ThreadPool>,
    pub audit_tx: SyncSender<AuditEvent>,
    pub audit_rx: std::sync::Mutex<Option<Receiver<AuditEvent>>>,
}

impl ProductionConfig {
    pub fn from_data_dir(data_dir: PathBuf, collection_name: String, dim: usize) -> Self {
        let segments_dir = data_dir.join("segments");
        let quarantine_dir = data_dir.join("quarantine");
        let snapshots_dir = data_dir.join("snapshots");
        let tmp_dir = data_dir.join("tmp");
        let logs_dir = data_dir.join("logs");

        Self {
            data_dir,
            segments_dir,
            quarantine_dir,
            snapshots_dir,
            tmp_dir,
            logs_dir,
            collection_name,
            dim,
            hnsw_m: 16,
            hnsw_ef_construction: 100,
            checkpoint_interval: Duration::from_secs(30),
            compaction_interval: Duration::from_secs(180),
            admin_checkpoint_cooldown_secs: 30,
            admin_compaction_cooldown_secs: 30,
            orphan_ttl: Duration::from_secs(168 * 3600),
            audio_colocated: true,
            public_health_metrics: true,
            catalog_read_timeout_ms: 5_000,
            sqlite_wal_autocheckpoint_pages: 1000,
            quarantine_max_files: 50,
            quarantine_max_bytes: 5 * 1024 * 1024 * 1024,
            background_io_mb_per_sec: 40,
            hot_index_shards: 64,
            ingest_queue_capacity: 1024,
            memory_budget_bytes: 8 * 1024 * 1024 * 1024,
            query_pool_threads: 0,
            flight_decode_pool_threads: 0,
            filter_parallel_min_shards: 8,
            max_tag_registry_size: 500_000,
            max_tags_per_vector: 64,
            max_tag_len: 64,
            max_idempotency_key_len: 64,
            unindexed_memory_limit_bytes: 512 * 1024 * 1024,
            flight_decode_warn_ms: 20,
            vector_madvise_mode: VectorMadviseMode::Normal,
            api_pepper: std::env::var("VIBRATO_API_PEPPER")
                .unwrap_or_else(|_| "dev-pepper".to_string()),
        }
    }

    pub fn catalog_path(&self) -> PathBuf {
        self.data_dir.join("catalog.sqlite3")
    }
}

impl ProductionState {
    fn wal_ipc_schema_for_dim(dim: usize) -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("vector_id", DataType::UInt64, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    dim as i32,
                ),
                false,
            ),
            Field::new("entity_id", DataType::UInt64, false),
            Field::new("sequence_ts", DataType::UInt64, false),
            Field::new("payload", DataType::Binary, false),
            Field::new(
                "tag_ids",
                DataType::List(Arc::new(Field::new("item", DataType::UInt32, true))),
                false,
            ),
            Field::new("idempotency_key", DataType::Utf8, true),
        ]))
    }

    #[inline]
    fn auto_query_pool_threads(available: usize) -> usize {
        if available <= 2 {
            1
        } else {
            ((available * 3) / 4).clamp(2, 12)
        }
    }

    #[inline]
    fn auto_flight_decode_pool_threads(available: usize, query_threads: usize) -> usize {
        query_threads.div_ceil(2).clamp(1, 8).min(available.max(1))
    }

    pub fn initialize(config: ProductionConfig, catalog: Arc<SqliteCatalog>) -> Result<Arc<Self>> {
        let collection = catalog.ensure_collection(&config.collection_name, config.dim)?;

        let shard_count = next_power_of_two_at_least_one(config.hot_index_shards);
        let shard_mask = shard_count - 1;
        let hot_shards = (0..shard_count)
            .map(|_| {
                RwLock::new(HotShard {
                    vectors: FastIdMap::default(),
                    dense_vectors: Vec::new(),
                    index: make_empty_hot_hnsw(config.hnsw_m, config.hnsw_ef_construction),
                    filter_index: HashMap::new(),
                })
            })
            .collect::<Vec<_>>();
        let metadata_cache = Arc::new(SccHashMap::new());
        let background_cancel = CancellationToken::new();
        let (audit_tx, audit_rx) = sync_channel(4096);
        let (ingest_writer_tx, ingest_writer_rx) =
            sync_channel(config.ingest_queue_capacity.max(1));
        let available = std::thread::available_parallelism()
            .map(|v| v.get())
            .unwrap_or(2);
        let query_threads = if config.query_pool_threads > 0 {
            config.query_pool_threads.min(available.max(1)).max(1)
        } else {
            Self::auto_query_pool_threads(available)
        };
        let query_pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(query_threads)
                .thread_name(|idx| format!("vibrato-query-{idx}"))
                .build()
                .context("building query pool")?,
        );
        let flight_decode_threads = if config.flight_decode_pool_threads > 0 {
            config
                .flight_decode_pool_threads
                .min(available.max(1))
                .max(1)
        } else {
            Self::auto_flight_decode_pool_threads(available, query_threads)
        };
        let flight_decode_pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(flight_decode_threads)
                .thread_name(|idx| format!("vibrato-flight-decode-{idx}"))
                .build()
                .context("building flight decode pool")?,
        );
        let background_io_throttle =
            if config.audio_colocated && config.background_io_mb_per_sec > 0 {
                Some(Arc::new(BackgroundIoThrottle::new(
                    config.background_io_mb_per_sec * 1024 * 1024,
                )))
            } else {
                None
            };
        let wal_ipc_schema = Self::wal_ipc_schema_for_dim(collection.dim);
        let unindexed_bytes = Arc::new(AtomicU64::new(0));

        let state = Arc::new(Self {
            config,
            catalog,
            collection,
            live: AtomicBool::new(true),
            ready: AtomicBool::new(false),
            recovery_report: RwLock::new("initializing".to_string()),
            hot_shards,
            shard_mask,
            metadata_cache,
            segments: ArcSwap::from_pointee(Vec::new()),
            archive_segments: ArcSwap::from_pointee(Vec::new()),
            retired_segments: std::sync::Mutex::new(HashMap::new()),
            metrics: Metrics::default(),
            wal_ipc_schema,
            unindexed: UnindexedState::new(),
            ingest_writer_tx,
            ingest_writer_handle: std::sync::Mutex::new(None),
            ingest_writer_failed: AtomicBool::new(false),
            ingest_queue_bytes: AtomicU64::new(0),
            unindexed_bytes,
            hot_vectors_bytes: AtomicU64::new(0),
            active_identify_bytes: AtomicU64::new(0),
            hnsw_graph_bytes: AtomicU64::new(0),
            metadata_cache_bytes: AtomicU64::new(0),
            active_identify_arena: ArcSwap::from_pointee(ActiveIdentifyArena::default()),
            hot_min_id: AtomicUsize::new(0),
            hot_max_id_exclusive: AtomicUsize::new(0),
            inflight_decode_bytes: AtomicU64::new(0),
            admin_ops_lock: std::sync::Mutex::new(()),
            checkpoint_lock: std::sync::Mutex::new(()),
            compaction_lock: std::sync::Mutex::new(()),
            last_checkpoint_started_unix: AtomicU64::new(0),
            last_compaction_started_unix: AtomicU64::new(0),
            background_cancel,
            background_worker_handles: std::sync::Mutex::new(Vec::new()),
            background_io_throttle,
            flight_decode_pool,
            query_pool,
            audit_tx,
            audit_rx: std::sync::Mutex::new(Some(audit_rx)),
        });

        state.start_ingest_writer(ingest_writer_rx)?;

        Ok(state)
    }

    fn start_ingest_writer(
        self: &Arc<Self>,
        ingest_writer_rx: Receiver<IngestWriteJob>,
    ) -> Result<()> {
        let state = Arc::clone(self);
        let writer_name = format!("vibrato-sqlite-writer-{}", state.collection.id);
        let handle = std::thread::Builder::new()
            .name(writer_name)
            .spawn(move || {
                if state.config.audio_colocated {
                    set_background_worker_priority();
                }
                let fault_panic_after =
                    std::env::var("VIBRATO_TEST_FAULT_INGEST_WRITER_PANIC_AFTER_JOBS")
                        .ok()
                        .and_then(|raw| raw.parse::<usize>().ok())
                        .filter(|value| *value > 0);
                let mut processed_jobs = 0usize;
                while let Ok(job) = ingest_writer_rx.recv() {
                    let entries = job.entries;
                    let result = state
                        .catalog
                        .ingest_wal_batch(&state.collection.id, &entries)
                        .map(|wal_results| IngestWriteOutcome {
                            wal_results,
                            entries,
                        });
                    atomic_saturating_sub(&state.ingest_queue_bytes, job.estimated_bytes);
                    let _ = job.result_tx.send(result);
                    processed_jobs = processed_jobs.saturating_add(1);
                    if fault_panic_after
                        .map(|threshold| processed_jobs >= threshold)
                        .unwrap_or(false)
                    {
                        panic!(
                            "fault injection: ingest writer panic after {} jobs",
                            processed_jobs
                        );
                    }
                }
            })
            .context("spawning ingest writer thread")?;
        *self
            .ingest_writer_handle
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(handle);
        Ok(())
    }

    fn refresh_ingest_writer_health(&self) {
        let mut handle_slot = self
            .ingest_writer_handle
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let finished = handle_slot
            .as_ref()
            .map(std::thread::JoinHandle::is_finished)
            .unwrap_or(false);
        if !finished {
            return;
        }

        if let Some(handle) = handle_slot.take() {
            match handle.join() {
                Ok(()) => {
                    tracing::error!("ingest writer thread exited unexpectedly");
                }
                Err(_) => {
                    tracing::error!("ingest writer thread panicked");
                }
            }
            self.ingest_writer_failed
                .store(true, AtomicOrdering::SeqCst);
            self.live.store(false, AtomicOrdering::SeqCst);
            self.background_cancel.cancel();
        }
    }

    fn estimate_ingest_metadata_bytes(metadata: &IngestMetadataV3Input) -> u64 {
        let mut bytes = metadata.payload.len() as u64;
        bytes = bytes
            .saturating_add(std::mem::size_of::<u64>() as u64)
            .saturating_add(std::mem::size_of::<u64>() as u64);
        for tag in &metadata.tags {
            bytes = bytes.saturating_add(tag.len() as u64);
        }
        bytes
    }

    fn estimate_cached_metadata_bytes(metadata: &VectorMetadataV3) -> u64 {
        let tags_bytes =
            (metadata.tags.len() as u64).saturating_mul(std::mem::size_of::<u32>() as u64);
        tags_bytes
            .saturating_add(metadata.payload.len() as u64)
            .saturating_add(std::mem::size_of::<u64>() as u64 * 2)
    }

    fn update_hot_window_from_created_range(&self, min_id: usize, max_id_exclusive: usize) {
        if min_id >= max_id_exclusive {
            return;
        }

        let current_max = self.hot_max_id_exclusive.load(AtomicOrdering::Relaxed);
        if current_max == 0 {
            self.hot_min_id.store(min_id, AtomicOrdering::Relaxed);
            self.hot_max_id_exclusive
                .store(max_id_exclusive, AtomicOrdering::Relaxed);
            return;
        }

        self.hot_max_id_exclusive
            .fetch_max(max_id_exclusive, AtomicOrdering::Relaxed);
        let mut current_min = self.hot_min_id.load(AtomicOrdering::Relaxed);
        while min_id < current_min {
            match self.hot_min_id.compare_exchange_weak(
                current_min,
                min_id,
                AtomicOrdering::Relaxed,
                AtomicOrdering::Relaxed,
            ) {
                Ok(_) => break,
                Err(observed) => current_min = observed,
            }
        }
    }

    fn estimate_hnsw_bytes_per_vector(&self) -> u64 {
        // Approximate graph edge memory: M 32-bit neighbor IDs plus small node overhead.
        let edges = self.config.hnsw_m as u64 * std::mem::size_of::<u32>() as u64;
        edges.saturating_add(32)
    }

    fn serialize_accepted_rows_to_ipc(
        &self,
        ordered_rows: &[&AcceptedWalBatchRow],
    ) -> Result<Vec<u8>> {
        let dim = self.collection.dim;
        let mut vector_id_builder = UInt64Builder::with_capacity(ordered_rows.len());
        let mut vector_builder = FixedSizeListBuilder::new(
            Float32Builder::with_capacity(ordered_rows.len().saturating_mul(dim)),
            dim as i32,
        );
        let mut entity_id_builder = UInt64Builder::with_capacity(ordered_rows.len());
        let mut sequence_ts_builder = UInt64Builder::with_capacity(ordered_rows.len());
        let payload_capacity = ordered_rows
            .iter()
            .map(|row| row.metadata.payload.len())
            .sum::<usize>();
        let mut payload_builder =
            BinaryBuilder::with_capacity(ordered_rows.len(), payload_capacity);
        let mut tag_ids_builder = ListBuilder::new(UInt32Builder::new());
        let mut idempotency_builder = StringBuilder::with_capacity(
            ordered_rows.len(),
            ordered_rows
                .iter()
                .map(|row| row.idempotency_key.as_ref().map(|v| v.len()).unwrap_or(0))
                .sum(),
        );

        for row in ordered_rows {
            vector_id_builder.append_value(row.vector_id);
            vector_builder.values().append_slice(&row.vector);
            vector_builder.append(true);
            entity_id_builder.append_value(row.metadata.entity_id);
            sequence_ts_builder.append_value(row.metadata.sequence_ts);
            payload_builder.append_value(&row.metadata.payload);

            tag_ids_builder.values().append_slice(&row.metadata.tags);
            tag_ids_builder.append(true);

            if let Some(key) = row.idempotency_key.as_ref() {
                idempotency_builder.append_value(key);
            } else {
                idempotency_builder.append_null();
            }
        }

        let batch = RecordBatch::try_new(
            self.wal_ipc_schema.clone(),
            vec![
                Arc::new(vector_id_builder.finish()),
                Arc::new(vector_builder.finish()),
                Arc::new(entity_id_builder.finish()),
                Arc::new(sequence_ts_builder.finish()),
                Arc::new(payload_builder.finish()),
                Arc::new(tag_ids_builder.finish()),
                Arc::new(idempotency_builder.finish()),
            ],
        )?;

        let mut blob = Vec::new();
        let mut writer = StreamWriter::try_new(&mut blob, &self.wal_ipc_schema)?;
        writer.write(&batch)?;
        writer.finish()?;
        drop(writer);
        Ok(blob)
    }

    fn append_wal_ipc_batch_for_accepted_rows(&self, rows: &[AcceptedWalBatchRow]) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }

        let mut ordered_rows = rows.iter().collect::<Vec<_>>();
        ordered_rows.sort_by_key(|row| row.vector_id);
        let vector_id_start = ordered_rows.first().map(|row| row.vector_id).unwrap_or(0);
        let vector_id_end = ordered_rows
            .last()
            .map(|row| row.vector_id)
            .unwrap_or(vector_id_start);
        for (offset, row) in ordered_rows.iter().enumerate() {
            let expected = vector_id_start.saturating_add(offset as u64);
            if row.vector_id != expected {
                return Err(anyhow!(
                    "data integrity fault: accepted ingest batch has non-contiguous vector ids start={} offset={} actual={}",
                    vector_id_start,
                    offset,
                    row.vector_id
                ));
            }
        }

        let ipc_blob = self.serialize_accepted_rows_to_ipc(&ordered_rows)?;
        self.catalog.ingest_wal_ipc_batch(
            &self.collection.id,
            ordered_rows.len(),
            vector_id_start,
            vector_id_end,
            &ipc_blob,
        )?;
        Ok(())
    }

    fn estimate_unindexed_chunk_bytes(
        &self,
        rows: &[UnindexedChunkRow],
        vector_capacity: usize,
        tag_allow_list: &HashMap<u32, BitmapSet>,
    ) -> u64 {
        let vector_bytes =
            (vector_capacity as u64).saturating_mul(std::mem::size_of::<f32>() as u64);
        let metadata_bytes = rows.iter().fold(0u64, |acc, row| {
            acc.saturating_add(Self::estimate_cached_metadata_bytes(&row.metadata))
        });
        let row_overhead = (rows.len() as u64).saturating_mul(std::mem::size_of::<UnindexedChunkRow>() as u64);
        let tag_index_bytes = (tag_allow_list.len() as u64)
            .saturating_mul(std::mem::size_of::<(u32, BitmapSet)>() as u64);
        vector_bytes
            .saturating_add(metadata_bytes)
            .saturating_add(row_overhead)
            .saturating_add(tag_index_bytes)
    }

    fn build_unindexed_chunk(
        &self,
        mut rows: Vec<AcceptedWalBatchRow>,
    ) -> Result<Arc<UnindexedChunk>> {
        if rows.is_empty() {
            return Err(anyhow!(
                "data integrity fault: attempted to build unindexed chunk from empty row set"
            ));
        }
        rows.sort_by_key(|row| row.vector_id);
        let vector_id_start = rows.first().map(|row| row.vector_id).unwrap_or(0);
        let vector_id_end = rows.last().map(|row| row.vector_id).unwrap_or(vector_id_start);

        let mut chunk_rows = Vec::with_capacity(rows.len());
        let needed_values = rows.len().saturating_mul(self.collection.dim);
        let mut vectors = self
            .unindexed
            .vector_pool
            .pop()
            .filter(|buf| buf.capacity() >= needed_values)
            .unwrap_or_else(|| Vec::with_capacity(needed_values));
        vectors.clear();
        let mut tag_allow_list: HashMap<u32, BitmapSet> = HashMap::new();
        for (offset, row) in rows.into_iter().enumerate() {
            let expected = vector_id_start.saturating_add(offset as u64);
            if row.vector_id != expected {
                return Err(anyhow!(
                    "data integrity fault: unindexed chunk has non-contiguous ids start={} offset={} actual={}",
                    vector_id_start,
                    offset,
                    row.vector_id
                ));
            }
            if row.vector.len() != self.collection.dim {
                return Err(anyhow!(
                    "data integrity fault: unindexed chunk dim mismatch id={} expected={} got={}",
                    row.vector_id,
                    self.collection.dim,
                    row.vector.len()
                ));
            }
            vectors.extend_from_slice(&row.vector);
            let node_idx = chunk_rows.len();
            for tag_id in &row.metadata.tags {
                tag_allow_list.entry(*tag_id).or_default().insert(node_idx);
            }
            chunk_rows.push(UnindexedChunkRow {
                vector_id: row.vector_id,
                metadata: row.metadata,
            });
        }

        // Track the backing allocation, not logical len(), so pooled oversized buffers
        // still count against the unindexed memory breaker until they are recycled.
        let approx_bytes =
            self.estimate_unindexed_chunk_bytes(&chunk_rows, vectors.capacity(), &tag_allow_list);
        Ok(Arc::new(UnindexedChunk {
            vector_id_start,
            vector_id_end,
            vectors,
            rows: chunk_rows,
            tag_allow_list,
            approx_bytes,
            recycle_pool: Arc::downgrade(&self.unindexed.vector_pool),
            unindexed_bytes: Arc::downgrade(&self.unindexed_bytes),
        }))
    }

    fn degrade_integrity_fault(&self, message: impl Into<String>) -> anyhow::Error {
        let message = message.into();
        tracing::error!("{message}");
        self.live.store(false, AtomicOrdering::SeqCst);
        self.set_ready(false, format!("degraded: {message}"));
        anyhow!(message)
    }

    fn insert_sorted_unindexed_chunks(
        chunks: &[Arc<UnindexedChunk>],
        chunk: &Arc<UnindexedChunk>,
    ) -> Result<Vec<Arc<UnindexedChunk>>> {
        let insert_idx = chunks.partition_point(|existing| existing.vector_id_start < chunk.vector_id_start);

        if let Some(prev) = insert_idx.checked_sub(1).and_then(|idx| chunks.get(idx)) {
            if prev.vector_id_end >= chunk.vector_id_start {
                return Err(anyhow!(
                    "data integrity fault: overlapping unindexed identify chunk prev=[{},{}] next=[{},{}]",
                    prev.vector_id_start,
                    prev.vector_id_end,
                    chunk.vector_id_start,
                    chunk.vector_id_end
                ));
            }
        }
        if let Some(next) = chunks.get(insert_idx) {
            if chunk.vector_id_end >= next.vector_id_start {
                return Err(anyhow!(
                    "data integrity fault: overlapping unindexed identify chunk next=[{},{}] incoming=[{},{}]",
                    next.vector_id_start,
                    next.vector_id_end,
                    chunk.vector_id_start,
                    chunk.vector_id_end
                ));
            }
        }

        let mut next_chunks = Vec::with_capacity(chunks.len() + 1);
        next_chunks.extend_from_slice(&chunks[..insert_idx]);
        next_chunks.push(Arc::clone(chunk));
        next_chunks.extend_from_slice(&chunks[insert_idx..]);
        Ok(next_chunks)
    }

    fn publish_unindexed_chunk(&self, chunk: Arc<UnindexedChunk>) -> Result<()> {
        self.unindexed_bytes
            .as_ref()
            .fetch_add(chunk.approx_bytes, AtomicOrdering::Relaxed);
        loop {
            let current = self.unindexed.chunks.load_full();
            let next_chunks =
                match Self::insert_sorted_unindexed_chunks(current.as_slice(), &chunk) {
                    Ok(next_chunks) => next_chunks,
                    Err(err) => {
                        atomic_saturating_sub(self.unindexed_bytes.as_ref(), chunk.approx_bytes);
                        return Err(self.degrade_integrity_fault(err.to_string()));
                    }
                };
            let next = Arc::new(next_chunks);
            let previous = self.unindexed.chunks.compare_and_swap(&current, next);
            if Arc::ptr_eq(&current, &*previous) {
                break;
            }
        }
        self.unindexed.index_queue.push(chunk);
        Ok(())
    }

    #[inline]
    fn active_identify_vector_bytes(vectors: &[f32]) -> u64 {
        (vectors.len() as u64).saturating_mul(std::mem::size_of::<f32>() as u64)
    }

    fn build_active_identify_arena(
        runs: Vec<Arc<ActiveIdentifyRun>>,
        dim: usize,
    ) -> ActiveIdentifyArena {
        let mut arena = ActiveIdentifyArena {
            runs,
            min_vector_id: usize::MAX,
            max_vector_id_exclusive: 0,
            total_vectors: 0,
        };
        for run in &arena.runs {
            let len = run.len(dim);
            if len == 0 {
                continue;
            }
            arena.total_vectors = arena.total_vectors.saturating_add(len);
            arena.min_vector_id = arena.min_vector_id.min(run.start_id);
            arena.max_vector_id_exclusive =
                arena.max_vector_id_exclusive.max(run.start_id.saturating_add(len));
        }
        if arena.total_vectors == 0 {
            arena.min_vector_id = 0;
            arena.max_vector_id_exclusive = 0;
        }
        arena
    }

    fn merge_active_identify_runs(
        runs: &[Arc<ActiveIdentifyRun>],
        start_id: usize,
        vectors: &[f32],
        dim: usize,
    ) -> Result<Vec<Arc<ActiveIdentifyRun>>> {
        let mut insert_idx = runs.partition_point(|run| run.start_id < start_id);
        let mut merged_start = start_id;
        let mut merged_vectors = vectors.to_vec();
        let mut merged_prev = false;

        if insert_idx > 0 {
            let prev = &runs[insert_idx - 1];
            let prev_end = prev.end_id_exclusive(dim);
            if prev_end > start_id {
                return Err(anyhow!(
                    "data integrity fault: active identify overlap prev_start={} prev_end={} start_id={}",
                    prev.start_id,
                    prev_end,
                    start_id
                ));
            }
            if prev_end == start_id {
                merged_start = prev.start_id;
                merged_vectors = prev.vectors.clone();
                merged_vectors.extend_from_slice(vectors);
                insert_idx -= 1;
                merged_prev = true;
            }
        }

        let mut remove_end = insert_idx + usize::from(merged_prev);
        while let Some(next) = runs.get(remove_end) {
            let merged_end = merged_start.saturating_add(merged_vectors.len() / dim);
            if merged_end < next.start_id {
                break;
            }
            if merged_end > next.start_id {
                return Err(anyhow!(
                    "data integrity fault: active identify overlap merged_end={} next_start={} next_end={}",
                    merged_end,
                    next.start_id,
                    next.end_id_exclusive(dim)
                ));
            }
            merged_vectors.extend_from_slice(&next.vectors);
            remove_end += 1;
        }

        let mut next_runs =
            Vec::with_capacity(runs.len().saturating_add(1).saturating_sub(remove_end - insert_idx));
        next_runs.extend_from_slice(&runs[..insert_idx]);
        next_runs.push(Arc::new(ActiveIdentifyRun {
            start_id: merged_start,
            vectors: merged_vectors,
        }));
        next_runs.extend_from_slice(&runs[remove_end..]);
        Ok(next_runs)
    }

    fn append_active_identify_run(&self, start_id: usize, vectors: &[f32]) -> Result<()> {
        if vectors.is_empty() {
            return Ok(());
        }
        if !vectors.len().is_multiple_of(self.collection.dim) {
            return Err(self.degrade_integrity_fault(format!(
                "data integrity fault: active identify run misaligned start_id={} dim={} len={}",
                start_id,
                self.collection.dim,
                vectors.len()
            )));
        }
        let added_bytes = Self::active_identify_vector_bytes(vectors);
        loop {
            let current = self.active_identify_arena.load_full();
            let next_runs = match Self::merge_active_identify_runs(
                current.runs.as_slice(),
                start_id,
                vectors,
                self.collection.dim,
            ) {
                Ok(next_runs) => next_runs,
                Err(err) => return Err(self.degrade_integrity_fault(err.to_string())),
            };
            let next = Arc::new(Self::build_active_identify_arena(next_runs, self.collection.dim));
            let previous = self.active_identify_arena.compare_and_swap(&current, next);
            if Arc::ptr_eq(&current, &*previous) {
                break;
            }
        }
        self.active_identify_bytes
            .fetch_add(added_bytes, AtomicOrdering::Relaxed);
        Ok(())
    }

    fn rebuild_active_identify_arena_from_hot(&self) -> Result<()> {
        let mut ordered = Vec::new();
        for shard_lock in &self.hot_shards {
            let shard_guard = shard_lock.read();
            for (id, vector) in &shard_guard.vectors {
                ordered.push((*id, vector.clone()));
            }
        }
        ordered.sort_by_key(|(id, _)| *id);

        let mut runs = Vec::new();
        let mut active_identify_bytes = 0u64;
        let mut current_run_start = None;
        let mut current_run_vectors = Vec::new();
        let mut expected_next_id = 0usize;

        for (id, vector) in ordered {
            if vector.len() != self.collection.dim {
                return Err(self.degrade_integrity_fault(format!(
                    "data integrity fault: active identify rebuild dim mismatch id={} expected={} got={}",
                    id,
                    self.collection.dim,
                    vector.len()
                )));
            }
            match current_run_start {
                None => {
                    current_run_start = Some(id);
                }
                Some(_) if id < expected_next_id => {
                    return Err(self.degrade_integrity_fault(format!(
                        "data integrity fault: active identify rebuild overlap id={} expected_next_id={}",
                        id, expected_next_id
                    )));
                }
                Some(_) if id != expected_next_id => {
                    if let Some(start_id) = current_run_start.take() {
                        active_identify_bytes = active_identify_bytes.saturating_add(
                            Self::active_identify_vector_bytes(&current_run_vectors),
                        );
                        runs.push(Arc::new(ActiveIdentifyRun {
                            start_id,
                            vectors: std::mem::take(&mut current_run_vectors),
                        }));
                    }
                    current_run_start = Some(id);
                }
                Some(_) => {}
            }
            current_run_vectors.extend_from_slice(vector.as_slice());
            expected_next_id = id.saturating_add(1);
        }

        if let Some(start_id) = current_run_start.take() {
            active_identify_bytes = active_identify_bytes
                .saturating_add(Self::active_identify_vector_bytes(&current_run_vectors));
            runs.push(Arc::new(ActiveIdentifyRun {
                start_id,
                vectors: current_run_vectors,
            }));
        }

        let arena = Self::build_active_identify_arena(runs, self.collection.dim);
        self.active_identify_arena.store(Arc::new(arena));
        self.active_identify_bytes
            .store(active_identify_bytes, AtomicOrdering::Relaxed);
        Ok(())
    }

    fn mark_unindexed_chunks_indexed(&self, chunks: &[Arc<UnindexedChunk>]) {
        if chunks.is_empty() {
            return;
        }
        let mut remove_ptrs = HashSet::with_capacity(chunks.len());
        for chunk in chunks {
            remove_ptrs.insert(Arc::as_ptr(chunk) as usize);
        }
        self.unindexed.chunks.rcu(|current| {
            let retained = current
                .iter()
                .filter(|existing| !remove_ptrs.contains(&(Arc::as_ptr(existing) as usize)))
                .cloned()
                .collect::<Vec<_>>();
            Arc::new(retained)
        });
    }

    fn query_unindexed_chunks(
        &self,
        query: &[f32],
        k: usize,
        resolved_filter: Option<&ResolvedTagFilter>,
    ) -> Vec<(usize, f32)> {
        if k == 0 {
            return Vec::new();
        }
        let snapshot = self.unindexed.chunks.load_full();
        if snapshot.is_empty() {
            return Vec::new();
        }
        let dim = self.collection.dim;
        let mut best: FastIdMap<f32> = FastIdMap::default();
        for chunk in snapshot.iter() {
            if let Some(resolved) = resolved_filter {
                if let Some(allow) = build_local_allow_set(&chunk.tag_allow_list, resolved) {
                    for row_idx in allow.iter_ids() {
                        let Some(row) = chunk.rows.get(row_idx) else {
                            continue;
                        };
                        let id = match usize::try_from(row.vector_id) {
                            Ok(id) => id,
                            Err(_) => continue,
                        };
                        let start = row_idx.saturating_mul(dim);
                        let end = start.saturating_add(dim);
                        if end > chunk.vectors.len() {
                            continue;
                        }
                        let score = dot_product(query, &chunk.vectors[start..end]);
                        merge_best(&mut best, id, score);
                    }
                    continue;
                }
            }
            for (row_idx, row) in chunk.rows.iter().enumerate() {
                let id = match usize::try_from(row.vector_id) {
                    Ok(id) => id,
                    Err(_) => continue,
                };
                let start = row_idx.saturating_mul(dim);
                let end = start.saturating_add(dim);
                if end > chunk.vectors.len() {
                    continue;
                }
                let score = dot_product(query, &chunk.vectors[start..end]);
                merge_best(&mut best, id, score);
            }
        }
        let mut out = best.into_iter().collect::<Vec<_>>();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        out.truncate(k);
        out
    }

    fn build_anchor_plan(query_sequence: &[Vec<f32>]) -> Option<AnchorPlan> {
        if query_sequence.is_empty() {
            return None;
        }

        let seq_len = query_sequence.len();
        let mut max_norm_sq = 0.0f32;
        for vector in query_sequence {
            let norm_sq = vector.iter().map(|lane| lane * lane).sum::<f32>();
            max_norm_sq = max_norm_sq.max(norm_sq);
        }
        if max_norm_sq < 1e-8 {
            return None;
        }

        if seq_len == 1 {
            return Some(AnchorPlan {
                primary_offset: 0,
                secondary_offset: None,
                max_delta_sq: 0.0,
                max_norm_sq,
                low_information: true,
            });
        }

        let mut salience = Vec::with_capacity(seq_len);
        salience.push((0usize, 0.0f32));
        let mut max_delta_sq = 0.0f32;
        for idx in 1..seq_len {
            let delta_sq = l2_distance_squared(&query_sequence[idx], &query_sequence[idx - 1]);
            max_delta_sq = max_delta_sq.max(delta_sq);
            salience.push((idx, delta_sq));
        }

        if max_delta_sq < 1e-5 {
            let primary_offset = seq_len / 2;
            let secondary_offset = (seq_len >= 8).then_some(seq_len / 4);
            return Some(AnchorPlan {
                primary_offset,
                secondary_offset: secondary_offset.filter(|offset| *offset != primary_offset),
                max_delta_sq,
                max_norm_sq,
                low_information: true,
            });
        }

        salience.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        let primary_offset = salience.first().map(|(idx, _)| *idx).unwrap_or(0);
        let min_gap = (seq_len / 4).max(4);
        let secondary_offset = salience
            .iter()
            .copied()
            .find_map(|(idx, _)| (idx != primary_offset && idx.abs_diff(primary_offset) >= min_gap).then_some(idx));

        Some(AnchorPlan {
            primary_offset,
            secondary_offset,
            max_delta_sq,
            max_norm_sq,
            low_information: false,
        })
    }

    #[inline(always)]
    fn identify_tile_width(dim: usize) -> usize {
        match dim {
            128 => 32,
            256 => 16,
            _ => 16,
        }
    }

    #[inline(always)]
    fn identify_known_mask_for_plan(anchor_plan: &AnchorPlan) -> u8 {
        let mut mask = 0b01;
        if anchor_plan.secondary_offset.is_some() {
            mask |= 0b10;
        }
        mask
    }

    #[inline(always)]
    fn identify_known_count(mask: u8) -> usize {
        mask.count_ones() as usize
    }

    fn locate_identify_run<V: IdentifyRunView + ?Sized>(
        view: &V,
        dim: usize,
        id: usize,
    ) -> Option<usize> {
        let mut lo = 0usize;
        let mut hi = view.run_count();
        while lo < hi {
            let mid = (lo + hi) / 2;
            if view.run_end_id_exclusive(mid, dim) <= id {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if lo < view.run_count()
            && id >= view.run_start_id(lo)
            && id < view.run_end_id_exclusive(lo, dim)
        {
            Some(lo)
        } else {
            None
        }
    }

    fn identify_candidate_anchor_score<V: IdentifyRunView + ?Sized>(
        view: &V,
        dim: usize,
        start_id: usize,
        anchor_offset: usize,
        query_vec: &[f32],
    ) -> Option<f32> {
        let target_id = start_id.saturating_add(anchor_offset);
        let run_idx = Self::locate_identify_run(view, dim, target_id)?;
        let local_idx = target_id.saturating_sub(view.run_start_id(run_idx));
        let stored = view.vector_at(run_idx, dim, local_idx)?;
        Some(dot_product(query_vec, stored))
    }

    fn verify_identify_candidate<V: IdentifyRunView + ?Sized>(
        view: &V,
        dim: usize,
        start_id: usize,
        query_sequence: &[Vec<f32>],
        anchor_plan: &AnchorPlan,
        known_mask: u8,
        known_sum: f32,
        worst_topk_score: f32,
        bound_epsilon: f32,
    ) -> Option<f32> {
        let seq_len = query_sequence.len();
        let mut run_idx = Self::locate_identify_run(view, dim, start_id)?;
        let mut current_run_start = view.run_start_id(run_idx);
        let mut current_run_len = view.run_len(run_idx, dim);
        let mut local_idx = start_id.saturating_sub(current_run_start);
        let mut total_sim = known_sum;

        for (query_idx, query_vec) in query_sequence.iter().enumerate() {
            if local_idx >= current_run_len {
                run_idx = run_idx.saturating_add(1);
                if run_idx >= view.run_count() {
                    return None;
                }
                current_run_start = view.run_start_id(run_idx);
                if current_run_start != start_id.saturating_add(query_idx) {
                    return None;
                }
                current_run_len = view.run_len(run_idx, dim);
                local_idx = 0;
            }

            let skip_primary = (known_mask & 0b01) != 0 && query_idx == anchor_plan.primary_offset;
            let skip_secondary = (known_mask & 0b10) != 0
                && anchor_plan
                    .secondary_offset
                    .map(|offset| offset == query_idx)
                    .unwrap_or(false);
            if !(skip_primary || skip_secondary) {
                let stored = view.vector_at(run_idx, dim, local_idx)?;
                total_sim += dot_product(query_vec, stored);
            }

            local_idx = local_idx.saturating_add(1);
            if worst_topk_score.is_finite() {
                let frames_remaining = seq_len.saturating_sub(query_idx.saturating_add(1));
                let max_possible_avg = (total_sim + frames_remaining as f32) / seq_len as f32;
                if max_possible_avg + bound_epsilon < worst_topk_score {
                    return None;
                }
            }
        }

        Some(total_sim / seq_len as f32)
    }

    fn identify_exact_over_runs<V: IdentifyRunView + ?Sized>(
        &self,
        view: &V,
        query_sequence: &[Vec<f32>],
        k: usize,
        anchor_plan: &AnchorPlan,
    ) -> Vec<(usize, f32)> {
        if view.run_count() == 0 || query_sequence.is_empty() || k == 0 {
            return Vec::new();
        }

        let dim = self.collection.dim;
        let anchor_query = &query_sequence[anchor_plan.primary_offset];
        let anchor_overfetch = (k.saturating_mul(16)).max(64);
        let total_vectors = (0..view.run_count())
            .map(|run_idx| view.run_len(run_idx, dim))
            .sum::<usize>();
        if total_vectors == 0 {
            return Vec::new();
        }

        #[derive(Clone, Copy, PartialEq)]
        struct MinScore {
            id: usize,
            score: f32,
        }
        impl Eq for MinScore {}
        impl PartialOrd for MinScore {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for MinScore {
            fn cmp(&self, other: &Self) -> Ordering {
                other
                    .score
                    .partial_cmp(&self.score)
                    .unwrap_or(Ordering::Equal)
            }
        }

        let tile_width = Self::identify_tile_width(dim);
        let seq_len_f32 = query_sequence.len() as f32;
        let bound_epsilon = 8.0 * f32::EPSILON * query_sequence.len() as f32;
        let known_mask_template = Self::identify_known_mask_for_plan(anchor_plan);

        IDENTIFY_CANDIDATE_SCRATCH.with(|candidate_scratch| {
            IDENTIFY_TILE_SCORE_SCRATCH.with(|tile_score_scratch| {
                let mut scratch = candidate_scratch.borrow_mut();
                let scratch_capacity = scratch.capacity();
                if scratch_capacity < total_vectors {
                    scratch.reserve(total_vectors - scratch_capacity);
                }
                scratch.resize(total_vectors, IdentifyScoreEntry::default());

                let mut tile_scores = tile_score_scratch.borrow_mut();
                let tile_scores_capacity = tile_scores.capacity();
                if tile_scores_capacity < tile_width {
                    tile_scores.reserve(tile_width - tile_scores_capacity);
                }
                tile_scores.resize(tile_width, 0.0);

                let mut candidate_count = 0usize;
                for run_idx in 0..view.run_count() {
                    let run_start_id = view.run_start_id(run_idx);
                    let run_vectors = view.run_vectors(run_idx);
                    let run_len = run_vectors.len() / dim;
                    let mut local_idx = 0usize;
                    while local_idx < run_len {
                        let batch_count = (run_len - local_idx).min(tile_width);
                        let start = local_idx * dim;
                        let end = start + batch_count * dim;
                        dot_product_scores(
                            anchor_query,
                            &run_vectors[start..end],
                            dim,
                            &mut tile_scores[..batch_count],
                        );
                        for lane_idx in 0..batch_count {
                            let anchor_id = run_start_id.saturating_add(local_idx + lane_idx);
                            if anchor_id < anchor_plan.primary_offset {
                                continue;
                            }
                            scratch[candidate_count] = IdentifyScoreEntry {
                                id: anchor_id - anchor_plan.primary_offset,
                                primary_score: tile_scores[lane_idx],
                                known_sum: tile_scores[lane_idx],
                                known_mask: 0b01,
                            };
                            candidate_count += 1;
                        }
                        local_idx += batch_count;
                    }
                }

                if candidate_count == 0 {
                    scratch.clear();
                    return Vec::new();
                }

                let shortlist_len = candidate_count.min(anchor_overfetch);
                let shortlist = &mut scratch[..candidate_count];
                if shortlist_len < shortlist.len() {
                    shortlist.select_nth_unstable_by(shortlist_len - 1, |a, b| {
                        b.primary_score
                            .partial_cmp(&a.primary_score)
                            .unwrap_or(Ordering::Equal)
                    });
                }
                let shortlist = &mut shortlist[..shortlist_len];
                shortlist.sort_unstable_by(|a, b| {
                    b.primary_score
                        .partial_cmp(&a.primary_score)
                        .unwrap_or(Ordering::Equal)
                });

                let mut topk = std::collections::BinaryHeap::with_capacity(k + 1);
                for candidate in shortlist.iter_mut() {
                    candidate.known_sum = candidate.primary_score;
                    candidate.known_mask = 0b01;

                    if let Some(secondary_offset) = anchor_plan.secondary_offset {
                        let Some(secondary_score) = Self::identify_candidate_anchor_score(
                            view,
                            dim,
                            candidate.id,
                            secondary_offset,
                            &query_sequence[secondary_offset],
                        ) else {
                            continue;
                        };
                        candidate.known_sum += secondary_score;
                        candidate.known_mask = known_mask_template;
                    }

                    if topk.len() >= k {
                        let remaining_frames =
                            query_sequence.len().saturating_sub(Self::identify_known_count(candidate.known_mask));
                        let upper_bound =
                            (candidate.known_sum + remaining_frames as f32) / seq_len_f32;
                        let worst_topk_score = topk
                            .peek()
                            .map(|entry: &MinScore| entry.score)
                            .unwrap_or(f32::NEG_INFINITY);
                        if upper_bound + bound_epsilon < worst_topk_score {
                            continue;
                        }
                    }

                    let worst_topk_score = if topk.len() < k {
                        f32::NEG_INFINITY
                    } else {
                        topk.peek()
                            .map(|entry: &MinScore| entry.score)
                            .unwrap_or(f32::NEG_INFINITY)
                    };

                    let Some(score) = Self::verify_identify_candidate(
                        view,
                        dim,
                        candidate.id,
                        query_sequence,
                        anchor_plan,
                        candidate.known_mask,
                        candidate.known_sum,
                        worst_topk_score,
                        bound_epsilon,
                    ) else {
                        continue;
                    };

                    if topk.len() < k {
                        topk.push(MinScore {
                            id: candidate.id,
                            score,
                        });
                    } else if let Some(worst) = topk.peek() {
                        if score > worst.score {
                            topk.pop();
                            topk.push(MinScore {
                                id: candidate.id,
                                score,
                            });
                        }
                    }
                }

                scratch.clear();
                let mut out = Vec::with_capacity(topk.len());
                while let Some(entry) = topk.pop() {
                    out.push((entry.id, entry.score));
                }
                out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
                out
            })
        })
    }

    fn identify_unindexed_sequences(
        &self,
        query_sequence: &[Vec<f32>],
        k: usize,
        anchor_plan: &AnchorPlan,
    ) -> Vec<(usize, f32)> {
        let snapshot = self.unindexed.chunks.load_full();
        if snapshot.is_empty() {
            return Vec::new();
        }
        self.identify_exact_over_runs(snapshot.as_slice(), query_sequence, k, anchor_plan)
    }

    fn identify_hot_sequences_exact(
        &self,
        query_sequence: &[Vec<f32>],
        k: usize,
        anchor_plan: &AnchorPlan,
    ) -> Vec<(usize, f32)> {
        const EXACT_HOT_IDENTIFY_MAX_VECTORS: usize = 16_384;

        let arena = self.active_identify_arena.load_full();
        if arena.total_vectors == 0 || arena.total_vectors > EXACT_HOT_IDENTIFY_MAX_VECTORS {
            return Vec::new();
        }
        self.identify_exact_over_runs(arena.runs.as_slice(), query_sequence, k, anchor_plan)
    }

    fn index_unindexed_chunk(&self, chunk: &Arc<UnindexedChunk>) -> Result<()> {
        if chunk.rows.is_empty() {
            return Ok(());
        }
        let dim = self.collection.dim;
        let mut by_shard: HashMap<usize, Vec<usize>> = HashMap::new();
        by_shard.reserve(chunk.rows.len().min(1024));
        for (row_idx, row) in chunk.rows.iter().enumerate() {
            let vector_id = usize::try_from(row.vector_id).map_err(|_| {
                anyhow!(
                    "data integrity fault: indexer vector_id exceeds usize on this platform vector_id={}",
                    row.vector_id
                )
            })?;
            by_shard
                .entry(vector_id & self.shard_mask)
                .or_default()
                .push(row_idx);
        }

        let mut inserted_count = 0usize;
        let mut vector_bytes_delta = 0u64;
        let mut min_created_id = usize::MAX;
        let mut max_created_id_exclusive = 0usize;

        for (shard, row_indices) in by_shard {
            for row_slice in row_indices.chunks(64) {
                let shard_lock = self
                    .hot_shards
                    .get(shard)
                    .unwrap_or_else(|| panic!("data integrity fault: missing hot shard {}", shard));
                let mut shard_guard = shard_lock.write();
                let HotShard {
                    vectors,
                    dense_vectors,
                    index,
                    filter_index,
                } = &mut *shard_guard;
                for row_idx in row_slice {
                    let row = &chunk.rows[*row_idx];
                    let vector_id = usize::try_from(row.vector_id).map_err(|_| {
                        anyhow!(
                            "data integrity fault: indexer vector_id exceeds usize on this platform vector_id={}",
                            row.vector_id
                        )
                    })?;
                    let start = row_idx.saturating_mul(dim);
                    let end = start.saturating_add(dim);
                    if end > chunk.vectors.len() {
                        return Err(anyhow!(
                            "data integrity fault: unindexed chunk vector range out of bounds row_idx={} dim={} len={}",
                            row_idx,
                            dim,
                            chunk.vectors.len()
                        ));
                    }
                    let vector_arc = Arc::new(chunk.vectors[start..end].to_vec());
                    let accessor = |node_idx: usize, sink: &mut dyn FnMut(&[f32])| {
                        let vector = dense_vectors.get(node_idx).unwrap_or_else(|| {
                            panic!(
                                "data integrity fault: indexer hot shard missing node_idx={} shard={}",
                                node_idx, shard
                            )
                        });
                        sink(vector.as_slice());
                    };
                    let node_idx =
                        index.insert_with_accessor(row.vector_id, vector_arc.as_slice(), &accessor);
                    if node_idx == dense_vectors.len() {
                        dense_vectors.push(Arc::clone(&vector_arc));
                        vectors.insert(vector_id, Arc::clone(&vector_arc));
                        for tag_id in &row.metadata.tags {
                            filter_index.entry(*tag_id).or_default().insert(node_idx);
                        }
                        inserted_count = inserted_count.saturating_add(1);
                        vector_bytes_delta = vector_bytes_delta.saturating_add(
                            (vector_arc.len() as u64)
                                .saturating_mul(std::mem::size_of::<f32>() as u64),
                        );
                        min_created_id = min_created_id.min(vector_id);
                        max_created_id_exclusive =
                            max_created_id_exclusive.max(vector_id.saturating_add(1));
                    } else if !vectors.contains_key(&vector_id) {
                        let existing = dense_vectors.get(node_idx).unwrap_or_else(|| {
                            panic!(
                                "data integrity fault: indexer missing duplicate node_idx={} shard={}",
                                node_idx, shard
                            )
                        });
                        vectors.insert(vector_id, Arc::clone(existing));
                    }
                }
                drop(shard_guard);
                std::thread::yield_now();
            }
        }

        if inserted_count > 0 {
            if inserted_count == chunk.rows.len() {
                self.append_active_identify_run(
                    usize::try_from(chunk.vector_id_start).unwrap_or_else(|_| {
                        panic!(
                            "data integrity fault: active identify start id overflow id={}",
                            chunk.vector_id_start
                        )
                    }),
                    chunk.vectors.as_slice(),
                )?;
            } else {
                self.rebuild_active_identify_arena_from_hot()?;
            }
            self.hot_vectors_bytes
                .fetch_add(vector_bytes_delta, AtomicOrdering::Relaxed);
            self.hnsw_graph_bytes.fetch_add(
                (inserted_count as u64).saturating_mul(self.estimate_hnsw_bytes_per_vector()),
                AtomicOrdering::Relaxed,
            );
            self.update_hot_window_from_created_range(min_created_id, max_created_id_exclusive);
        }
        Ok(())
    }

    fn enqueue_ingest_batch(
        &self,
        entries: Vec<(Vec<f32>, IngestMetadataV3Input, Option<String>)>,
        estimated_bytes: u64,
    ) -> Result<IngestWriteOutcome> {
        const INGEST_WRITER_RESULT_TIMEOUT: Duration = Duration::from_secs(120);
        self.refresh_ingest_writer_health();
        if self.ingest_writer_failed.load(AtomicOrdering::SeqCst) {
            return Err(anyhow!("ingest writer is unavailable"));
        }

        let (result_tx, result_rx) = sync_channel(1);
        self.ingest_queue_bytes
            .fetch_add(estimated_bytes, AtomicOrdering::Relaxed);
        let send_result = self.ingest_writer_tx.send(IngestWriteJob {
            entries,
            estimated_bytes,
            result_tx,
        });
        if send_result.is_err() {
            atomic_saturating_sub(&self.ingest_queue_bytes, estimated_bytes);
            return Err(anyhow!("ingest writer queue is unavailable"));
        }
        match result_rx.recv_timeout(INGEST_WRITER_RESULT_TIMEOUT) {
            Ok(inner) => inner,
            Err(RecvTimeoutError::Timeout) => {
                self.refresh_ingest_writer_health();
                Err(anyhow!(
                    "ingest writer timeout after {}s (queue_bytes={}, failed={})",
                    INGEST_WRITER_RESULT_TIMEOUT.as_secs(),
                    self.ingest_queue_bytes.load(AtomicOrdering::Relaxed),
                    self.ingest_writer_failed.load(AtomicOrdering::SeqCst)
                ))
            }
            Err(RecvTimeoutError::Disconnected) => {
                self.refresh_ingest_writer_health();
                Err(anyhow!("ingest writer disconnected before sending result"))
            }
        }
    }

    pub fn estimate_ingest_entry_bytes(
        vector: &[f32],
        metadata: &IngestMetadataV3Input,
        idempotency_key: Option<&str>,
    ) -> u64 {
        let vector_bytes = (vector.len() as u64).saturating_mul(std::mem::size_of::<f32>() as u64);
        let metadata_bytes = Self::estimate_ingest_metadata_bytes(metadata);
        let idempotency_bytes = idempotency_key.map(|v| v.len() as u64).unwrap_or(0);
        vector_bytes
            .saturating_add(metadata_bytes)
            .saturating_add(idempotency_bytes)
    }

    pub fn estimate_ingest_batch_bytes(
        entries: &[(Vec<f32>, IngestMetadataV3Input, Option<String>)],
    ) -> u64 {
        entries.iter().fold(0u64, |acc, (vector, metadata, key)| {
            acc.saturating_add(Self::estimate_ingest_entry_bytes(
                vector,
                metadata,
                key.as_deref(),
            ))
        })
    }

    pub fn memory_proxy_bytes(&self) -> u64 {
        self.ingest_queue_bytes
            .load(AtomicOrdering::Relaxed)
            .saturating_add(self.unindexed_bytes.load(AtomicOrdering::Relaxed))
            .saturating_add(self.hot_vectors_bytes.load(AtomicOrdering::Relaxed))
            .saturating_add(self.active_identify_bytes.load(AtomicOrdering::Relaxed))
            .saturating_add(self.hnsw_graph_bytes.load(AtomicOrdering::Relaxed))
            .saturating_add(self.metadata_cache_bytes.load(AtomicOrdering::Relaxed))
            .saturating_add(self.inflight_decode_bytes.load(AtomicOrdering::Relaxed))
    }

    pub fn ingest_backpressure_decision(&self, incoming_bytes: u64) -> IngestBackpressureDecision {
        let budget = self.config.memory_budget_bytes;
        if budget == 0 {
            return IngestBackpressureDecision::Allow;
        }
        let soft_limit = (budget as u128 * 70 / 100).min(u64::MAX as u128) as u64;
        let hard_limit = (budget as u128 * 85 / 100).min(u64::MAX as u128) as u64;
        let projected = self.memory_proxy_bytes().saturating_add(incoming_bytes);
        if projected > hard_limit {
            return IngestBackpressureDecision::Reject {
                projected_bytes: projected,
                hard_limit_bytes: hard_limit,
            };
        }
        if projected > soft_limit {
            let range = hard_limit.saturating_sub(soft_limit).max(1);
            let pressure = projected.saturating_sub(soft_limit);
            let delay_ms = (1 + pressure.saturating_mul(19) / range).clamp(1, 20);
            return IngestBackpressureDecision::Throttle {
                delay: Duration::from_millis(delay_ms),
                projected_bytes: projected,
                soft_limit_bytes: soft_limit,
                hard_limit_bytes: hard_limit,
            };
        }
        IngestBackpressureDecision::Allow
    }

    pub fn record_ingest_soft_throttle(&self, delay: Duration) {
        self.metrics
            .ingest_backpressure_soft_total
            .fetch_add(1, AtomicOrdering::Relaxed);
        self.metrics.ingest_semantic_throttle_ms_total.fetch_add(
            delay.as_millis().min(u64::MAX as u128) as u64,
            AtomicOrdering::Relaxed,
        );
    }

    pub fn record_ingest_hard_reject(&self) {
        self.metrics
            .ingest_backpressure_hard_total
            .fetch_add(1, AtomicOrdering::Relaxed);
    }

    pub fn set_ready(&self, ready: bool, report: impl Into<String>) {
        self.ready.store(ready, AtomicOrdering::SeqCst);
        *self.recovery_report.write() = report.into();
    }

    pub fn live_status(&self) -> bool {
        self.refresh_ingest_writer_health();
        self.live.load(AtomicOrdering::SeqCst)
    }

    fn observe_query_latency_us(&self, us: u64) {
        let bucket = &self.metrics;
        bucket
            .query_latency_count
            .fetch_add(1, AtomicOrdering::Relaxed);
        bucket
            .query_latency_seconds_sum
            .fetch_add(us, AtomicOrdering::Relaxed);
        if us <= 10 {
            bucket
                .query_latency_seconds_le_10us
                .fetch_add(1, AtomicOrdering::Relaxed);
        } else if us <= 25 {
            bucket
                .query_latency_seconds_le_25us
                .fetch_add(1, AtomicOrdering::Relaxed);
        } else if us <= 50 {
            bucket
                .query_latency_seconds_le_50us
                .fetch_add(1, AtomicOrdering::Relaxed);
        } else if us <= 100 {
            bucket
                .query_latency_seconds_le_100us
                .fetch_add(1, AtomicOrdering::Relaxed);
        } else if us <= 500 {
            bucket
                .query_latency_seconds_le_500us
                .fetch_add(1, AtomicOrdering::Relaxed);
        } else if us <= 1_000 {
            bucket
                .query_latency_seconds_le_1000us
                .fetch_add(1, AtomicOrdering::Relaxed);
        }
    }

    pub fn connection_opened(&self) {
        self.metrics
            .active_connections
            .fetch_add(1, AtomicOrdering::Relaxed);
    }

    pub fn connection_closed(&self) {
        let _ = self.metrics.active_connections.fetch_update(
            AtomicOrdering::Relaxed,
            AtomicOrdering::Relaxed,
            |current| Some(current.saturating_sub(1)),
        );
    }

    pub fn http_request_opened(&self) {
        self.metrics
            .active_http_requests
            .fetch_add(1, AtomicOrdering::Relaxed);
        self.connection_opened();
    }

    pub fn http_request_closed(&self) {
        let _ = self.metrics.active_http_requests.fetch_update(
            AtomicOrdering::Relaxed,
            AtomicOrdering::Relaxed,
            |current| Some(current.saturating_sub(1)),
        );
        self.connection_closed();
    }

    pub fn flight_stream_opened(&self) {
        self.metrics
            .active_flight_streams
            .fetch_add(1, AtomicOrdering::Relaxed);
        self.connection_opened();
    }

    pub fn flight_stream_closed(&self) {
        let _ = self.metrics.active_flight_streams.fetch_update(
            AtomicOrdering::Relaxed,
            AtomicOrdering::Relaxed,
            |current| Some(current.saturating_sub(1)),
        );
        self.connection_closed();
    }

    pub fn audit_event_best_effort(
        &self,
        request_id: &str,
        api_key_id: Option<&str>,
        endpoint: &str,
        action: &str,
        status_code: u16,
        latency_ms: f64,
        details: Value,
    ) {
        let event = AuditEvent {
            request_id: request_id.to_string(),
            api_key_id: api_key_id.map(|v| v.to_string()),
            endpoint: endpoint.to_string(),
            action: action.to_string(),
            status_code,
            latency_ms,
            details,
        };
        if self.audit_tx.try_send(event).is_err() {
            self.metrics
                .audit_failures_total
                .fetch_add(1, AtomicOrdering::Relaxed);
        }
    }

    pub fn insert_hot_vector(
        &self,
        vector_id: usize,
        vector: Vec<f32>,
        metadata: VectorMetadataV3,
    ) -> Result<()> {
        let vector_bytes = (vector.len() as u64).saturating_mul(std::mem::size_of::<f32>() as u64);
        let metadata_bytes = Self::estimate_cached_metadata_bytes(&metadata);
        let shard = vector_id & self.shard_mask;
        let shard_lock = self
            .hot_shards
            .get(shard)
            .unwrap_or_else(|| panic!("data integrity fault: missing hot shard {}", shard));
        let mut shard_guard = shard_lock.write();
        let HotShard {
            vectors,
            dense_vectors,
            index,
            filter_index,
        } = &mut *shard_guard;
        let vector_arc = Arc::new(vector);
        let accessor = |node_idx: usize, sink: &mut dyn FnMut(&[f32])| {
            let vector = dense_vectors.get(node_idx).unwrap_or_else(|| {
                panic!(
                    "data integrity fault: hot shard missing node_idx={} shard={}",
                    node_idx, shard
                )
            });
            sink(vector.as_slice());
        };
        let node_idx =
            index.insert_with_accessor(vector_id as u64, vector_arc.as_slice(), &accessor);
        if node_idx == dense_vectors.len() {
            dense_vectors.push(Arc::clone(&vector_arc));
            vectors.insert(vector_id, Arc::clone(&vector_arc));
            for tag_id in &metadata.tags {
                filter_index.entry(*tag_id).or_default().insert(node_idx);
            }
            self.metadata_cache.upsert(vector_id, metadata.clone());
        } else if !vectors.contains_key(&vector_id) {
            let existing = dense_vectors.get(node_idx).unwrap_or_else(|| {
                panic!(
                    "data integrity fault: hot shard missing duplicate node_idx={} shard={}",
                    node_idx, shard
                )
            });
            vectors.insert(vector_id, Arc::clone(existing));
            return Ok(());
        } else {
            return Ok(());
        }
        self.hot_vectors_bytes
            .fetch_add(vector_bytes, AtomicOrdering::Relaxed);
        self.metadata_cache_bytes
            .fetch_add(metadata_bytes, AtomicOrdering::Relaxed);
        self.hnsw_graph_bytes.fetch_add(
            self.estimate_hnsw_bytes_per_vector(),
            AtomicOrdering::Relaxed,
        );
        self.append_active_identify_run(vector_id, vector_arc.as_slice())?;
        self.update_hot_window_from_created_range(vector_id, vector_id.saturating_add(1));
        Ok(())
    }

    pub fn ingest_vector(
        &self,
        vector: &[f32],
        metadata: &IngestMetadataV3Input,
        idempotency_key: Option<&str>,
    ) -> Result<(usize, bool)> {
        if vector.len() != self.collection.dim {
            return Err(anyhow!(
                "dimension mismatch: expected {}, got {}",
                self.collection.dim,
                vector.len()
            ));
        }

        let mut entries = Vec::with_capacity(1);
        entries.push((
            vector.to_vec(),
            metadata.clone(),
            idempotency_key.map(|v| v.to_string()),
        ));
        let result = self.ingest_batch_owned(entries)?;
        result
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("ingest writer returned an empty batch result"))
    }

    /// Batch ingest: single SQLite transaction for N vectors, shard-grouped
    /// hot index updates. Returns (vector_id, created) for each entry.
    pub fn ingest_batch(
        &self,
        entries: &[(Vec<f32>, IngestMetadataV3Input, Option<String>)],
    ) -> Result<Vec<(usize, bool)>> {
        self.ingest_batch_owned(entries.to_vec())
    }

    /// Batch ingest with owned entries to avoid an additional full-batch clone
    /// before enqueueing to the dedicated SQLite writer lane.
    pub fn ingest_batch_owned(
        &self,
        entries: Vec<(Vec<f32>, IngestMetadataV3Input, Option<String>)>,
    ) -> Result<Vec<(usize, bool)>> {
        if entries.is_empty() {
            return Ok(Vec::new());
        }
        // Validate dimensions up front
        for (i, (vector, _, _)) in entries.iter().enumerate() {
            if vector.len() != self.collection.dim {
                return Err(anyhow!(
                    "dimension mismatch at index {}: expected {}, got {}",
                    i,
                    self.collection.dim,
                    vector.len()
                ));
            }
        }
        for (i, (_, _, key)) in entries.iter().enumerate() {
            if let Some(key) = key {
                if key.len() > self.config.max_idempotency_key_len {
                    return Err(anyhow!(
                        "idempotency key too long at index {}: max={} actual={}",
                        i,
                        self.config.max_idempotency_key_len,
                        key.len()
                    ));
                }
            }
        }

        let estimated_bytes = Self::estimate_ingest_batch_bytes(&entries);
        match self.ingest_backpressure_decision(estimated_bytes) {
            IngestBackpressureDecision::Reject {
                projected_bytes,
                hard_limit_bytes,
            } => {
                self.record_ingest_hard_reject();
                return Err(anyhow!(
                    "resource exhausted: memory proxy={} projected={} hard_limit={}",
                    self.memory_proxy_bytes(),
                    projected_bytes,
                    hard_limit_bytes
                ));
            }
            IngestBackpressureDecision::Throttle { .. } | IngestBackpressureDecision::Allow => {}
        }
        if self.config.unindexed_memory_limit_bytes > 0 {
            let current_unindexed = self.unindexed_bytes.load(AtomicOrdering::Relaxed);
            let projected_unindexed = current_unindexed.saturating_add(estimated_bytes);
            if projected_unindexed > self.config.unindexed_memory_limit_bytes {
                self.record_ingest_hard_reject();
                return Err(anyhow!(
                    "resource exhausted: indexing queue full unindexed_bytes={} projected={} limit={}",
                    current_unindexed,
                    projected_unindexed,
                    self.config.unindexed_memory_limit_bytes
                ));
            }
        }

        // 1. Bulk catalog write on dedicated SQLite writer lane.
        let IngestWriteOutcome {
            wal_results,
            entries,
        } = self.enqueue_ingest_batch(entries, estimated_bytes)?;

        // 2. Build accepted (created-only) row set.
        let mut accepted_rows = Vec::new();
        for (result, (vector, _, idempotency_key)) in wal_results.iter().zip(entries.into_iter()) {
            if result.created {
                let metadata = result.metadata.clone().unwrap_or_default();
                let vector_id = u64::try_from(result.vector_id).map_err(|_| {
                    anyhow!(
                        "data integrity fault: vector_id does not fit u64 result.vector_id={}",
                        result.vector_id
                    )
                })?;
                accepted_rows.push(AcceptedWalBatchRow {
                    vector_id,
                    vector,
                    metadata,
                    idempotency_key,
                });
            }
        }

        // 3. Persist deduplicated/id-stamped WAL batch blob in engine domain.
        self.append_wal_ipc_batch_for_accepted_rows(&accepted_rows)?;

        // 4. Publish accepted rows into lock-free unindexed read state.
        let created_count = accepted_rows.len();
        let mut metadata_updates = Vec::with_capacity(created_count);
        for row in &accepted_rows {
            let vector_id = usize::try_from(row.vector_id).map_err(|_| {
                anyhow!(
                    "data integrity fault: vector_id exceeds usize on this platform vector_id={}",
                    row.vector_id
                )
            })?;
            metadata_updates.push((vector_id, row.metadata.clone()));
        }
        if created_count > 0 {
            let chunk = self.build_unindexed_chunk(accepted_rows)?;
            self.publish_unindexed_chunk(chunk)?;
        }

        // 5. Update metadata cache for immediate metadata visibility.
        if created_count > 0 {
            let mut metadata_bytes_delta = 0u64;
            for (vector_id, metadata) in metadata_updates {
                metadata_bytes_delta = metadata_bytes_delta
                    .saturating_add(Self::estimate_cached_metadata_bytes(&metadata));
                self.metadata_cache.upsert(vector_id, metadata.clone());
            }

            self.metadata_cache_bytes
                .fetch_add(metadata_bytes_delta, AtomicOrdering::Relaxed);
            self.metrics
                .ingest_total
                .fetch_add(created_count as u64, AtomicOrdering::Relaxed);
        }

        // 6. Build output
        Ok(wal_results
            .iter()
            .map(|r| (r.vector_id, r.created))
            .collect())
    }

    pub fn query(&self, request: &QueryRequestV2) -> Result<QueryResponseV2> {
        let start = Instant::now();
        if request.vector.len() != self.collection.dim {
            return Err(anyhow!(
                "dimension mismatch: expected {}, got {}",
                self.collection.dim,
                request.vector.len()
            ));
        }

        let resolved_filter = if let Some(filter) = request.filter.as_ref() {
            self.resolve_filter_tag_ids(filter)?
        } else {
            None
        };

        let mut best: FastIdMap<f32> = FastIdMap::default();
        let hot_shard_ef = effective_hot_shard_ef(request.k, request.ef, self.hot_shards.len());

        if !matches!(request.search_tier, SearchTier::Archive) {
            let hot_results = self.query_pool.install(|| {
                self.hot_shards
                    .par_iter()
                    .enumerate()
                    .map(|(shard, shard_lock)| {
                        let shard_guard = shard_lock.read();
                        let HotShard {
                            dense_vectors,
                            index,
                            filter_index,
                            ..
                        } = &*shard_guard;
                        let local_allow = resolved_filter
                            .as_ref()
                            .and_then(|resolved| build_local_allow_set(filter_index, resolved));
                        let accessor = |node_idx: usize, sink: &mut dyn FnMut(&[f32])| {
                            let vector = dense_vectors.get(node_idx).unwrap_or_else(|| {
                                panic!(
                                    "data integrity fault: hot shard missing node_idx={} shard={}",
                                    node_idx, shard
                                )
                            });
                            sink(vector.as_slice());
                        };
                        search_index_with_dynamic_fallback(
                            index,
                            &request.vector,
                            request.k,
                            hot_shard_ef,
                            local_allow.as_ref(),
                            Some(FlatScanHint::HotShard {
                                shard,
                                shard_mask: self.shard_mask,
                            }),
                            Some(&accessor),
                        )
                    })
                    .collect::<Vec<_>>()
            });
            for results in hot_results {
                for (id, score) in results {
                    merge_best(&mut best, id, score);
                }
            }

            let unindexed_results =
                self.query_unindexed_chunks(&request.vector, request.k, resolved_filter.as_ref());
            for (id, score) in unindexed_results {
                merge_best(&mut best, id, score);
            }
        }

        let raw_tier_allowed = |level: i64| match request.search_tier {
            SearchTier::Active => level <= 1,
            SearchTier::All => true,
            SearchTier::Archive => level >= 2,
        };

        let segment_snapshot = self.segments.load_full();
        let segment_results = self.query_pool.install(|| {
            segment_snapshot
                .par_iter()
                .filter(|seg| raw_tier_allowed(seg.record.level))
                .map(|seg| {
                    let index = seg.index.read();
                    let local_allow = resolved_filter
                        .as_ref()
                        .and_then(|resolved| build_local_allow_set(&seg.filter_index, resolved));
                    search_index_with_dynamic_fallback(
                        &index,
                        &request.vector,
                        request.k,
                        request.ef,
                        local_allow.as_ref(),
                        None,
                        None,
                    )
                })
                .collect::<Vec<_>>()
        });
        for results in segment_results {
            for (id, score) in results {
                merge_best(&mut best, id, score);
            }
        }

        if !matches!(request.search_tier, SearchTier::Active) {
            let archive_snapshot = self.archive_segments.load_full();
            let archive_results = self.query_pool.install(|| {
                archive_snapshot
                    .par_iter()
                    .map(|seg| {
                        let local_allow = resolved_filter.as_ref().and_then(|resolved| {
                            build_local_allow_set(&seg.filter_index, resolved)
                        });
                        search_archive_pq_segment(
                            seg,
                            &request.vector,
                            request.k,
                            local_allow.as_ref(),
                        )
                    })
                    .collect::<Vec<_>>()
            });
            for results in archive_results {
                for (id, score) in results {
                    merge_best(&mut best, id, score);
                }
            }
        }

        #[derive(Clone, Copy)]
        struct MinScore {
            id: usize,
            score: f32,
        }
        impl PartialEq for MinScore {
            fn eq(&self, other: &Self) -> bool {
                self.score == other.score
            }
        }
        impl Eq for MinScore {}
        impl PartialOrd for MinScore {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for MinScore {
            fn cmp(&self, other: &Self) -> Ordering {
                // Reverse for min-heap behavior.
                other
                    .score
                    .partial_cmp(&self.score)
                    .unwrap_or(Ordering::Equal)
            }
        }

        let mut heap = std::collections::BinaryHeap::with_capacity(request.k + 1);
        for (id, score) in best {
            if heap.len() < request.k {
                heap.push(MinScore { id, score });
            } else if let Some(worst) = heap.peek() {
                if score > worst.score {
                    heap.pop();
                    heap.push(MinScore { id, score });
                }
            }
        }

        let mut merged = Vec::with_capacity(heap.len());
        while let Some(entry) = heap.pop() {
            merged.push((entry.id, entry.score));
        }
        merged.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        let metadata_map = if request.include_metadata {
            self.metadata_envelopes_for_ids(&merged.iter().map(|(id, _)| *id).collect::<Vec<_>>())?
        } else {
            HashMap::new()
        };

        self.metrics
            .query_total
            .fetch_add(1, AtomicOrdering::Relaxed);
        self.observe_query_latency_us(start.elapsed().as_micros() as u64);
        let query_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        Ok(QueryResponseV2 {
            results: merged
                .into_iter()
                .map(|(id, score)| QueryResultV2 {
                    id,
                    score,
                    metadata: if request.include_metadata {
                        metadata_map.get(&id).cloned()
                    } else {
                        None
                    },
                })
                .collect(),
            query_time_ms,
        })
    }

    pub fn identify(&self, request: &IdentifyRequestV2) -> Result<IdentifyResponseV2> {
        let start = Instant::now();
        if request.vectors.is_empty() {
            return Err(anyhow!("identify requires at least one vector"));
        }
        if request.k == 0 {
            return Ok(IdentifyResponseV2 {
                results: Vec::new(),
                query_time_ms: 0.0,
            });
        }
        for (i, vector) in request.vectors.iter().enumerate() {
            if vector.len() != self.collection.dim {
                return Err(anyhow!(
                    "dimension mismatch at vectors[{}]: expected {}, got {}",
                    i,
                    self.collection.dim,
                    vector.len()
                ));
            }
        }

        let Some(anchor_plan) = Self::build_anchor_plan(&request.vectors) else {
            self.metrics
                .query_total
                .fetch_add(1, AtomicOrdering::Relaxed);
            self.observe_query_latency_us(start.elapsed().as_micros() as u64);
            return Ok(IdentifyResponseV2 {
                results: Vec::new(),
                query_time_ms: start.elapsed().as_secs_f64() * 1000.0,
            });
        };

        let seq_len = request.vectors.len();
        let mut best: FastIdMap<f32> = FastIdMap::default();

        let raw_tier_allowed = |level: i64| match request.search_tier {
            SearchTier::Active => level <= 1,
            SearchTier::All => true,
            SearchTier::Archive => false,
        };

        if !matches!(request.search_tier, SearchTier::Archive) {
            let hot_min_id = self.hot_min_id.load(AtomicOrdering::Relaxed);
            let hot_max_id_exclusive = self.hot_max_id_exclusive.load(AtomicOrdering::Relaxed);
            let metadata_cache = self.metadata_cache.clone();
            if hot_max_id_exclusive > hot_min_id
                && seq_len <= hot_max_id_exclusive.saturating_sub(hot_min_id)
            {
                let hot_exact_results =
                    self.identify_hot_sequences_exact(&request.vectors, request.k, &anchor_plan);
                if !hot_exact_results.is_empty() {
                    for (start_id, score) in hot_exact_results {
                        if start_id < hot_min_id
                            || start_id.saturating_add(seq_len) > hot_max_id_exclusive
                        {
                            continue;
                        }
                        if (0..seq_len).any(|offset| !metadata_cache.contains(&(start_id + offset))) {
                            continue;
                        }
                        merge_best(&mut best, start_id, score);
                    }
                } else {
                    let results = self.query_pool.install(|| {
                        self.hot_shards
                            .par_iter()
                            .enumerate()
                            .map(|(shard_idx, shard_lock)| {
                                let shard_guard = shard_lock.read();
                                let HotShard {
                                    dense_vectors,
                                    index,
                                    ..
                                } = &*shard_guard;
                                let metadata_cache = metadata_cache.clone();
                                let accessor = |node_idx: usize, sink: &mut dyn FnMut(&[f32])| {
                                    let vector = dense_vectors.get(node_idx).unwrap_or_else(|| {
                                        panic!(
                                            "data integrity fault: hot shard missing node_idx={} shard={}",
                                            node_idx, shard_idx
                                        )
                                    });
                                    sink(vector.as_slice());
                                };
                                index.search_subsequence_with_predicate_and_accessor(
                                    &request.vectors,
                                    request.k,
                                    request.ef.max(request.k),
                                    hot_max_id_exclusive as u64,
                                    move |id| {
                                        id >= hot_min_id as u64
                                            && id < hot_max_id_exclusive as u64
                                            && usize::try_from(id)
                                                .ok()
                                                .map(|id| metadata_cache.contains(&id))
                                                .unwrap_or(false)
                                    },
                                    &accessor,
                                )
                            })
                            .collect::<Vec<_>>()
                    });
                    for shard_results in results {
                        for (start_id, score) in shard_results {
                            if start_id < hot_min_id as u64
                                || start_id.saturating_add(seq_len as u64)
                                    > hot_max_id_exclusive as u64
                            {
                                continue;
                            }
                            if (0..seq_len).any(|offset| {
                                let id = start_id + offset as u64;
                                id < hot_min_id as u64
                                    || id >= hot_max_id_exclusive as u64
                                    || usize::try_from(id)
                                        .ok()
                                        .map(|id| !metadata_cache.contains(&id))
                                        .unwrap_or(true)
                            }) {
                                continue;
                            }
                            let start_id = usize::try_from(start_id).unwrap_or_else(|_| {
                                panic!(
                                    "data integrity fault: identify start id overflow id={start_id}"
                                )
                            });
                            merge_best(&mut best, start_id, score);
                        }
                    }
                }
            }

            let segment_snapshot = self.segments.load_full();
            let segment_results = self.query_pool.install(|| {
                segment_snapshot
                    .par_iter()
                    .filter(|seg| raw_tier_allowed(seg.record.level))
                    .map(|seg| {
                        if seq_len > seg.record.row_count {
                            return Vec::new();
                        }

                        let min_start = seg.record.vector_id_start;
                        let max_exclusive = seg.record.vector_id_end.saturating_add(1);
                        let local = {
                            let index = seg.index.read();
                            index.search_subsequence_with_predicate(
                                &request.vectors,
                                request.k,
                                request.ef.max(request.k),
                                max_exclusive as u64,
                                move |id| id >= min_start as u64 && id < max_exclusive as u64,
                            )
                        };

                        local
                            .into_iter()
                            .filter(|(start_id, _)| {
                                *start_id >= min_start as u64
                                    && start_id.saturating_add(seq_len as u64)
                                        <= max_exclusive as u64
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            });

            for results in segment_results {
                for (start_id, score) in results {
                    let start_id = usize::try_from(start_id).unwrap_or_else(|_| {
                        panic!("data integrity fault: segment identify id overflow id={start_id}")
                    });
                    merge_best(&mut best, start_id, score);
                }
            }

            let unindexed_results =
                self.identify_unindexed_sequences(&request.vectors, request.k, &anchor_plan);
            for (start_id, score) in unindexed_results {
                merge_best(&mut best, start_id, score);
            }
        }

        if !matches!(request.search_tier, SearchTier::Active) {
            let archive_snapshot = self.archive_segments.load_full();
            let archive_results = self.query_pool.install(|| {
                archive_snapshot
                    .par_iter()
                    .map(|seg| {
                        identify_archive_pq_segment(seg, &request.vectors, request.k, request.ef)
                    })
                    .collect::<Vec<_>>()
            });
            for results in archive_results {
                for (start_id, score) in results {
                    merge_best(&mut best, start_id, score);
                }
            }
        }

        #[derive(Clone, Copy)]
        struct MinScore {
            id: usize,
            score: f32,
        }
        impl PartialEq for MinScore {
            fn eq(&self, other: &Self) -> bool {
                self.score == other.score
            }
        }
        impl Eq for MinScore {}
        impl PartialOrd for MinScore {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for MinScore {
            fn cmp(&self, other: &Self) -> Ordering {
                other
                    .score
                    .partial_cmp(&self.score)
                    .unwrap_or(Ordering::Equal)
            }
        }

        let mut heap = std::collections::BinaryHeap::with_capacity(request.k + 1);
        for (id, score) in best {
            if heap.len() < request.k {
                heap.push(MinScore { id, score });
            } else if let Some(worst) = heap.peek() {
                if score > worst.score {
                    heap.pop();
                    heap.push(MinScore { id, score });
                }
            }
        }

        let mut merged = Vec::with_capacity(heap.len());
        while let Some(entry) = heap.pop() {
            merged.push((entry.id, entry.score));
        }
        merged.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        let metadata_map = if request.include_metadata {
            self.metadata_envelopes_for_ids(&merged.iter().map(|(id, _)| *id).collect::<Vec<_>>())?
        } else {
            HashMap::new()
        };

        self.metrics
            .identify_total
            .fetch_add(1, AtomicOrdering::Relaxed);
        self.observe_query_latency_us(start.elapsed().as_micros() as u64);
        let query_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        let results = merged
            .into_iter()
            .map(|(id, score)| {
                let metadata = if request.include_metadata {
                    metadata_map.get(&id).cloned()
                } else {
                    None
                };
                let (start_timestamp_ms, duration_ms) = if let Some(meta) = metadata.as_ref() {
                    (meta.sequence_ts, 0u64.saturating_mul(seq_len as u64))
                } else {
                    (0, 0)
                };

                IdentifyResultV2 {
                    id,
                    start_timestamp_ms,
                    duration_ms,
                    score,
                    metadata,
                }
            })
            .collect();

        Ok(IdentifyResponseV2 {
            results,
            query_time_ms,
        })
    }

    pub fn stats(&self) -> Result<StatsResponseV2> {
        let active_segments = self.segments.load().len() + self.archive_segments.load().len();
        let obsolete_segments = self
            .catalog
            .list_segments_by_state(&self.collection.id, &["obsolete"])?
            .len();
        let failed_segments = self
            .catalog
            .list_segments_by_state(&self.collection.id, &["failed"])?
            .len();
        let hot_vectors = self
            .hot_shards
            .iter()
            .map(|shard| shard.read().vectors.len())
            .sum::<usize>();
        let wal_pending = self.catalog.count_wal_pending(&self.collection.id)?;
        let total_vectors = self.catalog.total_vectors(&self.collection.id)?;
        let checkpoint_jobs_inflight = self
            .catalog
            .list_checkpoint_jobs_by_state(&self.collection.id, &["building", "pending_activate"])?
            .len();
        let compaction_jobs_inflight = self
            .catalog
            .list_compaction_jobs_by_state(&self.collection.id, &["building", "pending_activate"])?
            .len();
        let quarantine_usage = self.catalog.quarantine_usage()?;

        Ok(StatsResponseV2 {
            ready: self.ready.load(AtomicOrdering::SeqCst),
            live: self.live_status(),
            active_segments,
            obsolete_segments,
            failed_segments,
            hot_vectors,
            wal_pending,
            total_vectors,
            checkpoint_jobs_inflight,
            compaction_jobs_inflight,
            sqlite_wal_bytes: self.catalog.sqlite_wal_bytes(),
            catalog_read_timeout_total: self.catalog.read_timeout_total(),
            quarantine_files: quarantine_usage.files,
            quarantine_bytes: quarantine_usage.bytes,
            quarantine_evictions_total: self.catalog.quarantine_evictions_total(),
            ingest_queue_bytes: self.ingest_queue_bytes.load(AtomicOrdering::Relaxed),
            memory_proxy_bytes: self.memory_proxy_bytes(),
            memory_budget_bytes: self.config.memory_budget_bytes,
        })
    }

    pub fn render_metrics(&self) -> Result<String> {
        let stats = self.stats()?;
        let mut out = String::new();
        out.push_str("# TYPE vibrato_query_requests_total counter\n");
        out.push_str(&format!(
            "vibrato_query_requests_total {}\n",
            self.metrics.query_total.load(AtomicOrdering::Relaxed)
        ));
        out.push_str("# TYPE vibrato_identify_requests_total counter\n");
        out.push_str(&format!(
            "vibrato_identify_requests_total {}\n",
            self.metrics.identify_total.load(AtomicOrdering::Relaxed)
        ));
        out.push_str("# TYPE vibrato_ingest_requests_total counter\n");
        out.push_str(&format!(
            "vibrato_ingest_requests_total {}\n",
            self.metrics.ingest_total.load(AtomicOrdering::Relaxed)
        ));
        out.push_str("# TYPE vibrato_flight_ingest_batches_total counter\n");
        out.push_str(&format!(
            "vibrato_flight_ingest_batches_total {}\n",
            self.metrics
                .flight_ingest_batches_total
                .load(AtomicOrdering::Relaxed)
        ));
        out.push_str("# TYPE vibrato_flight_ingest_rows_total counter\n");
        out.push_str(&format!(
            "vibrato_flight_ingest_rows_total {}\n",
            self.metrics
                .flight_ingest_rows_total
                .load(AtomicOrdering::Relaxed)
        ));
        out.push_str("# TYPE vibrato_flight_decode_us_total counter\n");
        out.push_str(&format!(
            "vibrato_flight_decode_us_total {}\n",
            self.metrics
                .flight_decode_us_total
                .load(AtomicOrdering::Relaxed)
        ));
        out.push_str("# TYPE vibrato_flight_commit_us_total counter\n");
        out.push_str(&format!(
            "vibrato_flight_commit_us_total {}\n",
            self.metrics
                .flight_commit_us_total
                .load(AtomicOrdering::Relaxed)
        ));
        out.push_str("# TYPE vibrato_flight_ack_us_total counter\n");
        out.push_str(&format!(
            "vibrato_flight_ack_us_total {}\n",
            self.metrics
                .flight_ack_us_total
                .load(AtomicOrdering::Relaxed)
        ));
        out.push_str("# TYPE vibrato_ingest_backpressure_soft_total counter\n");
        out.push_str(&format!(
            "vibrato_ingest_backpressure_soft_total {}\n",
            self.metrics
                .ingest_backpressure_soft_total
                .load(AtomicOrdering::Relaxed)
        ));
        out.push_str("# TYPE vibrato_ingest_backpressure_hard_total counter\n");
        out.push_str(&format!(
            "vibrato_ingest_backpressure_hard_total {}\n",
            self.metrics
                .ingest_backpressure_hard_total
                .load(AtomicOrdering::Relaxed)
        ));
        out.push_str("# TYPE vibrato_ingest_semantic_throttle_ms_total counter\n");
        out.push_str(&format!(
            "vibrato_ingest_semantic_throttle_ms_total {}\n",
            self.metrics
                .ingest_semantic_throttle_ms_total
                .load(AtomicOrdering::Relaxed)
        ));
        out.push_str("# TYPE vibrato_tag_reject_overflow_total counter\n");
        out.push_str(&format!(
            "vibrato_tag_reject_overflow_total {}\n",
            self.metrics
                .tag_reject_overflow_total
                .load(AtomicOrdering::Relaxed)
        ));
        out.push_str("# TYPE vibrato_tag_reject_invalid_total counter\n");
        out.push_str(&format!(
            "vibrato_tag_reject_invalid_total {}\n",
            self.metrics
                .tag_reject_invalid_total
                .load(AtomicOrdering::Relaxed)
        ));
        out.push_str("# TYPE vibrato_flight_decode_chunk_warn_total counter\n");
        out.push_str(&format!(
            "vibrato_flight_decode_chunk_warn_total {}\n",
            self.metrics
                .flight_decode_chunk_warn_total
                .load(AtomicOrdering::Relaxed)
        ));
        out.push_str("# TYPE vibrato_auth_failures_total counter\n");
        out.push_str(&format!(
            "vibrato_auth_failures_total {}\n",
            self.metrics
                .auth_failures_total
                .load(AtomicOrdering::Relaxed)
        ));
        out.push_str("# TYPE vibrato_audit_failures_total counter\n");
        out.push_str(&format!(
            "vibrato_audit_failures_total {}\n",
            self.metrics
                .audit_failures_total
                .load(AtomicOrdering::Relaxed)
        ));
        out.push_str("# TYPE vibrato_checkpoint_total counter\n");
        out.push_str(&format!(
            "vibrato_checkpoint_total {}\n",
            self.metrics.checkpoint_total.load(AtomicOrdering::Relaxed)
        ));
        out.push_str("# TYPE vibrato_compaction_total counter\n");
        out.push_str(&format!(
            "vibrato_compaction_total {}\n",
            self.metrics.compaction_total.load(AtomicOrdering::Relaxed)
        ));
        out.push_str("# TYPE vibrato_obsolete_files_deleted_total counter\n");
        out.push_str(&format!(
            "vibrato_obsolete_files_deleted_total {}\n",
            self.metrics
                .obsolete_files_deleted_total
                .load(AtomicOrdering::Relaxed)
        ));

        let b10 = self
            .metrics
            .query_latency_seconds_le_10us
            .load(AtomicOrdering::Relaxed);
        let b25 = b10
            + self
                .metrics
                .query_latency_seconds_le_25us
                .load(AtomicOrdering::Relaxed);
        let b50 = b25
            + self
                .metrics
                .query_latency_seconds_le_50us
                .load(AtomicOrdering::Relaxed);
        let b100 = b50
            + self
                .metrics
                .query_latency_seconds_le_100us
                .load(AtomicOrdering::Relaxed);
        let b500 = b100
            + self
                .metrics
                .query_latency_seconds_le_500us
                .load(AtomicOrdering::Relaxed);
        let b1000 = b500
            + self
                .metrics
                .query_latency_seconds_le_1000us
                .load(AtomicOrdering::Relaxed);
        let h_count = self
            .metrics
            .query_latency_count
            .load(AtomicOrdering::Relaxed);
        let h_sum_us = self
            .metrics
            .query_latency_seconds_sum
            .load(AtomicOrdering::Relaxed);
        let h_sum_seconds = h_sum_us as f64 / 1_000_000.0;

        out.push_str("# TYPE vibrato_query_latency_seconds histogram\n");
        out.push_str(&format!(
            "vibrato_query_latency_seconds_bucket{{le=\"0.00001\"}} {}\n",
            b10
        ));
        out.push_str(&format!(
            "vibrato_query_latency_seconds_bucket{{le=\"0.000025\"}} {}\n",
            b25
        ));
        out.push_str(&format!(
            "vibrato_query_latency_seconds_bucket{{le=\"0.00005\"}} {}\n",
            b50
        ));
        out.push_str(&format!(
            "vibrato_query_latency_seconds_bucket{{le=\"0.0001\"}} {}\n",
            b100
        ));
        out.push_str(&format!(
            "vibrato_query_latency_seconds_bucket{{le=\"0.0005\"}} {}\n",
            b500
        ));
        out.push_str(&format!(
            "vibrato_query_latency_seconds_bucket{{le=\"0.001\"}} {}\n",
            b1000
        ));
        out.push_str(&format!(
            "vibrato_query_latency_seconds_bucket{{le=\"+Inf\"}} {}\n",
            h_count
        ));
        out.push_str(&format!(
            "vibrato_query_latency_seconds_sum {}\n",
            h_sum_seconds
        ));
        out.push_str(&format!(
            "vibrato_query_latency_seconds_count {}\n",
            h_count
        ));

        let metadata_hits = self
            .metrics
            .metadata_cache_hits_total
            .load(AtomicOrdering::Relaxed);
        let metadata_misses = self
            .metrics
            .metadata_cache_misses_total
            .load(AtomicOrdering::Relaxed);
        let filter_hits = self
            .metrics
            .filter_allow_cache_hits_total
            .load(AtomicOrdering::Relaxed);
        let filter_misses = self
            .metrics
            .filter_allow_cache_misses_total
            .load(AtomicOrdering::Relaxed);
        let cache_hits = metadata_hits.saturating_add(filter_hits);
        let cache_misses = metadata_misses.saturating_add(filter_misses);
        let cache_total = cache_hits.saturating_add(cache_misses);
        let cache_hit_ratio = if cache_total == 0 {
            1.0
        } else {
            cache_hits as f64 / cache_total as f64
        };
        out.push_str("# TYPE vibrato_cache_hit_ratio gauge\n");
        out.push_str(&format!("vibrato_cache_hit_ratio {}\n", cache_hit_ratio));
        out.push_str("# TYPE vibrato_metadata_cache_hits_total counter\n");
        out.push_str(&format!(
            "vibrato_metadata_cache_hits_total {}\n",
            metadata_hits
        ));
        out.push_str("# TYPE vibrato_metadata_cache_misses_total counter\n");
        out.push_str(&format!(
            "vibrato_metadata_cache_misses_total {}\n",
            metadata_misses
        ));
        out.push_str("# TYPE vibrato_filter_allow_cache_hits_total counter\n");
        out.push_str(&format!(
            "vibrato_filter_allow_cache_hits_total {}\n",
            filter_hits
        ));
        out.push_str("# TYPE vibrato_filter_allow_cache_misses_total counter\n");
        out.push_str(&format!(
            "vibrato_filter_allow_cache_misses_total {}\n",
            filter_misses
        ));
        out.push_str("# TYPE vibrato_active_connections gauge\n");
        out.push_str(&format!(
            "vibrato_active_connections {}\n",
            self.metrics
                .active_connections
                .load(AtomicOrdering::Relaxed)
        ));
        out.push_str("# TYPE vibrato_active_http_requests gauge\n");
        out.push_str(&format!(
            "vibrato_active_http_requests {}\n",
            self.metrics
                .active_http_requests
                .load(AtomicOrdering::Relaxed)
        ));
        out.push_str("# TYPE vibrato_active_flight_streams gauge\n");
        out.push_str(&format!(
            "vibrato_active_flight_streams {}\n",
            self.metrics
                .active_flight_streams
                .load(AtomicOrdering::Relaxed)
        ));

        out.push_str("# TYPE vibrato_active_segments gauge\n");
        out.push_str(&format!(
            "vibrato_active_segments {}\n",
            stats.active_segments
        ));
        out.push_str("# TYPE vibrato_obsolete_segments gauge\n");
        out.push_str(&format!(
            "vibrato_obsolete_segments {}\n",
            stats.obsolete_segments
        ));
        out.push_str("# TYPE vibrato_failed_segments gauge\n");
        out.push_str(&format!(
            "vibrato_failed_segments {}\n",
            stats.failed_segments
        ));
        out.push_str("# TYPE vibrato_hot_vectors gauge\n");
        out.push_str(&format!("vibrato_hot_vectors {}\n", stats.hot_vectors));
        out.push_str("# TYPE vibrato_wal_pending gauge\n");
        out.push_str(&format!("vibrato_wal_pending {}\n", stats.wal_pending));
        out.push_str("# TYPE vibrato_wal_pending_rows gauge\n");
        out.push_str(&format!("vibrato_wal_pending_rows {}\n", stats.wal_pending));
        out.push_str("# TYPE vibrato_total_vectors gauge\n");
        out.push_str(&format!("vibrato_total_vectors {}\n", stats.total_vectors));
        out.push_str("# TYPE vibrato_checkpoint_jobs_inflight gauge\n");
        out.push_str(&format!(
            "vibrato_checkpoint_jobs_inflight {}\n",
            stats.checkpoint_jobs_inflight
        ));
        out.push_str("# TYPE vibrato_compaction_jobs_inflight gauge\n");
        out.push_str(&format!(
            "vibrato_compaction_jobs_inflight {}\n",
            stats.compaction_jobs_inflight
        ));
        out.push_str("# TYPE vibrato_sqlite_wal_bytes gauge\n");
        out.push_str(&format!(
            "vibrato_sqlite_wal_bytes {}\n",
            stats.sqlite_wal_bytes
        ));
        out.push_str("# TYPE vibrato_catalog_read_timeout_total counter\n");
        out.push_str(&format!(
            "vibrato_catalog_read_timeout_total {}\n",
            stats.catalog_read_timeout_total
        ));
        out.push_str("# TYPE vibrato_quarantine_files gauge\n");
        out.push_str(&format!(
            "vibrato_quarantine_files {}\n",
            stats.quarantine_files
        ));
        out.push_str("# TYPE vibrato_quarantine_bytes gauge\n");
        out.push_str(&format!(
            "vibrato_quarantine_bytes {}\n",
            stats.quarantine_bytes
        ));
        out.push_str("# TYPE vibrato_quarantine_evictions_total counter\n");
        out.push_str(&format!(
            "vibrato_quarantine_evictions_total {}\n",
            stats.quarantine_evictions_total
        ));
        out.push_str("# TYPE vibrato_ingest_queue_bytes gauge\n");
        out.push_str(&format!(
            "vibrato_ingest_queue_bytes {}\n",
            stats.ingest_queue_bytes
        ));
        out.push_str("# TYPE vibrato_unindexed_bytes gauge\n");
        out.push_str(&format!(
            "vibrato_unindexed_bytes {}\n",
            self.unindexed_bytes.load(AtomicOrdering::Relaxed)
        ));
        out.push_str("# TYPE vibrato_memory_proxy_bytes gauge\n");
        out.push_str(&format!(
            "vibrato_memory_proxy_bytes {}\n",
            stats.memory_proxy_bytes
        ));
        out.push_str("# TYPE vibrato_memory_budget_bytes gauge\n");
        out.push_str(&format!(
            "vibrato_memory_budget_bytes {}\n",
            stats.memory_budget_bytes
        ));
        out.push_str("# TYPE vibrato_ingest_ops_total counter\n");
        out.push_str(&format!(
            "vibrato_ingest_ops_total {}\n",
            self.metrics.ingest_total.load(AtomicOrdering::Relaxed)
        ));

        Ok(out)
    }

    pub fn checkpoint_once(&self) -> Result<JobResponseV2> {
        self.checkpoint_once_with_trigger(CheckpointTrigger::Admin)
    }

    pub fn checkpoint_once_with_trigger(
        &self,
        trigger: CheckpointTrigger,
    ) -> Result<JobResponseV2> {
        let _admin_guard = match trigger {
            CheckpointTrigger::Admin => Some(
                self.admin_ops_lock
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner()),
            ),
            CheckpointTrigger::Background => None,
        };
        let _guard = self
            .checkpoint_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let job_id = make_id("chk");
        let mut pending = self.catalog.pending_wal(&self.collection.id, 50_000)?;
        // Deterministic replay/build order across restarts.
        pending.sort_by_key(|e| e.lsn);
        if pending.is_empty() {
            return Ok(JobResponseV2 {
                job_id,
                state: "idle".to_string(),
                details: Some(json!({"reason": "no_pending_wal"})),
            });
        }
        if matches!(trigger, CheckpointTrigger::Admin) {
            let cooldown_secs = self.config.admin_checkpoint_cooldown_secs;
            let now = current_unix_ts();
            let last = self
                .last_checkpoint_started_unix
                .load(AtomicOrdering::Relaxed);
            if cooldown_secs > 0 && last > 0 && now.saturating_sub(last) < cooldown_secs {
                return Ok(JobResponseV2 {
                    job_id,
                    state: "idle".to_string(),
                    details: Some(json!({
                        "reason": "checkpoint_cooldown",
                        "cooldown_secs": cooldown_secs,
                    })),
                });
            }
            self.last_checkpoint_started_unix
                .store(now, AtomicOrdering::Relaxed);
        }

        let start_lsn = pending.first().map(|e| e.lsn).unwrap_or(0);
        let end_lsn = pending.last().map(|e| e.lsn).unwrap_or(start_lsn);
        let segment_id = make_id("seg");
        let final_path = self.config.segments_dir.join(format!("{}.vdb", segment_id));
        let tmp_path = self.config.tmp_dir.join(format!("{}.vdb.tmp", segment_id));

        self.catalog.upsert_checkpoint_job(&CheckpointJobRecord {
            id: job_id.clone(),
            collection_id: self.collection.id.clone(),
            state: "building".to_string(),
            start_lsn: Some(start_lsn),
            end_lsn: Some(end_lsn),
            details: json!({
                "phase": "start",
                "segment_id": segment_id,
                "tmp_path": tmp_path,
                "final_path": final_path,
                "start_lsn": start_lsn,
                "end_lsn": end_lsn,
            }),
            created_at: current_unix_ts() as i64,
            updated_at: current_unix_ts() as i64,
        })?;

        let result = (|| -> Result<JobResponseV2> {
            let (ids_raw, vectors_raw, metadata_raw) = wal_to_arrays(&pending);
            let (ids, vectors, metadata) = densify_id_space(ids_raw, vectors_raw, metadata_raw)?;
            let hnsw = build_index_from_pairs(
                self.config.hnsw_m,
                self.config.hnsw_ef_construction,
                &ids,
                &vectors,
            );
            self.throttle_background_io(estimate_raw_segment_io_bytes(
                self.collection.dim,
                vectors.len(),
                metadata.len(),
            ));

            write_segment(
                &self.config,
                self.collection.dim,
                &vectors,
                &metadata,
                &hnsw,
                &tmp_path,
            )?;

            self.catalog.insert_segment(&SegmentRecord {
                id: segment_id.clone(),
                collection_id: self.collection.id.clone(),
                level: 0,
                path: final_path.clone(),
                row_count: vectors.len(),
                vector_id_start: *ids.first().unwrap_or(&0),
                vector_id_end: *ids.last().unwrap_or(&0),
                created_lsn: end_lsn,
                state: "building".to_string(),
            })?;

            self.catalog.update_checkpoint_job_state(
                &job_id,
                "building",
                json!({
                    "phase": "segment_written",
                    "segment_id": segment_id,
                    "rows": vectors.len(),
                }),
            )?;

            std::fs::rename(&tmp_path, &final_path)
                .with_context(|| format!("renaming {:?} -> {:?}", tmp_path, final_path))?;
            sync_parent(&final_path)?;

            self.catalog.checkpoint_mark_pending_activate(
                &self.collection.id,
                &segment_id,
                &job_id,
                start_lsn,
                end_lsn,
                json!({
                    "phase": "pending_activate",
                    "segment_id": segment_id,
                    "start_lsn": start_lsn,
                    "end_lsn": end_lsn,
                }),
            )?;

            let handle = self.load_segment_handle(&SegmentRecord {
                id: segment_id.clone(),
                collection_id: self.collection.id.clone(),
                level: 0,
                path: final_path.clone(),
                row_count: vectors.len(),
                vector_id_start: *ids.first().unwrap_or(&0),
                vector_id_end: *ids.last().unwrap_or(&0),
                created_lsn: end_lsn,
                state: "pending_activate".to_string(),
            })?;
            prewarm_segment(&self.config, &handle);

            self.catalog.checkpoint_activate(
                &segment_id,
                &job_id,
                json!({
                    "phase": "completed",
                    "start_lsn": start_lsn,
                    "end_lsn": end_lsn,
                    "rows": vectors.len(),
                    "segment_id": segment_id,
                }),
            )?;
            self.publish_segments_with(|segments| {
                segments.push(handle);
            });

            self.rebuild_hot_from_pending()?;

            self.metrics
                .checkpoint_total
                .fetch_add(1, AtomicOrdering::Relaxed);

            Ok(JobResponseV2 {
                job_id: job_id.clone(),
                state: "completed".to_string(),
                details: Some(json!({
                    "start_lsn": start_lsn,
                    "end_lsn": end_lsn,
                    "rows": vectors.len(),
                    "segment_id": segment_id,
                })),
            })
        })();

        if let Err(err) = &result {
            let _ = self.catalog.update_checkpoint_job_state(
                &job_id,
                "failed",
                json!({
                    "phase": "failed",
                    "segment_id": segment_id,
                    "start_lsn": start_lsn,
                    "end_lsn": end_lsn,
                    "error": err.to_string(),
                }),
            );
            let _ = self.catalog.update_segment_state(&segment_id, "failed");
            let _ = std::fs::remove_file(&tmp_path);
        }

        result
    }

    pub fn compact_once(&self) -> Result<JobResponseV2> {
        const COMPACTION_SKIP_WAL_PENDING_THRESHOLD: usize = 4_096;

        let _admin_guard = self
            .admin_ops_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let _guard = self
            .compaction_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let job_id = make_id("cmp");
        let wal_pending = self.catalog.count_wal_pending(&self.collection.id)?;
        if wal_pending >= COMPACTION_SKIP_WAL_PENDING_THRESHOLD {
            return Ok(JobResponseV2 {
                job_id,
                state: "idle".to_string(),
                details: Some(json!({
                    "reason": "wal_backlog",
                    "wal_pending": wal_pending,
                    "threshold": COMPACTION_SKIP_WAL_PENDING_THRESHOLD,
                })),
            });
        }
        let now = current_unix_ts();
        let last = self
            .last_compaction_started_unix
            .load(AtomicOrdering::Relaxed);
        let cooldown_secs = self.config.admin_compaction_cooldown_secs;
        if cooldown_secs > 0 && last > 0 && now.saturating_sub(last) < cooldown_secs {
            return Ok(JobResponseV2 {
                job_id,
                state: "idle".to_string(),
                details: Some(json!({
                    "reason": "compaction_cooldown",
                    "cooldown_secs": cooldown_secs,
                })),
            });
        }
        self.last_compaction_started_unix
            .store(now, AtomicOrdering::Relaxed);

        let snapshot = self.segments.load_full();
        let l0 = snapshot
            .iter()
            .filter(|s| s.record.level == 0)
            .cloned()
            .collect::<Vec<_>>();
        let l1 = snapshot
            .iter()
            .filter(|s| s.record.level == 1)
            .cloned()
            .collect::<Vec<_>>();

        const MAX_COMPACTION_CANDIDATES: usize = 8;
        let (mut candidates, output_level) = if l0.len() >= 2 {
            (l0, 1_i64)
        } else if l1.len() >= 2 {
            (l1, 2_i64)
        } else {
            (Vec::new(), 0_i64)
        };
        // Sort by vector_id_start for contiguous selection, then cap to bound
        // memory, SQL query size, and lock-hold duration.
        candidates.sort_by_key(|s| s.record.vector_id_start);
        candidates.truncate(MAX_COMPACTION_CANDIDATES);

        if candidates.len() < 2 {
            return Ok(JobResponseV2 {
                job_id,
                state: "idle".to_string(),
                details: Some(json!({"reason": "not_enough_compaction_candidates"})),
            });
        }

        let out_segment_id = make_id("seg");
        let final_path = self
            .config
            .segments_dir
            .join(format!("{}.vdb", out_segment_id));
        let tmp_path = self
            .config
            .tmp_dir
            .join(format!("{}.vdb.tmp", out_segment_id));

        for seg in &candidates {
            let _ = self
                .catalog
                .update_segment_state(&seg.record.id, "compacting");
        }

        self.catalog.upsert_compaction_job(&CompactionJobRecord {
            id: job_id.clone(),
            collection_id: self.collection.id.clone(),
            state: "building".to_string(),
            details: json!({
                "phase": "start",
                "output_segment": out_segment_id,
                "output_tmp_path": tmp_path,
                "output_final_path": final_path,
                "input_segments": candidates.iter().map(|s| s.record.id.clone()).collect::<Vec<_>>(),
            }),
            created_at: current_unix_ts() as i64,
            updated_at: current_unix_ts() as i64,
        })?;

        let result = (|| -> Result<JobResponseV2> {
            // Candidates are already sorted by vector_id_start from the cap logic above.
            let ordered_segments = candidates.clone();
            for pair in ordered_segments.windows(2) {
                let prev = &pair[0].record;
                let next = &pair[1].record;
                if next.vector_id_start <= prev.vector_id_end {
                    tracing::warn!(
                        "compaction candidates overlap: {} [{}..={}] and {} [{}..={}]; dedup merge will resolve",
                        prev.id,
                        prev.vector_id_start,
                        prev.vector_id_end,
                        next.id,
                        next.vector_id_start,
                        next.vector_id_end
                    );
                } else if next.vector_id_start != prev.vector_id_end.saturating_add(1) {
                    tracing::warn!(
                        "compaction sparse id gap detected between {} [{}..={}] and {} [{}..={}]; operation will fail to protect integrity",
                        prev.id,
                        prev.vector_id_start,
                        prev.vector_id_end,
                        next.id,
                        next.vector_id_start,
                        next.vector_id_end
                    );
                }
            }
            let total_rows = ordered_segments
                .iter()
                .map(|seg| seg.record.row_count)
                .sum::<usize>();
            // Collect all (id, vector) pairs; later segments win on overlap (newest-wins).
            let mut merged: FastIdMap<Vec<f32>> = FastIdMap::default();
            merged.reserve(total_rows);
            let mut read_bytes_budget = 0u64;
            for seg in &ordered_segments {
                for offset in 0..seg.record.row_count {
                    let vector_id = seg.record.vector_id_start + offset;
                    merged.insert(vector_id, seg.store.get(offset).to_vec());
                    read_bytes_budget += (self.collection.dim as u64) * 4;
                    if read_bytes_budget >= 1 * 1024 * 1024 {
                        self.throttle_background_io(read_bytes_budget);
                        read_bytes_budget = 0;
                    }
                }
            }
            if read_bytes_budget > 0 {
                self.throttle_background_io(read_bytes_budget);
            }
            // Sort by ID for deterministic output.
            let mut ids_raw: Vec<usize> = merged.keys().copied().collect();
            ids_raw.sort_unstable();
            let vectors_raw: Vec<Vec<f32>> = ids_raw
                .iter()
                .map(|id| merged.remove(id).unwrap())
                .collect();
            drop(merged);
            // Use range-based metadata fetch: single SQL query per range.
            let range_start = *ids_raw.first().unwrap_or(&0);
            let range_end = *ids_raw.last().unwrap_or(&0);
            let metadata_map =
                self.catalog
                    .fetch_metadata_range(&self.collection.id, range_start, range_end)?;
            let metadata_raw = ids_raw
                .iter()
                .map(|id| metadata_map.get(id).cloned().unwrap_or_default())
                .collect::<Vec<_>>();
            let (ids, vectors, metadata) = densify_id_space(ids_raw, vectors_raw, metadata_raw)?;
            let use_archive_pq =
                output_level >= 2 && should_use_archive_pq(self.collection.dim, vectors.len());
            let write_budget = if use_archive_pq {
                estimate_archive_segment_io_bytes(
                    self.collection.dim,
                    vectors.len(),
                    metadata.len(),
                )
            } else {
                estimate_raw_segment_io_bytes(self.collection.dim, vectors.len(), metadata.len())
            };
            self.throttle_background_io(write_budget);
            if use_archive_pq {
                write_archive_pq_segment(self.collection.dim, &vectors, &metadata, &tmp_path)?;
            } else {
                let hnsw = build_index_from_pairs(
                    self.config.hnsw_m,
                    self.config.hnsw_ef_construction,
                    &ids,
                    &vectors,
                );
                write_segment(
                    &self.config,
                    self.collection.dim,
                    &vectors,
                    &metadata,
                    &hnsw,
                    &tmp_path,
                )?;
            }

            let output_record = SegmentRecord {
                id: out_segment_id.clone(),
                collection_id: self.collection.id.clone(),
                level: output_level,
                path: final_path.clone(),
                row_count: vectors.len(),
                vector_id_start: *ids.first().unwrap_or(&0),
                vector_id_end: *ids.last().unwrap_or(&0),
                created_lsn: 0,
                state: "building".to_string(),
            };
            self.catalog.insert_segment(&output_record)?;

            std::fs::rename(&tmp_path, &final_path)?;
            sync_parent(&final_path)?;
            self.catalog.compaction_mark_pending_activate(
                &out_segment_id,
                &job_id,
                json!({
                    "phase": "pending_activate",
                    "output_segment": out_segment_id,
                    "input_segments": candidates.iter().map(|s| s.record.id.clone()).collect::<Vec<_>>(),
                }),
            )?;

            let mut pending_record = output_record.clone();
            pending_record.state = "pending_activate".to_string();
            let loaded_raw = if use_archive_pq {
                None
            } else {
                let handle = self.load_segment_handle(&pending_record)?;
                prewarm_segment(&self.config, &handle);
                Some(handle)
            };
            let loaded_archive = if use_archive_pq {
                Some(self.load_archive_segment_handle(&pending_record)?)
            } else {
                None
            };

            self.catalog.compaction_activate(
                &out_segment_id,
                &candidates
                    .iter()
                    .map(|s| s.record.id.clone())
                    .collect::<Vec<_>>(),
                &job_id,
                json!({
                    "phase": "completed",
                    "input_segments": candidates.iter().map(|s| s.record.id.clone()).collect::<Vec<_>>(),
                    "output_segment": out_segment_id,
                }),
            )?;

            let candidate_ids = candidates
                .iter()
                .map(|s| s.record.id.clone())
                .collect::<HashSet<_>>();
            let archive_out = loaded_archive;
            self.publish_segments_with(|segments| {
                for seg in segments.iter() {
                    if candidate_ids.contains(&seg.record.id) {
                        self.retired_segments
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner())
                            .insert(seg.record.id.clone(), Arc::downgrade(seg));
                    }
                }
                segments.retain(|seg| !candidate_ids.contains(&seg.record.id));
                if let Some(handle) = loaded_raw.clone() {
                    segments.push(handle);
                }
            });
            self.publish_archive_segments_with(|segments| {
                segments.retain(|seg| !candidate_ids.contains(&seg.record.id));
                if let Some(handle) = archive_out.clone() {
                    segments.push(handle);
                }
            });

            let deleted = self.gc_obsolete_segment_files()?;
            if deleted > 0 {
                self.metrics
                    .obsolete_files_deleted_total
                    .fetch_add(deleted as u64, AtomicOrdering::Relaxed);
            }

            self.metrics
                .compaction_total
                .fetch_add(1, AtomicOrdering::Relaxed);

            Ok(JobResponseV2 {
                job_id: job_id.clone(),
                state: "completed".to_string(),
                details: Some(json!({
                    "input_segments": candidates.iter().map(|s| s.record.id.clone()).collect::<Vec<_>>(),
                    "output_segment": out_segment_id,
                    "output_level": output_level,
                    "archive_pq": use_archive_pq,
                })),
            })
        })();

        if let Err(err) = &result {
            for seg in &candidates {
                let _ = self.catalog.update_segment_state(&seg.record.id, "active");
            }
            let _ = self.catalog.update_compaction_job_state(
                &job_id,
                "failed",
                json!({
                    "phase": "failed",
                    "output_segment": out_segment_id,
                    "error": err.to_string(),
                }),
            );
            let _ = self.catalog.update_segment_state(&out_segment_id, "failed");
            let _ = std::fs::remove_file(&tmp_path);
        }

        result
    }

    fn throttle_background_io(&self, bytes: u64) {
        if let Some(throttle) = &self.background_io_throttle {
            throttle.consume(bytes);
        }
    }

    pub fn load_active_segments_from_catalog(&self) -> Result<()> {
        let segments = self
            .catalog
            .list_segments_by_state(&self.collection.id, &["active"])?;
        let mut handles = Vec::with_capacity(segments.len());
        let mut archive_handles = Vec::new();
        for seg in segments {
            match read_v2_header(&seg.path) {
                Ok(header) if header.is_pq_enabled() => {
                    let handle = self.load_archive_segment_handle(&seg)?;
                    archive_handles.push(handle);
                }
                _ => {
                    let handle = self.load_segment_handle(&seg)?;
                    prewarm_segment(&self.config, &handle);
                    handles.push(handle);
                }
            }
        }
        self.segments.store(Arc::new(handles));
        self.archive_segments.store(Arc::new(archive_handles));
        self.retired_segments
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
        Ok(())
    }

    pub fn rebuild_hot_from_pending(&self) -> Result<()> {
        let mut pending = self.catalog.pending_wal_after_lsn(&self.collection.id, 0)?;
        // Enforce LSN order even if upstream query ordering changes.
        pending.sort_by_key(|e| e.lsn);
        let max_lsn = pending.last().map(|e| e.lsn).unwrap_or(0);
        let shard_count = self.hot_shards.len();
        let mut pending_by_shard: Vec<Vec<WalEntry>> = vec![Vec::new(); shard_count];
        for entry in pending {
            let shard = entry.vector_id & self.shard_mask;
            if let Some(bucket) = pending_by_shard.get_mut(shard) {
                bucket.push(entry);
            }
        }

        // 1. Build shadow shard map/index pairs in parallel.
        let shadow_shards: Vec<HotShard> = pending_by_shard
            .into_par_iter()
            .map(|mut entries| {
                let mut vectors: FastIdMap<Arc<Vec<f32>>> = FastIdMap::default();
                vectors.reserve(entries.len());
                let mut dense_vectors: Vec<Arc<Vec<f32>>> = Vec::with_capacity(entries.len());
                let mut filter_index: HashMap<u32, BitmapSet> = HashMap::new();
                entries.sort_by_key(|e| e.vector_id);

                let mut index =
                    make_empty_hot_hnsw(self.config.hnsw_m, self.config.hnsw_ef_construction);
                for entry in entries {
                    let vector_id = entry.vector_id;
                    let vector_arc = Arc::new(entry.vector);
                    let accessor = |node_idx: usize, sink: &mut dyn FnMut(&[f32])| {
                        let vector = dense_vectors.get(node_idx).unwrap_or_else(|| {
                            panic!(
                                "data integrity fault: rebuilding hot shard missing node_idx={node_idx}"
                            )
                        });
                        sink(vector.as_slice());
                    };
                    let node_idx =
                        index.insert_with_accessor(vector_id as u64, vector_arc.as_slice(), &accessor);
                    if node_idx == dense_vectors.len() {
                        dense_vectors.push(Arc::clone(&vector_arc));
                        vectors.insert(vector_id, Arc::clone(&vector_arc));
                        for tag_id in &entry.metadata.tags {
                            filter_index
                                .entry(*tag_id)
                                .or_default()
                                .insert(node_idx);
                        }
                    }
                }

                HotShard {
                    vectors,
                    dense_vectors,
                    index,
                    filter_index,
                }
            })
            .collect();

        // 2. Atomic shard swap (vectors+index move together per shard).
        for (idx, shadow) in shadow_shards.into_iter().enumerate() {
            let mut guard = self.hot_shards[idx].write();
            *guard = shadow;
        }

        // 3. Catch-up phase: replay rows committed while rebuilding shadow shards.
        let mut pending_catchup = self
            .catalog
            .pending_wal_after_lsn(&self.collection.id, max_lsn)?;
        pending_catchup.sort_by_key(|e| e.lsn);

        let mut catchup_by_shard: Vec<Vec<WalEntry>> = vec![Vec::new(); shard_count];
        for entry in pending_catchup {
            let shard = entry.vector_id & self.shard_mask;
            if let Some(bucket) = catchup_by_shard.get_mut(shard) {
                bucket.push(entry);
            }
        }

        for (shard_idx, entries) in catchup_by_shard.into_iter().enumerate() {
            if entries.is_empty() {
                continue;
            }
            let shard_lock = &self.hot_shards[shard_idx];
            let mut shard_guard = shard_lock.write();
            let HotShard {
                vectors,
                dense_vectors,
                index,
                filter_index,
            } = &mut *shard_guard;
            for entry in entries {
                let vector_arc = Arc::new(entry.vector);
                let accessor = |node_idx: usize, sink: &mut dyn FnMut(&[f32])| {
                    let vector = dense_vectors.get(node_idx).unwrap_or_else(|| {
                        panic!(
                            "data integrity fault: catch-up hot shard missing node_idx={} shard={}",
                            node_idx, shard_idx
                        )
                    });
                    sink(vector.as_slice());
                };
                let node_idx = index.insert_with_accessor(
                    entry.vector_id as u64,
                    vector_arc.as_slice(),
                    &accessor,
                );
                if node_idx == dense_vectors.len() {
                    dense_vectors.push(Arc::clone(&vector_arc));
                    vectors.insert(entry.vector_id, Arc::clone(&vector_arc));
                    for tag_id in &entry.metadata.tags {
                        filter_index.entry(*tag_id).or_default().insert(node_idx);
                    }
                } else if !vectors.contains_key(&entry.vector_id) {
                    let existing = dense_vectors.get(node_idx).unwrap_or_else(|| {
                        panic!(
                            "data integrity fault: catch-up missing duplicate node_idx={} shard={}",
                            node_idx, shard_idx
                        )
                    });
                    vectors.insert(entry.vector_id, Arc::clone(existing));
                }
            }
        }

        let mut hot_vector_count = 0u64;
        let mut hot_vector_bytes = 0u64;
        let mut hot_min_id = usize::MAX;
        let mut hot_max_id_exclusive = 0usize;
        for shard_lock in &self.hot_shards {
            let shard_guard = shard_lock.read();
            for (id, vector) in &shard_guard.vectors {
                hot_vector_count = hot_vector_count.saturating_add(1);
                hot_vector_bytes = hot_vector_bytes.saturating_add(
                    (vector.len() as u64).saturating_mul(std::mem::size_of::<f32>() as u64),
                );
                hot_min_id = hot_min_id.min(*id);
                hot_max_id_exclusive = hot_max_id_exclusive.max(id.saturating_add(1));
            }
        }
        self.hot_vectors_bytes
            .store(hot_vector_bytes, AtomicOrdering::Relaxed);
        self.hnsw_graph_bytes.store(
            hot_vector_count.saturating_mul(self.estimate_hnsw_bytes_per_vector()),
            AtomicOrdering::Relaxed,
        );
        if hot_min_id == usize::MAX {
            self.hot_min_id.store(0, AtomicOrdering::Relaxed);
            self.hot_max_id_exclusive.store(0, AtomicOrdering::Relaxed);
        } else {
            self.hot_min_id.store(hot_min_id, AtomicOrdering::Relaxed);
            self.hot_max_id_exclusive
                .store(hot_max_id_exclusive, AtomicOrdering::Relaxed);
        }
        self.rebuild_active_identify_arena_from_hot()?;

        Ok(())
    }

    pub fn rebuild_filter_index(&self) -> Result<()> {
        let entries = self.catalog.fetch_all_metadata(&self.collection.id)?;

        self.metadata_cache.clear();
        let mut metadata_bytes = 0u64;
        for (id, meta) in entries {
            metadata_bytes =
                metadata_bytes.saturating_add(Self::estimate_cached_metadata_bytes(&meta));
            self.metadata_cache.upsert(id, meta.clone());
        }
        self.metadata_cache_bytes
            .store(metadata_bytes, AtomicOrdering::Relaxed);

        // Rebuild localized hot-shard filter indexes using dense node_idx bitmaps.
        for shard_lock in &self.hot_shards {
            let mut shard_guard = shard_lock.write();
            shard_guard.filter_index.clear();
            let ids = shard_guard.vectors.keys().copied().collect::<Vec<_>>();
            for vector_id in ids {
                let Some(metadata) = self.metadata_cache.read(&vector_id, |_, meta| meta.clone())
                else {
                    continue;
                };
                let Some(node_idx) = shard_guard.index.node_index_for_id(vector_id as u64) else {
                    continue;
                };
                for tag_id in metadata.tags {
                    shard_guard
                        .filter_index
                        .entry(tag_id)
                        .or_default()
                        .insert(node_idx);
                }
            }
        }
        Ok(())
    }

    fn metadata_for_ids_internal(&self, ids: &[usize]) -> Result<HashMap<usize, VectorMetadataV3>> {
        if ids.is_empty() {
            return Ok(HashMap::new());
        }

        let mut out = HashMap::with_capacity(ids.len());
        let mut missing = Vec::new();
        for id in ids {
            if let Some(meta) = self.metadata_cache.read(id, |_, meta| meta.clone()) {
                self.metrics
                    .metadata_cache_hits_total
                    .fetch_add(1, AtomicOrdering::Relaxed);
                out.insert(*id, meta);
                continue;
            }
            self.metrics
                .metadata_cache_misses_total
                .fetch_add(1, AtomicOrdering::Relaxed);
            missing.push(*id);
        }

        if !missing.is_empty() {
            let fetched = self.catalog.fetch_metadata(&self.collection.id, &missing)?;
            if !fetched.is_empty() {
                for (id, meta) in &fetched {
                    self.metadata_cache.upsert(*id, meta.clone());
                }
            }
            for (id, meta) in fetched {
                out.insert(id, meta);
            }
        }

        Ok(out)
    }

    fn metadata_envelopes_for_ids(
        &self,
        ids: &[usize],
    ) -> Result<HashMap<usize, MetadataEnvelopeV3>> {
        let metadata_map = self.metadata_for_ids_internal(ids)?;
        if metadata_map.is_empty() {
            return Ok(HashMap::new());
        }

        let mut tag_ids = metadata_map
            .values()
            .flat_map(|meta| meta.tags.iter().copied())
            .collect::<Vec<_>>();
        tag_ids.sort_unstable();
        tag_ids.dedup();
        let tag_texts = self
            .catalog
            .resolve_tag_texts(&self.collection.id, &tag_ids)?;

        let mut out = HashMap::with_capacity(metadata_map.len());
        for (id, meta) in metadata_map {
            let tags = meta
                .tags
                .iter()
                .filter_map(|tag_id| tag_texts.get(tag_id).cloned())
                .collect::<Vec<_>>();
            out.insert(id, MetadataEnvelopeV3::from_internal(&meta, tags));
        }
        Ok(out)
    }

    fn resolve_filter_tag_ids(
        &self,
        filter: &super::model::QueryFilter,
    ) -> Result<Option<ResolvedTagFilter>> {
        if filter.is_empty() {
            return Ok(None);
        }

        let mut resolved = ResolvedTagFilter::default();

        if !filter.tags_all.is_empty() {
            let mut normalized = filter
                .tags_all
                .iter()
                .map(|t| t.trim().to_ascii_lowercase())
                .filter(|t| !t.is_empty())
                .collect::<Vec<_>>();
            normalized.sort();
            normalized.dedup();
            let ids = self
                .catalog
                .resolve_tag_ids_readonly(&self.collection.id, &normalized)?;
            if ids.len() < normalized.len() {
                resolved.impossible = true;
                return Ok(Some(resolved));
            }
            resolved.tags_all_ids = ids;
        }

        if !filter.tags_any.is_empty() {
            let mut normalized = filter
                .tags_any
                .iter()
                .map(|t| t.trim().to_ascii_lowercase())
                .filter(|t| !t.is_empty())
                .collect::<Vec<_>>();
            normalized.sort();
            normalized.dedup();
            resolved.tags_any_ids = self
                .catalog
                .resolve_tag_ids_readonly(&self.collection.id, &normalized)?;
            if !normalized.is_empty() && resolved.tags_any_ids.is_empty() {
                resolved.impossible = true;
                return Ok(Some(resolved));
            }
        }

        Ok(Some(resolved))
    }

    pub fn gc_obsolete_segment_files(&self) -> Result<usize> {
        let obsolete = self
            .catalog
            .list_segments_by_state(&self.collection.id, &["obsolete"])?;
        if obsolete.is_empty() {
            return Ok(0);
        }

        let active = self.segments.load_full();
        let active_ids = active
            .iter()
            .map(|s| s.record.id.clone())
            .collect::<HashSet<_>>();

        let mut deleted = 0usize;
        let mut retired = self
            .retired_segments
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        for seg in obsolete {
            if active_ids.contains(&seg.id) {
                continue;
            }
            let can_delete = retired
                .get(&seg.id)
                .map(|weak| weak.upgrade().is_none())
                .unwrap_or(true);
            if !can_delete {
                continue;
            }
            if seg.path.exists() && std::fs::remove_file(&seg.path).is_ok() {
                deleted += 1;
            }
            retired.remove(&seg.id);
        }

        Ok(deleted)
    }

    pub fn start_background_workers(self: &Arc<Self>) {
        if let Some(audit_rx) = self
            .audit_rx
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
        {
            let audit_state = Arc::clone(self);
            let colocated = self.config.audio_colocated;
            let _ = std::thread::Builder::new()
                .name("vibrato-audit".to_string())
                .spawn(move || {
                    if colocated {
                        set_background_worker_priority();
                    }
                    audit_worker_loop(audit_state, audit_rx);
                });
        }

        let mut guard = self
            .background_worker_handles
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if !guard.is_empty() {
            return;
        }

        let colocated = self.config.audio_colocated;
        let checkpoint_cancel = self.background_cancel.child_token();
        let compaction_cancel = self.background_cancel.child_token();
        let indexer_cancel = self.background_cancel.child_token();
        let checkpoint_state = Arc::clone(self);
        let checkpoint_handle = match std::thread::Builder::new()
            .name("vibrato-bg-checkpoint".to_string())
            .spawn(move || {
                if colocated {
                    set_background_worker_priority();
                }
                checkpoint_loop(checkpoint_state, checkpoint_cancel);
            }) {
            Ok(handle) => handle,
            Err(err) => {
                tracing::error!("failed to spawn checkpoint worker: {err}");
                return;
            }
        };

        let compaction_state = Arc::clone(self);
        let compaction_handle = match std::thread::Builder::new()
            .name("vibrato-bg-compaction".to_string())
            .spawn(move || {
                if colocated {
                    set_background_worker_priority();
                }
                compaction_loop(compaction_state, compaction_cancel);
            }) {
            Ok(handle) => handle,
            Err(err) => {
                tracing::error!("failed to spawn compaction worker: {err}");
                guard.push(checkpoint_handle);
                return;
            }
        };

        let indexer_state = Arc::clone(self);
        let indexer_handle = match std::thread::Builder::new()
            .name("vibrato-bg-indexer".to_string())
            .spawn(move || {
                if colocated {
                    set_background_worker_priority();
                }
                unindexed_index_loop(indexer_state, indexer_cancel);
            }) {
            Ok(handle) => handle,
            Err(err) => {
                tracing::error!("failed to spawn unindexed indexer worker: {err}");
                guard.push(checkpoint_handle);
                guard.push(compaction_handle);
                return;
            }
        };

        guard.push(checkpoint_handle);
        guard.push(compaction_handle);
        guard.push(indexer_handle);
    }

    fn load_segment_handle(&self, seg: &SegmentRecord) -> Result<Arc<SegmentHandle>> {
        let store = Arc::new(VectorStore::open(&seg.path).with_context(|| {
            format!(
                "opening segment store {:?} for segment {}",
                seg.path, seg.id
            )
        })?);

        let mut file = File::open(&seg.path)?;
        let mut header_bytes = [0u8; 64];
        file.read_exact(&mut header_bytes)?;
        let header = VdbHeaderV2::from_bytes(&header_bytes)?;

        let start = seg.vector_id_start;
        let count = store.count;
        let store_for_accessor = store.clone();
        let segment_id = seg.id.clone();
        let accessor = move |node_idx: usize, sink: &mut dyn FnMut(&[f32])| {
            if node_idx < count {
                sink(store_for_accessor.get(node_idx));
            } else {
                panic!(
                    "data integrity fault: segment '{}' missing node_idx={} count={}",
                    segment_id, node_idx, count
                );
            }
        };

        let index = if header.has_graph() && header.graph_offset > 0 {
            file.seek(std::io::SeekFrom::Start(header.graph_offset))?;
            let mut reader = BufReader::new(file);
            HNSW::load_from_reader_with_accessor(&mut reader, accessor)?
        } else {
            let mut idx = HNSW::new_with_accessor(
                self.config.hnsw_m,
                self.config.hnsw_ef_construction,
                accessor,
            );
            for offset in 0..count {
                idx.insert((start + offset) as u64, store.get(offset));
            }
            idx
        };

        let mut filter_index: HashMap<u32, BitmapSet> = HashMap::new();
        let filter_rows = self.catalog.fetch_filter_rows(&self.collection.id)?;
        for row in filter_rows {
            if row.vector_id < start || row.vector_id >= start + count {
                continue;
            }
            let node_idx = row.vector_id - start;
            for tag_id in row.tag_ids {
                filter_index.entry(tag_id).or_default().insert(node_idx);
            }
        }

        Ok(Arc::new(SegmentHandle {
            record: seg.clone(),
            store,
            index: Arc::new(RwLock::new(index)),
            filter_index,
        }))
    }

    fn load_archive_segment_handle(&self, seg: &SegmentRecord) -> Result<Arc<ArchivePqSegment>> {
        let bytes = std::fs::read(&seg.path)
            .with_context(|| format!("reading archive segment {:?}", seg.path))?;
        let (pq, codes, nsub) = parse_archive_segment_bytes(&seg.id, &bytes)?;
        let start = seg.vector_id_start;
        let end_exclusive = seg.vector_id_end.saturating_add(1);
        let mut filter_index: HashMap<u32, BitmapSet> = HashMap::new();
        let filter_rows = self.catalog.fetch_filter_rows(&self.collection.id)?;
        for row in filter_rows {
            if row.vector_id < start || row.vector_id >= end_exclusive {
                continue;
            }
            let node_idx = row.vector_id - start;
            for tag_id in row.tag_ids {
                filter_index.entry(tag_id).or_default().insert(node_idx);
            }
        }

        Ok(Arc::new(ArchivePqSegment {
            record: seg.clone(),
            pq,
            codes: Arc::new(codes),
            num_subspaces: nsub,
            filter_index,
        }))
    }

    fn publish_segments_with<F>(&self, mutator: F)
    where
        F: FnOnce(&mut Vec<Arc<SegmentHandle>>),
    {
        let mut next = (*self.segments.load_full()).clone();
        mutator(&mut next);
        self.segments.store(Arc::new(next));
    }

    fn publish_archive_segments_with<F>(&self, mutator: F)
    where
        F: FnOnce(&mut Vec<Arc<ArchivePqSegment>>),
    {
        let mut next = (*self.archive_segments.load_full()).clone();
        mutator(&mut next);
        self.archive_segments.store(Arc::new(next));
    }
}

fn write_segment(
    config: &ProductionConfig,
    dim: usize,
    vectors: &[Vec<f32>],
    metadata: &[VectorMetadataV3],
    graph: &HNSW,
    output_path: &std::path::Path,
) -> Result<()> {
    let empty_base_path = config.tmp_dir.join(format!("empty_base_{}.vdb", dim));
    if !empty_base_path.exists() {
        let writer = VdbWriterV2::new_raw(&empty_base_path, dim)?;
        writer.finish()?;
    }

    let base_store = VectorStore::open(&empty_base_path)?;
    let legacy_metadata = metadata
        .iter()
        .map(legacy_segment_metadata)
        .collect::<Vec<_>>();
    VdbWriterV2::merge_with_metadata(&base_store, vectors, &legacy_metadata, graph, output_path)?;

    File::open(output_path)?.sync_all()?;
    sync_parent(output_path)?;
    Ok(())
}

fn write_archive_pq_segment(
    dim: usize,
    vectors: &[Vec<f32>],
    metadata: &[VectorMetadataV3],
    output_path: &std::path::Path,
) -> Result<()> {
    if vectors.is_empty() {
        return Err(anyhow!(
            "cannot build archive pq segment from empty vector set"
        ));
    }
    if metadata.len() != vectors.len() {
        return Err(anyhow!(
            "archive pq metadata mismatch: vectors={} metadata={}",
            vectors.len(),
            metadata.len()
        ));
    }

    let nsub = choose_pq_subspaces(dim);
    let training_cfg = TrainingConfig {
        num_subspaces: nsub,
        max_iters: 15,
        tolerance: 1e-4,
        min_vectors: 256,
        seed: Some(42),
    };
    let pq = train_pq(vectors, dim, &training_cfg).with_context(|| {
        format!(
            "training archive pq codebook (vectors={}, dim={}, nsub={})",
            vectors.len(),
            dim,
            nsub
        )
    })?;

    let mut file = File::create(output_path)
        .with_context(|| format!("creating archive pq segment {:?}", output_path))?;
    let code_len = vectors.len() * nsub;
    let codebook_bytes = pq.codebook_bytes();

    let mut metadata_builder = MetadataBuilder::new();
    for item in metadata {
        let legacy = legacy_segment_metadata(item);
        let tags: Vec<&str> = legacy.tags.iter().map(|s| s.as_str()).collect();
        metadata_builder.add_entry(
            &legacy.source_file,
            legacy.start_time_ms,
            legacy.duration_ms,
            legacy.bpm,
            &tags,
        );
    }
    let metadata_bytes = if metadata_builder.is_empty() {
        Vec::new()
    } else {
        metadata_builder.build()
    };

    let vectors_offset = 64u64;
    let codebook_offset = vectors_offset + code_len as u64;
    let metadata_offset = if metadata_bytes.is_empty() {
        0
    } else {
        codebook_offset + codebook_bytes.len() as u64
    };

    let mut flags = vibrato_core::format_v2::flags::PQ_ENABLED;
    if !metadata_bytes.is_empty() {
        flags |= vibrato_core::format_v2::flags::HAS_METADATA;
    }

    let header = VdbHeaderV2 {
        version: 2,
        flags,
        count: vectors.len() as u32,
        dimensions: dim as u32,
        pq_subspaces: nsub as u32,
        vectors_offset,
        codebook_offset,
        metadata_offset,
        graph_offset: 0,
    };
    use std::io::Write;
    file.write_all(&header.to_bytes())?;

    for vector in vectors {
        let codes = pq.encode(vector);
        file.write_all(&codes)?;
    }
    file.write_all(codebook_bytes)?;
    if !metadata_bytes.is_empty() {
        file.write_all(&metadata_bytes)?;
    }
    file.sync_all()?;
    sync_parent(output_path)?;
    Ok(())
}

fn should_use_archive_pq(dim: usize, vector_count: usize) -> bool {
    vector_count >= 256 && choose_pq_subspaces(dim) > 1
}

fn estimate_raw_segment_io_bytes(dim: usize, rows: usize, metadata_rows: usize) -> u64 {
    let vector_bytes = (dim as u64) * (rows as u64) * 4;
    let graph_overhead = (rows as u64) * 128;
    let metadata_estimate = (metadata_rows as u64) * 256;
    64 + vector_bytes + graph_overhead + metadata_estimate
}

fn estimate_archive_segment_io_bytes(dim: usize, rows: usize, metadata_rows: usize) -> u64 {
    let nsub = choose_pq_subspaces(dim).max(1);
    let code_bytes = (rows as u64) * (nsub as u64);
    let codebook_bytes = (dim as u64) * 256;
    let metadata_estimate = (metadata_rows as u64) * 256;
    64 + code_bytes + codebook_bytes + metadata_estimate
}

fn choose_pq_subspaces(dim: usize) -> usize {
    for candidate in [16usize, 12, 10, 8, 6, 5, 4, 3, 2] {
        if dim % candidate == 0 {
            return candidate;
        }
    }
    1
}

fn read_v2_header(path: &std::path::Path) -> Result<VdbHeaderV2> {
    let mut file = File::open(path).with_context(|| format!("opening {:?}", path))?;
    let mut header_bytes = [0u8; 64];
    file.read_exact(&mut header_bytes)?;
    let header = VdbHeaderV2::from_bytes(&header_bytes)?;
    Ok(header)
}

fn parse_archive_segment_bytes(
    segment_label: &str,
    bytes: &[u8],
) -> Result<(ProductQuantizer, Vec<u8>, usize)> {
    let header = VdbHeaderV2::from_bytes(bytes)?;
    if !header.is_pq_enabled() {
        return Err(anyhow!(
            "segment {} is not pq-enabled archive data",
            segment_label
        ));
    }

    let nsub = header.pq_subspaces as usize;
    if nsub == 0 {
        return Err(anyhow!("segment {} has zero pq subspaces", segment_label));
    }

    let code_len = (header.count as usize)
        .checked_mul(nsub)
        .ok_or_else(|| anyhow!("segment {} pq code length overflow", segment_label))?;
    let codes_start = usize::try_from(header.vectors_offset)
        .map_err(|_| anyhow!("segment {} vectors_offset overflow", segment_label))?;
    let codes_end = codes_start
        .checked_add(code_len)
        .ok_or_else(|| anyhow!("segment {} codes range overflow", segment_label))?;
    if codes_end > bytes.len() {
        return Err(anyhow!(
            "invalid pq codes range in segment {}: {}..{} > {}",
            segment_label,
            codes_start,
            codes_end,
            bytes.len()
        ));
    }

    if header.codebook_offset == 0 {
        return Err(anyhow!(
            "segment {} missing pq codebook section",
            segment_label
        ));
    }
    let codebook_start = usize::try_from(header.codebook_offset)
        .map_err(|_| anyhow!("segment {} codebook_offset overflow", segment_label))?;
    let codebook_end = if header.metadata_offset > 0 {
        usize::try_from(header.metadata_offset)
            .map_err(|_| anyhow!("segment {} metadata_offset overflow", segment_label))?
    } else if header.graph_offset > 0 {
        usize::try_from(header.graph_offset)
            .map_err(|_| anyhow!("segment {} graph_offset overflow", segment_label))?
    } else {
        bytes.len()
    };
    if codebook_end <= codebook_start || codebook_end > bytes.len() {
        return Err(anyhow!(
            "invalid codebook range in segment {}: {}..{}",
            segment_label,
            codebook_start,
            codebook_end
        ));
    }

    let pq = ProductQuantizer::from_codebook_bytes(
        header.dimensions as usize,
        nsub,
        &bytes[codebook_start..codebook_end],
    )
    .with_context(|| format!("decoding pq codebook for segment {}", segment_label))?;

    Ok((pq, bytes[codes_start..codes_end].to_vec(), nsub))
}

#[cfg(any(test, feature = "fuzzing"))]
pub fn fuzz_read_v2_header_bytes(bytes: &[u8]) -> Result<()> {
    let mut path = std::env::temp_dir();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    path.push(format!("vibrato_fuzz_header_{nonce}.vdb"));
    std::fs::write(&path, bytes)?;
    let result = read_v2_header(&path).map(|_| ());
    let _ = std::fs::remove_file(&path);
    result
}

#[cfg(any(test, feature = "fuzzing"))]
pub fn fuzz_parse_archive_segment_bytes(bytes: &[u8]) -> Result<()> {
    parse_archive_segment_bytes("fuzz", bytes).map(|_| ())
}

fn build_index_from_pairs(
    m: usize,
    ef_construction: usize,
    ids: &[usize],
    vectors: &[Vec<f32>],
) -> HNSW {
    let dense_vectors = Arc::new(vectors.to_vec());
    let mut hnsw = HNSW::new_with_accessor(m, ef_construction, {
        let dense_vectors = dense_vectors.clone();
        move |node_idx, sink| sink(&dense_vectors[node_idx])
    });

    for (idx, id) in ids.iter().enumerate() {
        hnsw.insert(*id as u64, &vectors[idx]);
    }
    hnsw
}

fn densify_id_space(
    ids: Vec<usize>,
    vectors: Vec<Vec<f32>>,
    metadata: Vec<VectorMetadataV3>,
) -> Result<(Vec<usize>, Vec<Vec<f32>>, Vec<VectorMetadataV3>)> {
    if ids.len() != vectors.len() || ids.len() != metadata.len() {
        return Err(anyhow!(
            "id/vector/metadata length mismatch: ids={} vectors={} metadata={}",
            ids.len(),
            vectors.len(),
            metadata.len()
        ));
    }
    if ids.is_empty() {
        return Ok((Vec::new(), Vec::new(), Vec::new()));
    }

    let mut rows = ids
        .into_iter()
        .zip(vectors)
        .zip(metadata)
        .map(|((id, vector), meta)| (id, vector, meta))
        .collect::<Vec<_>>();
    rows.sort_by_key(|(id, _, _)| *id);

    for pair in rows.windows(2) {
        if pair[1].0 == pair[0].0 {
            return Err(anyhow!(
                "duplicate vector_id {} while densifying",
                pair[0].0
            ));
        }
    }

    let start_id = rows.first().map(|r| r.0).unwrap_or(0);
    let dense_len = rows.len();
    let mut out_ids = Vec::with_capacity(dense_len);
    let mut out_vectors = Vec::with_capacity(dense_len);
    let mut out_metadata = Vec::with_capacity(dense_len);

    let mut cursor = start_id;
    for (id, vector, meta) in rows {
        while cursor < id {
            return Err(anyhow!(
                "data integrity fault: sparse vector id gap while building segment: missing_id={} next_present_id={}",
                cursor,
                id
            ));
        }
        out_ids.push(id);
        out_vectors.push(vector);
        out_metadata.push(meta);
        cursor = id.saturating_add(1);
    }

    Ok((out_ids, out_vectors, out_metadata))
}

fn wal_to_arrays(entries: &[WalEntry]) -> (Vec<usize>, Vec<Vec<f32>>, Vec<VectorMetadataV3>) {
    let mut ids = Vec::with_capacity(entries.len());
    let mut vectors = Vec::with_capacity(entries.len());
    let mut metadata = Vec::with_capacity(entries.len());
    for e in entries {
        ids.push(e.vector_id);
        vectors.push(e.vector.clone());
        metadata.push(e.metadata.clone());
    }
    (ids, vectors, metadata)
}

fn legacy_segment_metadata(metadata: &VectorMetadataV3) -> VectorMetadata {
    if !metadata.payload.is_empty() {
        if let Ok(decoded) = rmp_serde::from_slice::<VectorMetadata>(&metadata.payload) {
            return decoded;
        }
    }
    VectorMetadata {
        source_file: String::new(),
        start_time_ms: metadata.sequence_ts.min(u32::MAX as u64) as u32,
        duration_ms: 0,
        bpm: 0.0,
        tags: metadata.tags.iter().map(|id| id.to_string()).collect(),
    }
}

fn merge_best<S>(best: &mut HashMap<usize, f32, S>, id: usize, score: f32)
where
    S: BuildHasher,
{
    match best.get_mut(&id) {
        Some(existing) => {
            if score > *existing {
                *existing = score;
            }
        }
        None => {
            best.insert(id, score);
        }
    }
}

#[inline]
fn effective_hot_shard_ef(k: usize, requested_ef: usize, shard_count: usize) -> usize {
    let requested = requested_ef.max(k).max(1);
    if shard_count <= 1 {
        return requested;
    }
    // Distributed ANN heuristic: keep per-shard ef proportional to 1/sqrt(shards)
    // to avoid evaluating requested_ef candidates on every shard.
    let scaled = (requested as f32 / (shard_count as f32).sqrt()).ceil() as usize;
    scaled.max(k).max(1)
}

fn salient_anchor_offsets(query_sequence: &[Vec<f32>]) -> Vec<usize> {
    if query_sequence.is_empty() {
        return Vec::new();
    }
    let seq_len = query_sequence.len();
    let probe_count = seq_len.min(3);
    let min_anchor_gap = if seq_len >= 48 {
        48
    } else if seq_len >= 32 {
        32
    } else if seq_len >= 16 {
        16
    } else {
        1
    };

    let mut salience = query_sequence
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let norm_sq: f32 = dot_product(v, v);
            (i, norm_sq)
        })
        .collect::<Vec<_>>();
    let top_partition = (probe_count.saturating_mul(8))
        .min(salience.len())
        .max(probe_count);
    let cmp_desc =
        |a: &(usize, f32), b: &(usize, f32)| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal);
    if top_partition < salience.len() {
        let (top, _, _) = salience.select_nth_unstable_by(top_partition - 1, cmp_desc);
        top.sort_by(cmp_desc);
    } else {
        salience.sort_by(cmp_desc);
    }

    let mut anchors = Vec::with_capacity(probe_count);
    for (idx, _) in salience.iter().take(top_partition).copied() {
        if anchors
            .iter()
            .all(|selected: &usize| selected.abs_diff(idx) >= min_anchor_gap)
        {
            anchors.push(idx);
            if anchors.len() >= probe_count {
                break;
            }
        }
    }
    while anchors.len() < probe_count {
        let mut best_candidate = None;
        let mut best_salience = f32::NEG_INFINITY;
        for (idx, score) in salience.iter().copied() {
            if anchors.contains(&idx) {
                continue;
            }
            if anchors
                .iter()
                .all(|selected: &usize| selected.abs_diff(idx) >= min_anchor_gap)
                && score > best_salience
            {
                best_salience = score;
                best_candidate = Some(idx);
            }
        }
        let Some(idx) = best_candidate else {
            break;
        };
        anchors.push(idx);
    }
    if anchors.is_empty() {
        anchors.push(0);
    }
    anchors
}

fn identify_archive_pq_segment(
    segment: &ArchivePqSegment,
    query_sequence: &[Vec<f32>],
    k: usize,
    ef: usize,
) -> Vec<(usize, f32)> {
    if k == 0 || query_sequence.is_empty() || segment.record.row_count < query_sequence.len() {
        return Vec::new();
    }

    let seq_len = query_sequence.len();
    let nsub = segment.num_subspaces;
    if nsub == 0 {
        return Vec::new();
    }
    let codes = &segment.codes;
    if codes.len() < segment.record.row_count * nsub {
        return Vec::new();
    }

    let anchors = salient_anchor_offsets(query_sequence);
    let anchor_overfetch = ef
        .max(k)
        .saturating_mul(4)
        .max(32)
        .min(segment.record.row_count);
    let seg_start = segment.record.vector_id_start;
    let seg_end_exclusive = segment.record.vector_id_end.saturating_add(1);

    #[derive(Clone, Copy)]
    struct AnchorCandidate {
        offset: usize,
        dist: f32,
    }
    impl PartialEq for AnchorCandidate {
        fn eq(&self, other: &Self) -> bool {
            self.dist == other.dist
        }
    }
    impl Eq for AnchorCandidate {}
    impl PartialOrd for AnchorCandidate {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for AnchorCandidate {
        fn cmp(&self, other: &Self) -> Ordering {
            self.dist
                .partial_cmp(&other.dist)
                .unwrap_or(Ordering::Equal)
        }
    }

    let mut best_by_start: HashMap<usize, f32> = HashMap::new();
    for salient_offset in anchors {
        let table = segment
            .pq
            .compute_distance_table(&query_sequence[salient_offset]);
        let mut heap = std::collections::BinaryHeap::with_capacity(anchor_overfetch + 1);

        for offset in 0..segment.record.row_count {
            let code_start = offset * nsub;
            let code_end = code_start + nsub;
            let dist = ProductQuantizer::adc_distance(&table, &codes[code_start..code_end], nsub);
            if heap.len() < anchor_overfetch {
                heap.push(AnchorCandidate { offset, dist });
            } else if let Some(worst) = heap.peek() {
                if dist < worst.dist {
                    heap.pop();
                    heap.push(AnchorCandidate { offset, dist });
                }
            }
        }

        while let Some(candidate) = heap.pop() {
            let anchor_id = seg_start + candidate.offset;
            if anchor_id < salient_offset {
                continue;
            }
            let start_id = anchor_id - salient_offset;
            if start_id < seg_start || start_id.saturating_add(seq_len) > seg_end_exclusive {
                continue;
            }

            let mut total_sim = 0.0f32;
            let mut valid = true;
            for (offset, query_vec) in query_sequence.iter().enumerate() {
                let vec_id = start_id + offset;
                if vec_id < seg_start || vec_id >= seg_end_exclusive {
                    valid = false;
                    break;
                }
                let local = vec_id - seg_start;
                let code_start = local * nsub;
                let code_end = code_start + nsub;
                if code_end > codes.len() {
                    valid = false;
                    break;
                }
                let reconstructed = segment.pq.decode(&codes[code_start..code_end]);
                total_sim += dot_product(query_vec, &reconstructed);
            }
            if !valid {
                continue;
            }

            let avg_sim = total_sim / seq_len as f32;
            merge_best(&mut best_by_start, start_id, avg_sim);
        }
    }

    let mut out = best_by_start.into_iter().collect::<Vec<_>>();
    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    out.truncate(k);
    out
}

fn search_archive_pq_segment(
    segment: &ArchivePqSegment,
    query: &[f32],
    k: usize,
    allow_set: Option<&BitmapSet>,
) -> Vec<(usize, f32)> {
    if k == 0 || segment.record.row_count == 0 {
        return Vec::new();
    }

    #[derive(Clone, Copy)]
    struct Candidate {
        id: usize,
        dist: f32,
    }
    impl PartialEq for Candidate {
        fn eq(&self, other: &Self) -> bool {
            self.dist == other.dist
        }
    }
    impl Eq for Candidate {}
    impl PartialOrd for Candidate {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Candidate {
        fn cmp(&self, other: &Self) -> Ordering {
            self.dist
                .partial_cmp(&other.dist)
                .unwrap_or(Ordering::Equal)
        }
    }

    let nsub = segment.num_subspaces;
    if nsub == 0 {
        return Vec::new();
    }
    let codes = &segment.codes;
    if codes.len() < segment.record.row_count * nsub {
        return Vec::new();
    }

    let table = segment.pq.compute_distance_table(query);
    let overfetch = (k.saturating_mul(4)).max(k).min(segment.record.row_count);
    let mut heap = std::collections::BinaryHeap::with_capacity(overfetch + 1);

    for offset in 0..segment.record.row_count {
        let vector_id = segment.record.vector_id_start + offset;
        if let Some(allow) = allow_set {
            if !allow.contains(offset) {
                continue;
            }
        }

        let code_start = offset * nsub;
        let code_end = code_start + nsub;
        let dist = ProductQuantizer::adc_distance(&table, &codes[code_start..code_end], nsub);
        if heap.len() < overfetch {
            heap.push(Candidate {
                id: vector_id,
                dist,
            });
        } else if let Some(worst) = heap.peek() {
            if dist < worst.dist {
                heap.pop();
                heap.push(Candidate {
                    id: vector_id,
                    dist,
                });
            }
        }
    }

    let mut reranked = Vec::with_capacity(heap.len());
    while let Some(candidate) = heap.pop() {
        let offset = candidate.id.saturating_sub(segment.record.vector_id_start);
        let code_start = offset * nsub;
        if code_start + nsub > codes.len() {
            continue;
        }
        let reconstructed = segment.pq.decode(&codes[code_start..code_start + nsub]);
        let score = dot_product(query, &reconstructed);
        reranked.push((candidate.id, score));
    }

    reranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    reranked.truncate(k);
    reranked
}

fn search_index_with_dynamic_fallback(
    index: &HNSW,
    query: &[f32],
    k: usize,
    ef: usize,
    allow_set: Option<&BitmapSet>,
    flat_scan_hint: Option<FlatScanHint>,
    accessor: Option<&(dyn Fn(usize, &mut dyn FnMut(&[f32])) + Send + Sync)>,
) -> Vec<(usize, f32)> {
    let to_usize_hits = |hits: Vec<(u64, f32)>| {
        hits.into_iter()
            .map(|(id, score)| {
                (
                    usize::try_from(id).unwrap_or_else(|_| {
                        panic!("data integrity fault: search id overflow id={id}")
                    }),
                    score,
                )
            })
            .collect::<Vec<_>>()
    };
    if allow_set.is_none() {
        return if let Some(accessor) = accessor {
            to_usize_hits(index.search_with_accessor(query, k, ef, accessor))
        } else {
            to_usize_hits(index.search(query, k, ef))
        };
    }

    let allow = allow_set.unwrap();
    let cardinality = allow.cardinality();
    if cardinality == 0 {
        return Vec::new();
    }
    if k == 0 {
        return Vec::new();
    }
    let target_hits = k.min(cardinality);

    // Selective-filter fast path: bypass HNSW entirely and run a direct SIMD scan
    // over the allowed ID bitmap to avoid "swiss-cheese" graph routing.
    const FLAT_SCAN_MAX_IDS: usize = 5_000;
    const FLAT_SCAN_HARD_CAP_IDS: usize = 50_000;
    const FLAT_SCAN_SELECTIVITY_PERCENT: usize = 5;
    let graph_nodes = index.len().max(1);
    let selective = cardinality <= FLAT_SCAN_MAX_IDS
        || (cardinality <= FLAT_SCAN_HARD_CAP_IDS
            && cardinality.saturating_mul(100)
                <= graph_nodes.saturating_mul(FLAT_SCAN_SELECTIVITY_PERCENT));
    if selective {
        return flat_scan_index_with_allow_set(index, query, k, allow, flat_scan_hint, accessor);
    }

    // Stage A (strict): evaluate only allowed IDs.
    let strict = if let Some(accessor) = accessor {
        index.search_filtered_with_accessor(
            query,
            k,
            ef.max(k),
            |node_idx| allow.contains(node_idx),
            accessor,
        )
    } else {
        index.search_filtered(query, k, ef.max(k), |node_idx| allow.contains(node_idx))
    };
    let strict = to_usize_hits(strict);
    if strict.len() >= target_hits {
        return strict;
    }

    // Stage B1: progressively expand ef for sparse filters.
    let expansions: [usize; 3] = if cardinality <= k.saturating_mul(2) {
        [8, 16, 32]
    } else if cardinality <= k.saturating_mul(8) {
        [4, 8, 16]
    } else {
        [2, 4, 8]
    };
    let mut best = strict;
    for mult in expansions {
        let expanded_ef = ef.max(k).saturating_mul(mult);
        let expanded = if let Some(accessor) = accessor {
            to_usize_hits(index.search_filtered_with_accessor(
                query,
                k,
                expanded_ef,
                |node_idx| allow.contains(node_idx),
                accessor,
            ))
        } else {
            to_usize_hits(
                index.search_filtered(query, k, expanded_ef, |node_idx| allow.contains(node_idx)),
            )
        };
        if expanded.len() > best.len() {
            best = expanded;
        }
        if best.len() >= target_hits {
            return best;
        }
    }

    // Stage B2 (route-through): unfiltered expansion to preserve connectivity, then filter.
    // This handles disconnected low-cardinality allow-sets where strict traversal under-fills.
    let desired = k.max(target_hits);
    let route_multiplier = if cardinality <= k.saturating_mul(2) {
        64
    } else {
        32
    };
    let node_cap = index.len().max(k);
    let overfetch_k = desired
        .saturating_mul(route_multiplier)
        .min(4096)
        .min(node_cap);
    let route_ef = ef.max(overfetch_k).saturating_mul(2).min(8192);

    let routed = (if let Some(accessor) = accessor {
        index.search_with_accessor(query, overfetch_k, route_ef, accessor)
    } else {
        index.search(query, overfetch_k, route_ef)
    })
    .into_iter()
    .filter(|(id, _)| {
        index
            .node_index_for_id(*id)
            .map(|node_idx| allow.contains(node_idx))
            .unwrap_or(false)
    })
    .map(|(id, score)| {
        (
            usize::try_from(id).unwrap_or_else(|_| {
                panic!("data integrity fault: routed search id overflow id={id}")
            }),
            score,
        )
    })
    .take(k)
    .collect::<Vec<_>>();

    if routed.len() > best.len() {
        routed
    } else {
        best
    }
}

fn flat_scan_index_with_allow_set(
    index: &HNSW,
    query: &[f32],
    k: usize,
    allow: &BitmapSet,
    flat_scan_hint: Option<FlatScanHint>,
    accessor: Option<&(dyn Fn(usize, &mut dyn FnMut(&[f32])) + Send + Sync)>,
) -> Vec<(usize, f32)> {
    if k == 0 {
        return Vec::new();
    }

    #[derive(Clone, Copy)]
    struct MinScore {
        id: usize,
        score: f32,
    }
    impl PartialEq for MinScore {
        fn eq(&self, other: &Self) -> bool {
            self.score == other.score
        }
    }
    impl Eq for MinScore {}
    impl PartialOrd for MinScore {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for MinScore {
        fn cmp(&self, other: &Self) -> Ordering {
            // Reverse for min-heap behavior.
            other
                .score
                .partial_cmp(&self.score)
                .unwrap_or(Ordering::Equal)
        }
    }

    let mut heap = std::collections::BinaryHeap::with_capacity(k + 1);
    for node_idx in allow.iter_ids() {
        let id_u64 = if let Some(id) = index.id_for_node_idx(node_idx) {
            id
        } else {
            continue;
        };
        let id = usize::try_from(id_u64)
            .unwrap_or_else(|_| panic!("data integrity fault: flat-scan id overflow id={id_u64}"));
        if let Some(FlatScanHint::HotShard { shard, shard_mask }) = flat_scan_hint {
            if (id & shard_mask) != shard {
                continue;
            }
        }
        let maybe_score = if let Some(accessor) = accessor {
            index.score_for_node_idx_with_accessor(query, node_idx, accessor)
        } else {
            index.score_for_node_idx(query, node_idx)
        };
        let Some(score) = maybe_score else {
            continue;
        };
        if heap.len() < k {
            heap.push(MinScore { id, score });
        } else if let Some(worst) = heap.peek() {
            if score > worst.score {
                heap.pop();
                heap.push(MinScore { id, score });
            }
        }
    }

    let mut out = Vec::with_capacity(heap.len());
    while let Some(result) = heap.pop() {
        out.push((result.id, result.score));
    }
    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    out
}

#[derive(Clone, Copy, Debug)]
enum FlatScanHint {
    HotShard { shard: usize, shard_mask: usize },
}

fn checkpoint_loop(state: Arc<ProductionState>, cancel: CancellationToken) {
    const RAPID_DRAIN_WAL_PENDING_THRESHOLD: usize = 50_000;
    const RAPID_DRAIN_SLEEP: Duration = Duration::from_millis(100);
    const RAPID_DRAIN_ERROR_BACKOFF: Duration = Duration::from_secs(5);

    loop {
        if cancel.is_cancelled() {
            break;
        }
        let pending = match state.catalog.count_wal_pending(&state.collection.id) {
            Ok(v) => v,
            Err(err) => {
                tracing::warn!("background checkpoint pending-count failed: {err}");
                std::thread::sleep(state.config.checkpoint_interval);
                continue;
            }
        };
        let rapid_mode = pending >= RAPID_DRAIN_WAL_PENDING_THRESHOLD;
        if rapid_mode {
            std::thread::sleep(RAPID_DRAIN_SLEEP);
        } else {
            std::thread::sleep(state.config.checkpoint_interval);
        }
        if cancel.is_cancelled() {
            break;
        }
        if let Err(err) = state.checkpoint_once_with_trigger(CheckpointTrigger::Background) {
            tracing::warn!("background checkpoint failed: {err}");
            if rapid_mode {
                std::thread::sleep(RAPID_DRAIN_ERROR_BACKOFF);
            }
        }
    }
}

fn compaction_loop(state: Arc<ProductionState>, cancel: CancellationToken) {
    loop {
        if cancel.is_cancelled() {
            break;
        }
        std::thread::sleep(state.config.compaction_interval);
        if cancel.is_cancelled() {
            break;
        }
        if let Err(err) = state.compact_once() {
            tracing::warn!("background compaction failed: {err}");
        }
    }
}

fn unindexed_index_loop(state: Arc<ProductionState>, cancel: CancellationToken) {
    const IDLE_SLEEP: Duration = Duration::from_millis(50);
    const MAX_DRAIN_PER_TICK: usize = 64;

    loop {
        if cancel.is_cancelled() {
            break;
        }

        let mut drained = Vec::with_capacity(MAX_DRAIN_PER_TICK);
        while drained.len() < MAX_DRAIN_PER_TICK {
            let Some(chunk) = state.unindexed.index_queue.pop() else {
                break;
            };
            drained.push(chunk);
        }

        if drained.is_empty() {
            std::thread::sleep(IDLE_SLEEP);
            continue;
        }

        let mut indexed = Vec::with_capacity(drained.len());
        for chunk in drained {
            if cancel.is_cancelled() {
                break;
            }
            match state.index_unindexed_chunk(&chunk) {
                Ok(()) => indexed.push(chunk),
                Err(err) => {
                    if !indexed.is_empty() {
                        state.mark_unindexed_chunks_indexed(&indexed);
                    }
                    tracing::error!("background unindexed indexer fatal failure: {err}");
                    state.set_ready(false, format!("degraded: background indexer failure: {err}"));
                    state.live.store(false, AtomicOrdering::SeqCst);
                    state.background_cancel.cancel();
                    return;
                }
            }
        }

        if !indexed.is_empty() {
            state.mark_unindexed_chunks_indexed(&indexed);
        }
    }
}

fn audit_worker_loop(state: Arc<ProductionState>, rx: Receiver<AuditEvent>) {
    let mut buffer = Vec::with_capacity(1000);
    let batch_size = 500;
    let batch_timeout = Duration::from_millis(250);
    let mut last_flush = Instant::now();

    loop {
        let now = Instant::now();
        let elapsed = now.duration_since(last_flush);
        let timeout = if elapsed >= batch_timeout {
            Duration::from_millis(0)
        } else {
            batch_timeout - elapsed
        };

        match rx.recv_timeout(timeout) {
            Ok(event) => {
                buffer.push(event);
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                // Continue to flush check
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                if !buffer.is_empty() {
                    if let Err(e) = state.catalog.audit_events_batch(&buffer) {
                        tracing::error!("final audit batch failed: {}", e);
                        state
                            .metrics
                            .audit_failures_total
                            .fetch_add(buffer.len() as u64, AtomicOrdering::Relaxed);
                    }
                }
                break;
            }
        }

        if !buffer.is_empty()
            && (buffer.len() >= batch_size || last_flush.elapsed() >= batch_timeout)
        {
            // Optimization: No retries. If the DB is locked for 30s, retrying for 20ms is useless.
            // Drop the batch to save the system.
            if let Err(e) = state.catalog.audit_events_batch(&buffer) {
                tracing::error!("audit batch failed, dropped {} events: {}", buffer.len(), e);
                state
                    .metrics
                    .audit_failures_total
                    .fetch_add(buffer.len() as u64, AtomicOrdering::Relaxed);
            }
            buffer.clear();
            last_flush = Instant::now();
        }
    }
}

fn make_empty_hot_hnsw(m: usize, ef_construction: usize) -> HNSW {
    HNSW::new_with_accessor(m, ef_construction, |id, _| {
        panic!("data integrity fault: hot shard accessor was not provided for id={id}");
    })
}

fn next_power_of_two_at_least_one(v: usize) -> usize {
    let n = v.max(1);
    if n.is_power_of_two() {
        n
    } else {
        n.next_power_of_two()
    }
}

fn prewarm_segment(config: &ProductionConfig, segment: &SegmentHandle) {
    let mapped = segment.store.mapped_bytes();
    if mapped.len() < 64 {
        return;
    }

    let header = VdbHeaderV2::from_bytes(&mapped[..64]);
    if let Ok(header) = header {
        if header.metadata_offset > 0 {
            let metadata_start = header.metadata_offset as usize;
            let metadata_end = if header.graph_offset > 0 {
                header.graph_offset as usize
            } else {
                mapped.len()
            };
            if metadata_start < metadata_end && metadata_end <= mapped.len() {
                let metadata = &mapped[metadata_start..metadata_end];
                advise_memory(metadata, libc_advice_willneed());
                pretouch_sampled_pages(metadata, 64);
            }
        }
        if header.graph_offset > 0 {
            let graph_start = header.graph_offset as usize;
            if graph_start < mapped.len() {
                let graph = &mapped[graph_start..];
                advise_memory(graph, libc_advice_willneed());
                pretouch_sampled_pages(graph, 64);
            }
        }
    }

    let vector_bytes = segment.store.vector_bytes();
    let vector_advice = match config.vector_madvise_mode {
        VectorMadviseMode::Normal => libc_advice_normal(),
        VectorMadviseMode::Random => libc_advice_random(),
    };
    advise_memory(vector_bytes, vector_advice);
    pretouch_sampled_pages(vector_bytes, 128);
}

fn pretouch_sampled_pages(bytes: &[u8], sample_pages: usize) {
    if bytes.is_empty() {
        return;
    }
    let page_size = 4096usize;
    let total_pages = (bytes.len() + page_size - 1) / page_size;
    let stride_pages = (total_pages / sample_pages.max(1)).max(1);
    let stride = stride_pages * page_size;
    for i in (0..bytes.len()).step_by(stride).take(sample_pages.max(1)) {
        std::hint::black_box(bytes[i]);
    }
}

fn libc_advice_normal() -> i32 {
    #[cfg(unix)]
    {
        libc::MADV_NORMAL
    }
    #[cfg(not(unix))]
    {
        0
    }
}

fn libc_advice_random() -> i32 {
    #[cfg(unix)]
    {
        libc::MADV_RANDOM
    }
    #[cfg(not(unix))]
    {
        0
    }
}

fn libc_advice_willneed() -> i32 {
    #[cfg(unix)]
    {
        libc::MADV_WILLNEED
    }
    #[cfg(not(unix))]
    {
        0
    }
}

fn advise_memory(bytes: &[u8], advice: i32) {
    #[cfg(unix)]
    unsafe {
        if bytes.is_empty() {
            return;
        }
        let _ = libc::madvise(bytes.as_ptr() as *mut libc::c_void, bytes.len(), advice);
    }
}

fn atomic_saturating_sub(counter: &AtomicU64, amount: u64) {
    loop {
        let current = counter.load(AtomicOrdering::Relaxed);
        let next = current.saturating_sub(amount);
        if counter
            .compare_exchange(
                current,
                next,
                AtomicOrdering::Relaxed,
                AtomicOrdering::Relaxed,
            )
            .is_ok()
        {
            break;
        }
    }
}

fn sync_parent(path: &std::path::Path) -> Result<()> {
    #[cfg(unix)]
    {
        if let Some(parent) = path.parent() {
            File::open(parent)?.sync_all()?;
        }
    }
    Ok(())
}

fn set_background_worker_priority() {
    #[cfg(target_os = "linux")]
    {
        use libc::{c_int, c_long, pid_t};

        const IOPRIO_CLASS_SHIFT: c_int = 13;
        const IOPRIO_CLASS_IDLE: c_int = 3;
        const IOPRIO_WHO_PROCESS: c_int = 1;

        unsafe {
            // Linux NPTL treats nice values as per-thread scheduling hints.
            let tid = libc::syscall(libc::SYS_gettid as c_long) as pid_t;
            let _ = libc::setpriority(libc::PRIO_PROCESS, tid as u32, 15);

            // Keep background checkpoint/compaction IO idle under DAW workloads.
            let ioprio = (IOPRIO_CLASS_IDLE << IOPRIO_CLASS_SHIFT) | 0;
            let _ = libc::syscall(
                libc::SYS_ioprio_set as c_long,
                IOPRIO_WHO_PROCESS as c_long,
                tid as c_long,
                ioprio as c_long,
            );
        }
    }
}

fn make_id(prefix: &str) -> String {
    use rand::RngCore;
    let mut bytes = [0u8; 8];
    rand::thread_rng().fill_bytes(&mut bytes);
    let mut suffix = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        suffix.push_str(&format!("{:02x}", b));
    }
    format!("{}_{}_{}", prefix, current_unix_ts(), suffix)
}

fn current_unix_ts() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prod::{bootstrap_data_dirs, CatalogOptions};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use tempfile::{tempdir, TempDir};
    use vibrato_core::simd::l2_normalized;

    fn random_vector(dim: usize, rng: &mut StdRng) -> Vec<f32> {
        let raw: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect();
        l2_normalized(&raw)
    }

    fn test_state(dim: usize) -> (Arc<ProductionState>, TempDir) {
        let dir = tempdir().expect("tempdir");
        let mut config =
            ProductionConfig::from_data_dir(dir.path().join("engine_tests"), "tests".to_string(), dim);
        config.public_health_metrics = false;
        config.audio_colocated = false;
        config.checkpoint_interval = Duration::from_secs(3600);
        config.compaction_interval = Duration::from_secs(3600);
        config.hot_index_shards = 4;
        config.query_pool_threads = 0;
        config.flight_decode_pool_threads = 1;
        bootstrap_data_dirs(&config).expect("bootstrap data dirs");
        let catalog = Arc::new(
            SqliteCatalog::open_with_options(
                &config.catalog_path(),
                CatalogOptions {
                    read_timeout_ms: config.catalog_read_timeout_ms,
                    wal_autocheckpoint_pages: config.sqlite_wal_autocheckpoint_pages,
                    max_tag_registry_size: config.max_tag_registry_size,
                },
            )
            .expect("open catalog"),
        );
        let state = ProductionState::initialize(config, catalog).expect("initialize state");
        (state, dir)
    }

    fn normalized_sequence_with_spike(dim: usize, seq_len: usize, spike_idx: usize) -> Vec<Vec<f32>> {
        let base = l2_normalized(
            &(0..dim)
                .map(|idx| if idx % 2 == 0 { 1.0 } else { 0.25 })
                .collect::<Vec<_>>(),
        );
        let spike = l2_normalized(
            &(0..dim)
                .map(|idx| if idx == (spike_idx % dim) { 2.0 } else { -0.5 })
                .collect::<Vec<_>>(),
        );
        let mut out = Vec::with_capacity(seq_len);
        for idx in 0..seq_len {
            out.push(if idx == spike_idx {
                spike.clone()
            } else {
                base.clone()
            });
        }
        out
    }

    #[test]
    fn sparse_filter_fallback_returns_only_allowed_ids() {
        let dim = 32;
        let total = 1024;
        let mut rng = StdRng::seed_from_u64(42);
        let vectors = Arc::new(
            (0..total)
                .map(|_| random_vector(dim, &mut rng))
                .collect::<Vec<_>>(),
        );

        let accessor_vectors = vectors.clone();
        let mut index = HNSW::new_with_accessor_and_seed(
            16,
            100,
            move |id, sink| sink(&accessor_vectors[id]),
            11,
        );
        for id in 0..total {
            index.insert(id as u64, &vectors[id]);
        }

        let target_id = total - 1;
        let query = vectors[target_id].clone();
        let mut allow_set = BitmapSet::with_capacity(total);
        allow_set.grow(total);
        allow_set.insert(target_id);

        let results =
            search_index_with_dynamic_fallback(&index, &query, 10, 1, Some(&allow_set), None, None);
        assert!(!results.is_empty(), "expected at least one filtered hit");
        assert!(
            results.iter().all(|(id, _)| allow_set.contains(*id)),
            "fallback must not leak non-filtered IDs"
        );
        assert_eq!(results[0].0, target_id);
    }

    #[test]
    fn hot_shard_ef_scales_with_sqrt_of_shard_count() {
        let ef = effective_hot_shard_ef(10, 64, 4);
        // ceil(64 / sqrt(4)) = 32
        assert_eq!(ef, 32);
    }

    #[test]
    fn hot_shard_ef_never_drops_below_k() {
        let ef = effective_hot_shard_ef(50, 64, 8);
        assert_eq!(ef, 50);
    }

    #[test]
    fn anchor_plan_picks_transient_anchor() {
        let query = normalized_sequence_with_spike(16, 12, 7);
        let plan = ProductionState::build_anchor_plan(&query).expect("anchor plan");
        assert_eq!(plan.primary_offset, 7);
        assert!(!plan.low_information);
    }

    #[test]
    fn anchor_plan_uses_deterministic_fallback_for_low_information() {
        let frame = l2_normalized(&[1.0, 2.0, 3.0, 4.0]);
        let query = vec![frame; 12];
        let plan = ProductionState::build_anchor_plan(&query).expect("anchor plan");
        assert!(plan.low_information);
        assert_eq!(plan.primary_offset, 6);
        assert_eq!(plan.secondary_offset, Some(3));
    }

    #[test]
    fn anchor_plan_rejects_all_zero_query() {
        let query = vec![vec![0.0f32; 8]; 6];
        assert!(ProductionState::build_anchor_plan(&query).is_none());
    }

    #[test]
    fn exact_cascade_matches_single_anchor_results() {
        let (state, _dir) = test_state(8);
        let runs = vec![Arc::new(ActiveIdentifyRun {
            start_id: 0,
            vectors: (0..64)
                .flat_map(|idx| {
                    let mut raw = vec![0.0f32; 8];
                    raw[idx % 8] = 1.0;
                    if idx % 9 == 0 {
                        raw[(idx + 3) % 8] = 0.5;
                    }
                    l2_normalized(&raw)
                })
                .collect::<Vec<_>>(),
        })];
        let query = (10..16)
            .map(|idx| {
                let start = idx * 8;
                runs[0].vectors[start..start + 8].to_vec()
            })
            .collect::<Vec<_>>();
        let plan = ProductionState::build_anchor_plan(&query).expect("anchor plan");
        let mut single_anchor = plan;
        single_anchor.secondary_offset = None;
        let exact = state.identify_exact_over_runs(runs.as_slice(), &query, 5, &plan);
        let single = state.identify_exact_over_runs(runs.as_slice(), &query, 5, &single_anchor);
        assert_eq!(exact, single);
    }

    #[test]
    fn insert_sorted_unindexed_chunks_orders_out_of_order_runs() {
        let pool = Arc::new(SegQueue::new());
        let metric = Arc::new(AtomicU64::new(0));
        let mk_chunk = |start: u64, end: u64| {
            Arc::new(UnindexedChunk {
                vector_id_start: start,
                vector_id_end: end,
                vectors: vec![0.0; (((end - start + 1) as usize).max(1)) * 4],
                rows: (start..=end)
                    .map(|id| UnindexedChunkRow {
                        vector_id: id,
                        metadata: VectorMetadataV3::default(),
                    })
                    .collect(),
                tag_allow_list: HashMap::new(),
                approx_bytes: 0,
                recycle_pool: Arc::downgrade(&pool),
                unindexed_bytes: Arc::downgrade(&metric),
            })
        };

        let one = mk_chunk(10, 19);
        let two = mk_chunk(0, 9);
        let sorted = ProductionState::insert_sorted_unindexed_chunks(&[one.clone()], &two)
            .expect("sorted insertion");
        assert_eq!(sorted[0].vector_id_start, 0);
        assert_eq!(sorted[1].vector_id_start, 10);
    }

    #[test]
    fn publish_unindexed_chunk_overlap_degrades_state() {
        let (state, _dir) = test_state(4);
        let pool = state.unindexed.vector_pool.clone();
        let metric = Arc::new(AtomicU64::new(0));
        let mk_chunk = |start: u64, end: u64| {
            Arc::new(UnindexedChunk {
                vector_id_start: start,
                vector_id_end: end,
                vectors: vec![0.0; (((end - start + 1) as usize).max(1)) * 4],
                rows: (start..=end)
                    .map(|id| UnindexedChunkRow {
                        vector_id: id,
                        metadata: VectorMetadataV3::default(),
                    })
                    .collect(),
                tag_allow_list: HashMap::new(),
                approx_bytes: 0,
                recycle_pool: Arc::downgrade(&pool),
                unindexed_bytes: Arc::downgrade(&metric),
            })
        };

        state
            .publish_unindexed_chunk(mk_chunk(0, 9))
            .expect("first publish");
        let err = state
            .publish_unindexed_chunk(mk_chunk(5, 14))
            .expect_err("overlap must fail");
        assert!(err.to_string().contains("overlapping unindexed identify chunk"));
        assert!(!state.live.load(AtomicOrdering::SeqCst));
        assert!(!state.ready.load(AtomicOrdering::SeqCst));
    }

    #[test]
    fn append_active_identify_run_overlap_degrades_state() {
        let (state, _dir) = test_state(4);
        let first = vec![0.25f32; 8];
        let second = vec![0.5f32; 8];
        state
            .append_active_identify_run(10, &first)
            .expect("initial append");
        let err = state
            .append_active_identify_run(11, &second)
            .expect_err("overlap must fail");
        assert!(err.to_string().contains("active identify overlap"));
        assert!(!state.live.load(AtomicOrdering::SeqCst));
        assert!(!state.ready.load(AtomicOrdering::SeqCst));
    }
}
