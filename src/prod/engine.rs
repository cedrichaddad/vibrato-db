use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, Read, Seek};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering as AtomicOrdering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use arc_swap::ArcSwap;
use parking_lot::RwLock;
use rayon::prelude::*;
use rayon::ThreadPool;
use serde_json::{json, Value};
use vibrato_core::format_v2::{VdbHeaderV2, VdbWriterV2};
use vibrato_core::hnsw::HNSW;
use vibrato_core::metadata::{MetadataBuilder, VectorMetadata};
use vibrato_core::pq::ProductQuantizer;
use vibrato_core::simd::dot_product;
use vibrato_core::store::VectorStore;
use vibrato_core::training::{train_pq, TrainingConfig};

use super::catalog::{
    CatalogStore, CheckpointJobRecord, CollectionRecord, CompactionJobRecord, SegmentRecord,
    SqliteCatalog, WalEntry,
};
use super::filter::{BitmapSet, FilterIndex};
use super::model::{
    AuditEvent, IdentifyRequestV2, IdentifyResponseV2, IdentifyResultV2, JobResponseV2,
    QueryRequestV2, QueryResponseV2, QueryResultV2, SearchTier, StatsResponseV2,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorMadviseMode {
    Normal,
    Random,
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
    pub orphan_ttl: Duration,
    pub audio_colocated: bool,
    pub public_health_metrics: bool,
    pub catalog_read_timeout_ms: u64,
    pub sqlite_wal_autocheckpoint_pages: u32,
    pub quarantine_max_files: usize,
    pub quarantine_max_bytes: u64,
    pub background_io_mb_per_sec: u64,
    pub hot_index_shards: usize,
    pub vector_madvise_mode: VectorMadviseMode,
    pub api_pepper: String,
}

#[derive(Default)]
pub struct Metrics {
    pub query_total: AtomicU64,
    pub identify_total: AtomicU64,
    pub ingest_total: AtomicU64,
    pub auth_failures_total: AtomicU64,
    pub audit_failures_total: AtomicU64,
    pub checkpoint_total: AtomicU64,
    pub compaction_total: AtomicU64,
    pub obsolete_files_deleted_total: AtomicU64,
    pub query_latency_count: AtomicU64,
    pub query_latency_us_sum: AtomicU64,
    pub query_latency_us_le_10: AtomicU64,
    pub query_latency_us_le_25: AtomicU64,
    pub query_latency_us_le_50: AtomicU64,
    pub query_latency_us_le_100: AtomicU64,
    pub query_latency_us_le_250: AtomicU64,
    pub query_latency_us_le_500: AtomicU64,
    pub query_latency_us_le_1000: AtomicU64,
    pub query_latency_us_le_2500: AtomicU64,
    pub query_latency_us_le_5000: AtomicU64,
    pub query_latency_us_gt_5000: AtomicU64,
}

pub struct SegmentHandle {
    pub record: SegmentRecord,
    pub store: Arc<VectorStore>,
    pub index: Arc<RwLock<HNSW>>,
}

pub struct ArchivePqSegment {
    pub record: SegmentRecord,
    pub pq: ProductQuantizer,
    pub codes: Arc<Vec<u8>>,
    pub num_subspaces: usize,
}

pub struct ProductionState {
    pub config: ProductionConfig,
    pub catalog: Arc<SqliteCatalog>,
    pub collection: CollectionRecord,
    pub live: AtomicBool,
    pub ready: AtomicBool,
    pub recovery_report: RwLock<String>,

    pub hot_vector_shards: Vec<ArcSwap<RwLock<HashMap<usize, Arc<Vec<f32>>>>>>,
    pub hot_indices: Vec<RwLock<HNSW>>,
    pub shard_mask: usize,
    pub metadata_cache: Arc<RwLock<HashMap<usize, VectorMetadata>>>,
    pub segments: ArcSwap<Vec<Arc<SegmentHandle>>>,
    pub archive_segments: ArcSwap<Vec<Arc<ArchivePqSegment>>>,
    pub filter_index: Arc<RwLock<FilterIndex>>,
    pub retired_segments: std::sync::Mutex<HashMap<String, Weak<SegmentHandle>>>,

    pub metrics: Metrics,
    pub admin_ops_lock: std::sync::Mutex<()>,
    pub checkpoint_lock: std::sync::Mutex<()>,
    pub compaction_lock: std::sync::Mutex<()>,
    pub background_pool: std::sync::Mutex<Option<Arc<ThreadPool>>>,
    background_io_throttle: Option<Arc<BackgroundIoThrottle>>,
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
            orphan_ttl: Duration::from_secs(168 * 3600),
            audio_colocated: true,
            public_health_metrics: true,
            catalog_read_timeout_ms: 5_000,
            sqlite_wal_autocheckpoint_pages: 1000,
            quarantine_max_files: 50,
            quarantine_max_bytes: 5 * 1024 * 1024 * 1024,
            background_io_mb_per_sec: 40,
            hot_index_shards: 8,
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
    pub fn initialize(config: ProductionConfig, catalog: Arc<SqliteCatalog>) -> Result<Arc<Self>> {
        let collection = catalog.ensure_collection(&config.collection_name, config.dim)?;

        let shard_count = next_power_of_two_at_least_one(config.hot_index_shards);
        let shard_mask = shard_count - 1;
        let hot_vector_shards = (0..shard_count)
            .map(|_| ArcSwap::new(Arc::new(RwLock::new(HashMap::new()))))
            .collect::<Vec<_>>();
        let hot_indices = hot_vector_shards
            .iter()
            .map(|shard_vectors| {
                let accessor = make_hot_accessor(shard_vectors.load_full(), config.dim);
                RwLock::new(HNSW::new_with_accessor(
                    config.hnsw_m,
                    config.hnsw_ef_construction,
                    accessor,
                ))
            })
            .collect::<Vec<_>>();
        let metadata_cache = Arc::new(RwLock::new(HashMap::new()));
        let (audit_tx, audit_rx) = sync_channel(4096);
        let available = std::thread::available_parallelism()
            .map(|v| v.get())
            .unwrap_or(2);
        let query_threads = (available / 2).clamp(1, 4);
        let query_pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(query_threads)
                .thread_name(|idx| format!("vibrato-query-{idx}"))
                .build()
                .context("building query pool")?,
        );
        let background_io_throttle =
            if config.audio_colocated && config.background_io_mb_per_sec > 0 {
                Some(Arc::new(BackgroundIoThrottle::new(
                    config.background_io_mb_per_sec * 1024 * 1024,
                )))
            } else {
                None
            };

        let state = Arc::new(Self {
            config,
            catalog,
            collection,
            live: AtomicBool::new(true),
            ready: AtomicBool::new(false),
            recovery_report: RwLock::new("initializing".to_string()),
            hot_vector_shards,
            hot_indices,
            shard_mask,
            metadata_cache,
            segments: ArcSwap::from_pointee(Vec::new()),
            archive_segments: ArcSwap::from_pointee(Vec::new()),
            filter_index: Arc::new(RwLock::new(FilterIndex::default())),
            retired_segments: std::sync::Mutex::new(HashMap::new()),
            metrics: Metrics::default(),
            admin_ops_lock: std::sync::Mutex::new(()),
            checkpoint_lock: std::sync::Mutex::new(()),
            compaction_lock: std::sync::Mutex::new(()),
            background_pool: std::sync::Mutex::new(None),
            background_io_throttle,
            query_pool,
            audit_tx,
            audit_rx: std::sync::Mutex::new(Some(audit_rx)),
        });

        Ok(state)
    }

    pub fn set_ready(&self, ready: bool, report: impl Into<String>) {
        self.ready.store(ready, AtomicOrdering::SeqCst);
        *self.recovery_report.write() = report.into();
    }

    fn observe_query_latency_us(&self, us: u64) {
        let bucket = &self.metrics;
        bucket
            .query_latency_count
            .fetch_add(1, AtomicOrdering::Relaxed);
        bucket
            .query_latency_us_sum
            .fetch_add(us, AtomicOrdering::Relaxed);
        if us <= 10 {
            bucket
                .query_latency_us_le_10
                .fetch_add(1, AtomicOrdering::Relaxed);
        } else if us <= 25 {
            bucket
                .query_latency_us_le_25
                .fetch_add(1, AtomicOrdering::Relaxed);
        } else if us <= 50 {
            bucket
                .query_latency_us_le_50
                .fetch_add(1, AtomicOrdering::Relaxed);
        } else if us <= 100 {
            bucket
                .query_latency_us_le_100
                .fetch_add(1, AtomicOrdering::Relaxed);
        } else if us <= 250 {
            bucket
                .query_latency_us_le_250
                .fetch_add(1, AtomicOrdering::Relaxed);
        } else if us <= 500 {
            bucket
                .query_latency_us_le_500
                .fetch_add(1, AtomicOrdering::Relaxed);
        } else if us <= 1_000 {
            bucket
                .query_latency_us_le_1000
                .fetch_add(1, AtomicOrdering::Relaxed);
        } else if us <= 2_500 {
            bucket
                .query_latency_us_le_2500
                .fetch_add(1, AtomicOrdering::Relaxed);
        } else if us <= 5_000 {
            bucket
                .query_latency_us_le_5000
                .fetch_add(1, AtomicOrdering::Relaxed);
        } else {
            bucket
                .query_latency_us_gt_5000
                .fetch_add(1, AtomicOrdering::Relaxed);
        }
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

    pub fn insert_hot_vector(&self, vector_id: usize, vector: Vec<f32>, metadata: VectorMetadata) {
        let shard = vector_id & self.shard_mask;
        if let Some(shard_vectors) = self.hot_vector_shards.get(shard) {
            shard_vectors
                .load()
                .write()
                .insert(vector_id, Arc::new(vector));
        }
        if let Some(index) = self.hot_indices.get(shard) {
            index.write().insert(vector_id);
        }
        self.metadata_cache
            .write()
            .insert(vector_id, metadata.clone());
        self.filter_index.write().add(vector_id, &metadata);
    }

    pub fn ingest_vector(
        &self,
        vector: &[f32],
        metadata: &VectorMetadata,
        idempotency_key: Option<&str>,
    ) -> Result<(usize, bool)> {
        if vector.len() != self.collection.dim {
            return Err(anyhow!(
                "dimension mismatch: expected {}, got {}",
                self.collection.dim,
                vector.len()
            ));
        }

        let ingest = self.catalog.ingest_wal_atomic(
            &self.collection.id,
            vector,
            metadata,
            idempotency_key,
        )?;
        if ingest.created {
            self.insert_hot_vector(ingest.vector_id, vector.to_vec(), metadata.clone());
            self.metrics
                .ingest_total
                .fetch_add(1, AtomicOrdering::Relaxed);
        }
        Ok((ingest.vector_id, ingest.created))
    }

    /// Batch ingest: single SQLite transaction for N vectors, shard-grouped
    /// hot index updates. Returns (vector_id, created) for each entry.
    pub fn ingest_batch(
        &self,
        entries: &[(Vec<f32>, VectorMetadata, Option<String>)],
    ) -> Result<Vec<(usize, bool)>> {
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

        // 1. Bulk catalog write — single BEGIN...COMMIT
        let wal_results = self
            .catalog
            .ingest_wal_batch(&self.collection.id, entries)?;

        // 2. Group by shard for efficient lock acquisition
        let mut by_shard: HashMap<usize, Vec<(usize, &[f32], &VectorMetadata)>> = HashMap::new();
        for (i, result) in wal_results.iter().enumerate() {
            if result.created {
                let shard = result.vector_id & self.shard_mask;
                let (ref vec, ref meta, _) = entries[i];
                by_shard
                    .entry(shard)
                    .or_default()
                    .push((result.vector_id, vec.as_slice(), meta));
            }
        }

        // 3. Insert into hot index shards in parallel — one lock acquisition per shard
        by_shard.par_iter().for_each(|(shard, items)| {
            if let Some(shard_vectors) = self.hot_vector_shards.get(*shard) {
                let guard = shard_vectors.load();
                let mut vec_guard = guard.write();
                for &(vid, vec, _) in items {
                    vec_guard.insert(vid, Arc::new(vec.to_vec()));
                }
            }
            if let Some(index) = self.hot_indices.get(*shard) {
                let mut idx_guard = index.write();
                for &(vid, _, _) in items {
                    idx_guard.insert(vid);
                }
            }
        });

        // 4. Update metadata cache + filter index in one pass
        let created_count = by_shard.values().map(|v| v.len()).sum::<usize>();
        if created_count > 0 {
            let mut meta_cache = self.metadata_cache.write();
            let mut filter_idx = self.filter_index.write();
            for items in by_shard.values() {
                for &(vid, _, meta) in items {
                    meta_cache.insert(vid, meta.clone());
                    filter_idx.add(vid, meta);
                }
            }
            self.metrics
                .ingest_total
                .fetch_add(created_count as u64, AtomicOrdering::Relaxed);
        }

        // 5. Build output
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

        let allow = request
            .filter
            .as_ref()
            .map(|f| self.filter_index.read().build_allow_set(f));

        let allow_bitmap = allow.flatten();

        let mut best: HashMap<usize, f32> = HashMap::new();

        if !matches!(request.search_tier, SearchTier::Archive) {
            let hot_results = self.query_pool.install(|| {
                self.hot_indices
                    .par_iter()
                    .map(|shard_index| {
                        let index = shard_index.read();
                        search_index_with_dynamic_fallback(
                            &index,
                            &request.vector,
                            request.k,
                            request.ef,
                            allow_bitmap.as_ref(),
                        )
                    })
                    .collect::<Vec<_>>()
            });
            for results in hot_results {
                for (id, score) in results {
                    merge_best(&mut best, id, score);
                }
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
                    search_index_with_dynamic_fallback(
                        &index,
                        &request.vector,
                        request.k,
                        request.ef,
                        allow_bitmap.as_ref(),
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
                        search_archive_pq_segment(
                            seg,
                            &request.vector,
                            request.k,
                            allow_bitmap.as_ref(),
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
            self.metadata_for_ids(&merged.iter().map(|(id, _)| *id).collect::<Vec<_>>())?
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

        let seq_len = request.vectors.len();
        let mut best: HashMap<usize, f32> = HashMap::new();

        let raw_tier_allowed = |level: i64| match request.search_tier {
            SearchTier::Active => level <= 1,
            SearchTier::All => true,
            SearchTier::Archive => false,
        };

        if !matches!(request.search_tier, SearchTier::Archive) {
            let hot_snapshot = {
                let total_hot = self
                    .hot_vector_shards
                    .iter()
                    .map(|shard| shard.load().read().len())
                    .sum::<usize>();
                if total_hot == 0 {
                    None
                } else {
                    let mut min_id = usize::MAX;
                    let mut max_id = 0usize;
                    let mut ids = HashSet::with_capacity(total_hot);
                    for shard in &self.hot_vector_shards {
                        let arc_guard = shard.load();
                        let guard = arc_guard.read();
                        for id in guard.keys().copied() {
                            min_id = min_id.min(id);
                            max_id = max_id.max(id);
                            ids.insert(id);
                        }
                    }
                    Some((min_id, max_id, Arc::new(ids)))
                }
            };

            if let Some((min_id, max_id, hot_ids)) = hot_snapshot {
                let total_vectors = max_id.saturating_add(1);
                let results = self.query_pool.install(|| {
                    self.hot_indices
                        .par_iter()
                        .map(|shard_index| {
                            let index = shard_index.read();
                            let hot_ids = hot_ids.clone();
                            index.search_subsequence_with_predicate(
                                &request.vectors,
                                request.k,
                                request.ef.max(request.k),
                                total_vectors,
                                move |id| hot_ids.contains(&id),
                            )
                        })
                        .collect::<Vec<_>>()
                });
                for shard_results in results {
                    for (start_id, score) in shard_results {
                        if start_id < min_id || start_id.saturating_add(seq_len) > total_vectors {
                            continue;
                        }
                        if (0..seq_len).any(|offset| !hot_ids.contains(&(start_id + offset))) {
                            continue;
                        }
                        merge_best(&mut best, start_id, score);
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
                            index.search_subsequence(
                                &request.vectors,
                                request.k,
                                request.ef.max(request.k),
                                max_exclusive,
                            )
                        };

                        local
                            .into_iter()
                            .filter(|(start_id, _)| {
                                *start_id >= min_start
                                    && start_id.saturating_add(seq_len) <= max_exclusive
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            });

            for results in segment_results {
                for (start_id, score) in results {
                    merge_best(&mut best, start_id, score);
                }
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
            self.metadata_for_ids(&merged.iter().map(|(id, _)| *id).collect::<Vec<_>>())?
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
                    (
                        meta.start_time_ms as u64,
                        (meta.duration_ms as u64).saturating_mul(seq_len as u64),
                    )
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
            .hot_vector_shards
            .iter()
            .map(|shard| shard.load().read().len())
            .sum();
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
            live: self.live.load(AtomicOrdering::SeqCst),
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
            .query_latency_us_le_10
            .load(AtomicOrdering::Relaxed);
        let b25 = b10
            + self
                .metrics
                .query_latency_us_le_25
                .load(AtomicOrdering::Relaxed);
        let b50 = b25
            + self
                .metrics
                .query_latency_us_le_50
                .load(AtomicOrdering::Relaxed);
        let b100 = b50
            + self
                .metrics
                .query_latency_us_le_100
                .load(AtomicOrdering::Relaxed);
        let b250 = b100
            + self
                .metrics
                .query_latency_us_le_250
                .load(AtomicOrdering::Relaxed);
        let b500 = b250
            + self
                .metrics
                .query_latency_us_le_500
                .load(AtomicOrdering::Relaxed);
        let b1000 = b500
            + self
                .metrics
                .query_latency_us_le_1000
                .load(AtomicOrdering::Relaxed);
        let b2500 = b1000
            + self
                .metrics
                .query_latency_us_le_2500
                .load(AtomicOrdering::Relaxed);
        let b5000 = b2500
            + self
                .metrics
                .query_latency_us_le_5000
                .load(AtomicOrdering::Relaxed);
        let h_count = self
            .metrics
            .query_latency_count
            .load(AtomicOrdering::Relaxed);
        let h_sum = self
            .metrics
            .query_latency_us_sum
            .load(AtomicOrdering::Relaxed);

        out.push_str("# TYPE vibrato_query_latency_us histogram\n");
        out.push_str(&format!(
            "vibrato_query_latency_us_bucket{{le=\"10\"}} {}\n",
            b10
        ));
        out.push_str(&format!(
            "vibrato_query_latency_us_bucket{{le=\"25\"}} {}\n",
            b25
        ));
        out.push_str(&format!(
            "vibrato_query_latency_us_bucket{{le=\"50\"}} {}\n",
            b50
        ));
        out.push_str(&format!(
            "vibrato_query_latency_us_bucket{{le=\"100\"}} {}\n",
            b100
        ));
        out.push_str(&format!(
            "vibrato_query_latency_us_bucket{{le=\"250\"}} {}\n",
            b250
        ));
        out.push_str(&format!(
            "vibrato_query_latency_us_bucket{{le=\"500\"}} {}\n",
            b500
        ));
        out.push_str(&format!(
            "vibrato_query_latency_us_bucket{{le=\"1000\"}} {}\n",
            b1000
        ));
        out.push_str(&format!(
            "vibrato_query_latency_us_bucket{{le=\"2500\"}} {}\n",
            b2500
        ));
        out.push_str(&format!(
            "vibrato_query_latency_us_bucket{{le=\"5000\"}} {}\n",
            b5000
        ));
        out.push_str(&format!(
            "vibrato_query_latency_us_bucket{{le=\"+Inf\"}} {}\n",
            h_count
        ));
        out.push_str(&format!("vibrato_query_latency_us_sum {}\n", h_sum));
        out.push_str(&format!("vibrato_query_latency_us_count {}\n", h_count));

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

        Ok(out)
    }

    pub fn checkpoint_once(&self) -> Result<JobResponseV2> {
        let _admin_guard = self
            .admin_ops_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
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
            for pair in ids_raw.windows(2) {
                if pair[1] > pair[0].saturating_add(1) {
                    tracing::warn!(
                        "checkpoint gap detected between vector_id {} and {}; filling tombstones",
                        pair[0],
                        pair[1]
                    );
                }
            }
            let (ids, vectors, metadata) =
                densify_id_space(self.collection.dim, ids_raw, vectors_raw, metadata_raw)?;
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
        let _admin_guard = self
            .admin_ops_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let _guard = self
            .compaction_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let job_id = make_id("cmp");

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
                        "compaction gap detected between {} [{}..={}] and {} [{}..={}]; filling tombstones",
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
            let mut merged: HashMap<usize, Vec<f32>> = HashMap::with_capacity(total_rows);
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
            let (ids, vectors, metadata) =
                densify_id_space(self.collection.dim, ids_raw, vectors_raw, metadata_raw)?;
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

        // 1. Create shadow shards (Arc<RwLock<HashMap>>)
        let mut new_shards = Vec::with_capacity(self.hot_vector_shards.len());
        for _ in 0..self.hot_vector_shards.len() {
            new_shards.push(Arc::new(RwLock::new(HashMap::new())));
        }

        // 2. Populate shadow shards
        // This is done without holding any locks on the main index
        for (shard_idx, shard_vectors) in new_shards.iter().enumerate() {
            let mut vectors = shard_vectors.write();
            for e in &pending {
                if (e.vector_id & self.shard_mask) == shard_idx {
                    vectors.insert(e.vector_id, Arc::new(e.vector.clone()));
                }
            }
        }

        // 3. Build shadow HNSW indices
        // HNSW is built against the SHADOW shards
        let mut new_indices = Vec::with_capacity(self.hot_indices.len());
        for (shard_idx, shard_vectors) in new_shards.iter().enumerate() {
            let mut hnsw = HNSW::new_with_accessor(
                self.config.hnsw_m,
                self.config.hnsw_ef_construction,
                make_hot_accessor(shard_vectors.clone(), self.collection.dim),
            );
            // Populate HNSW from pending
            for e in &pending {
                if (e.vector_id & self.shard_mask) == shard_idx {
                    hnsw.insert(e.vector_id);
                }
            }
            new_indices.push(hnsw);
        }

        // 4. Atomic Swap
        // We swap the vector shards FIRST, then the indices.
        // HNSW indices hold a reference to the vector shards they were built with.
        // By swapping the shards into the global ArcSwap, we ensure new ingests
        // go to the same place the new indices are reading from.
        for (i, shard) in new_shards.into_iter().enumerate() {
            self.hot_vector_shards[i].store(shard);
        }
        for (i, hnsw) in new_indices.into_iter().enumerate() {
            *self.hot_indices[i].write() = hnsw;
        }

        Ok(())
    }

    pub fn rebuild_filter_index(&self) -> Result<()> {
        let entries = self.catalog.fetch_all_metadata(&self.collection.id)?;
        let tag_dictionary = self.catalog.fetch_tag_dictionary(&self.collection.id)?;
        let filter_rows = self.catalog.fetch_filter_rows(&self.collection.id)?;

        let mut index = FilterIndex::with_dictionary(tag_dictionary);
        for row in filter_rows {
            index.add_with_tag_ids(row.vector_id, row.bpm, &row.tag_ids);
        }
        let mut metadata_cache = HashMap::with_capacity(entries.len());
        for (id, meta) in entries {
            metadata_cache.insert(id, meta);
        }
        *self.filter_index.write() = index;
        *self.metadata_cache.write() = metadata_cache;
        Ok(())
    }

    fn metadata_for_ids(&self, ids: &[usize]) -> Result<HashMap<usize, VectorMetadata>> {
        if ids.is_empty() {
            return Ok(HashMap::new());
        }

        let mut out = HashMap::with_capacity(ids.len());
        let mut missing = Vec::new();
        {
            let cache = self.metadata_cache.read();
            for id in ids {
                if let Some(meta) = cache.get(id) {
                    out.insert(*id, meta.clone());
                } else {
                    missing.push(*id);
                }
            }
        }

        if !missing.is_empty() {
            let fetched = self.catalog.fetch_metadata(&self.collection.id, &missing)?;
            if !fetched.is_empty() {
                let mut cache = self.metadata_cache.write();
                for (id, meta) in &fetched {
                    cache.insert(*id, meta.clone());
                }
            }
            for (id, meta) in fetched {
                out.insert(id, meta);
            }
        }

        Ok(out)
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
            .background_pool
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if guard.is_some() {
            return;
        }

        let colocated = self.config.audio_colocated;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .thread_name(|idx| {
                if idx == 0 {
                    "vibrato-bg-checkpoint".to_string()
                } else {
                    "vibrato-bg-compaction".to_string()
                }
            })
            .start_handler(move |_| {
                if colocated {
                    set_background_worker_priority();
                }
            })
            .build();

        let pool = match pool {
            Ok(pool) => Arc::new(pool),
            Err(err) => {
                tracing::error!("failed to create background worker pool: {err}");
                return;
            }
        };

        let checkpoint_state = Arc::clone(self);
        pool.spawn(move || checkpoint_loop(checkpoint_state));

        let compaction_state = Arc::clone(self);
        pool.spawn(move || compaction_loop(compaction_state));

        *guard = Some(pool);
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
        let dim = store.dim;
        let store_for_accessor = store.clone();
        let zero_fallback = vec![0.0f32; dim];
        let accessor = move |id: usize, sink: &mut dyn FnMut(&[f32])| {
            if id >= start && id < start + count {
                sink(store_for_accessor.get(id - start));
            } else {
                sink(&zero_fallback);
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
                idx.insert(start + offset);
            }
            idx
        };

        Ok(Arc::new(SegmentHandle {
            record: seg.clone(),
            store,
            index: Arc::new(RwLock::new(index)),
        }))
    }

    fn load_archive_segment_handle(&self, seg: &SegmentRecord) -> Result<Arc<ArchivePqSegment>> {
        let bytes = std::fs::read(&seg.path)
            .with_context(|| format!("reading archive segment {:?}", seg.path))?;
        let (pq, codes, nsub) = parse_archive_segment_bytes(&seg.id, &bytes)?;

        Ok(Arc::new(ArchivePqSegment {
            record: seg.clone(),
            pq,
            codes: Arc::new(codes),
            num_subspaces: nsub,
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
    metadata: &[VectorMetadata],
    graph: &HNSW,
    output_path: &std::path::Path,
) -> Result<()> {
    let empty_base_path = config.tmp_dir.join(format!("empty_base_{}.vdb", dim));
    if !empty_base_path.exists() {
        let writer = VdbWriterV2::new_raw(&empty_base_path, dim)?;
        writer.finish()?;
    }

    let base_store = VectorStore::open(&empty_base_path)?;
    VdbWriterV2::merge_with_metadata(&base_store, vectors, metadata, graph, output_path)?;

    File::open(output_path)?.sync_all()?;
    sync_parent(output_path)?;
    Ok(())
}

fn write_archive_pq_segment(
    dim: usize,
    vectors: &[Vec<f32>],
    metadata: &[VectorMetadata],
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
        let tags: Vec<&str> = item.tags.iter().map(|s| s.as_str()).collect();
        metadata_builder.add_entry(
            &item.source_file,
            item.start_time_ms,
            item.duration_ms,
            item.bpm,
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
    let mut map = HashMap::with_capacity(ids.len());
    for (id, vec) in ids.iter().zip(vectors.iter()) {
        map.insert(*id, vec.clone());
    }
    let map = Arc::new(map);

    let mut hnsw = HNSW::new_with_accessor(m, ef_construction, {
        let map = map.clone();
        move |id, sink| {
            if let Some(v) = map.get(&id) {
                sink(v);
            }
        }
    });

    for id in ids {
        hnsw.insert(*id);
    }
    hnsw
}

fn wal_to_arrays(entries: &[WalEntry]) -> (Vec<usize>, Vec<Vec<f32>>, Vec<VectorMetadata>) {
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

fn tombstone_metadata() -> VectorMetadata {
    VectorMetadata {
        source_file: "__vibrato_tombstone__".to_string(),
        start_time_ms: 0,
        duration_ms: 0,
        bpm: 0.0,
        tags: vec!["__vibrato_tombstone__".to_string()],
    }
}

fn densify_id_space(
    dim: usize,
    ids: Vec<usize>,
    vectors: Vec<Vec<f32>>,
    metadata: Vec<VectorMetadata>,
) -> Result<(Vec<usize>, Vec<Vec<f32>>, Vec<VectorMetadata>)> {
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
    let end_id = rows.last().map(|r| r.0).unwrap_or(start_id);
    let dense_len = end_id.saturating_sub(start_id) + 1;
    let mut out_ids = Vec::with_capacity(dense_len);
    let mut out_vectors = Vec::with_capacity(dense_len);
    let mut out_metadata = Vec::with_capacity(dense_len);

    let mut cursor = start_id;
    for (id, vector, meta) in rows {
        while cursor < id {
            out_ids.push(cursor);
            out_vectors.push(vec![0.0f32; dim]);
            out_metadata.push(tombstone_metadata());
            cursor = cursor.saturating_add(1);
        }
        out_ids.push(id);
        out_vectors.push(vector);
        out_metadata.push(meta);
        cursor = id.saturating_add(1);
    }

    Ok((out_ids, out_vectors, out_metadata))
}

fn merge_best(best: &mut HashMap<usize, f32>, id: usize, score: f32) {
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
            let norm_sq: f32 = v.iter().map(|x| x * x).sum();
            (i, norm_sq)
        })
        .collect::<Vec<_>>();
    salience.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    let mut anchors = Vec::with_capacity(probe_count);
    for (idx, _) in salience {
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
            if !allow.contains(vector_id) {
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
) -> Vec<(usize, f32)> {
    if allow_set.is_none() {
        return index.search(query, k, ef);
    }

    let allow = allow_set.unwrap();
    let cardinality = allow.count_ones(..);
    if cardinality == 0 {
        return Vec::new();
    }
    let target_hits = k.min(cardinality);

    // Stage A (strict): evaluate only allowed IDs.
    let strict = index.search_filtered(query, k, ef.max(k), |id| allow.contains(id));
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
        let expanded = index.search_filtered(query, k, expanded_ef, |id| allow.contains(id));
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

    let routed = index
        .search(query, overfetch_k, route_ef)
        .into_iter()
        .filter(|(id, _)| allow.contains(*id))
        .take(k)
        .collect::<Vec<_>>();

    if routed.len() > best.len() {
        routed
    } else {
        best
    }
}

fn checkpoint_loop(state: Arc<ProductionState>) {
    loop {
        std::thread::sleep(state.config.checkpoint_interval);
        if let Err(err) = state.checkpoint_once() {
            tracing::warn!("background checkpoint failed: {err}");
        }
    }
}

fn compaction_loop(state: Arc<ProductionState>) {
    loop {
        std::thread::sleep(state.config.compaction_interval);
        if let Err(err) = state.compact_once() {
            tracing::warn!("background compaction failed: {err}");
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

fn make_hot_accessor(
    hot_vector_shard: Arc<RwLock<HashMap<usize, Arc<Vec<f32>>>>>,
    dim: usize,
) -> impl Fn(usize, &mut dyn FnMut(&[f32])) + Send + Sync + 'static {
    let zero_fallback = Arc::new(vec![0.0f32; dim]);
    move |id, sink| {
        // Important: do not hold the shard lock while executing `sink`.
        // HNSW insert/search can invoke nested accessor callbacks.
        let maybe_vec = {
            let guard = hot_vector_shard.read();
            guard.get(&id).cloned()
        };
        if let Some(v) = maybe_vec {
            sink(v.as_slice());
        } else {
            sink(zero_fallback.as_slice());
        }
    }
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
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use vibrato_core::simd::l2_normalized;

    fn random_vector(dim: usize, rng: &mut StdRng) -> Vec<f32> {
        let raw: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect();
        l2_normalized(&raw)
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
            index.insert(id);
        }

        let target_id = total - 1;
        let query = vectors[target_id].clone();
        let mut allow_set = BitmapSet::with_capacity(total);
        allow_set.grow(total);
        allow_set.insert(target_id);

        let results = search_index_with_dynamic_fallback(&index, &query, 10, 1, Some(&allow_set));
        assert!(!results.is_empty(), "expected at least one filtered hit");
        assert!(
            results.iter().all(|(id, _)| allow_set.contains(*id)),
            "fallback must not leak non-filtered IDs"
        );
        assert_eq!(results[0].0, target_id);
    }
}
