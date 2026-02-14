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
    JobResponseV2, QueryRequestV2, QueryResponseV2, QueryResultV2, SearchTier, StatsResponseV2,
};

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
    pub api_pepper: String,
}

#[derive(Default)]
pub struct Metrics {
    pub query_total: AtomicU64,
    pub ingest_total: AtomicU64,
    pub auth_failures_total: AtomicU64,
    pub audit_failures_total: AtomicU64,
    pub checkpoint_total: AtomicU64,
    pub compaction_total: AtomicU64,
    pub obsolete_files_deleted_total: AtomicU64,
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

#[derive(Debug, Clone)]
pub struct AuditEvent {
    pub request_id: String,
    pub api_key_id: Option<String>,
    pub endpoint: String,
    pub action: String,
    pub status_code: u16,
    pub latency_ms: f64,
    pub details: Value,
}

pub struct ProductionState {
    pub config: ProductionConfig,
    pub catalog: Arc<SqliteCatalog>,
    pub collection: CollectionRecord,
    pub live: AtomicBool,
    pub ready: AtomicBool,
    pub recovery_report: RwLock<String>,

    pub hot_vectors: Arc<RwLock<HashMap<usize, Vec<f32>>>>,
    pub hot_index: Arc<RwLock<HNSW>>,
    pub metadata_cache: Arc<RwLock<HashMap<usize, VectorMetadata>>>,
    pub segments: ArcSwap<Vec<Arc<SegmentHandle>>>,
    pub archive_segments: ArcSwap<Vec<Arc<ArchivePqSegment>>>,
    pub filter_index: Arc<RwLock<FilterIndex>>,
    pub retired_segments: std::sync::Mutex<HashMap<String, Weak<SegmentHandle>>>,

    pub metrics: Metrics,
    pub checkpoint_lock: std::sync::Mutex<()>,
    pub compaction_lock: std::sync::Mutex<()>,
    pub background_pool: std::sync::Mutex<Option<Arc<ThreadPool>>>,
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

        let hot_vectors = Arc::new(RwLock::new(HashMap::new()));
        let hot_accessor = make_hot_accessor(hot_vectors.clone(), config.dim);
        let hot_index = Arc::new(RwLock::new(HNSW::new_with_accessor(
            config.hnsw_m,
            config.hnsw_ef_construction,
            hot_accessor,
        )));
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

        let state = Arc::new(Self {
            config,
            catalog,
            collection,
            live: AtomicBool::new(true),
            ready: AtomicBool::new(false),
            recovery_report: RwLock::new("initializing".to_string()),
            hot_vectors,
            hot_index,
            metadata_cache,
            segments: ArcSwap::from_pointee(Vec::new()),
            archive_segments: ArcSwap::from_pointee(Vec::new()),
            filter_index: Arc::new(RwLock::new(FilterIndex::default())),
            retired_segments: std::sync::Mutex::new(HashMap::new()),
            metrics: Metrics::default(),
            checkpoint_lock: std::sync::Mutex::new(()),
            compaction_lock: std::sync::Mutex::new(()),
            background_pool: std::sync::Mutex::new(None),
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
        self.hot_vectors.write().insert(vector_id, vector);
        self.hot_index.write().insert(vector_id);
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
            let hot_results = {
                let index = self.hot_index.read();
                search_index_with_dynamic_fallback(
                    &index,
                    &request.vector,
                    request.k,
                    request.ef,
                    allow_bitmap.as_ref(),
                )
            };
            for (id, score) in hot_results {
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
        let hot_vectors = self.hot_vectors.read().len();
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
        })
    }

    pub fn render_metrics(&self) -> Result<String> {
        let stats = self.stats()?;
        Ok(format!(
            concat!(
                "# TYPE vibrato_query_requests_total counter\n",
                "vibrato_query_requests_total {}\n",
                "# TYPE vibrato_ingest_requests_total counter\n",
                "vibrato_ingest_requests_total {}\n",
                "# TYPE vibrato_auth_failures_total counter\n",
                "vibrato_auth_failures_total {}\n",
                "# TYPE vibrato_audit_failures_total counter\n",
                "vibrato_audit_failures_total {}\n",
                "# TYPE vibrato_checkpoint_total counter\n",
                "vibrato_checkpoint_total {}\n",
                "# TYPE vibrato_compaction_total counter\n",
                "vibrato_compaction_total {}\n",
                "# TYPE vibrato_obsolete_files_deleted_total counter\n",
                "vibrato_obsolete_files_deleted_total {}\n",
                "# TYPE vibrato_active_segments gauge\n",
                "vibrato_active_segments {}\n",
                "# TYPE vibrato_obsolete_segments gauge\n",
                "vibrato_obsolete_segments {}\n",
                "# TYPE vibrato_failed_segments gauge\n",
                "vibrato_failed_segments {}\n",
                "# TYPE vibrato_hot_vectors gauge\n",
                "vibrato_hot_vectors {}\n",
                "# TYPE vibrato_wal_pending gauge\n",
                "vibrato_wal_pending {}\n",
                "# TYPE vibrato_total_vectors gauge\n",
                "vibrato_total_vectors {}\n",
                "# TYPE vibrato_checkpoint_jobs_inflight gauge\n",
                "vibrato_checkpoint_jobs_inflight {}\n",
                "# TYPE vibrato_compaction_jobs_inflight gauge\n",
                "vibrato_compaction_jobs_inflight {}\n"
            ),
            self.metrics.query_total.load(AtomicOrdering::Relaxed),
            self.metrics.ingest_total.load(AtomicOrdering::Relaxed),
            self.metrics
                .auth_failures_total
                .load(AtomicOrdering::Relaxed),
            self.metrics
                .audit_failures_total
                .load(AtomicOrdering::Relaxed),
            self.metrics.checkpoint_total.load(AtomicOrdering::Relaxed),
            self.metrics.compaction_total.load(AtomicOrdering::Relaxed),
            self.metrics
                .obsolete_files_deleted_total
                .load(AtomicOrdering::Relaxed),
            stats.active_segments,
            stats.obsolete_segments,
            stats.failed_segments,
            stats.hot_vectors,
            stats.wal_pending,
            stats.total_vectors,
            stats.checkpoint_jobs_inflight,
            stats.compaction_jobs_inflight,
        ))
    }

    pub fn checkpoint_once(&self) -> Result<JobResponseV2> {
        let _guard = self.checkpoint_lock.lock().unwrap();
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
            let (ids, vectors, metadata) = wal_to_arrays(&pending);
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
            prewarm_segment(&handle);

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
        let _guard = self.compaction_lock.lock().unwrap();
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

        let (candidates, output_level) = if l0.len() >= 2 {
            (l0, 1_i64)
        } else if l1.len() >= 2 {
            (l1, 2_i64)
        } else {
            (Vec::new(), 0_i64)
        };

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
            let mut pairs: Vec<(usize, Vec<f32>)> = Vec::new();
            for seg in &candidates {
                for offset in 0..seg.record.row_count {
                    let vector_id = seg.record.vector_id_start + offset;
                    pairs.push((vector_id, seg.store.get(offset).to_vec()));
                }
            }
            pairs.sort_by_key(|(id, _)| *id);

            let ids = pairs.iter().map(|(id, _)| *id).collect::<Vec<_>>();
            let vectors = pairs.into_iter().map(|(_, v)| v).collect::<Vec<_>>();
            let metadata_map = self.catalog.fetch_metadata(&ids)?;
            let metadata = ids
                .iter()
                .map(|id| metadata_map.get(id).cloned().unwrap_or_default())
                .collect::<Vec<_>>();
            let use_archive_pq =
                output_level >= 2 && should_use_archive_pq(self.collection.dim, vectors.len());
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
                prewarm_segment(&handle);
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
                            .unwrap()
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
                    prewarm_segment(&handle);
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
        {
            let mut hv = self.hot_vectors.write();
            hv.clear();
            for e in &pending {
                hv.insert(e.vector_id, e.vector.clone());
            }
        }

        let new_index = HNSW::new_with_accessor(
            self.config.hnsw_m,
            self.config.hnsw_ef_construction,
            make_hot_accessor(self.hot_vectors.clone(), self.collection.dim),
        );
        {
            let mut idx = self.hot_index.write();
            *idx = new_index;
            for e in &pending {
                idx.insert(e.vector_id);
            }
        }
        Ok(())
    }

    pub fn rebuild_filter_index(&self) -> Result<()> {
        let entries = self.catalog.fetch_all_metadata(&self.collection.id)?;
        let mut index = FilterIndex::default();
        let mut metadata_cache = HashMap::with_capacity(entries.len());
        for (id, meta) in entries {
            index.add(id, &meta);
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
            let fetched = self.catalog.fetch_metadata(&missing)?;
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
        let accessor = move |id: usize, sink: &mut dyn FnMut(&[f32])| {
            if id >= start && id < start + count {
                sink(store_for_accessor.get(id - start));
            } else {
                let fallback = vec![0.0; dim];
                sink(&fallback);
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
        let header = VdbHeaderV2::from_bytes(&bytes)?;
        if !header.is_pq_enabled() {
            return Err(anyhow!("segment {} is not pq-enabled archive data", seg.id));
        }

        let nsub = header.pq_subspaces as usize;
        let code_len = header.count as usize * nsub;
        let codes_start = header.vectors_offset as usize;
        let codes_end = codes_start + code_len;
        if codes_end > bytes.len() {
            return Err(anyhow!(
                "invalid pq codes range in segment {}: {}..{} > {}",
                seg.id,
                codes_start,
                codes_end,
                bytes.len()
            ));
        }

        if header.codebook_offset == 0 {
            return Err(anyhow!("segment {} missing pq codebook section", seg.id));
        }
        let codebook_start = header.codebook_offset as usize;
        let codebook_end = if header.metadata_offset > 0 {
            header.metadata_offset as usize
        } else if header.graph_offset > 0 {
            header.graph_offset as usize
        } else {
            bytes.len()
        };
        if codebook_end <= codebook_start || codebook_end > bytes.len() {
            return Err(anyhow!(
                "invalid codebook range in segment {}: {}..{}",
                seg.id,
                codebook_start,
                codebook_end
            ));
        }

        let pq = ProductQuantizer::from_codebook_bytes(
            header.dimensions as usize,
            nsub,
            &bytes[codebook_start..codebook_end],
        )
        .with_context(|| format!("decoding pq codebook for segment {}", seg.id))?;

        Ok(Arc::new(ArchivePqSegment {
            record: seg.clone(),
            pq,
            codes: Arc::new(bytes[codes_start..codes_end].to_vec()),
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
    while let Ok(event) = rx.recv() {
        let mut written = false;
        for attempt in 0..3 {
            let result = state.catalog.audit_event(
                &event.request_id,
                event.api_key_id.as_deref(),
                &event.endpoint,
                &event.action,
                event.status_code,
                event.latency_ms,
                event.details.clone(),
            );
            if result.is_ok() {
                written = true;
                break;
            }
            if attempt < 2 {
                std::thread::sleep(Duration::from_millis(20 * (attempt + 1) as u64));
            }
        }
        if !written {
            state
                .metrics
                .audit_failures_total
                .fetch_add(1, AtomicOrdering::Relaxed);
        }
    }
}

fn make_hot_accessor(
    hot_vectors: Arc<RwLock<HashMap<usize, Vec<f32>>>>,
    dim: usize,
) -> impl Fn(usize, &mut dyn FnMut(&[f32])) + Send + Sync + 'static {
    move |id, sink| {
        let guard = hot_vectors.read();
        if let Some(v) = guard.get(&id) {
            sink(v);
        } else {
            let fallback = vec![0.0f32; dim];
            sink(&fallback);
        }
    }
}

fn prewarm_segment(segment: &SegmentHandle) {
    if segment.store.count == 0 {
        return;
    }

    let stride = (segment.store.count / 128).max(1);
    for i in (0..segment.store.count).step_by(stride).take(256) {
        let v = segment.store.get(i);
        std::hint::black_box(v[0]);
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
