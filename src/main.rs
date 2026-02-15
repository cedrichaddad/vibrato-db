//! Vibrato-DB CLI
//!
//! A persistent, disk-backed vector search engine.
//!
//! # Usage
//!
//! ```bash
//! # Start the server
//! vibrato-db serve --data data.vdb --port 8080
//!
//! # Build an index from a .vdb file
//! vibrato-db build --data data.vdb --output index.idx --m 16 --ef 100
//!
//! # Download inference models for /ingest audio_path
//! vibrato-db setup-models --model-dir ./models
//! ```

use std::io::{BufReader, Read, Seek};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use clap::{Parser, Subcommand};
use parking_lot::RwLock;
use tracing_subscriber::EnvFilter;

use vibrato_db::format_v2::{VdbHeaderV2, VdbWriterV2};
use vibrato_db::hnsw::HNSW;
use vibrato_db::prod::{
    bootstrap_data_dirs, create_snapshot, create_v2_router, migrate_existing_vdb_to_segment,
    recover_state, replay_to_lsn, restore_snapshot, CatalogOptions, CatalogStore, ProductionConfig,
    ProductionState, Role, SqliteCatalog, VectorMadviseMode,
};
use vibrato_db::server::{
    load_store_metadata, serve, AppState, SearchRequest, SearchResponse, SharedStore,
};
use vibrato_db::store::VectorStore;
use vibrato_neural::inference::InferenceEngine;

#[derive(Parser)]
#[command(name = "vibrato-db")]
#[command(about = "A persistent, disk-backed vector search engine")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the HTTP server
    Serve {
        /// Path to the .vdb vector data file
        #[arg(short, long)]
        data: PathBuf,

        /// Path to the .idx index file (optional, will build if missing)
        #[arg(short, long)]
        index: Option<PathBuf>,

        /// Server port
        #[arg(short, long, default_value = "8080")]
        port: u16,

        /// Host to bind to
        #[arg(long, default_value = "0.0.0.0")]
        host: String,

        /// HNSW M parameter (max neighbors per layer)
        #[arg(long, default_value = "16")]
        m: usize,

        /// HNSW ef_construction parameter (search depth during build)
        #[arg(long, default_value = "100")]
        ef_construction: usize,

        /// Maximum number of unflushed vectors kept in RAM before /ingest returns 503.
        #[arg(long, default_value = "100000")]
        max_ram_buffer_vectors: usize,
    },

    /// Build an HNSW index from a .vdb file
    Build {
        /// Path to the .vdb vector data file
        #[arg(short, long)]
        data: PathBuf,

        /// Path to output .idx index file
        #[arg(short, long)]
        output: PathBuf,

        /// HNSW M parameter (max neighbors per layer)
        #[arg(short, long, default_value = "16")]
        m: usize,

        /// HNSW ef_construction parameter (search depth during build)
        #[arg(short, long, default_value = "100")]
        ef_construction: usize,
    },

    /// Display statistics about a .vdb file or .idx index
    Stats {
        /// Path to .vdb or .idx file
        #[arg(short, long)]
        file: PathBuf,
    },

    /// Create a .vdb file from a JSON list of vectors
    ///
    /// Input format: JSON array of arrays [[0.1, ...], [0.2, ...]]
    Ingest {
        /// Input JSON file
        #[arg(short, long)]
        input: PathBuf,

        /// Output .vdb file
        #[arg(short, long)]
        output: PathBuf,
    },

    /// Search for nearest neighbors using the HTTP server
    Search {
        /// Server URL
        #[arg(long, default_value = "http://localhost:8080")]
        server: String,

        /// Query vector (comma separated floats)
        #[arg(short, long, value_parser = parse_vector)]
        query: Vec<f32>,

        /// Number of results
        #[arg(short = 'k', long, default_value = "10")]
        k: usize,

        /// Search depth
        #[arg(long, default_value = "50")]
        ef: usize,
    },

    /// Download and verify inference models ahead of serving
    SetupModels {
        /// Model directory (defaults to ./models)
        #[arg(long, default_value = "models")]
        model_dir: PathBuf,
    },

    /// Start the production v2 server (SQLite catalog + WAL + segments)
    ServeV2 {
        /// Root data directory (catalog, segments, tmp, quarantine)
        #[arg(long, default_value = "vibrato_data")]
        data_dir: PathBuf,

        /// Collection name
        #[arg(long, default_value = "default")]
        collection: String,

        /// Vector dimensionality for bootstrap/new collections
        #[arg(long, default_value = "128")]
        dim: usize,

        /// Server port
        #[arg(short, long, default_value = "8080")]
        port: u16,

        /// Host to bind to
        #[arg(long, default_value = "0.0.0.0")]
        host: String,

        /// Background checkpoint interval in seconds
        #[arg(long, default_value = "30")]
        checkpoint_interval_secs: u64,

        /// Background compaction interval in seconds
        #[arg(long, default_value = "180")]
        compaction_interval_secs: u64,

        /// Quarantine GC TTL in hours
        #[arg(long, default_value = "168")]
        orphan_ttl_hours: u64,

        /// Assume co-location with audio workloads and lower maintenance worker priority
        #[arg(long, default_value_t = true)]
        audio_colocated: bool,

        /// If false, /v2/health/* and /v2/metrics require API auth
        #[arg(long, default_value_t = true)]
        public_health_metrics: bool,

        /// Catalog read timeout in ms (guards against stuck readers / WAL bloat)
        #[arg(long, default_value = "500")]
        catalog_read_timeout_ms: u64,

        /// SQLite internal WAL autocheckpoint pages
        #[arg(long, default_value = "1000")]
        sqlite_wal_autocheckpoint_pages: u32,

        /// Max quarantined orphan files retained
        #[arg(long, default_value = "50")]
        quarantine_max_files: usize,

        /// Max quarantined orphan bytes retained
        #[arg(long, default_value = "5368709120")]
        quarantine_max_bytes: u64,

        /// Background IO throttle for checkpoint/compaction workers in MB/s (0 disables)
        #[arg(long, default_value = "40")]
        background_io_mb_per_sec: u64,

        /// Number of hot-index shards (rounded up to next power-of-two)
        #[arg(long, default_value = "8")]
        hot_index_shards: usize,

        /// Vector mmap advise mode for active segment vectors: normal|random
        #[arg(long, default_value = "normal")]
        vector_madvise_mode: String,

        /// Bootstrap first admin API key if no keys exist in catalog
        #[arg(long, default_value_t = false)]
        bootstrap_admin_key: bool,
    },

    /// Create an API key for v2 server auth
    KeyCreate {
        #[arg(long, default_value = "vibrato_data")]
        data_dir: PathBuf,

        #[arg(long)]
        name: String,

        #[arg(long, default_value = "admin,query,ingest")]
        roles: String,
    },

    /// Revoke an API key by ID
    KeyRevoke {
        #[arg(long, default_value = "vibrato_data")]
        data_dir: PathBuf,

        #[arg(long)]
        key_id: String,
    },

    /// Migrate an existing single .vdb into v2 catalog + segment layout
    MigrateV2 {
        #[arg(long, default_value = "vibrato_data")]
        data_dir: PathBuf,

        #[arg(long)]
        input: PathBuf,

        #[arg(long, default_value = "default")]
        collection: String,

        #[arg(long, default_value = "128")]
        dim: usize,

        #[arg(long, default_value = "1")]
        level: i64,
    },

    /// Create a local snapshot (catalog + active segments)
    SnapshotCreate {
        #[arg(long, default_value = "vibrato_data")]
        data_dir: PathBuf,

        #[arg(long, default_value = "default")]
        collection: String,
    },

    /// Restore a previously created snapshot
    SnapshotRestore {
        #[arg(long, default_value = "vibrato_data")]
        data_dir: PathBuf,

        #[arg(long)]
        snapshot_dir: PathBuf,
    },

    /// Replay catalog/WAL to a target LSN (marks current active segments obsolete)
    ReplayToLsn {
        #[arg(long, default_value = "vibrato_data")]
        data_dir: PathBuf,

        #[arg(long, default_value = "default")]
        collection: String,

        #[arg(long)]
        target_lsn: u64,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Serve {
            data,
            index,
            port,
            host,
            m,
            ef_construction,
            max_ram_buffer_vectors,
        } => {
            tracing::info!("Loading vector data from {:?}", data);
            let store = Arc::new(VectorStore::open(&data)?);
            // Create SharedStore (Arc<RwLock<Arc<VectorStore>>>)
            let shared_store = Arc::new(arc_swap::ArcSwap::from(store.clone()));
            tracing::info!("Loaded {} vectors of dimension {}", store.count, store.dim);

            // Create dynamic components
            let dynamic_store = Arc::new(RwLock::new(Vec::new()));
            let dynamic_metadata = Arc::new(RwLock::new(Vec::new()));
            let persisted_metadata = Arc::new(arc_swap::ArcSwap::from(Arc::new(
                load_store_metadata(&store),
            )));

            if index.is_some() {
                tracing::warn!(
                    "`--index` is deprecated. Use a single .vdb at `--data`; `--index` is used only as fallback input."
                );
            }

            // Initialize Inference Engine. Startup stays online for search even if models are missing.
            let model_dir = PathBuf::from("models"); // Default to local models dir
            let inference = match InferenceEngine::new(&model_dir) {
                Ok(engine) => {
                    tracing::info!("Inference engine loaded from {:?}", model_dir);
                    Some(Arc::new(engine))
                }
                Err(e) => {
                    tracing::warn!(
                        "Inference engine unavailable: {}. Server will run in search-only mode.",
                        e
                    );
                    None
                }
            };

            let mut hnsw = if let Some(hnsw) = try_load_graph_from_vdb(
                &data,
                create_accessor(shared_store.clone(), dynamic_store.clone()),
            )? {
                let stats = hnsw.stats();
                tracing::info!(
                    "Loaded graph from V2 container: {} nodes, {} layers",
                    stats.num_nodes,
                    stats.max_layer + 1
                );
                hnsw
            } else if let Some(index_path) = &index {
                if index_path.exists() {
                    tracing::info!("Loading legacy index from {:?}", index_path);
                    let accessor = create_accessor(shared_store.clone(), dynamic_store.clone());
                    let hnsw = HNSW::load_with_accessor(index_path, accessor)?;
                    let stats = hnsw.stats();
                    tracing::info!(
                        "Legacy index loaded: {} nodes, {} layers",
                        stats.num_nodes,
                        stats.max_layer + 1
                    );
                    hnsw
                } else {
                    tracing::info!("No persisted graph found, building index...");
                    let accessor = create_accessor(shared_store.clone(), dynamic_store.clone());
                    let mut hnsw = HNSW::new_with_accessor(m, ef_construction, accessor);
                    for i in 0..store.count as usize {
                        hnsw.insert(i);
                    }
                    hnsw
                }
            } else {
                tracing::info!("No persisted graph found, building index...");
                let accessor = create_accessor(shared_store.clone(), dynamic_store.clone());
                let mut hnsw = HNSW::new_with_accessor(m, ef_construction, accessor);
                for i in 0..store.count as usize {
                    hnsw.insert(i);
                }
                hnsw
            };

            if let Err(err) = validate_graph_bounds(&hnsw, store.count) {
                tracing::warn!(
                    "Persisted graph failed integrity check ({}). Rebuilding from vectors.",
                    err
                );
                hnsw = build_hnsw_from_store(
                    store.count,
                    m,
                    ef_construction,
                    create_accessor(shared_store.clone(), dynamic_store.clone()),
                );
            }

            let state = Arc::new(AppState {
                index: RwLock::new(hnsw),
                store: shared_store,
                dynamic_store,
                persisted_metadata,
                dynamic_metadata,
                inference,
                flush_mutex: RwLock::new(()),
                flush_status: Arc::new(RwLock::new(vibrato_db::server::FlushStatus {
                    state: "idle".to_string(),
                    job_id: 0,
                    snapshot_vectors: 0,
                    persisted_vectors: store.count,
                    dynamic_vectors: 0,
                    error: None,
                })),
                flush_job_seq: std::sync::atomic::AtomicU64::new(0),
                data_path: data.clone(),
                max_dynamic_vectors: max_ram_buffer_vectors,
            });

            let addr: SocketAddr = format!("{}:{}", host, port).parse()?;
            serve(state, addr).await?;
        }

        Commands::Build {
            data,
            output,
            m,
            ef_construction,
        } => {
            tracing::info!("Loading vector data from {:?}", data);
            let store = Arc::new(VectorStore::open(&data)?);
            tracing::info!("Loaded {} vectors of dimension {}", store.count, store.dim);

            let hnsw = build_index(store, m, ef_construction, Some(&output))?;
            let stats = hnsw.stats();
            tracing::info!(
                "Index built: {} nodes, {} layers, {} edges",
                stats.num_nodes,
                stats.max_layer + 1,
                stats.total_edges
            );
        }

        Commands::Stats { file } => {
            let extension = file.extension().and_then(|e| e.to_str());

            match extension {
                Some("vdb") => {
                    let store = VectorStore::open(&file)?;
                    println!("Vector Data File: {:?}", file);
                    println!("  Vectors: {}", store.count);
                    println!("  Dimensions: {}", store.dim);
                    println!(
                        "  File Size: {:.2} MB",
                        store.memory_bytes() as f64 / (1024.0 * 1024.0)
                    );
                }
                Some("idx") => {
                    // We need a dummy vector fn to load
                    let hnsw = HNSW::load(&file, |_| vec![0.0f32; 128])?;
                    let stats = hnsw.stats();
                    println!("Index File: {:?}", file);
                    println!("  Nodes: {}", stats.num_nodes);
                    println!("  Max Layer: {}", stats.max_layer);
                    println!("  Total Edges: {}", stats.total_edges);
                    println!("  M: {}", stats.m);
                    println!("  ef_construction: {}", stats.ef_construction);
                    println!("  Layer Distribution:");
                    for (i, count) in stats.layer_counts.iter().enumerate() {
                        if *count > 0 {
                            println!("    Layer {}: {} nodes", i, count);
                        }
                    }
                }
                _ => {
                    eprintln!("Unknown file type. Expected .vdb or .idx");
                    std::process::exit(1);
                }
            }
        }

        Commands::Ingest { input, output } => {
            tracing::info!("Reading vectors from {:?}", input);
            let file = std::fs::File::open(&input)?;
            let reader = std::io::BufReader::new(file);
            let vectors: Vec<Vec<f32>> = serde_json::from_reader(reader)?;

            if vectors.is_empty() {
                anyhow::bail!("No vectors found in input");
            }

            let dim = vectors[0].len();
            tracing::info!("Found {} vectors of dimension {}", vectors.len(), dim);

            let mut writer = VdbWriterV2::new_raw(&output, dim)?;
            for (i, vec) in vectors.iter().enumerate() {
                if vec.len() != dim {
                    anyhow::bail!("Vector {} has dimension {}, expected {}", i, vec.len(), dim);
                }
                writer.write_vector(vec)?;
            }
            writer.finish()?;
            tracing::info!("Wrote .vdb file to {:?}", output);
        }

        Commands::Search {
            server,
            query,
            k,
            ef,
        } => {
            let client = reqwest::Client::new();
            let url = format!("{}/search", server.trim_end_matches('/'));

            let request = SearchRequest {
                vector: query,
                k,
                ef,
            };

            let response = client.post(&url).json(&request).send().await?;

            if !response.status().is_success() {
                let error: serde_json::Value = response.json().await?;
                eprintln!("Error: {}", error);
                std::process::exit(1);
            }

            let result: SearchResponse = response.json().await?;
            println!("Query time: {:.2}ms", result.query_time_ms);
            println!("Results:");
            for res in result.results {
                println!("  ID: {}, Score: {:.4}", res.id, res.score);
            }
        }

        Commands::SetupModels { model_dir } => {
            tracing::info!("Setting up models in {:?}", model_dir);
            InferenceEngine::setup_models(&model_dir)?;
            println!("Models downloaded and verified in {:?}", model_dir);
        }

        Commands::ServeV2 {
            data_dir,
            collection,
            dim,
            port,
            host,
            checkpoint_interval_secs,
            compaction_interval_secs,
            orphan_ttl_hours,
            audio_colocated,
            public_health_metrics,
            catalog_read_timeout_ms,
            sqlite_wal_autocheckpoint_pages,
            quarantine_max_files,
            quarantine_max_bytes,
            background_io_mb_per_sec,
            hot_index_shards,
            vector_madvise_mode,
            bootstrap_admin_key,
        } => {
            let mut config = ProductionConfig::from_data_dir(data_dir, collection, dim);
            config.checkpoint_interval = std::time::Duration::from_secs(checkpoint_interval_secs);
            config.compaction_interval = std::time::Duration::from_secs(compaction_interval_secs);
            config.orphan_ttl = std::time::Duration::from_secs(orphan_ttl_hours * 3600);
            config.audio_colocated = audio_colocated;
            config.public_health_metrics = public_health_metrics;
            config.catalog_read_timeout_ms = catalog_read_timeout_ms;
            config.sqlite_wal_autocheckpoint_pages = sqlite_wal_autocheckpoint_pages;
            config.quarantine_max_files = quarantine_max_files;
            config.quarantine_max_bytes = quarantine_max_bytes;
            config.background_io_mb_per_sec = background_io_mb_per_sec;
            config.hot_index_shards = hot_index_shards;
            config.vector_madvise_mode = parse_vector_madvise_mode(&vector_madvise_mode)?;

            bootstrap_data_dirs(&config)?;
            let catalog = Arc::new(SqliteCatalog::open_with_options(
                &config.catalog_path(),
                CatalogOptions {
                    read_timeout_ms: config.catalog_read_timeout_ms,
                    wal_autocheckpoint_pages: config.sqlite_wal_autocheckpoint_pages,
                },
            )?);
            let state = ProductionState::initialize(config.clone(), catalog.clone())?;

            if bootstrap_admin_key && catalog.count_api_keys()? == 0 {
                let key = catalog.create_api_key(
                    "bootstrap-admin",
                    &[Role::Admin, Role::Query, Role::Ingest],
                    &config.api_pepper,
                )?;
                println!("BOOTSTRAP_ADMIN_KEY={}", key.token);
            }

            let report = recover_state(&state)?;
            tracing::info!("{}", report.report);

            state.start_background_workers();

            let router = create_v2_router(state);
            let addr: SocketAddr = format!("{}:{}", host, port).parse()?;
            tracing::info!("Starting Vibrato v2 server on {}", addr);
            let listener = tokio::net::TcpListener::bind(addr).await?;
            axum::serve(listener, router).await?;
        }

        Commands::KeyCreate {
            data_dir,
            name,
            roles,
        } => {
            let config = ProductionConfig::from_data_dir(data_dir, "default".to_string(), 128);
            bootstrap_data_dirs(&config)?;
            let catalog = SqliteCatalog::open(&config.catalog_path())?;
            let parsed_roles = Role::parse_csv(&roles);
            if parsed_roles.is_empty() {
                anyhow::bail!(
                    "No valid roles parsed from '{}'. Use comma-separated query,ingest,admin",
                    roles
                );
            }
            let key = catalog.create_api_key(&name, &parsed_roles, &config.api_pepper)?;
            println!("id={}", key.id);
            println!("token={}", key.token);
        }

        Commands::KeyRevoke { data_dir, key_id } => {
            let config = ProductionConfig::from_data_dir(data_dir, "default".to_string(), 128);
            bootstrap_data_dirs(&config)?;
            let catalog = SqliteCatalog::open(&config.catalog_path())?;
            catalog.revoke_api_key(&key_id)?;
            println!("revoked={}", key_id);
        }

        Commands::MigrateV2 {
            data_dir,
            input,
            collection,
            dim,
            level,
        } => {
            let config = ProductionConfig::from_data_dir(data_dir, collection, dim);
            bootstrap_data_dirs(&config)?;
            let catalog = Arc::new(SqliteCatalog::open(&config.catalog_path())?);
            let state = ProductionState::initialize(config.clone(), catalog)?;

            let segment_id = migrate_existing_vdb_to_segment(&state, &input, level)?;
            let report = recover_state(&state)?;
            println!("migrated_segment={}", segment_id);
            println!("{}", report.report);
        }

        Commands::SnapshotCreate {
            data_dir,
            collection,
        } => {
            let config = ProductionConfig::from_data_dir(data_dir, collection.clone(), 128);
            bootstrap_data_dirs(&config)?;
            let catalog = SqliteCatalog::open(&config.catalog_path())?;
            let snapshot = create_snapshot(&config, &catalog, &collection)?;
            println!("snapshot_id={}", snapshot.snapshot_id);
            println!("snapshot_dir={}", snapshot.snapshot_dir.display());
            println!("segments={}", snapshot.segments);
        }

        Commands::SnapshotRestore {
            data_dir,
            snapshot_dir,
        } => {
            let config = ProductionConfig::from_data_dir(data_dir, "default".to_string(), 128);
            bootstrap_data_dirs(&config)?;
            restore_snapshot(&config, &snapshot_dir)?;
            println!("restored_snapshot={}", snapshot_dir.display());
        }

        Commands::ReplayToLsn {
            data_dir,
            collection,
            target_lsn,
        } => {
            let config = ProductionConfig::from_data_dir(data_dir, collection.clone(), 128);
            bootstrap_data_dirs(&config)?;
            let catalog = SqliteCatalog::open(&config.catalog_path())?;
            replay_to_lsn(&catalog, &collection, target_lsn)?;
            println!("replayed_to_lsn={}", target_lsn);
            println!("collection={}", collection);
        }
    }

    Ok(())
}

fn create_accessor(
    store_handle: SharedStore,
    dynamic: Arc<RwLock<Vec<Vec<f32>>>>,
) -> impl Fn(usize, &mut dyn FnMut(&[f32])) + Send + Sync + 'static {
    move |id: usize, sink: &mut dyn FnMut(&[f32])| {
        let store_guard = store_handle.load();
        if id < store_guard.count {
            sink(store_guard.get(id));
        } else {
            let offset = id - store_guard.count;
            let guard = dynamic.read();
            if offset < guard.len() {
                sink(&guard[offset]);
            } else {
                tracing::error!(
                    "Vector ID {} out of bounds (store={}, dynamic={})",
                    id,
                    store_guard.count,
                    guard.len()
                );
                let fallback = vec![0.0; store_guard.dim];
                sink(&fallback);
            }
        }
    }
}

fn build_hnsw_from_store<F>(count: usize, m: usize, ef_construction: usize, vector_fn: F) -> HNSW
where
    F: Fn(usize, &mut dyn FnMut(&[f32])) + Send + Sync + 'static,
{
    let mut hnsw = HNSW::new_with_accessor(m, ef_construction, vector_fn);
    for i in 0..count {
        hnsw.insert(i);
    }
    hnsw
}

fn validate_graph_bounds(hnsw: &HNSW, max_id_exclusive: usize) -> anyhow::Result<()> {
    if let Some(entry) = hnsw.entry_point {
        if entry >= max_id_exclusive {
            anyhow::bail!(
                "entry_point {} out of bounds for {} vectors",
                entry,
                max_id_exclusive
            );
        }
    }

    for node in &hnsw.nodes {
        if node.id >= max_id_exclusive {
            anyhow::bail!(
                "node id {} out of bounds for {} vectors",
                node.id,
                max_id_exclusive
            );
        }

        for neighbors in &node.layers {
            for &neighbor in neighbors {
                if neighbor >= max_id_exclusive {
                    anyhow::bail!(
                        "neighbor id {} out of bounds for {} vectors",
                        neighbor,
                        max_id_exclusive
                    );
                }
            }
        }
    }

    Ok(())
}

fn try_load_graph_from_vdb<F>(path: &PathBuf, vector_fn: F) -> anyhow::Result<Option<HNSW>>
where
    F: Fn(usize, &mut dyn FnMut(&[f32])) + Send + Sync + 'static,
{
    let mut file = std::fs::File::open(path)?;
    let mut header_bytes = [0u8; 64];
    if file.read_exact(&mut header_bytes).is_err() {
        return Ok(None);
    }

    let Ok(header) = VdbHeaderV2::from_bytes(&header_bytes) else {
        return Ok(None);
    };
    if !header.has_graph() || header.graph_offset == 0 {
        return Ok(None);
    }

    file.seek(std::io::SeekFrom::Start(header.graph_offset))?;
    let mut reader = BufReader::new(file);
    let hnsw = HNSW::load_from_reader_with_accessor(&mut reader, vector_fn)?;
    Ok(Some(hnsw))
}

fn build_index(
    store: Arc<VectorStore>,
    m: usize,
    ef_construction: usize,
    save_path: Option<&PathBuf>,
) -> anyhow::Result<HNSW> {
    let store_for_hnsw = store.clone();
    let mut hnsw = HNSW::new_with_accessor(m, ef_construction, move |id, sink| {
        sink(store_for_hnsw.get(id))
    });

    let total = store.count;
    let progress_interval = (total / 100).max(1);

    for i in 0..total {
        hnsw.insert(i);

        if i % progress_interval == 0 {
            let pct = (i as f64 / total as f64) * 100.0;
            tracing::info!("Building index: {:.0}% ({}/{})", pct, i, total);
        }
    }

    if let Some(path) = save_path {
        tracing::info!("Saving index to {:?}", path);
        hnsw.save(path)?;
    }

    Ok(hnsw)
}

fn parse_vector(s: &str) -> Result<Vec<f32>, String> {
    s.split(',')
        .map(|v| v.trim().parse::<f32>().map_err(|e| e.to_string()))
        .collect()
}

fn parse_vector_madvise_mode(s: &str) -> anyhow::Result<VectorMadviseMode> {
    match s.trim().to_ascii_lowercase().as_str() {
        "normal" => Ok(VectorMadviseMode::Normal),
        "random" => Ok(VectorMadviseMode::Random),
        other => anyhow::bail!(
            "invalid --vector-madvise-mode '{}', expected normal|random",
            other
        ),
    }
}
