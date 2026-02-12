//! Vibrato-DB CLI
//!
//! A persistent, disk-backed vector search engine.
//!
//! # Usage
//!
//! ```bash
//! # Start the server
//! vibrato-db serve --data data.vdb --index index.idx --port 8080
//!
//! # Build an index from a .vdb file
//! vibrato-db build --data data.vdb --output index.idx --m 16 --ef 100
//! ```

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use clap::{Parser, Subcommand};
use parking_lot::RwLock;
use tracing_subscriber::EnvFilter;

use vibrato_db::hnsw::HNSW;
use vibrato_db::server::{serve, AppState, SearchRequest, SearchResponse};
use vibrato_db::store::VectorStore;
use vibrato_db::format_v2::VdbWriterV2;
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
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info")),
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
        } => {
            tracing::info!("Loading vector data from {:?}", data);
            let store = Arc::new(VectorStore::open(&data)?);
            tracing::info!(
                "Loaded {} vectors of dimension {}",
                store.count,
                store.dim
            );

            // Create dynamic components
            let dynamic_store = Arc::new(RwLock::new(Vec::new()));
            // Initialize Inference Engine (mock or real)
            let model_dir = PathBuf::from("models"); // Default to local models dir
            let inference = Arc::new(InferenceEngine::new(&model_dir).unwrap_or_else(|e| {
                tracing::warn!("Failed to load inference engine: {}. Using mock.", e);
                // Creating a dummy engine pointing to temp (which will fail nicely or fallback)
                InferenceEngine::new(&std::env::temp_dir()).unwrap() 
            }));

            // Helper to create the combined vector accessor
            let create_accessor = |store: Arc<VectorStore>, dynamic: Arc<RwLock<Vec<Vec<f32>>>>| {
                move |id: usize| {
                    if id < store.count as usize {
                        // Access mmap store (zero-copy-ish, strictly we need owned Vec for HNSW currently)
                        store.get(id).to_vec()
                    } else {
                        // Access dynamic store
                        let offset = id - store.count as usize;
                        let guard = dynamic.read();
                        if offset < guard.len() {
                            guard[offset].clone()
                        } else {
                            // Should not happen if HNSW is consistent
                             vec![0.0; store.dim] 
                        }
                    }
                }
            };

            let hnsw = if let Some(index_path) = &index {
                if index_path.exists() {
                    tracing::info!("Loading index from {:?}", index_path);
                    let accessor = create_accessor(store.clone(), dynamic_store.clone());
                    let hnsw = HNSW::load(index_path, accessor)?;
                    
                    let stats = hnsw.stats();
                    tracing::info!(
                        "Index loaded: {} nodes, {} layers",
                        stats.num_nodes,
                        stats.max_layer + 1
                    );
                    hnsw
                } else {
                    tracing::info!("Index not found, building new index...");
                    // We need to build using the combined accessor, though dynamic is empty now.
                    let accessor = create_accessor(store.clone(), dynamic_store.clone());
                    // Build logic adapted to use accessor
                    // But build_index helper uses &VectorStore. We should inline or refactor.
                    // For simplicity, we assume we are building from just the static store initially
                    // BUT we must supply the *final* accessor that supports dynamic growth.
                    // HNSW::new takes the accessor.
                    let mut hnsw = HNSW::new(m, ef_construction, accessor);
                    for i in 0..store.count as usize {
                         hnsw.insert(i);
                    }
                    if let Some(path) = index {
                        hnsw.save(path)?;
                    }
                    hnsw
                }
            } else {
                tracing::info!("No index path specified, building in-memory index...");
                let accessor = create_accessor(store.clone(), dynamic_store.clone());
                let mut hnsw = HNSW::new(m, ef_construction, accessor);
                 for i in 0..store.count as usize {
                     hnsw.insert(i);
                 }
                 hnsw
            };

            let state = Arc::new(AppState {
                index: RwLock::new(hnsw),
                store,
                dynamic_store,
                inference,
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
            let store = VectorStore::open(&data)?;
            tracing::info!(
                "Loaded {} vectors of dimension {}",
                store.count,
                store.dim
            );

            let hnsw = build_index(&store, m, ef_construction, Some(&output))?;
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

        Commands::Search { server, query, k, ef } => {
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
    }

    Ok(())
}

fn build_index(
    store: &VectorStore,
    m: usize,
    ef_construction: usize,
    save_path: Option<&PathBuf>,
) -> anyhow::Result<HNSW> {
    // We need to copy vectors to a Vec since HNSW needs owned vectors
    // In a real implementation, you'd use Arc<VectorStore> or similar
    let vectors: Vec<Vec<f32>> = (0..store.count)
        .map(|i| store.get(i).to_vec())
        .collect();

    let vectors = Arc::new(vectors);
    let vectors_for_hnsw = vectors.clone();

    let mut hnsw = HNSW::new(m, ef_construction, move |id| {
        vectors_for_hnsw[id].clone()
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
