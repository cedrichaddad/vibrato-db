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
use vibrato_db::server::{serve, AppState};
use vibrato_db::store::VectorStore;

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
            let store = VectorStore::open(&data)?;
            tracing::info!(
                "Loaded {} vectors of dimension {}",
                store.count,
                store.dim
            );

            let hnsw = if let Some(index_path) = &index {
                if index_path.exists() {
                    tracing::info!("Loading index from {:?}", index_path);
                    let store_ref = Arc::new(store);
                    let store_for_hnsw = store_ref.clone();
                    let hnsw = HNSW::load(index_path, move |id| {
                        store_for_hnsw.get(id).to_vec()
                    })?;
                    let stats = hnsw.stats();
                    tracing::info!(
                        "Index loaded: {} nodes, {} layers",
                        stats.num_nodes,
                        stats.max_layer + 1
                    );
                    // We need to get store back
                    drop(store_ref);
                    let store = VectorStore::open(&data)?;
                    let store_ref = Arc::new(store);
                    let store_for_hnsw = store_ref.clone();
                    let hnsw = HNSW::load(index_path, move |id| {
                        store_for_hnsw.get(id).to_vec()
                    })?;
                    let state = Arc::new(AppState {
                        index: RwLock::new(hnsw),
                        store: VectorStore::open(&data)?,
                    });
                    let addr: SocketAddr = format!("{}:{}", host, port).parse()?;
                    return Ok(serve(state, addr).await?);
                } else {
                    tracing::info!("Index not found, building new index...");
                    build_index(&store, m, ef_construction, Some(index_path))?
                }
            } else {
                tracing::info!("No index path specified, building in-memory index...");
                build_index(&store, m, ef_construction, None)?
            };

            let store = VectorStore::open(&data)?;
            let state = Arc::new(AppState {
                index: RwLock::new(hnsw),
                store,
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
