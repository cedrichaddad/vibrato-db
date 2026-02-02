//! Vibrato-DB: A persistent, disk-backed vector search engine
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                      HTTP API (axum)                        │
//! │                    POST /search, GET /health                │
//! └─────────────────────────────────────────────────────────────┘
//!                              │
//!                              ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    HNSW Indexing Engine                     │
//! │         Hierarchical Navigable Small World Graph            │
//! └─────────────────────────────────────────────────────────────┘
//!                              │
//!                              ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │                     VectorStore (mmap)                      │
//! │              Zero-copy access to .vdb files                 │
//! └─────────────────────────────────────────────────────────────┘
//! ```

pub mod format;
pub mod hnsw;
pub mod server;
pub mod simd;
pub mod store;

pub use format::{VdbHeader, VdbWriter};
pub use hnsw::HNSW;
pub use simd::{dot_product, l2_distance};
pub use store::VectorStore;
