//! Vibrato Core – Storage engine, HNSW index, SIMD vector math, and Product Quantization
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    HNSW Indexing Engine                     │
//! │   ANN Search · Hybrid Filtered Search · Sub-sequence       │
//! ├─────────────────────────────────────────────────────────────┤
//! │         Product Quantization (PQ + ADC + SIMD)             │
//! ├─────────────────────────────────────────────────────────────┤
//! │           VectorStore (mmap zero-copy .vdb V2)             │
//! └─────────────────────────────────────────────────────────────┘
//! ```

pub mod format;
pub mod format_v2;
pub mod hnsw;
pub mod metadata;
pub mod pq;
pub mod simd;
pub mod store;
pub mod training;

pub use format::{VdbHeader, VdbWriter};
pub use hnsw::HNSW;
pub use simd::{dot_product, l2_distance};
pub use store::VectorStore;
