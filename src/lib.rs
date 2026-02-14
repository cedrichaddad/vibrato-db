//! Vibrato-DB: A persistent, disk-backed vector search engine
//!
//! This is the root binary crate. All core functionality lives in `vibrato-core`.
//! The neural pipeline lives in `vibrato-neural`.

pub mod prod;
pub mod server;

// Re-export core types for backward compatibility
pub use vibrato_core::format;
pub use vibrato_core::format_v2; // New V2 format
pub use vibrato_core::hnsw;
pub use vibrato_core::simd;
pub use vibrato_core::store;
pub use vibrato_core::{dot_product, l2_distance, VdbHeader, VdbWriter, VectorStore, HNSW};
