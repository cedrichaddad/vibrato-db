//! HNSW (Hierarchical Navigable Small World) Index
//!
//! A multi-layer graph structure for approximate nearest neighbor search.
//!
//! # Architecture
//!
//! ```text
//! Layer 3: ○─────────────────────○ (few nodes, long-range)
//!          │                     │
//! Layer 2: ●───────○─────────────● (more nodes)
//!          │       │             │
//! Layer 1: ●───○───●───○───○─────● (even more)
//!          │   │   │   │   │     │
//! Layer 0: ●─●─●─●─●─●─●─●─●─●─●─● (all nodes)
//! ```

mod index;
mod node;
mod serialize;
mod visited;

pub use index::HNSW;
pub use node::Node;
pub use visited::VisitedPool;
