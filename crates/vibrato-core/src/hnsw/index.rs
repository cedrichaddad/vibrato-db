//! HNSW Index Implementation
//!
//! The core HNSW algorithm with:
//! - Random layer assignment (exponential distribution)
//! - Diversity-preserving neighbor selection heuristic
//! - Greedy beam search with BitSet visited tracking
//!
//! # Algorithm Overview
//!
//! **Insert**: Assign random layer L, search top-down from entry point to L,
//! then wire connections on all layers from L down to 0.
//!
//! **Search**: Start at entry point, greedy descent to layer 0, then beam search
//! on layer 0 with ef candidates.

use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use rustc_hash::{FxHashMap, FxHashSet};

use super::node::Node;
use super::visited::{VisitedGuard, VisitedSet};
use crate::simd::dot_product;

/// Candidate for search (min-heap)
#[derive(Clone, Copy)]
struct Candidate {
    idx: usize,
    distance: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for Candidate {}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse for min-heap (smaller distance = higher priority)
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Result from search (max-heap for keeping top-k worst)
#[derive(Clone, Copy)]
struct SearchResult {
    idx: usize,
    distance: f32,
}

impl PartialEq for SearchResult {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for SearchResult {}

impl Ord for SearchResult {
    fn cmp(&self, other: &Self) -> Ordering {
        // Normal order for max-heap (larger distance = higher priority)
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for SearchResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Reusable heap buffers for `search_layer` to avoid per-call allocations.
struct SearchScratch {
    candidates: BinaryHeap<Candidate>,
    results: BinaryHeap<SearchResult>,
}

impl SearchScratch {
    #[inline]
    fn new(initial_ef: usize) -> Self {
        let ef = initial_ef.max(1);
        Self {
            candidates: BinaryHeap::with_capacity(ef),
            results: BinaryHeap::with_capacity(ef + 1),
        }
    }

    #[inline]
    fn prepare(&mut self, ef: usize) {
        let ef = ef.max(1);
        self.candidates.clear();
        self.results.clear();

        if self.candidates.capacity() < ef {
            self.candidates.reserve(ef - self.candidates.capacity());
        }
        if self.results.capacity() < ef + 1 {
            self.results.reserve((ef + 1) - self.results.capacity());
        }
    }
}

struct QueryScratch {
    search: SearchScratch,
    layer_candidates: Vec<(usize, f32)>,
}

impl QueryScratch {
    #[inline]
    fn new() -> Self {
        Self {
            search: SearchScratch::new(32),
            layer_candidates: Vec::with_capacity(128),
        }
    }
}

struct InsertScratch {
    search: SearchScratch,
    layer_candidates: Vec<(usize, f32)>,
    reverse_edges: Vec<(usize, usize, u32)>,
    prune_ops: Vec<(usize, usize, Vec<u32>)>,
    neighbor_vec: Vec<f32>,
    all_neighbors: Vec<u32>,
    neighbor_candidates: Vec<(usize, f32)>,
}

impl InsertScratch {
    #[inline]
    fn new(initial_ef: usize) -> Self {
        Self {
            search: SearchScratch::new(initial_ef.max(1)),
            layer_candidates: Vec::with_capacity(256),
            reverse_edges: Vec::with_capacity(256),
            prune_ops: Vec::with_capacity(128),
            neighbor_vec: Vec::with_capacity(64),
            all_neighbors: Vec::with_capacity(64),
            neighbor_candidates: Vec::with_capacity(64),
        }
    }
}

thread_local! {
    static QUERY_SCRATCH: RefCell<QueryScratch> = RefCell::new(QueryScratch::new());
    static INSERT_SCRATCH: RefCell<InsertScratch> = RefCell::new(InsertScratch::new(64));
}

/// HNSW Index
///
/// A hierarchical graph structure for approximate nearest neighbor search.
pub struct HNSW {
    /// All nodes in the graph
    pub nodes: Vec<Node>,

    /// Map from external/global ID to dense node index.
    id_to_index: FxHashMap<u64, usize>,

    /// Entry point node index (node on the highest layer)
    pub entry_point: Option<usize>,

    /// Maximum layer currently in the graph
    pub max_layer: usize,

    /// Max neighbors per layer (M)
    pub m: usize,

    /// Max neighbors for layer 0 (usually 2*M)
    pub m0: usize,

    /// Level multiplier for random layer assignment (1/ln(M))
    pub ml: f64,

    /// Search depth during construction
    pub ef_construction: usize,

    /// Vector accessor function
    vectors: VectorAccessor,

    /// RNG for layer assignment (stored to avoid per-insert thread RNG overhead)
    rng: StdRng,
}

/// Type-erased vector accessor.
///
/// Callback style avoids allocating/cloning vectors on the hot search path.
/// The accessor key is the dense internal `node_idx` (0..nodes.len()).
type VectorAccessor = Box<dyn Fn(usize, &mut dyn FnMut(&[f32])) + Send + Sync>;
type VectorAccessorRef<'a> = &'a (dyn Fn(usize, &mut dyn FnMut(&[f32])) + Send + Sync);

impl HNSW {
    /// Create a new HNSW index
    ///
    /// # Parameters
    /// - `m`: Max neighbors per layer (typically 12-48)
    /// - `ef_construction`: Search depth during build (typically 100-200)
    /// - `vector_fn`: Function to get vector by dense node index
    pub fn new<F>(m: usize, ef_construction: usize, vector_fn: F) -> Self
    where
        F: Fn(usize) -> Vec<f32> + Send + Sync + 'static,
    {
        Self::new_with_seed(m, ef_construction, vector_fn, rand::random())
    }

    /// Create a new HNSW index with deterministic RNG seed.
    pub fn new_with_seed<F>(m: usize, ef_construction: usize, vector_fn: F, seed: u64) -> Self
    where
        F: Fn(usize) -> Vec<f32> + Send + Sync + 'static,
    {
        Self::new_with_accessor_and_seed(
            m,
            ef_construction,
            move |id, sink| {
                let v = vector_fn(id);
                sink(&v);
            },
            seed,
        )
    }

    /// Create a new HNSW index with zero-copy accessor callback.
    pub fn new_with_accessor<F>(m: usize, ef_construction: usize, vector_fn: F) -> Self
    where
        F: Fn(usize, &mut dyn FnMut(&[f32])) + Send + Sync + 'static,
    {
        Self::new_with_accessor_and_seed(m, ef_construction, vector_fn, rand::random())
    }

    /// Create a new HNSW index with zero-copy accessor callback and deterministic seed.
    pub fn new_with_accessor_and_seed<F>(
        m: usize,
        ef_construction: usize,
        vector_fn: F,
        seed: u64,
    ) -> Self
    where
        F: Fn(usize, &mut dyn FnMut(&[f32])) + Send + Sync + 'static,
    {
        Self {
            nodes: Vec::new(),
            id_to_index: FxHashMap::default(),
            entry_point: None,
            max_layer: 0,
            m,
            m0: m * 2,
            ml: 1.0 / (m as f64).ln(),
            ef_construction,
            vectors: Box::new(move |id, sink| vector_fn(id, sink)),
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Reconstruct an HNSW index from its parts with zero-copy vector accessor.
    pub(crate) fn from_parts_with_accessor<F>(
        nodes: Vec<Node>,
        entry_point: Option<usize>,
        max_layer: usize,
        m: usize,
        m0: usize,
        ml: f64,
        ef_construction: usize,
        vector_fn: F,
    ) -> Self
    where
        F: Fn(usize, &mut dyn FnMut(&[f32])) + Send + Sync + 'static,
    {
        // Build id_to_index map from existing nodes
        let id_to_index: FxHashMap<u64, usize> = nodes
            .iter()
            .enumerate()
            .map(|(idx, node)| (node.id, idx))
            .collect();

        Self {
            nodes,
            id_to_index,
            entry_point,
            max_layer,
            m,
            m0,
            ml,
            ef_construction,
            vectors: Box::new(move |id, sink| vector_fn(id, sink)),
            rng: StdRng::seed_from_u64(rand::random()),
        }
    }

    #[inline]
    fn with_vector_at_idx(
        &self,
        node_idx: usize,
        accessor: Option<VectorAccessorRef<'_>>,
        sink: &mut dyn FnMut(&[f32]),
    ) {
        if let Some(accessor) = accessor {
            accessor(node_idx, sink);
        } else {
            (self.vectors)(node_idx, sink);
        }
    }

    /// Returns true if `id` is present in the graph.
    #[inline]
    pub fn contains_id(&self, id: u64) -> bool {
        self.id_to_index.contains_key(&id)
    }

    /// Resolve external/global ID to dense internal node index.
    #[inline]
    pub fn node_index_for_id(&self, id: u64) -> Option<usize> {
        self.id_to_index.get(&id).copied()
    }

    /// Resolve dense internal node index to external/global ID.
    #[inline]
    pub fn id_for_node_idx(&self, node_idx: usize) -> Option<u64> {
        self.nodes.get(node_idx).map(|n| n.id)
    }

    /// Score a node ID against a query vector if that node exists.
    #[inline]
    pub fn score_for_id(&self, query: &[f32], id: u64) -> Option<f32> {
        self.score_for_id_with_optional_accessor(query, id, None)
    }

    #[inline]
    pub fn score_for_id_with_accessor(
        &self,
        query: &[f32],
        id: u64,
        accessor: VectorAccessorRef<'_>,
    ) -> Option<f32> {
        self.score_for_id_with_optional_accessor(query, id, Some(accessor))
    }

    #[inline]
    fn score_for_id_with_optional_accessor(
        &self,
        query: &[f32],
        id: u64,
        accessor: Option<VectorAccessorRef<'_>>,
    ) -> Option<f32> {
        let node_idx = *self.id_to_index.get(&id)?;
        let mut score = 0.0f32;
        self.with_vector_at_idx(node_idx, accessor, &mut |node_vec| {
            score = dot_product(query, node_vec);
        });
        Some(score)
    }

    /// Score a dense internal node index.
    #[inline]
    pub fn score_for_node_idx(&self, query: &[f32], node_idx: usize) -> Option<f32> {
        self.score_for_node_idx_with_optional_accessor(query, node_idx, None)
    }

    #[inline]
    pub fn score_for_node_idx_with_accessor(
        &self,
        query: &[f32],
        node_idx: usize,
        accessor: VectorAccessorRef<'_>,
    ) -> Option<f32> {
        self.score_for_node_idx_with_optional_accessor(query, node_idx, Some(accessor))
    }

    #[inline]
    fn score_for_node_idx_with_optional_accessor(
        &self,
        query: &[f32],
        node_idx: usize,
        accessor: Option<VectorAccessorRef<'_>>,
    ) -> Option<f32> {
        if self.nodes.get(node_idx).is_none() {
            return None;
        }
        let mut score = 0.0f32;
        self.with_vector_at_idx(node_idx, accessor, &mut |node_vec| {
            score = dot_product(query, node_vec);
        });
        Some(score)
    }

    #[inline]
    fn copy_vector_into_with_optional_accessor(
        &self,
        node_idx: usize,
        out: &mut Vec<f32>,
        accessor: Option<VectorAccessorRef<'_>>,
    ) {
        out.clear();
        self.with_vector_at_idx(node_idx, accessor, &mut |v| {
            if out.capacity() < v.len() {
                out.reserve(v.len() - out.capacity());
            }
            out.extend_from_slice(v);
        });
    }

    /// Compute distance between query and node (dot product for normalized vectors).
    /// Returns negative similarity so smaller = more similar, without extra subtraction.
    #[inline]
    fn distance_with_optional_accessor(
        &self,
        query: &[f32],
        node_idx: usize,
        accessor: Option<VectorAccessorRef<'_>>,
    ) -> f32 {
        let mut dist = 0.0f32;
        self.with_vector_at_idx(node_idx, accessor, &mut |node_vec| {
            dist = -dot_product(query, node_vec);
        });
        dist
    }

    /// Assign a random layer based on exponential distribution
    fn random_layer(&mut self) -> usize {
        let r: f64 = self.rng.gen();
        (-r.ln() * self.ml).floor() as usize
    }

    /// Insert a vector into the index
    ///
    /// # Parameters
    /// - `id`: External/global vector ID
    /// - `query`: Vector payload for the inserted ID
    pub fn insert(&mut self, id: u64, query: &[f32]) -> usize {
        self.insert_with_optional_accessor(id, query, None)
    }

    pub fn insert_with_accessor(
        &mut self,
        id: u64,
        query: &[f32],
        accessor: VectorAccessorRef<'_>,
    ) -> usize {
        self.insert_with_optional_accessor(id, query, Some(accessor))
    }

    fn insert_with_optional_accessor(
        &mut self,
        id: u64,
        query: &[f32],
        accessor: Option<VectorAccessorRef<'_>>,
    ) -> usize {
        // Idempotent insert guard for checkpoint catch-up/replay paths.
        if let Some(existing_idx) = self.id_to_index.get(&id).copied() {
            return existing_idx;
        }
        if self.nodes.len() >= u32::MAX as usize {
            panic!(
                "data integrity fault: hnsw node index overflow (len={} exceeds u32::MAX)",
                self.nodes.len()
            );
        }

        INSERT_SCRATCH.with(|slot| {
            let mut slot = slot.borrow_mut();
            let InsertScratch {
                search,
                layer_candidates,
                reverse_edges,
                prune_ops,
                neighbor_vec,
                all_neighbors,
                neighbor_candidates,
            } = &mut *slot;

            layer_candidates.clear();
            reverse_edges.clear();
            prune_ops.clear();
            all_neighbors.clear();
            neighbor_candidates.clear();

            let node_layer = self.random_layer();
            let new_node_idx = self.nodes.len();
            let new_node_idx_u32 = u32::try_from(new_node_idx)
                .expect("data integrity fault: node index conversion overflow");

            // Create node
            let mut node = Node::new(id, node_layer);

            // First node becomes entry point
            if self.entry_point.is_none() {
                self.entry_point = Some(0);
                self.max_layer = node_layer;
                self.id_to_index.insert(id, 0); // First node always at index 0
                self.nodes.push(node);
                return 0;
            }

            let entry_point = self.entry_point.unwrap();
            let mut current_node = entry_point;
            let mut visited = VisitedGuard::new(self.nodes.len().saturating_add(1));

            // Phase 1: Zoom in from top layer to node_layer + 1
            // Greedy search, single best neighbor per layer
            for layer in (node_layer + 1..=self.max_layer).rev() {
                self.search_layer_into_with_accessor(
                    query,
                    &[current_node],
                    1,
                    layer,
                    &mut visited,
                    search,
                    layer_candidates,
                    accessor,
                );
                if let Some((nearest_idx, _)) = layer_candidates.first() {
                    current_node = *nearest_idx;
                }
            }

            // Phase 2: Insert at layers from min(node_layer, max_layer) down to 0
            // We collect all the updates needed, then apply them.
            let start_layer = node_layer.min(self.max_layer);

            for layer in (0..=start_layer).rev() {
                let m_layer = if layer == 0 { self.m0 } else { self.m };

                // Find candidates for this layer.
                self.search_layer_into_with_accessor(
                    query,
                    &[current_node],
                    self.ef_construction,
                    layer,
                    &mut visited,
                    search,
                    layer_candidates,
                    accessor,
                );

                // Select neighbors using the heuristic.
                let neighbors = self.select_neighbors_with_accessor(
                    query,
                    layer_candidates,
                    m_layer,
                    accessor,
                    None,
                );

                // Add forward edges to new node.
                for &(neighbor_idx, _) in &neighbors {
                    // `select_neighbors` returns unique IDs, so this is duplicate-safe.
                    let neighbor_idx_u32 = u32::try_from(neighbor_idx)
                        .expect("data integrity fault: neighbor index overflow");
                    node.add_neighbor_unchecked(layer, neighbor_idx_u32);

                    reverse_edges.push((neighbor_idx, layer, new_node_idx_u32));

                    // Check if pruning will be needed.
                    let current_neighbors = self.nodes[neighbor_idx].neighbors(layer);
                    if current_neighbors.len() >= m_layer {
                        // Will need pruning after adding new edge.
                        all_neighbors.clear();
                        all_neighbors.extend_from_slice(current_neighbors);
                        all_neighbors.push(new_node_idx_u32);

                        // Compute distances and select.
                        neighbor_candidates.clear();
                        if neighbor_candidates.capacity() < all_neighbors.len() {
                            neighbor_candidates
                                .reserve(all_neighbors.len() - neighbor_candidates.capacity());
                        }
                        self.copy_vector_into_with_optional_accessor(
                            neighbor_idx,
                            neighbor_vec,
                            accessor,
                        );
                        for &n in all_neighbors.iter() {
                            let n_idx = n as usize;
                            let mut dist = 0.0f32;
                            if n_idx == new_node_idx {
                                dist = -dot_product(neighbor_vec.as_slice(), query);
                            } else {
                                self.with_vector_at_idx(n_idx, accessor, &mut |v| {
                                    dist = -dot_product(neighbor_vec.as_slice(), v);
                                });
                            }
                            neighbor_candidates.push((n_idx, dist));
                        }

                        let pruned = self.select_neighbors_with_accessor(
                            &[],
                            neighbor_candidates,
                            m_layer,
                            accessor,
                            Some((new_node_idx, query)),
                        );
                        let pruned_ids: Vec<u32> = pruned
                            .iter()
                            .map(|(idx, _)| {
                                u32::try_from(*idx).expect("data integrity fault: prune idx overflow")
                            })
                            .collect();
                        prune_ops.push((neighbor_idx, layer, pruned_ids));
                    }
                }

                // Use first candidate as entry for next layer.
                if let Some((first_idx, _)) = layer_candidates.first() {
                    current_node = *first_idx;
                }
            }

            // Apply reverse edges (avoiding those that will be overwritten by pruning).
            let prune_targets: FxHashSet<(usize, usize)> = prune_ops
                .iter()
                .map(|(idx, layer, _)| (*idx, *layer))
                .collect();

            for &(node_idx, layer, neighbor_id) in reverse_edges.iter() {
                if !prune_targets.contains(&(node_idx, layer)) {
                    // New node ID was not present before this insert, so edge is unique.
                    self.nodes[node_idx].add_neighbor_unchecked(layer, neighbor_id);
                }
            }

            // Apply pruning operations.
            for (node_idx, layer, new_neighbors) in prune_ops.iter() {
                if let Some(layer_neighbors) = self.nodes[*node_idx].neighbors_mut(*layer) {
                    layer_neighbors.clear();
                    layer_neighbors.extend(new_neighbors.iter().copied());
                }
            }

            // Update entry point if new node has higher layer.
            if node_layer > self.max_layer {
                self.max_layer = node_layer;
                self.entry_point = Some(new_node_idx);
            }

            // Add to id_to_index map before pushing.
            self.id_to_index.insert(id, new_node_idx);
            self.nodes.push(node);
            new_node_idx
        })
    }

    /// Search for nearest neighbors on a single layer
    ///
    /// Greedy beam search with `ef` candidates.
    fn search_layer_into_with_accessor<V: VisitedSet>(
        &self,
        query: &[f32],
        entry_points: &[usize],
        ef: usize,
        layer: usize,
        visited: &mut V,
        scratch: &mut SearchScratch,
        out: &mut Vec<(usize, f32)>,
        accessor: Option<VectorAccessorRef<'_>>,
    ) {
        let ef = ef.max(1);
        visited.clear();
        scratch.prepare(ef);
        let candidates = &mut scratch.candidates;
        let results = &mut scratch.results;
        let mut worst_distance = f32::INFINITY;

        // Initialize with entry points
        for &ep in entry_points {
            if !visited.is_visited(ep) {
                visited.visit(ep);
                let dist = self.distance_with_optional_accessor(query, ep, accessor);
                candidates.push(Candidate {
                    idx: ep,
                    distance: dist,
                });
                results.push(SearchResult {
                    idx: ep,
                    distance: dist,
                });
                worst_distance = results.peek().map(|r| r.distance).unwrap_or(f32::INFINITY);
            }
        }

        // Greedy search
        while let Some(current) = candidates.pop() {
            // Pruning: if current is worse than worst result, stop
            if results.len() >= ef && current.distance > worst_distance {
                break;
            }

            // Explore neighbors via dense node-index adjacency.
            if let Some(node) = self.nodes.get(current.idx) {
                // PERF NOTE: prefetching Node structs here is ineffective because
                // Node contains Vec<Vec<u32>> — the actual neighbor data lives
                // on the heap behind pointers, not inline in the struct.
                // True fix: linearize the graph into a flat Vec<u32> (V3 scope).
                for &neighbor in node.neighbors(layer) {
                    let neighbor_idx = neighbor as usize;
                    if visited.is_visited(neighbor_idx) {
                        continue;
                    }
                    visited.visit(neighbor_idx);

                    let dist = self.distance_with_optional_accessor(query, neighbor_idx, accessor);

                    // Add to candidates if promising
                    let dominated = results.len() >= ef && dist > worst_distance;
                    if !dominated {
                        candidates.push(Candidate {
                            idx: neighbor_idx,
                            distance: dist,
                        });
                        results.push(SearchResult {
                            idx: neighbor_idx,
                            distance: dist,
                        });

                        // Keep only top ef results
                        if results.len() > ef {
                            results.pop();
                        }
                        worst_distance =
                            results.peek().map(|r| r.distance).unwrap_or(f32::INFINITY);
                    }
                }
            }
        }

        // Convert to sorted vec while keeping heap allocations for reuse.
        out.clear();
        out.extend(results.iter().map(|r| (r.idx, r.distance)));
        out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    }

    /// Select neighbors using diversity-preserving heuristic
    ///
    /// Instead of just picking the closest M neighbors, we ensure diversity:
    /// A candidate is added only if it's closer to query than to any selected neighbor.
    fn select_neighbors_with_accessor(
        &self,
        _query: &[f32],
        candidates: &[(usize, f32)],
        m: usize,
        accessor: Option<VectorAccessorRef<'_>>,
        virtual_candidate: Option<(usize, &[f32])>,
    ) -> Vec<(usize, f32)> {
        if candidates.is_empty() {
            return Vec::new();
        }

        // Sort by distance to query
        let mut sorted: Vec<_> = candidates.to_vec();
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        let mut result: Vec<(usize, f32)> = Vec::with_capacity(m);
        if let Some((virtual_idx, virtual_vec)) = virtual_candidate {
            for &(candidate_idx, candidate_dist) in &sorted {
                if result.len() >= m {
                    break;
                }

                // Check if candidate is closer to query than to any existing result
                let mut is_diverse = true;
                if candidate_idx == virtual_idx {
                    for &(existing_idx, _) in &result {
                        let dist_to_existing: f32 = if existing_idx == virtual_idx {
                            -dot_product(virtual_vec, virtual_vec)
                        } else {
                            let mut dist = 0.0f32;
                            self.with_vector_at_idx(existing_idx, accessor, &mut |existing_vec| {
                                dist = -dot_product(virtual_vec, existing_vec);
                            });
                            dist
                        };

                        if dist_to_existing < candidate_dist {
                            // Candidate is closer to existing neighbor than to query.
                            // This means the existing neighbor "covers" this direction.
                            is_diverse = false;
                            break;
                        }
                    }
                } else {
                    self.with_vector_at_idx(candidate_idx, accessor, &mut |candidate_vec| {
                        for &(existing_idx, _) in &result {
                            let dist_to_existing: f32 = if existing_idx == virtual_idx {
                                -dot_product(candidate_vec, virtual_vec)
                            } else {
                                let mut dist = 0.0f32;
                                self.with_vector_at_idx(existing_idx, accessor, &mut |existing_vec| {
                                    dist = -dot_product(candidate_vec, existing_vec);
                                });
                                dist
                            };

                            if dist_to_existing < candidate_dist {
                                // Candidate is closer to existing neighbor than to query.
                                // This means the existing neighbor "covers" this direction.
                                is_diverse = false;
                                break;
                            }
                        }
                    });
                }

                if is_diverse {
                    result.push((candidate_idx, candidate_dist));
                }
            }
        } else {
            for &(candidate_idx, candidate_dist) in &sorted {
                if result.len() >= m {
                    break;
                }
                let mut is_diverse = true;
                self.with_vector_at_idx(candidate_idx, accessor, &mut |candidate_vec| {
                    for &(existing_idx, _) in &result {
                        let mut dist_to_existing = 0.0f32;
                        self.with_vector_at_idx(existing_idx, accessor, &mut |existing_vec| {
                            dist_to_existing = -dot_product(candidate_vec, existing_vec);
                        });

                        if dist_to_existing < candidate_dist {
                            is_diverse = false;
                            break;
                        }
                    }
                });
                if is_diverse {
                    result.push((candidate_idx, candidate_dist));
                }
            }
        }

        // If we don't have enough diverse neighbors, fill with closest
        if result.len() < m {
            for &(candidate_idx, candidate_dist) in &sorted {
                if result.len() >= m {
                    break;
                }
                if !result.iter().any(|(idx, _)| *idx == candidate_idx) {
                    result.push((candidate_idx, candidate_dist));
                }
            }
        }

        result
    }

    /// Search for k nearest neighbors to a query vector
    ///
    /// # Parameters
    /// - `query`: Query vector
    /// - `k`: Number of neighbors to return
    /// - `ef`: Search depth (higher = better recall, slower)
    ///
    /// # Returns
    /// Vector of (id, score) pairs, sorted by score descending
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<(u64, f32)> {
        self.search_with_optional_accessor(query, k, ef, None)
    }

    pub fn search_with_accessor(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        accessor: VectorAccessorRef<'_>,
    ) -> Vec<(u64, f32)> {
        self.search_with_optional_accessor(query, k, ef, Some(accessor))
    }

    fn search_with_optional_accessor(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        accessor: Option<VectorAccessorRef<'_>>,
    ) -> Vec<(u64, f32)> {
        if self.entry_point.is_none() {
            return Vec::new();
        }

        QUERY_SCRATCH.with(|slot| {
            let mut slot = slot.borrow_mut();
            let QueryScratch {
                search,
                layer_candidates,
            } = &mut *slot;
            let entry_point = self.entry_point.unwrap();
            let mut current_node = entry_point;
            let mut visited = VisitedGuard::new(self.nodes.len());

            // Phase 1: Greedy descent from top layer to layer 1
            for layer in (1..=self.max_layer).rev() {
                self.search_layer_into_with_accessor(
                    query,
                    &[current_node],
                    1,
                    layer,
                    &mut visited,
                    search,
                    layer_candidates,
                    accessor,
                );
                if let Some((nearest_idx, _)) = layer_candidates.first() {
                    current_node = *nearest_idx;
                }
            }

            // Phase 2: Beam search on layer 0
            self.search_layer_into_with_accessor(
                query,
                &[current_node],
                ef.max(k),
                0,
                &mut visited,
                search,
                layer_candidates,
                accessor,
            );

            // Return top k, convert distance to similarity score
            layer_candidates
                .iter()
                .take(k)
                .filter_map(|(idx, dist)| self.nodes.get(*idx).map(|node| (node.id, -*dist)))
                .collect()
        })
    }

    /// Search with metadata predicate filter (hybrid search)
    ///
    /// Traverses the graph normally but only emits results that pass the
    /// predicate. Uses over-fetch (ef × 2) to compensate for filtered candidates.
    ///
    /// # Parameters
    /// - `query`: Query vector
    /// - `k`: Number of results to return
    /// - `ef`: Base search depth (will be doubled internally for over-fetch)
    /// - `predicate`: Closure that returns true for IDs to keep
    ///
    /// # Example
    /// ```ignore
    /// // Find kicks near a query, filtering by metadata
    /// let results = hnsw.search_filtered(&query, 10, 100, |id| {
    ///     metadata.get(id).map(|m| m.tags.contains("kick")).unwrap_or(false)
    /// });
    /// ```
    pub fn search_filtered<P>(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        predicate: P,
    ) -> Vec<(u64, f32)>
    where
        P: Fn(usize) -> bool,
    {
        self.search_filtered_with_optional_accessor(query, k, ef, predicate, None)
    }

    pub fn search_filtered_with_accessor<P>(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        predicate: P,
        accessor: VectorAccessorRef<'_>,
    ) -> Vec<(u64, f32)>
    where
        P: Fn(usize) -> bool,
    {
        self.search_filtered_with_optional_accessor(query, k, ef, predicate, Some(accessor))
    }

    fn search_filtered_with_optional_accessor<P>(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        predicate: P,
        accessor: Option<VectorAccessorRef<'_>>,
    ) -> Vec<(u64, f32)>
    where
        P: Fn(usize) -> bool,
    {
        if self.entry_point.is_none() {
            return Vec::new();
        }

        let candidates = QUERY_SCRATCH.with(|slot| {
            let mut slot = slot.borrow_mut();
            let QueryScratch {
                search,
                layer_candidates,
            } = &mut *slot;
            let entry_point = self.entry_point.unwrap();
            let mut current_node = entry_point;
            let mut visited = VisitedGuard::new(self.nodes.len());

            // Phase 1: Greedy descent (unfiltered — we don't filter hub nodes)
            for layer in (1..=self.max_layer).rev() {
                self.search_layer_into_with_accessor(
                    query,
                    &[current_node],
                    1,
                    layer,
                    &mut visited,
                    search,
                    layer_candidates,
                    accessor,
                );
                if let Some((nearest_idx, _)) = layer_candidates.first() {
                    current_node = *nearest_idx;
                }
            }

            // Phase 2: Over-fetch beam search on layer 0
            // We fetch ef*2 candidates because many will be filtered out
            let over_fetch = (ef * 2).max(k * 4);
            self.search_layer_into_with_accessor(
                query,
                &[current_node],
                over_fetch,
                0,
                &mut visited,
                search,
                layer_candidates,
                accessor,
            );

            layer_candidates.clone()
        });

        // Apply predicate filter outside the scratch borrow scope so predicate
        // closures may safely perform re-entrant searches.
        candidates
            .into_iter()
            .filter(|(idx, _)| predicate(*idx))
            .take(k)
            .filter_map(|(idx, dist)| self.nodes.get(idx).map(|node| (node.id, -dist)))
            .collect()
    }

    /// Sub-sequence search for multi-vector audio queries
    ///
    /// Given a sequence of query vectors (e.g., consecutive audio frames),
    /// finds the best matching contiguous sub-sequence in the index.
    ///
    /// Algorithm (optimized per review feedback):
    /// 1. Find the most "salient" query vector (highest L2 norm = most distinct)
    /// 2. Search HNSW for that single vector → get candidate anchor positions
    /// 3. For each candidate, brute-force verify the full sequence alignment
    ///
    /// # Parameters
    /// - `query_sequence`: Ordered sequence of query vectors
    /// - `k`: Number of results
    /// - `ef`: Search depth
    /// - `total_vectors`: Total number of vectors in the store (for bounds checking)
    ///
    /// # Invariants
    ///
    /// **CRITICAL**: This method assumes that vectors belonging to a temporal sequence (e.g., audio frames)
    /// are stored with **contiguous, sequential IDs** in the `VectorStore`.
    ///
    /// The heuristic `start_id = anchor_id - salient_offset` relies on the assumption that if
    /// a database vector at `anchor_id` matches the query vector at `salient_offset`, then the
    /// preceding query vectors match `anchor_id - 1`, `anchor_id - 2`, etc.
    ///
    /// If your data is not stored sequentially (e.g. shuffled or random IDs), this method will return
    /// incorrect or empty results.
    ///
    /// # Returns
    /// Vector of (start_id, average_score) pairs for best matching sub-sequences
    pub fn search_subsequence(
        &self,
        query_sequence: &[Vec<f32>],
        k: usize,
        ef: usize,
        total_vectors: usize,
    ) -> Vec<(u64, f32)> {
        self.search_subsequence_with_predicate_and_optional_accessor(
            query_sequence,
            k,
            ef,
            total_vectors,
            |_| true,
            None,
        )
    }

    pub fn search_subsequence_with_accessor(
        &self,
        query_sequence: &[Vec<f32>],
        k: usize,
        ef: usize,
        total_vectors: usize,
        accessor: VectorAccessorRef<'_>,
    ) -> Vec<(u64, f32)> {
        self.search_subsequence_with_predicate_and_optional_accessor(
            query_sequence,
            k,
            ef,
            total_vectors,
            |_| true,
            Some(accessor),
        )
    }

    /// Sub-sequence search with explicit validity predicate.
    ///
    /// `is_valid_id` allows callers to enforce tombstone/hole checks for append-only ID invariants.
    pub fn search_subsequence_with_predicate<F>(
        &self,
        query_sequence: &[Vec<f32>],
        k: usize,
        ef: usize,
        total_vectors: usize,
        is_valid_id: F,
    ) -> Vec<(u64, f32)>
    where
        F: FnMut(u64) -> bool,
    {
        self.search_subsequence_with_predicate_and_optional_accessor(
            query_sequence,
            k,
            ef,
            total_vectors,
            is_valid_id,
            None,
        )
    }

    pub fn search_subsequence_with_predicate_and_accessor<F>(
        &self,
        query_sequence: &[Vec<f32>],
        k: usize,
        ef: usize,
        total_vectors: usize,
        is_valid_id: F,
        accessor: VectorAccessorRef<'_>,
    ) -> Vec<(u64, f32)>
    where
        F: FnMut(u64) -> bool,
    {
        self.search_subsequence_with_predicate_and_optional_accessor(
            query_sequence,
            k,
            ef,
            total_vectors,
            is_valid_id,
            Some(accessor),
        )
    }

    fn search_subsequence_with_predicate_and_optional_accessor<F>(
        &self,
        query_sequence: &[Vec<f32>],
        k: usize,
        ef: usize,
        total_vectors: usize,
        mut is_valid_id: F,
        accessor: Option<VectorAccessorRef<'_>>,
    ) -> Vec<(u64, f32)>
    where
        F: FnMut(u64) -> bool,
    {
        if query_sequence.is_empty() || self.entry_point.is_none() {
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
        salience.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        let mut anchor_offsets: Vec<usize> = Vec::with_capacity(probe_count);
        for (idx, _) in &salience {
            if anchor_offsets
                .iter()
                .all(|selected| selected.abs_diff(*idx) >= min_anchor_gap)
            {
                anchor_offsets.push(*idx);
                if anchor_offsets.len() >= probe_count {
                    break;
                }
            }
        }
        if anchor_offsets.is_empty() {
            anchor_offsets.push(salience.first().map(|(i, _)| *i).unwrap_or(0));
        }

        let mut best_by_start: FxHashMap<u64, f32> = FxHashMap::default();
        let anchor_overfetch = (k.saturating_mul(6)).max(32);

        for salient_offset in anchor_offsets {
            let salient_query = &query_sequence[salient_offset];
            let anchor_candidates =
                self.search_with_optional_accessor(salient_query, anchor_overfetch, ef, accessor);

            for (anchor_id, _anchor_score) in &anchor_candidates {
                let start_id = if *anchor_id >= salient_offset as u64 {
                    anchor_id - salient_offset as u64
                } else {
                    continue;
                };

                if start_id.saturating_add(seq_len as u64) > total_vectors as u64 {
                    continue;
                }

                let mut total_sim = 0.0f32;
                let mut valid = true;
                for (offset, query_vec) in query_sequence.iter().enumerate() {
                    let vec_id = start_id.saturating_add(offset as u64);
                    if vec_id >= total_vectors as u64 || !is_valid_id(vec_id) {
                        valid = false;
                        break;
                    }
                    let vec_idx = *self.id_to_index.get(&vec_id).unwrap_or_else(|| {
                        panic!("data integrity fault: subsequence missing vector id={vec_id}")
                    });
                    self.with_vector_at_idx(vec_idx, accessor, &mut |stored_vec| {
                        total_sim += dot_product(query_vec, stored_vec);
                    });
                }
                if !valid {
                    continue;
                }

                let avg_sim = total_sim / seq_len as f32;
                match best_by_start.get_mut(&start_id) {
                    Some(existing) => {
                        if avg_sim > *existing {
                            *existing = avg_sim;
                        }
                    }
                    None => {
                        best_by_start.insert(start_id, avg_sim);
                    }
                }
            }
        }

        let mut sequence_results = best_by_start.into_iter().collect::<Vec<_>>();
        sequence_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        sequence_results.truncate(k);
        sequence_results
    }

    /// Number of indexed nodes.
    #[inline]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns true when the index is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Get statistics about the index
    pub fn stats(&self) -> HNSWStats {
        let mut layer_counts = vec![0usize; self.max_layer + 1];
        let mut total_edges = 0;

        for node in &self.nodes {
            for (layer, neighbors) in node.layers.iter().enumerate() {
                if !neighbors.is_empty() {
                    if layer < layer_counts.len() {
                        layer_counts[layer] += 1;
                    }
                    total_edges += neighbors.len();
                }
            }
        }

        HNSWStats {
            num_nodes: self.nodes.len(),
            max_layer: self.max_layer,
            layer_counts,
            total_edges,
            m: self.m,
            ef_construction: self.ef_construction,
        }
    }
}

/// Statistics about the HNSW index
#[derive(Debug, Clone)]
pub struct HNSWStats {
    pub num_nodes: usize,
    pub max_layer: usize,
    pub layer_counts: Vec<usize>,
    pub total_edges: usize,
    pub m: usize,
    pub ef_construction: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::l2_normalized;
    use std::collections::HashMap;

    fn random_vector(dim: usize) -> Vec<f32> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect();
        l2_normalized(&v)
    }

    #[test]
    fn test_insert_single() {
        let vectors = vec![random_vector(128)];
        let vectors_clone = vectors.clone();

        let mut hnsw = HNSW::new(16, 100, move |id| vectors_clone[id].clone());
        hnsw.insert(0, &vectors[0]);

        assert_eq!(hnsw.nodes.len(), 1);
        assert_eq!(hnsw.entry_point, Some(0));
    }

    #[test]
    fn test_insert_multiple() {
        let vectors: Vec<_> = (0..100).map(|_| random_vector(128)).collect();
        let vectors_clone = vectors.clone();

        let mut hnsw = HNSW::new(16, 100, move |id| vectors_clone[id].clone());

        for i in 0..100 {
            hnsw.insert(i as u64, &vectors[i]);
        }

        assert_eq!(hnsw.nodes.len(), 100);
        assert!(hnsw.entry_point.is_some());
    }

    #[test]
    fn test_search_basic() {
        let vectors: Vec<_> = (0..100).map(|_| random_vector(128)).collect();
        let query = vectors[42].clone();
        let vectors_clone = vectors.clone();

        let mut hnsw = HNSW::new(16, 100, move |id| vectors_clone[id].clone());

        for i in 0..100 {
            hnsw.insert(i as u64, &vectors[i]);
        }

        let results = hnsw.search(&query, 5, 50);

        // The exact query vector should be the top result
        assert!(!results.is_empty());
        assert_eq!(results[0].0, 42);
        assert!((results[0].1 - 1.0).abs() < 0.001); // Score should be ~1.0
    }

    #[test]
    fn test_recall() {
        // Generate 1000 random vectors
        let vectors: Vec<_> = (0..1000).map(|_| random_vector(128)).collect();
        let vectors_clone = vectors.clone();

        let mut hnsw = HNSW::new(16, 100, move |id| vectors_clone[id].clone());

        for i in 0..1000 {
            hnsw.insert(i as u64, &vectors[i]);
        }

        // Test recall with 10 random queries
        let mut total_recall = 0.0;
        let num_queries = 10;
        let k = 10;

        for _ in 0..num_queries {
            let query = random_vector(128);

            // Brute-force ground truth
            let mut ground_truth: Vec<(usize, f32)> = vectors
                .iter()
                .enumerate()
                .map(|(id, v)| (id, dot_product(&query, v)))
                .collect();
            ground_truth.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let ground_truth_ids: std::collections::HashSet<u64> =
                ground_truth.iter().take(k).map(|(id, _)| *id as u64).collect();

            // HNSW search
            let hnsw_results = hnsw.search(&query, k, 50);
            let hnsw_ids: std::collections::HashSet<u64> =
                hnsw_results.iter().map(|(id, _)| *id).collect();

            // Calculate recall
            let intersection = ground_truth_ids.intersection(&hnsw_ids).count();
            total_recall += intersection as f64 / k as f64;
        }

        let avg_recall = total_recall / num_queries as f64;
        println!("Average recall@{}: {:.2}%", k, avg_recall * 100.0);
        assert!(
            avg_recall > 0.8,
            "Recall should be > 80%, got {:.2}%",
            avg_recall * 100.0
        );
    }

    // ============== Edge Case Tests ==============

    #[test]
    fn test_search_empty_index() {
        let hnsw = HNSW::new(16, 100, |_id| vec![0.0f32; 128]);
        let query = random_vector(128);
        let results = hnsw.search(&query, 5, 50);
        assert!(
            results.is_empty(),
            "Search on empty index should return empty"
        );
    }

    #[test]
    fn test_search_k_greater_than_count() {
        let vectors: Vec<_> = (0..10).map(|_| random_vector(64)).collect();
        let vectors_clone = vectors.clone();

        let mut hnsw = HNSW::new(8, 50, move |id| vectors_clone[id].clone());
        for i in 0..10 {
            hnsw.insert(i as u64, &vectors[i]);
        }

        // Request more results than exist
        let query = random_vector(64);
        let results = hnsw.search(&query, 100, 50);

        // Should return all 10 vectors, not panic
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn test_search_k_zero() {
        let vectors: Vec<_> = (0..10).map(|_| random_vector(64)).collect();
        let vectors_clone = vectors.clone();

        let mut hnsw = HNSW::new(8, 50, move |id| vectors_clone[id].clone());
        for i in 0..10 {
            hnsw.insert(i as u64, &vectors[i]);
        }

        let query = random_vector(64);
        let results = hnsw.search(&query, 0, 50);

        // k=0 should return empty
        assert!(results.is_empty());
    }

    #[test]
    fn test_large_ef_value() {
        let vectors: Vec<_> = (0..50).map(|_| random_vector(64)).collect();
        let vectors_clone = vectors.clone();

        let mut hnsw = HNSW::new(8, 50, move |id| vectors_clone[id].clone());
        for i in 0..50 {
            hnsw.insert(i as u64, &vectors[i]);
        }

        // Very large ef should not panic
        let query = random_vector(64);
        let results = hnsw.search(&query, 5, 10000);

        assert!(!results.is_empty());
    }

    #[test]
    fn test_id_to_index_consistency() {
        let vectors: Vec<_> = (0..100).map(|_| random_vector(64)).collect();
        let vectors_clone = vectors.clone();

        let mut hnsw = HNSW::new(16, 50, move |id| vectors_clone[id].clone());
        for i in 0..100 {
            hnsw.insert(i as u64, &vectors[i]);
        }

        // Verify every node has correct id_to_index mapping
        for (idx, node) in hnsw.nodes.iter().enumerate() {
            let mapped_idx = hnsw.id_to_index.get(&node.id);
            assert_eq!(
                mapped_idx,
                Some(&idx),
                "id_to_index mismatch for node {} at index {}",
                node.id,
                idx
            );
        }
    }

    #[test]
    fn test_stats_accuracy() {
        let vectors: Vec<_> = (0..100).map(|_| random_vector(64)).collect();
        let vectors_clone = vectors.clone();

        let mut hnsw = HNSW::new(16, 50, move |id| vectors_clone[id].clone());
        for i in 0..100 {
            hnsw.insert(i as u64, &vectors[i]);
        }

        let stats = hnsw.stats();
        assert_eq!(stats.num_nodes, 100);
        assert_eq!(stats.m, 16);
        assert_eq!(stats.ef_construction, 50);
        assert!(stats.total_edges > 0, "Should have edges");
        assert!(!stats.layer_counts.is_empty(), "Should have layer counts");
    }

    #[test]
    fn test_single_vector_search() {
        let vectors = vec![random_vector(64)];
        let query = vectors[0].clone();
        let vectors_clone = vectors.clone();

        let mut hnsw = HNSW::new(8, 50, move |id| vectors_clone[id].clone());
        hnsw.insert(0, &vectors[0]);

        let results = hnsw.search(&query, 5, 50);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0);
        assert!((results[0].1 - 1.0).abs() < 0.001); // Exact match
    }

    #[test]
    fn test_search_self_similarity() {
        // Test that vectors indexed can find themselves
        let vectors: Vec<_> = (0..20).map(|_| random_vector(128)).collect();
        let vectors_clone = vectors.clone();

        let mut hnsw = HNSW::new(16, 100, move |id| vectors_clone[id].clone());
        for i in 0..20 {
            hnsw.insert(i as u64, &vectors[i]);
        }

        // Each vector should be its own top-1 result
        for i in 0..20 {
            let results = hnsw.search(&vectors[i], 1, 100);
            assert!(!results.is_empty(), "Should find at least one result");
            assert_eq!(
                results[0].0, i as u64,
                "Vector {} should find itself as top result",
                i
            );
        }
    }

    #[test]
    fn test_deterministic_with_same_vectors() {
        // Two identical indices should give same search results
        let vectors: Vec<_> = (0..50).map(|_| random_vector(64)).collect();
        let vectors1 = vectors.clone();
        let vectors2 = vectors.clone();

        let mut hnsw1 = HNSW::new(16, 50, move |id| vectors1[id].clone());
        let mut hnsw2 = HNSW::new(16, 50, move |id| vectors2[id].clone());

        for i in 0..50 {
            hnsw1.insert(i as u64, &vectors[i]);
            hnsw2.insert(i as u64, &vectors[i]);
        }

        // Note: Due to RNG in layer assignment, graphs may differ
        // But both should find the same vector given exact match query
        let query = vectors[25].clone();
        let results1 = hnsw1.search(&query, 1, 100);
        let results2 = hnsw2.search(&query, 1, 100);

        assert_eq!(results1[0].0, 25);
        assert_eq!(results2[0].0, 25);
    }

    #[test]
    fn test_search_filtered_basic() {
        let vectors: Vec<_> = (0..100).map(|_| random_vector(128)).collect();
        let query = vectors[42].clone();
        let vectors_clone = vectors.clone();

        let mut hnsw = HNSW::new(16, 100, move |id| vectors_clone[id].clone());
        for i in 0..100 {
            hnsw.insert(i as u64, &vectors[i]);
        }

        // Filter to only even IDs
        let results = hnsw.search_filtered(&query, 5, 100, |id| id % 2 == 0);

        // All results should have even IDs
        for (id, _score) in &results {
            assert_eq!(
                *id % 2,
                0,
                "Filtered results should all be even, got {}",
                id
            );
        }
        // Vector 42 is even, so it should be the top result
        assert!(!results.is_empty());
        assert_eq!(results[0].0, 42);
    }

    #[test]
    fn test_search_filtered_empty_predicate() {
        let vectors: Vec<_> = (0..50).map(|_| random_vector(64)).collect();
        let vectors_clone = vectors.clone();

        let mut hnsw = HNSW::new(8, 50, move |id| vectors_clone[id].clone());
        for i in 0..50 {
            hnsw.insert(i as u64, &vectors[i]);
        }

        // Filter that rejects everything
        let query = random_vector(64);
        let results = hnsw.search_filtered(&query, 5, 100, |_| false);
        assert!(
            results.is_empty(),
            "Should be empty when predicate rejects all"
        );
    }

    #[test]
    fn test_search_subsequence_basic() {
        // Create a sequence of vectors where IDs 10..15 form a distinct sequence
        let vectors: Vec<_> = (0..50).map(|_| random_vector(128)).collect();
        let vectors_clone = vectors.clone();

        let mut hnsw = HNSW::new(16, 100, move |id| vectors_clone[id].clone());
        for i in 0..50 {
            hnsw.insert(i as u64, &vectors[i]);
        }

        // Query with the exact sub-sequence at positions 10..15
        let query_seq: Vec<Vec<f32>> = (10..15).map(|i| vectors[i].clone()).collect();
        let results = hnsw.search_subsequence(&query_seq, 3, 100, 50);

        // The top result should start at index 10
        assert!(!results.is_empty(), "Should find at least one sub-sequence");
        assert_eq!(results[0].0, 10, "Best sub-sequence should start at 10");
        assert!(
            results[0].1 > 0.95,
            "Exact match should have high similarity"
        );
    }

    #[test]
    fn test_search_subsequence_empty() {
        let hnsw = HNSW::new(16, 100, |_id| vec![0.0f32; 128]);

        let query_seq: Vec<Vec<f32>> = vec![random_vector(128)];
        let results = hnsw.search_subsequence(&query_seq, 5, 100, 0);
        assert!(results.is_empty(), "Empty index should return no results");
    }

    #[test]
    fn test_search_subsequence_bounds_are_safe() {
        let vectors: Vec<_> = (0..5).map(|_| random_vector(16)).collect();
        let vectors_clone = vectors.clone();
        let mut hnsw = HNSW::new(16, 100, move |id| vectors_clone[id].clone());
        for i in 0..vectors.len() {
            hnsw.insert(i as u64, &vectors[i]);
        }

        // Query length exceeds total vectors; function must return without panic.
        let query_seq: Vec<Vec<f32>> = (0..8).map(|_| random_vector(16)).collect();
        let results = hnsw.search_subsequence(&query_seq, 3, 50, vectors.len());
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_subsequence_multi_probe_results_are_deduped_and_bounded() {
        fn deterministic_vector(i: usize, dim: usize) -> Vec<f32> {
            let mut out = Vec::with_capacity(dim);
            for j in 0..dim {
                let x = (((i + 1) as f32) * ((j + 1) as f32) * 0.173).sin();
                out.push(x);
            }
            l2_normalized(&out)
        }

        let dim = 32usize;
        let vectors: Vec<_> = (0..120).map(|i| deterministic_vector(i, dim)).collect();
        let vectors_clone = vectors.clone();

        let mut hnsw = HNSW::new(16, 120, move |id| vectors_clone[id].clone());
        for i in 0..vectors.len() {
            hnsw.insert(i as u64, &vectors[i]);
        }

        let mut query_seq: Vec<Vec<f32>> = (40..52).map(|i| vectors[i].clone()).collect();
        // Inject a single high-norm outlier so the highest-salience anchor is likely wrong.
        query_seq[6] = vec![50.0; dim];

        let results = hnsw.search_subsequence(&query_seq, 5, 300, vectors.len());
        assert!(results.len() <= 5, "results must be bounded by k");
        let mut seen = std::collections::HashSet::new();
        assert!(
            results.iter().all(|(start, _)| seen.insert(*start)),
            "multi-probe merge should dedupe repeated start_ids"
        );
    }

    #[test]
    fn test_search_with_sparse_ids_is_safe() {
        let ids = [0u64, 1024, 1028, 2056];
        let mut map = HashMap::new();
        for id in ids {
            map.insert(id, random_vector(32));
        }

        let map_for_accessor = map.clone();
        let zero = vec![0.0f32; 32];
        let ordered_ids = [0u64, 1024, 1028, 2056];
        let mut hnsw = HNSW::new_with_accessor(16, 100, move |node_idx, sink| {
            let id = ordered_ids[node_idx];
            if let Some(v) = map_for_accessor.get(&id) {
                sink(v);
            } else {
                sink(&zero);
            }
        });
        for id in ordered_ids {
            let query = map.get(&id).expect("sparse id must exist");
            hnsw.insert(id, query);
        }

        let query = map.get(&1028).unwrap().clone();
        let results = hnsw.search(&query, 3, 128);
        assert!(
            !results.is_empty(),
            "sparse-id search should return results"
        );
        assert_eq!(
            results[0].0, 1028,
            "exact sparse-id vector should be top-1 result"
        );
    }

    #[test]
    fn test_search_subsequence_predicate_blocks_holes() {
        let vectors: Vec<Vec<f32>> = (0..40)
            .map(|i| {
                let mut v = vec![0.0f32; 8];
                v[i % 8] = 1.0;
                v
            })
            .collect();
        let vectors_clone = vectors.clone();
        let mut hnsw = HNSW::new(8, 64, move |id| vectors_clone[id].clone());
        for i in 0..vectors.len() {
            hnsw.insert(i as u64, &vectors[i]);
        }

        let query_seq = vectors[10..15].to_vec();
        let open = hnsw.search_subsequence(&query_seq, 5, 100, vectors.len());
        assert!(
            open.iter().any(|(start, _)| *start == 10),
            "baseline subsequence should find the exact start"
        );

        let blocked =
            hnsw.search_subsequence_with_predicate(&query_seq, 5, 100, vectors.len(), |id| {
                id != 12
            });
        assert!(
            blocked.iter().all(|(start, _)| *start != 10),
            "predicate hole should invalidate exact sequence alignment"
        );
    }
}
