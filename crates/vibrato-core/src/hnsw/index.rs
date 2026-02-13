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

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

use super::node::Node;
use super::visited::VisitedGuard;
use crate::simd::dot_product;


/// Candidate for search (min-heap)
#[derive(Clone, Copy)]
struct Candidate {
    id: usize,
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
    id: usize,
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

/// HNSW Index
///
/// A hierarchical graph structure for approximate nearest neighbor search.
pub struct HNSW {
    /// All nodes in the graph
    pub nodes: Vec<Node>,

    /// Map from node ID to index in nodes vector (O(1) lookup)
    id_to_index: HashMap<usize, usize>,

    /// Entry point node ID (node on the highest layer)
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

/// Type-erased vector accessor
type VectorAccessor = Box<dyn Fn(usize) -> Vec<f32> + Send + Sync>;

impl HNSW {
    /// Create a new HNSW index
    ///
    /// # Parameters
    /// - `m`: Max neighbors per layer (typically 12-48)
    /// - `ef_construction`: Search depth during build (typically 100-200)
    /// - `vector_fn`: Function to get vector by ID
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
        Self {
            nodes: Vec::new(),
            id_to_index: HashMap::new(),
            entry_point: None,
            max_layer: 0,
            m,
            m0: m * 2,
            ml: 1.0 / (m as f64).ln(),
            ef_construction,
            vectors: Box::new(vector_fn),
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Reconstruct an HNSW index from its parts (used by serialization)
    pub(crate) fn from_parts<F>(
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
        F: Fn(usize) -> Vec<f32> + Send + Sync + 'static,
    {
        // Build id_to_index map from existing nodes
        let id_to_index: HashMap<usize, usize> = nodes
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
            vectors: Box::new(vector_fn),
            rng: StdRng::seed_from_u64(rand::random()),
        }
    }

    /// Get node index by ID (O(1) lookup)
    #[inline]
    fn get_node_index(&self, id: usize) -> Option<usize> {
        self.id_to_index.get(&id).copied()
    }

    /// Get node by ID (O(1) lookup)
    #[inline]
    fn get_node(&self, id: usize) -> Option<&Node> {
        self.id_to_index.get(&id).map(|&idx| &self.nodes[idx])
    }

    /// Get mutable node by ID (O(1) lookup)
    #[inline]
    fn get_node_mut(&mut self, id: usize) -> Option<&mut Node> {
        if let Some(&idx) = self.id_to_index.get(&id) {
            Some(&mut self.nodes[idx])
        } else {
            None
        }
    }

    /// Get vector for a node
    #[inline]
    fn get_vector(&self, id: usize) -> Vec<f32> {
        (self.vectors)(id)
    }

    /// Compute distance between query and node (dot product for normalized vectors)
    /// Returns negative dot product so smaller = more similar
    #[inline]
    fn distance(&self, query: &[f32], node_id: usize) -> f32 {
        let node_vec = self.get_vector(node_id);
        // For normalized vectors, higher dot product = more similar
        // We return 1 - dot_product to convert to a distance metric
        1.0 - dot_product(query, &node_vec)
    }

    /// Assign a random layer based on exponential distribution
    fn random_layer(&mut self) -> usize {
        let r: f64 = self.rng.gen();
        (-r.ln() * self.ml).floor() as usize
    }

    /// Insert a vector into the index
    ///
    /// # Parameters
    /// - `id`: Vector ID (index into VectorStore)
    pub fn insert(&mut self, id: usize) {
        let query = self.get_vector(id);
        let node_layer = self.random_layer();

        // Create node
        let mut node = Node::new(id, node_layer);

        // First node becomes entry point
        if self.entry_point.is_none() {
            self.entry_point = Some(id);
            self.max_layer = node_layer;
            self.id_to_index.insert(id, 0);  // First node always at index 0
            self.nodes.push(node);
            return;
        }

        let entry_point = self.entry_point.unwrap();
        let mut current_node = entry_point;

        // Phase 1: Zoom in from top layer to node_layer + 1
        // Greedy search, single best neighbor per layer
        for layer in (node_layer + 1..=self.max_layer).rev() {
            let nearest = self.search_layer(&query, &[current_node], 1, layer);
            if let Some((nearest_id, _)) = nearest.first() {
                current_node = *nearest_id;
            }
        }

        // Phase 2: Insert at layers from min(node_layer, max_layer) down to 0
        // We collect all the updates needed, then apply them
        let start_layer = node_layer.min(self.max_layer);
        
        // Collect updates to apply: (node_index, layer, neighbor_id_to_add)
        let mut reverse_edges: Vec<(usize, usize, usize)> = Vec::new();
        // Collect pruning operations: (node_index, layer, new_neighbors)
        let mut prune_ops: Vec<(usize, usize, Vec<usize>)> = Vec::new();

        for layer in (0..=start_layer).rev() {
            let m_layer = if layer == 0 { self.m0 } else { self.m };

            // Find candidates for this layer
            let candidates = self.search_layer(&query, &[current_node], self.ef_construction, layer);

            // Select neighbors using the heuristic
            let neighbors = self.select_neighbors(&query, &candidates, m_layer);

            // Add forward edges to new node
            for &(neighbor_id, _) in &neighbors {
                node.add_neighbor(layer, neighbor_id);

                // Find the node index for this neighbor using O(1) HashMap
                if let Some(node_idx) = self.get_node_index(neighbor_id) {
                    reverse_edges.push((node_idx, layer, id));
                    
                    // Check if pruning will be needed
                    let current_neighbors = self.nodes[node_idx].neighbors(layer);
                    if current_neighbors.len() >= m_layer {
                        // Will need pruning after adding new edge
                        let neighbor_vec = self.get_vector(neighbor_id);
                        let mut all_neighbors: Vec<usize> = current_neighbors.to_vec();
                        all_neighbors.push(id);
                        
                        // Compute distances and select
                        let neighbor_candidates: Vec<(usize, f32)> = all_neighbors
                            .iter()
                            .map(|&n| {
                                let v = self.get_vector(n);
                                (n, 1.0 - dot_product(&neighbor_vec, &v))
                            })
                            .collect();
                        
                        let pruned = self.select_neighbors(&neighbor_vec, &neighbor_candidates, m_layer);
                        let pruned_ids: Vec<usize> = pruned.iter().map(|(id, _)| *id).collect();
                        prune_ops.push((node_idx, layer, pruned_ids));
                    }
                }
            }

            // Use first candidate as entry for next layer
            if let Some((first_id, _)) = candidates.first() {
                current_node = *first_id;
            }
        }

        // Apply reverse edges (avoiding those that will be overwritten by pruning)
        let prune_targets: std::collections::HashSet<(usize, usize)> = 
            prune_ops.iter().map(|(idx, layer, _)| (*idx, *layer)).collect();
        
        for (node_idx, layer, neighbor_id) in reverse_edges {
            if !prune_targets.contains(&(node_idx, layer)) {
                self.nodes[node_idx].add_neighbor(layer, neighbor_id);
            }
        }

        // Apply pruning operations
        for (node_idx, layer, new_neighbors) in prune_ops {
            if let Some(layer_neighbors) = self.nodes[node_idx].neighbors_mut(layer) {
                layer_neighbors.clear();
                layer_neighbors.extend(new_neighbors);
            }
        }

        // Update entry point if new node has higher layer
        if node_layer > self.max_layer {
            self.max_layer = node_layer;
            self.entry_point = Some(id);
        }

        // Add to id_to_index map before pushing
        let node_idx = self.nodes.len();
        self.id_to_index.insert(id, node_idx);
        self.nodes.push(node);
    }

    /// Search for nearest neighbors on a single layer
    ///
    /// Greedy beam search with `ef` candidates.
    fn search_layer(
        &self,
        query: &[f32],
        entry_points: &[usize],
        ef: usize,
        layer: usize,
    ) -> Vec<(usize, f32)> {
        let mut visited = VisitedGuard::new(self.nodes.len().max(1024));

        // Candidates to explore (min-heap by distance)
        // Pre-allocate for performance
        let mut candidates: BinaryHeap<Candidate> = BinaryHeap::with_capacity(ef);

        // Found neighbors (max-heap to track worst)
        let mut results: BinaryHeap<SearchResult> = BinaryHeap::with_capacity(ef + 1);

        // Initialize with entry points
        for &ep in entry_points {
            if !visited.is_visited(ep) {
                visited.visit(ep);
                let dist = self.distance(query, ep);
                candidates.push(Candidate { id: ep, distance: dist });
                results.push(SearchResult { id: ep, distance: dist });
            }
        }

        // Greedy search
        while let Some(current) = candidates.pop() {
            // Pruning: if current is worse than worst result, stop
            if let Some(worst) = results.peek() {
                if current.distance > worst.distance && results.len() >= ef {
                    break;
                }
            }

            // Explore neighbors using O(1) HashMap lookup
            if let Some(node) = self.get_node(current.id) {
                // PERF NOTE: prefetching Node structs here is ineffective because
                // Node contains Vec<Vec<usize>> — the actual neighbor data lives
                // on the heap behind pointers, not inline in the struct.
                // True fix: linearize the graph into a flat Vec<u32> (V3 scope).
                for &neighbor_id in node.neighbors(layer) {
                    if visited.is_visited(neighbor_id) {
                        continue;
                    }
                    visited.visit(neighbor_id);

                    let dist = self.distance(query, neighbor_id);

                    // Add to candidates if promising
                    let dominated = results.len() >= ef && dist > results.peek().unwrap().distance;
                    if !dominated {
                        candidates.push(Candidate {
                            id: neighbor_id,
                            distance: dist,
                        });
                        results.push(SearchResult {
                            id: neighbor_id,
                            distance: dist,
                        });

                        // Keep only top ef results
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }

        // Convert to sorted vec
        let mut result_vec: Vec<_> = results.into_iter().map(|r| (r.id, r.distance)).collect();
        result_vec.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        result_vec
    }

    /// Select neighbors using diversity-preserving heuristic
    ///
    /// Instead of just picking the closest M neighbors, we ensure diversity:
    /// A candidate is added only if it's closer to query than to any selected neighbor.
    fn select_neighbors(
        &self,
        _query: &[f32],
        candidates: &[(usize, f32)],
        m: usize,
    ) -> Vec<(usize, f32)> {
        if candidates.is_empty() {
            return Vec::new();
        }

        // Sort by distance to query
        let mut sorted: Vec<_> = candidates.to_vec();
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        let mut result = Vec::with_capacity(m);

        for &(candidate_id, candidate_dist) in &sorted {
            if result.len() >= m {
                break;
            }

            // Fetch candidate vector once (outside inner loop)
            let candidate_vec = self.get_vector(candidate_id);

            // Check if candidate is closer to query than to any existing result
            let mut is_diverse = true;
            for &(existing_id, _) in &result {
                let existing_vec = self.get_vector(existing_id);
                let dist_to_existing = 1.0 - dot_product(&candidate_vec, &existing_vec);

                if dist_to_existing < candidate_dist {
                    // Candidate is closer to existing neighbor than to query
                    // This means the existing neighbor "covers" this direction
                    is_diverse = false;
                    break;
                }
            }

            if is_diverse {
                result.push((candidate_id, candidate_dist));
            }
        }

        // If we don't have enough diverse neighbors, fill with closest
        if result.len() < m {
            for &(candidate_id, candidate_dist) in &sorted {
                if result.len() >= m {
                    break;
                }
                if !result.iter().any(|(id, _)| *id == candidate_id) {
                    result.push((candidate_id, candidate_dist));
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
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<(usize, f32)> {
        if self.entry_point.is_none() {
            return Vec::new();
        }

        let entry_point = self.entry_point.unwrap();
        let mut current_node = entry_point;

        // Phase 1: Greedy descent from top layer to layer 1
        for layer in (1..=self.max_layer).rev() {
            let nearest = self.search_layer(query, &[current_node], 1, layer);
            if let Some((nearest_id, _)) = nearest.first() {
                current_node = *nearest_id;
            }
        }

        // Phase 2: Beam search on layer 0
        let candidates = self.search_layer(query, &[current_node], ef.max(k), 0);

        // Return top k, convert distance to similarity score
        candidates
            .into_iter()
            .take(k)
            .map(|(id, dist)| (id, 1.0 - dist)) // Convert back to similarity
            .collect()
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
    ) -> Vec<(usize, f32)>
    where
        P: Fn(usize) -> bool,
    {
        if self.entry_point.is_none() {
            return Vec::new();
        }

        let entry_point = self.entry_point.unwrap();
        let mut current_node = entry_point;

        // Phase 1: Greedy descent (unfiltered — we don't filter hub nodes)
        for layer in (1..=self.max_layer).rev() {
            let nearest = self.search_layer(query, &[current_node], 1, layer);
            if let Some((nearest_id, _)) = nearest.first() {
                current_node = *nearest_id;
            }
        }

        // Phase 2: Over-fetch beam search on layer 0
        // We fetch ef*2 candidates because many will be filtered out
        let over_fetch = (ef * 2).max(k * 4);
        let candidates = self.search_layer(query, &[current_node], over_fetch, 0);

        // Apply predicate filter and return top k
        candidates
            .into_iter()
            .filter(|(id, _)| predicate(*id))
            .take(k)
            .map(|(id, dist)| (id, 1.0 - dist))
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
    ) -> Vec<(usize, f32)> {
        if query_sequence.is_empty() || self.entry_point.is_none() {
            return Vec::new();
        }

        let seq_len = query_sequence.len();

        // Step 1: Find the most salient vector (highest L2 norm)
        let (salient_offset, _) = query_sequence
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let norm_sq: f32 = v.iter().map(|x| x * x).sum();
                (i, norm_sq)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
            .unwrap();

        let salient_query = &query_sequence[salient_offset];

        // Step 2: HNSW search for the salient vector (over-fetch for candidates)
        let anchor_candidates = self.search(salient_query, k * 4, ef);

        // Step 3: For each candidate anchor, compute the start_id and verify
        // the full sequence alignment via brute-force
        let mut sequence_results: Vec<(usize, f32)> = Vec::new();

        for (anchor_id, _anchor_score) in &anchor_candidates {
            // The anchor matches query_sequence[salient_offset],
            // so the sequence would start at anchor_id - salient_offset
            let start_id = if *anchor_id >= salient_offset {
                anchor_id - salient_offset
            } else {
                continue; // Can't fit the sequence
            };

            // Check bounds: sequence must fit within total vectors
            if start_id + seq_len > total_vectors {
                continue;
            }

            // Brute-force verify: compute average similarity across all positions
            let mut total_sim = 0.0f32;
            for (offset, query_vec) in query_sequence.iter().enumerate() {
                let vec_id = start_id + offset;
                let stored_vec = self.get_vector(vec_id);
                total_sim += dot_product(query_vec, &stored_vec);
            }
            let avg_sim = total_sim / seq_len as f32;

            sequence_results.push((start_id, avg_sim));
        }

        // Sort by similarity (descending) and return top k
        sequence_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        sequence_results.truncate(k);
        sequence_results
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
        hnsw.insert(0);

        assert_eq!(hnsw.nodes.len(), 1);
        assert_eq!(hnsw.entry_point, Some(0));
    }

    #[test]
    fn test_insert_multiple() {
        let vectors: Vec<_> = (0..100).map(|_| random_vector(128)).collect();
        let vectors_clone = vectors.clone();

        let mut hnsw = HNSW::new(16, 100, move |id| vectors_clone[id].clone());

        for i in 0..100 {
            hnsw.insert(i);
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
            hnsw.insert(i);
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
            hnsw.insert(i);
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
            let ground_truth_ids: std::collections::HashSet<_> =
                ground_truth.iter().take(k).map(|(id, _)| *id).collect();

            // HNSW search
            let hnsw_results = hnsw.search(&query, k, 50);
            let hnsw_ids: std::collections::HashSet<_> =
                hnsw_results.iter().map(|(id, _)| *id).collect();

            // Calculate recall
            let intersection = ground_truth_ids.intersection(&hnsw_ids).count();
            total_recall += intersection as f64 / k as f64;
        }

        let avg_recall = total_recall / num_queries as f64;
        println!("Average recall@{}: {:.2}%", k, avg_recall * 100.0);
        assert!(avg_recall > 0.8, "Recall should be > 80%, got {:.2}%", avg_recall * 100.0);
    }

    // ============== Edge Case Tests ==============

    #[test]
    fn test_search_empty_index() {
        let hnsw = HNSW::new(16, 100, |_id| vec![0.0f32; 128]);
        let query = random_vector(128);
        let results = hnsw.search(&query, 5, 50);
        assert!(results.is_empty(), "Search on empty index should return empty");
    }

    #[test]
    fn test_search_k_greater_than_count() {
        let vectors: Vec<_> = (0..10).map(|_| random_vector(64)).collect();
        let vectors_clone = vectors.clone();
        
        let mut hnsw = HNSW::new(8, 50, move |id| vectors_clone[id].clone());
        for i in 0..10 {
            hnsw.insert(i);
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
            hnsw.insert(i);
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
            hnsw.insert(i);
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
            hnsw.insert(i);
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
            hnsw.insert(i);
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
        hnsw.insert(0);
        
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
            hnsw.insert(i);
        }
        
        // Each vector should be its own top-1 result
        for i in 0..20 {
            let results = hnsw.search(&vectors[i], 1, 100);
            assert!(!results.is_empty(), "Should find at least one result");
            assert_eq!(results[0].0, i, "Vector {} should find itself as top result", i);
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
            hnsw1.insert(i);
            hnsw2.insert(i);
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
            hnsw.insert(i);
        }

        // Filter to only even IDs
        let results = hnsw.search_filtered(&query, 5, 100, |id| id % 2 == 0);

        // All results should have even IDs
        for (id, _score) in &results {
            assert_eq!(*id % 2, 0, "Filtered results should all be even, got {}", id);
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
            hnsw.insert(i);
        }

        // Filter that rejects everything
        let query = random_vector(64);
        let results = hnsw.search_filtered(&query, 5, 100, |_| false);
        assert!(results.is_empty(), "Should be empty when predicate rejects all");
    }

    #[test]
    fn test_search_subsequence_basic() {
        // Create a sequence of vectors where IDs 10..15 form a distinct sequence
        let vectors: Vec<_> = (0..50).map(|_| random_vector(128)).collect();
        let vectors_clone = vectors.clone();

        let mut hnsw = HNSW::new(16, 100, move |id| vectors_clone[id].clone());
        for i in 0..50 {
            hnsw.insert(i);
        }

        // Query with the exact sub-sequence at positions 10..15
        let query_seq: Vec<Vec<f32>> = (10..15).map(|i| vectors[i].clone()).collect();
        let results = hnsw.search_subsequence(&query_seq, 3, 100, 50);

        // The top result should start at index 10
        assert!(!results.is_empty(), "Should find at least one sub-sequence");
        assert_eq!(results[0].0, 10, "Best sub-sequence should start at 10");
        assert!(results[0].1 > 0.95, "Exact match should have high similarity");
    }

    #[test]
    fn test_search_subsequence_empty() {
        let hnsw = HNSW::new(16, 100, |_id| vec![0.0f32; 128]);

        let query_seq: Vec<Vec<f32>> = vec![random_vector(128)];
        let results = hnsw.search_subsequence(&query_seq, 5, 100, 0);
        assert!(results.is_empty(), "Empty index should return no results");
    }
}
