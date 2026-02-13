//! Node representation in the HNSW graph

use serde::{Deserialize, Serialize};

/// A node in the HNSW graph
///
/// Each node exists on one or more layers. Layer 0 (base layer) contains all nodes.
/// Higher layers contain progressively fewer nodes for "express" navigation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    /// Index into the VectorStore (not the vector itself - zero-copy)
    pub id: usize,

    /// Neighbors at each layer
    /// - `layers[0]` = neighbors at layer 0 (base, everyone)
    /// - `layers[n]` = neighbors at layer n (express, fewer nodes)
    pub layers: Vec<Vec<usize>>,
}

impl Node {
    /// Create a new node with the given ID and number of layers
    pub fn new(id: usize, max_layer: usize) -> Self {
        Self {
            id,
            layers: vec![Vec::new(); max_layer + 1],
        }
    }

    /// Get the maximum layer this node exists on
    pub fn max_layer(&self) -> usize {
        self.layers.len().saturating_sub(1)
    }

    /// Get neighbors at a specific layer
    pub fn neighbors(&self, layer: usize) -> &[usize] {
        self.layers.get(layer).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Get mutable neighbors at a specific layer
    pub fn neighbors_mut(&mut self, layer: usize) -> Option<&mut Vec<usize>> {
        self.layers.get_mut(layer)
    }

    /// Add a neighbor at a specific layer
    pub fn add_neighbor(&mut self, layer: usize, neighbor_id: usize) {
        if let Some(neighbors) = self.layers.get_mut(layer) {
            if !neighbors.contains(&neighbor_id) {
                neighbors.push(neighbor_id);
            }
        }
    }

    /// Remove a neighbor at a specific layer
    pub fn remove_neighbor(&mut self, layer: usize, neighbor_id: usize) {
        if let Some(neighbors) = self.layers.get_mut(layer) {
            neighbors.retain(|&n| n != neighbor_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_creation() {
        let node = Node::new(42, 3);
        assert_eq!(node.id, 42);
        assert_eq!(node.layers.len(), 4); // Layers 0, 1, 2, 3
        assert_eq!(node.max_layer(), 3);
    }

    #[test]
    fn test_add_neighbor() {
        let mut node = Node::new(0, 2);
        node.add_neighbor(0, 1);
        node.add_neighbor(0, 2);
        node.add_neighbor(1, 3);

        assert_eq!(node.neighbors(0), &[1, 2]);
        assert_eq!(node.neighbors(1), &[3]);
        assert_eq!(node.neighbors(2), &[] as &[usize]);
    }

    #[test]
    fn test_remove_neighbor() {
        let mut node = Node::new(0, 1);
        node.add_neighbor(0, 1);
        node.add_neighbor(0, 2);
        node.add_neighbor(0, 3);

        node.remove_neighbor(0, 2);
        assert_eq!(node.neighbors(0), &[1, 3]);
    }

    #[test]
    fn test_no_duplicate_neighbors() {
        let mut node = Node::new(0, 0);
        node.add_neighbor(0, 1);
        node.add_neighbor(0, 1); // Duplicate

        assert_eq!(node.neighbors(0), &[1]);
    }
}
