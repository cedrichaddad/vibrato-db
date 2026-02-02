//! HNSW serialization using bincode
//!
//! Saves and loads the graph structure to/from disk.

use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::node::Node;

#[derive(Error, Debug)]
pub enum SerializeError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Bincode error: {0}")]
    Bincode(#[from] bincode::Error),
}

/// Serializable HNSW structure (without the vector accessor)
#[derive(Serialize, Deserialize)]
pub struct HNSWData {
    pub nodes: Vec<Node>,
    pub entry_point: Option<usize>,
    pub max_layer: usize,
    pub m: usize,
    pub m0: usize,
    pub ml: f64,
    pub ef_construction: usize,
}

impl HNSWData {
    /// Save to a file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), SerializeError> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, self)?;
        Ok(())
    }

    /// Load from a file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, SerializeError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let data = bincode::deserialize_from(reader)?;
        Ok(data)
    }
}

impl super::HNSW {
    /// Export serializable data
    pub fn to_data(&self) -> HNSWData {
        HNSWData {
            nodes: self.nodes.clone(),
            entry_point: self.entry_point,
            max_layer: self.max_layer,
            m: self.m,
            m0: self.m0,
            ml: self.ml,
            ef_construction: self.ef_construction,
        }
    }

    /// Import from serialized data
    pub fn from_data<F>(data: HNSWData, vector_fn: F) -> Self
    where
        F: Fn(usize) -> Vec<f32> + Send + Sync + 'static,
    {
        Self::from_parts(
            data.nodes,
            data.entry_point,
            data.max_layer,
            data.m,
            data.m0,
            data.ml,
            data.ef_construction,
            vector_fn,
        )
    }

    /// Save index to file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), SerializeError> {
        self.to_data().save(path)
    }

    /// Load index from file
    pub fn load<P: AsRef<Path>, F>(path: P, vector_fn: F) -> Result<Self, SerializeError>
    where
        F: Fn(usize) -> Vec<f32> + Send + Sync + 'static,
    {
        let data = HNSWData::load(path)?;
        Ok(Self::from_data(data, vector_fn))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::l2_normalized;
    use tempfile::tempdir;

    fn random_vector(dim: usize) -> Vec<f32> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect();
        l2_normalized(&v)
    }

    #[test]
    fn test_save_load_roundtrip() {
        let vectors: Vec<_> = (0..100).map(|_| random_vector(64)).collect();
        let vectors_clone = vectors.clone();
        let vectors_clone2 = vectors.clone();

        // Create and populate index
        let mut hnsw = super::super::HNSW::new(16, 50, move |id| vectors_clone[id].clone());
        for i in 0..100 {
            hnsw.insert(i);
        }

        // Save
        let dir = tempdir().unwrap();
        let path = dir.path().join("index.idx");
        hnsw.save(&path).unwrap();

        // Load
        let loaded = super::super::HNSW::load(&path, move |id| vectors_clone2[id].clone()).unwrap();

        // Verify
        assert_eq!(loaded.nodes.len(), hnsw.nodes.len());
        assert_eq!(loaded.entry_point, hnsw.entry_point);
        assert_eq!(loaded.max_layer, hnsw.max_layer);

        // Test search still works
        let query = &vectors[42];
        let results = loaded.search(query, 5, 50);
        assert!(!results.is_empty());
        assert_eq!(results[0].0, 42);
    }
}
