//! Product Quantization (PQ)
//!
//! Compresses 128-dimensional f32 vectors (512 bytes) to 16 bytes using 16 subspaces
//! with 256 centroids each. Distance computation uses Asymmetric Distance Computation (ADC)
//! with pre-computed lookup tables.
//!
//! # Architecture
//!
//! ```text
//! Vector [f32; 128] → split into 16 sub-vectors of 8 floats each
//!                      ↓
//! Each sub-vector → quantized to nearest centroid index (u8)
//!                      ↓
//! PQ Code [u8; 16]  (32x compression)
//! ```
//!
//! # Distance Computation (ADC)
//!
//! ```text
//! Query → compute distance table: [16 subspaces × 256 distances]
//!       → for any PQ code, distance = sum of 16 table lookups
//! ```

use crate::simd;

/// Default number of subspaces for PQ
pub const DEFAULT_NUM_SUBSPACES: usize = 16;

/// Number of centroids per subspace (fixed at 256 for u8 codes)
pub const NUM_CENTROIDS: usize = 256;

/// Product Quantizer
///
/// After training, can encode vectors to PQ codes and compute distances.
#[derive(Clone)]
pub struct ProductQuantizer {
    /// Number of subspaces (typically 16)
    pub num_subspaces: usize,

    /// Dimension of the original vector
    pub dimension: usize,

    /// Dimension of each sub-vector (dimension / num_subspaces)
    pub sub_dimension: usize,

    /// Codebook: [num_subspaces][NUM_CENTROIDS][sub_dimension]
    /// Flattened as: codebook[subspace * NUM_CENTROIDS * sub_dim + centroid * sub_dim + d]
    pub codebook: Vec<f32>,
}

impl ProductQuantizer {
    /// Create a new ProductQuantizer with a pre-trained codebook
    ///
    /// # Panics
    /// Panics if `dimension` is not evenly divisible by `num_subspaces`
    pub fn new(dimension: usize, num_subspaces: usize, codebook: Vec<f32>) -> Self {
        assert!(
            dimension % num_subspaces == 0,
            "Dimension {} must be divisible by num_subspaces {}",
            dimension,
            num_subspaces
        );
        let sub_dimension = dimension / num_subspaces;
        let expected_codebook_size = num_subspaces * NUM_CENTROIDS * sub_dimension;
        assert_eq!(
            codebook.len(),
            expected_codebook_size,
            "Codebook size mismatch: expected {}, got {}",
            expected_codebook_size,
            codebook.len()
        );

        Self {
            num_subspaces,
            dimension,
            sub_dimension,
            codebook,
        }
    }

    /// Get centroid vector for a given subspace and centroid index
    #[inline]
    fn centroid(&self, subspace: usize, centroid_idx: u8) -> &[f32] {
        let offset = (subspace * NUM_CENTROIDS + centroid_idx as usize) * self.sub_dimension;
        &self.codebook[offset..offset + self.sub_dimension]
    }

    /// Encode a single vector into PQ codes
    ///
    /// For each subspace, finds the nearest centroid and stores its index.
    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        debug_assert_eq!(vector.len(), self.dimension);
        let mut codes = Vec::with_capacity(self.num_subspaces);

        for s in 0..self.num_subspaces {
            let sub_vec = &vector[s * self.sub_dimension..(s + 1) * self.sub_dimension];
            let mut best_idx = 0u8;
            let mut best_dist = f32::MAX;

            for c in 0..NUM_CENTROIDS {
                let centroid = self.centroid(s, c as u8);
                let dist = simd::l2_distance_squared(sub_vec, centroid);
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = c as u8;
                }
            }

            codes.push(best_idx);
        }

        codes
    }

    /// Encode multiple vectors into PQ codes
    pub fn encode_batch(&self, vectors: &[Vec<f32>]) -> Vec<Vec<u8>> {
        vectors.iter().map(|v| self.encode(v)).collect()
    }

    /// Pre-compute the ADC distance table for a query vector
    ///
    /// Returns a table of shape [num_subspaces][256] where
    /// table[s][c] = L2 distance between query sub-vector s and centroid c.
    ///
    /// This table is reused for all PQ codes in a single search.
    pub fn compute_distance_table(&self, query: &[f32]) -> Vec<f32> {
        debug_assert_eq!(query.len(), self.dimension);
        let mut table = vec![0.0f32; self.num_subspaces * NUM_CENTROIDS];

        for s in 0..self.num_subspaces {
            let query_sub = &query[s * self.sub_dimension..(s + 1) * self.sub_dimension];
            let table_offset = s * NUM_CENTROIDS;

            for c in 0..NUM_CENTROIDS {
                let centroid = self.centroid(s, c as u8);
                table[table_offset + c] = simd::l2_distance_squared(query_sub, centroid);
            }
        }

        table
    }

    /// Compute ADC distance between a query (via pre-computed table) and a PQ code
    ///
    /// This is the hot path — just 16 table lookups summed.
    #[inline(always)]
    pub fn adc_distance(table: &[f32], codes: &[u8], num_subspaces: usize) -> f32 {
        debug_assert_eq!(codes.len(), num_subspaces);
        debug_assert_eq!(table.len(), num_subspaces * NUM_CENTROIDS);

        let mut distance = 0.0f32;
        for s in 0..num_subspaces {
            // Safety: codes[s] is u8 (0..255), table offset is s*256 + code
            distance += unsafe {
                *table.get_unchecked(s * NUM_CENTROIDS + *codes.get_unchecked(s) as usize)
            };
        }
        distance
    }

    /// Batch ADC: compute distances from one query to many PQ codes
    ///
    /// More efficient than calling `adc_distance` in a loop because the
    /// distance table stays hot in L1 cache.
    pub fn adc_distance_batch(
        table: &[f32],
        codes_batch: &[&[u8]],
        num_subspaces: usize,
    ) -> Vec<f32> {
        codes_batch
            .iter()
            .map(|codes| Self::adc_distance(table, codes, num_subspaces))
            .collect()
    }

    /// Reconstruct (approximate) the original vector from PQ codes
    ///
    /// Useful for debugging and visualization.
    pub fn decode(&self, codes: &[u8]) -> Vec<f32> {
        debug_assert_eq!(codes.len(), self.num_subspaces);
        let mut vector = Vec::with_capacity(self.dimension);

        for s in 0..self.num_subspaces {
            let centroid = self.centroid(s, codes[s]);
            vector.extend_from_slice(centroid);
        }

        vector
    }

    /// Serialize codebook to bytes for .vdb V2 format
    pub fn codebook_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.codebook)
    }

    /// Deserialize codebook from bytes
    pub fn from_codebook_bytes(
        dimension: usize,
        num_subspaces: usize,
        bytes: &[u8],
    ) -> Result<Self, PqError> {
        let codebook: &[f32] = bytemuck::try_cast_slice(bytes)
            .map_err(|_| PqError::AlignmentError)?;
        let sub_dimension = dimension / num_subspaces;
        let expected = num_subspaces * NUM_CENTROIDS * sub_dimension;
        if codebook.len() != expected {
            return Err(PqError::CodebookSizeMismatch {
                expected,
                actual: codebook.len(),
            });
        }
        Ok(Self::new(dimension, num_subspaces, codebook.to_vec()))
    }
}

/// Errors from PQ operations
#[derive(Debug, thiserror::Error)]
pub enum PqError {
    #[error("Codebook alignment error")]
    AlignmentError,

    #[error("Codebook size mismatch: expected {expected}, got {actual}")]
    CodebookSizeMismatch { expected: usize, actual: usize },

    #[error("Not enough training data: {count} vectors (minimum: {minimum})")]
    InsufficientData { count: usize, minimum: usize },
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    fn random_codebook(num_subspaces: usize, sub_dim: usize) -> Vec<f32> {
        let mut rng = rand::thread_rng();
        (0..num_subspaces * NUM_CENTROIDS * sub_dim)
            .map(|_| rng.gen::<f32>() - 0.5)
            .collect()
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let dim = 128;
        let nsub = 16;
        let sub_dim = dim / nsub;
        let codebook = random_codebook(nsub, sub_dim);
        let pq = ProductQuantizer::new(dim, nsub, codebook);

        // Create a vector that IS a centroid (exact match expected)
        let codes: Vec<u8> = (0..nsub).map(|s| (s % 256) as u8).collect();
        let original = pq.decode(&codes);
        let re_encoded = pq.encode(&original);

        assert_eq!(re_encoded, codes, "Centroid vector should encode back to same codes");
    }

    #[test]
    fn test_adc_distance_correctness() {
        let dim = 32;
        let nsub = 4;
        let sub_dim = dim / nsub;
        let codebook = random_codebook(nsub, sub_dim);
        let pq = ProductQuantizer::new(dim, nsub, codebook);

        let mut rng = rand::thread_rng();
        let query: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
        let codes: Vec<u8> = (0..nsub).map(|_| rng.gen::<u8>()).collect();

        // Compute ADC distance
        let table = pq.compute_distance_table(&query);
        let adc_dist = ProductQuantizer::adc_distance(&table, &codes, nsub);

        // Compute brute-force distance from query to reconstructed vector
        let reconstructed = pq.decode(&codes);
        let bf_dist = simd::l2_distance_squared(&query, &reconstructed);

        assert!(
            (adc_dist - bf_dist).abs() < 1e-4,
            "ADC distance {} should match brute-force {}",
            adc_dist,
            bf_dist
        );
    }

    #[test]
    fn test_adc_self_distance_zero() {
        let dim = 32;
        let nsub = 4;
        let sub_dim = dim / nsub;
        let codebook = random_codebook(nsub, sub_dim);
        let pq = ProductQuantizer::new(dim, nsub, codebook);

        // When query IS the decoded vector, ADC distance should be 0
        let codes: Vec<u8> = vec![0; nsub];
        let query = pq.decode(&codes);
        let table = pq.compute_distance_table(&query);
        let dist = ProductQuantizer::adc_distance(&table, &codes, nsub);

        assert!(dist.abs() < 1e-5, "Self-distance should be ~0, got {}", dist);
    }

    #[test]
    fn test_codebook_serialization() {
        let dim = 64;
        let nsub = 8;
        let sub_dim = dim / nsub;
        let codebook = random_codebook(nsub, sub_dim);
        let pq = ProductQuantizer::new(dim, nsub, codebook.clone());

        let bytes = pq.codebook_bytes();
        let pq2 = ProductQuantizer::from_codebook_bytes(dim, nsub, bytes).unwrap();

        assert_eq!(pq2.codebook, codebook);
    }

    #[test]
    fn test_batch_adc_distance() {
        let dim = 32;
        let nsub = 4;
        let codebook = random_codebook(nsub, dim / nsub);
        let pq = ProductQuantizer::new(dim, nsub, codebook);

        let mut rng = rand::thread_rng();
        let query: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
        let table = pq.compute_distance_table(&query);

        let codes: Vec<Vec<u8>> = (0..100)
            .map(|_| (0..nsub).map(|_| rng.gen::<u8>()).collect())
            .collect();
        let code_refs: Vec<&[u8]> = codes.iter().map(|c| c.as_slice()).collect();

        let batch_dists = ProductQuantizer::adc_distance_batch(&table, &code_refs, nsub);
        assert_eq!(batch_dists.len(), 100);

        // Verify first few match individual calls
        for i in 0..5 {
            let individual = ProductQuantizer::adc_distance(&table, &codes[i], nsub);
            assert!((batch_dists[i] - individual).abs() < 1e-6);
        }
    }
}
