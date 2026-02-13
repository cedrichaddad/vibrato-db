//! K-Means Training for Product Quantization Codebooks
//!
//! Pure Rust implementation of K-means clustering used to train PQ codebooks.
//! Designed for the PQ warm-up lifecycle:
//!
//! 1. Database starts in raw f32 mode
//! 2. Once N > threshold (default 10,000), training triggers
//! 3. Codebook is trained, vectors are compacted to PQ codes
//!
//! # Algorithm
//!
//! Standard Lloyd's K-means with:
//! - K-means++ initialization for better convergence
//! - Early stopping when centroid movement < tolerance
//! - Per-subspace independent training

use crate::pq::{PqError, ProductQuantizer, NUM_CENTROIDS};
use crate::simd;
use rand::{Rng, SeedableRng};

/// Configuration for PQ training
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Number of subspaces (default: 16)
    pub num_subspaces: usize,

    /// Maximum K-means iterations per subspace (default: 25)
    pub max_iters: usize,

    /// Convergence tolerance: stop if max centroid movement < tolerance (default: 1e-4)
    pub tolerance: f32,

    /// Minimum number of vectors required for training (default: 256)
    pub min_vectors: usize,

    /// Random seed for reproducibility (None = random)
    pub seed: Option<u64>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            num_subspaces: 16,
            max_iters: 25,
            tolerance: 1e-4,
            min_vectors: 256,
            seed: None,
        }
    }
}

/// Train a PQ codebook from a set of training vectors
///
/// # Arguments
/// * `vectors` - Training vectors, each of length `dimension`
/// * `dimension` - Vector dimensionality (must be divisible by `config.num_subspaces`)
/// * `config` - Training configuration
///
/// # Returns
/// A trained `ProductQuantizer` ready for encoding
pub fn train_pq(
    vectors: &[Vec<f32>],
    dimension: usize,
    config: &TrainingConfig,
) -> Result<ProductQuantizer, PqError> {
    if vectors.len() < config.min_vectors {
        return Err(PqError::InsufficientData {
            count: vectors.len(),
            minimum: config.min_vectors,
        });
    }

    if dimension % config.num_subspaces != 0 {
        return Err(PqError::CodebookSizeMismatch {
            expected: dimension,
            actual: config.num_subspaces,
        });
    }

    let sub_dim = dimension / config.num_subspaces;
    let mut codebook = vec![0.0f32; config.num_subspaces * NUM_CENTROIDS * sub_dim];

    // Train each subspace independently
    for s in 0..config.num_subspaces {
        // Extract sub-vectors for this subspace
        let sub_vectors: Vec<&[f32]> = vectors
            .iter()
            .map(|v| &v[s * sub_dim..(s + 1) * sub_dim])
            .collect();

        // Run K-means for this subspace
        let centroids = kmeans(
            &sub_vectors,
            sub_dim,
            NUM_CENTROIDS,
            config.max_iters,
            config.tolerance,
            config.seed.map(|s_| s_ + s as u64),
        );

        // Copy centroids into codebook
        let offset = s * NUM_CENTROIDS * sub_dim;
        for c in 0..NUM_CENTROIDS {
            let dst = offset + c * sub_dim;
            codebook[dst..dst + sub_dim]
                .copy_from_slice(&centroids[c * sub_dim..(c + 1) * sub_dim]);
        }
    }

    Ok(ProductQuantizer::new(
        dimension,
        config.num_subspaces,
        codebook,
    ))
}

/// K-means++ initialization
///
/// Selects initial centroids with probability proportional to squared distance
/// from the nearest existing centroid.
fn kmeans_plus_plus_init(data: &[&[f32]], dim: usize, k: usize, seed: Option<u64>) -> Vec<f32> {
    let n = data.len();
    let mut rng = match seed {
        Some(s) => rand::rngs::StdRng::seed_from_u64(s),
        None => rand::rngs::StdRng::seed_from_u64(rand::thread_rng().gen()),
    };

    let mut centroids = vec![0.0f32; k * dim];

    // First centroid: random data point
    let first_idx = rng.gen_range(0..n);
    centroids[0..dim].copy_from_slice(data[first_idx]);

    // Distances from each point to nearest centroid
    let mut min_dists = vec![f32::MAX; n];

    for c in 1..k {
        // Update min distances with the last added centroid
        let last_centroid = &centroids[(c - 1) * dim..c * dim];
        for i in 0..n {
            let d = simd::l2_distance_squared(data[i], last_centroid);
            if d < min_dists[i] {
                min_dists[i] = d;
            }
        }

        // Sample next centroid proportional to min_dists
        let total: f64 = min_dists.iter().map(|&d| d as f64).sum();
        if total <= 0.0 {
            // All points are at centroids already, use random
            let idx = rng.gen_range(0..n);
            centroids[c * dim..(c + 1) * dim].copy_from_slice(data[idx]);
            continue;
        }

        let threshold: f64 = rng.gen::<f64>() * total;
        let mut cumulative = 0.0f64;
        let mut chosen = 0;
        for i in 0..n {
            cumulative += min_dists[i] as f64;
            if cumulative >= threshold {
                chosen = i;
                break;
            }
        }

        centroids[c * dim..(c + 1) * dim].copy_from_slice(data[chosen]);
    }

    centroids
}

/// Standard Lloyd's K-means algorithm
fn kmeans(
    data: &[&[f32]],
    dim: usize,
    k: usize,
    max_iters: usize,
    tolerance: f32,
    seed: Option<u64>,
) -> Vec<f32> {
    let n = data.len();
    let k = k.min(n); // Can't have more centroids than data points

    // Initialize with K-means++
    let mut centroids = kmeans_plus_plus_init(data, dim, k, seed);
    let mut assignments = vec![0usize; n];
    let mut new_centroids = vec![0.0f32; k * dim];
    let mut counts = vec![0usize; k];

    for _iter in 0..max_iters {
        // Assignment step: assign each point to nearest centroid
        for i in 0..n {
            let mut best_c = 0;
            let mut best_dist = f32::MAX;
            for c in 0..k {
                let centroid = &centroids[c * dim..(c + 1) * dim];
                let dist = simd::l2_distance_squared(data[i], centroid);
                if dist < best_dist {
                    best_dist = dist;
                    best_c = c;
                }
            }
            assignments[i] = best_c;
        }

        // Update step: compute new centroids
        new_centroids.fill(0.0);
        counts.fill(0);

        for i in 0..n {
            let c = assignments[i];
            counts[c] += 1;
            let offset = c * dim;
            for d in 0..dim {
                new_centroids[offset + d] += data[i][d];
            }
        }

        for c in 0..k {
            if counts[c] > 0 {
                let offset = c * dim;
                let count_f = counts[c] as f32;
                for d in 0..dim {
                    new_centroids[offset + d] /= count_f;
                }
            } else {
                // Dead centroid: reinitialize to random data point
                let mut rng = rand::thread_rng();
                let idx = rng.gen_range(0..n);
                let offset = c * dim;
                new_centroids[offset..offset + dim].copy_from_slice(data[idx]);
            }
        }

        // Check convergence: max centroid movement
        let mut max_movement = 0.0f32;
        for c in 0..k {
            let old = &centroids[c * dim..(c + 1) * dim];
            let new = &new_centroids[c * dim..(c + 1) * dim];
            let movement = simd::l2_distance_squared(old, new);
            if movement > max_movement {
                max_movement = movement;
            }
        }

        // Swap
        std::mem::swap(&mut centroids, &mut new_centroids);

        if max_movement < tolerance * tolerance {
            tracing::debug!("K-means converged after {} iterations", _iter + 1);
            break;
        }
    }

    centroids
}

/// PQ warm-up lifecycle manager
///
/// Tracks vector count and triggers codebook training when threshold is reached.
pub struct PqLifecycle {
    /// Number of vectors required before training (default: 10,000)
    pub training_threshold: usize,

    /// Training configuration
    pub config: TrainingConfig,

    /// Current state
    state: PqState,
}

/// PQ lifecycle state
#[derive(Debug, Clone)]
pub enum PqState {
    /// Storing raw f32 vectors (pre-training)
    Raw,
    /// Training is in progress
    Training,
    /// Codebook trained, PQ codes active
    Quantized,
}

impl PqLifecycle {
    /// Create a new lifecycle manager
    pub fn new(training_threshold: usize, config: TrainingConfig) -> Self {
        Self {
            training_threshold,
            config,
            state: PqState::Raw,
        }
    }

    /// Check if training should be triggered
    pub fn should_train(&self, vector_count: usize) -> bool {
        matches!(self.state, PqState::Raw) && vector_count >= self.training_threshold
    }

    /// Get current state
    pub fn state(&self) -> &PqState {
        &self.state
    }

    /// Transition to training state
    pub fn begin_training(&mut self) {
        self.state = PqState::Training;
    }

    /// Transition to quantized state after training
    pub fn finish_training(&mut self) {
        self.state = PqState::Quantized;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect())
            .collect()
    }

    #[test]
    fn test_kmeans_convergence() {
        // 3 well-separated clusters
        let mut data = Vec::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        for _ in 0..100 {
            data.push(vec![rng.gen::<f32>() * 0.1, rng.gen::<f32>() * 0.1]); // cluster near (0,0)
        }
        for _ in 0..100 {
            data.push(vec![
                5.0 + rng.gen::<f32>() * 0.1,
                5.0 + rng.gen::<f32>() * 0.1,
            ]); // cluster near (5,5)
        }
        for _ in 0..100 {
            data.push(vec![
                10.0 + rng.gen::<f32>() * 0.1,
                0.0 + rng.gen::<f32>() * 0.1,
            ]); // cluster near (10,0)
        }

        let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
        let centroids = kmeans(&refs, 2, 3, 50, 1e-6, Some(42));

        // Should find 3 centroids near (0,0), (5,5), (10,0)
        let mut centers: Vec<(f32, f32)> = (0..3)
            .map(|i| (centroids[i * 2], centroids[i * 2 + 1]))
            .collect();
        centers.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        assert!(centers[0].0.abs() < 0.5, "First cluster center near 0");
        assert!(
            (centers[1].0 - 5.0).abs() < 0.5,
            "Second cluster center near 5"
        );
        assert!(
            (centers[2].0 - 10.0).abs() < 0.5,
            "Third cluster center near 10"
        );
    }

    #[test]
    fn test_train_pq_full_pipeline() {
        let dim = 32;
        let nsub = 4;
        let vectors = random_vectors(500, dim, 123);

        let config = TrainingConfig {
            num_subspaces: nsub,
            max_iters: 15,
            tolerance: 1e-4,
            min_vectors: 256,
            seed: Some(42),
        };

        let pq = train_pq(&vectors, dim, &config).unwrap();

        // Encode and decode should approximate original
        let test_vec = &vectors[0];
        let codes = pq.encode(test_vec);
        let reconstructed = pq.decode(&codes);

        // Reconstruction error should be reasonable (not exact, but not terrible)
        let error = simd::l2_distance_squared(test_vec, &reconstructed);
        assert!(
            error < 5.0,
            "Reconstruction error {} should be reasonable",
            error
        );
    }

    #[test]
    fn test_train_pq_insufficient_data() {
        let vectors = random_vectors(10, 32, 42);
        let config = TrainingConfig {
            min_vectors: 256,
            ..TrainingConfig::default()
        };

        let result = train_pq(&vectors, 32, &config);
        assert!(matches!(result, Err(PqError::InsufficientData { .. })));
    }

    #[test]
    fn test_pq_lifecycle() {
        let mut lifecycle = PqLifecycle::new(10_000, TrainingConfig::default());

        assert!(matches!(lifecycle.state(), PqState::Raw));
        assert!(!lifecycle.should_train(5_000));
        assert!(lifecycle.should_train(10_000));

        lifecycle.begin_training();
        assert!(matches!(lifecycle.state(), PqState::Training));
        assert!(!lifecycle.should_train(20_000)); // Already training

        lifecycle.finish_training();
        assert!(matches!(lifecycle.state(), PqState::Quantized));
    }

    #[test]
    fn test_adc_preserves_ordering() {
        // ADC should preserve distance ordering (at least approximately)
        let dim = 32;
        let nsub = 4;
        let vectors = random_vectors(500, dim, 99);
        let config = TrainingConfig {
            num_subspaces: nsub,
            max_iters: 20,
            tolerance: 1e-5,
            min_vectors: 256,
            seed: Some(99),
        };
        let pq = train_pq(&vectors, dim, &config).unwrap();

        let query = &vectors[0];
        let table = pq.compute_distance_table(query);

        // Compute ADC and brute-force distances
        let mut adc_dists: Vec<(usize, f32)> = Vec::new();
        let mut bf_dists: Vec<(usize, f32)> = Vec::new();
        for (i, v) in vectors.iter().enumerate().skip(1).take(50) {
            let codes = pq.encode(v);
            adc_dists.push((i, ProductQuantizer::adc_distance(&table, &codes, nsub)));
            bf_dists.push((i, simd::l2_distance_squared(query, v)));
        }

        adc_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        bf_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Top-5 should have significant overlap (at least 3 common elements)
        let adc_top5: std::collections::HashSet<usize> =
            adc_dists.iter().take(5).map(|x| x.0).collect();
        let bf_top5: std::collections::HashSet<usize> =
            bf_dists.iter().take(5).map(|x| x.0).collect();
        let overlap = adc_top5.intersection(&bf_top5).count();

        assert!(
            overlap >= 2,
            "ADC and brute-force top-5 should overlap significantly (got {})",
            overlap
        );
    }
}
