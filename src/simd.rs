//! SIMD-optimized distance functions
//!
//! These functions use iterator patterns that LLVM auto-vectorizes when
//! compiled with `-C target-cpu=native`.
//!
//! # Performance
//!
//! For L2-normalized vectors (unit length), dot product equals cosine similarity:
//! ```text
//! cos(θ) = A · B  when ||A|| = ||B|| = 1
//! ```
//! Dot product is much faster than cosine similarity (no sqrt, no division).

/// Compute dot product of two vectors
///
/// For L2-normalized vectors, this equals cosine similarity.
/// LLVM will auto-vectorize this to SIMD instructions with `-C target-cpu=native`.
#[inline(always)]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vector length mismatch");

    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Compute squared L2 (Euclidean) distance between two vectors
///
/// Returns ||a - b||² (no square root for performance).
#[inline(always)]
pub fn l2_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vector length mismatch");

    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum()
}

/// Compute L2 (Euclidean) distance between two vectors
///
/// Returns ||a - b||
#[inline(always)]
pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    l2_distance_squared(a, b).sqrt()
}

/// L2 normalize a vector in place
///
/// After normalization, ||v|| = 1
pub fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// L2 normalize a vector, returning a new vector
pub fn l2_normalized(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}

/// Check if a vector is L2 normalized (unit length)
#[inline]
pub fn is_normalized(v: &[f32], tolerance: f32) -> bool {
    let norm_sq: f32 = v.iter().map(|x| x * x).sum();
    (norm_sq - 1.0).abs() < tolerance
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product_basic() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];

        // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
        assert!((dot_product(&a, &b) - 70.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product_normalized() {
        let a = l2_normalized(&[1.0, 0.0, 0.0]);
        let b = l2_normalized(&[0.0, 1.0, 0.0]);

        // Orthogonal vectors have dot product = 0
        assert!((dot_product(&a, &b) - 0.0).abs() < 1e-6);

        // Same vector has dot product = 1 (when normalized)
        assert!((dot_product(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_distance() {
        let a = [0.0, 0.0, 0.0];
        let b = [3.0, 4.0, 0.0];

        // 3-4-5 triangle
        assert!((l2_distance(&a, &b) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_normalize() {
        let mut v = vec![3.0, 4.0];
        l2_normalize(&mut v);

        // Should be unit length
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);

        // Direction preserved: 3/5, 4/5
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_is_normalized() {
        let normalized = l2_normalized(&[1.0, 2.0, 3.0]);
        assert!(is_normalized(&normalized, 1e-5));

        let not_normalized = [1.0, 2.0, 3.0];
        assert!(!is_normalized(&not_normalized, 1e-5));
    }

    #[test]
    fn test_dot_product_large() {
        // Test with 128 dimensions (like VGGish)
        let a: Vec<f32> = (0..128).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..128).map(|i| (127 - i) as f32).collect();

        let result = dot_product(&a, &b);

        // Verify against naive implementation
        let expected: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        assert!((result - expected).abs() < 1e-3);
    }

    // ============== Edge Case Tests ==============

    #[test]
    fn test_dot_product_single_element() {
        let a = [3.0f32];
        let b = [4.0f32];
        assert!((dot_product(&a, &b) - 12.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product_zeros() {
        let a = [0.0f32; 128];
        let b = [1.0f32; 128];
        assert!((dot_product(&a, &b) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product_negative_values() {
        let a = vec![-1.0f32, -2.0, -3.0];
        let b = vec![1.0f32, 2.0, 3.0];
        // -1 + -4 + -9 = -14
        assert!((dot_product(&a, &b) - (-14.0)).abs() < 1e-6);
    }

    #[test]
    fn test_l2_normalize_zero_vector() {
        // Zero vector should remain zero (avoid division by zero)
        let mut v = vec![0.0f32; 64];
        l2_normalize(&mut v);
        
        assert!(v.iter().all(|&x| x == 0.0), "Zero vector should remain zero");
    }

    #[test]
    fn test_l2_normalized_zero_vector() {
        let v = vec![0.0f32; 64];
        let result = l2_normalized(&v);
        
        assert!(result.iter().all(|&x| x == 0.0), "Zero vector should remain zero");
    }

    #[test]
    fn test_l2_distance_same_point() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        assert!((l2_distance(&a, &a) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_distance_squared_symmetry() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 5.0, 6.0];
        assert!((l2_distance_squared(&a, &b) - l2_distance_squared(&b, &a)).abs() < 1e-6);
    }

    #[test]
    fn test_is_normalized_edge_cases() {
        // Exactly normalized
        let unit_x = [1.0f32, 0.0, 0.0];
        assert!(is_normalized(&unit_x, 1e-6));
        
        // Very close to normalized
        let almost = [0.9999999f32, 0.0, 0.0];
        assert!(is_normalized(&almost, 1e-5));
        
        // Not normalized
        let half = [0.5f32, 0.0, 0.0];
        assert!(!is_normalized(&half, 1e-5));
    }

    #[test]
    fn test_large_dimension_performance() {
        // Test with high dimensions (768 like BERT, 1536 like ada-002)
        let a: Vec<f32> = (0..1536).map(|i| (i as f32) / 1536.0).collect();
        let b: Vec<f32> = (0..1536).map(|i| ((1536 - i) as f32) / 1536.0).collect();
        
        // Just verify it completes and returns reasonable value
        let result = dot_product(&a, &b);
        assert!(result.is_finite(), "Result should be finite");
    }
}

