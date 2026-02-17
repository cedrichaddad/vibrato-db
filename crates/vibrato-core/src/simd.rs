//! SIMD-optimized distance functions
//!
//! Provides explicit SIMD intrinsics for aarch64 (NEON) and x86_64 (AVX2),
//! with a scalar fallback that LLVM auto-vectorizes with `-C target-cpu=native`.
//!
//! # Performance
//!
//! For L2-normalized vectors (unit length), dot product equals cosine similarity:
//! ```text
//! cos(θ) = A · B  when ||A|| = ||B|| = 1
//! ```
//! Dot product is much faster than cosine similarity (no sqrt, no division).
//!
//! # Architecture Selection
//!
//! | Platform     | ISA        | Width  | Functions used                    |
//! |-------------|------------|--------|-----------------------------------|
//! | Apple M1+   | NEON       | 128-bit (4×f32) | `vld1q_f32`, `vfmaq_f32`, `vaddvq_f32` |
//! | x86_64+AVX2 | AVX2+FMA   | 256-bit (8×f32) | `_mm256_fmadd_ps`, `_mm256_sub_ps`      |
//! | other       | scalar     | 1×f32  | auto-vectorized iterator          |

// ============================================================================
// aarch64 NEON intrinsics
// ============================================================================

/// NEON dot product: processes 4 floats per iteration
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut acc = vdupq_n_f32(0.0);

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let va = vld1q_f32(a_ptr.add(i * 4));
        let vb = vld1q_f32(b_ptr.add(i * 4));
        acc = vfmaq_f32(acc, va, vb); // acc += va * vb
    }

    let mut sum = vaddvq_f32(acc); // horizontal add

    // Handle remainder
    let tail_start = chunks * 4;
    for i in 0..remainder {
        sum += a[tail_start + i] * b[tail_start + i];
    }

    sum
}

/// NEON L2 distance squared: processes 4 floats per iteration
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn l2_distance_squared_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut acc = vdupq_n_f32(0.0);

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let va = vld1q_f32(a_ptr.add(i * 4));
        let vb = vld1q_f32(b_ptr.add(i * 4));
        let diff = vsubq_f32(va, vb);
        acc = vfmaq_f32(acc, diff, diff); // acc += diff * diff
    }

    let mut sum = vaddvq_f32(acc);

    let tail_start = chunks * 4;
    for i in 0..remainder {
        let d = a[tail_start + i] - b[tail_start + i];
        sum += d * d;
    }

    sum
}

// ============================================================================
// x86_64 AVX2 intrinsics (runtime feature detection)
// ============================================================================

/// AVX2+FMA dot product: processes 8 floats per iteration (unaligned load)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let chunks = n / 8;
    let remainder = n % 8;

    let mut acc = _mm256_setzero_ps();

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let va = _mm256_loadu_ps(a_ptr.add(i * 8));
        let vb = _mm256_loadu_ps(b_ptr.add(i * 8));
        acc = _mm256_fmadd_ps(va, vb, acc); // acc += va * vb
    }

    // Horizontal sum of 8 floats → 1 float
    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let result = _mm_add_ss(sums, shuf2);
    let mut sum = _mm_cvtss_f32(result);

    // Handle remainder
    let tail_start = chunks * 8;
    for i in 0..remainder {
        sum += a[tail_start + i] * b[tail_start + i];
    }

    sum
}

/// AVX2+FMA dot product: processes 8 floats per iteration (aligned load)
///
/// CAUTION: Usage requires `a` and `b` to be 32-byte aligned AND the stride
/// (vector length in bytes) to be a multiple of 32 if used in a loop over
/// contiguous memory blocks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn dot_product_avx2_aligned(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let chunks = n / 8;
    let remainder = n % 8;

    let mut acc = _mm256_setzero_ps();

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let va = _mm256_load_ps(a_ptr.add(i * 8));
        let vb = _mm256_load_ps(b_ptr.add(i * 8));
        acc = _mm256_fmadd_ps(va, vb, acc); // acc += va * vb
    }

    // Horizontal sum
    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let result = _mm_add_ss(sums, shuf2);
    let mut sum = _mm_cvtss_f32(result);

    // Handle remainder
    let tail_start = chunks * 8;
    for i in 0..remainder {
        sum += a[tail_start + i] * b[tail_start + i];
    }

    sum
}

/// AVX2+FMA L2 distance squared: processes 8 floats per iteration (unaligned)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn l2_distance_squared_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let chunks = n / 8;
    let remainder = n % 8;

    let mut acc = _mm256_setzero_ps();

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let va = _mm256_loadu_ps(a_ptr.add(i * 8));
        let vb = _mm256_loadu_ps(b_ptr.add(i * 8));
        let diff = _mm256_sub_ps(va, vb);
        acc = _mm256_fmadd_ps(diff, diff, acc); // acc += diff * diff
    }

    // Horizontal sum
    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let result = _mm_add_ss(sums, shuf2);
    let mut sum = _mm_cvtss_f32(result);

    let tail_start = chunks * 8;
    for i in 0..remainder {
        let d = a[tail_start + i] - b[tail_start + i];
        sum += d * d;
    }

    sum
}

/// AVX2+FMA L2 distance squared: processes 8 floats per iteration (aligned)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn l2_distance_squared_avx2_aligned(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let chunks = n / 8;
    let remainder = n % 8;

    let mut acc = _mm256_setzero_ps();

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let va = _mm256_load_ps(a_ptr.add(i * 8));
        let vb = _mm256_load_ps(b_ptr.add(i * 8));
        let diff = _mm256_sub_ps(va, vb);
        acc = _mm256_fmadd_ps(diff, diff, acc); // acc += diff * diff
    }

    // Horizontal sum
    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let result = _mm_add_ss(sums, shuf2);
    let mut sum = _mm_cvtss_f32(result);

    let tail_start = chunks * 8;
    for i in 0..remainder {
        let d = a[tail_start + i] - b[tail_start + i];
        sum += d * d;
    }

    sum
}

// ============================================================================
// Scalar fallback (auto-vectorized by LLVM)
// ============================================================================

#[inline(always)]
fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[inline(always)]
fn l2_distance_squared_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum()
}

// ============================================================================
// Public dispatch functions
// ============================================================================

/// Compute dot product of two vectors
///
/// For L2-normalized vectors, this equals cosine similarity.
/// Uses NEON on aarch64, AVX2+FMA on x86_64, or scalar fallback.
#[inline(always)]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vector length mismatch");

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is always available on aarch64, handles aligned/unaligned well
        return unsafe { dot_product_neon(a, b) };
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // Check for 32-byte alignment of both pointers AND stride alignment
            // (a.len() * 4) must be a multiple of 32 for stride to be aligned across vectors
            let ptr_a = a.as_ptr() as usize;
            let ptr_b = b.as_ptr() as usize;
            let byte_len = a.len() * 4;

            if ptr_a % 32 == 0 && ptr_b % 32 == 0 && byte_len % 32 == 0 {
                return unsafe { dot_product_avx2_aligned(a, b) };
            } else {
                return unsafe { dot_product_avx2(a, b) };
            }
        }
    }

    #[allow(unreachable_code)]
    dot_product_scalar(a, b)
}

/// Compute squared L2 (Euclidean) distance between two vectors
///
/// Returns ||a - b||² (no square root for performance).
/// Uses NEON on aarch64, AVX2+FMA on x86_64, or scalar fallback.
#[inline(always)]
pub fn l2_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vector length mismatch");

    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { l2_distance_squared_neon(a, b) };
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // Check for 32-byte alignment of both pointers AND stride alignment
            let ptr_a = a.as_ptr() as usize;
            let ptr_b = b.as_ptr() as usize;
            let byte_len = a.len() * 4;

            if ptr_a % 32 == 0 && ptr_b % 32 == 0 && byte_len % 32 == 0 {
                return unsafe { l2_distance_squared_avx2_aligned(a, b) };
            } else {
                return unsafe { l2_distance_squared_avx2(a, b) };
            }
        }
    }

    #[allow(unreachable_code)]
    l2_distance_squared_scalar(a, b)
}

/// Compute L2 (Euclidean) distance between two vectors
///
/// Returns ||a - b||
#[inline(always)]
pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    l2_distance_squared(a, b).sqrt()
}

/// Compute L2² distances from one query to many targets (batch)
///
/// Used for PQ distance table computation where one query sub-vector
/// is compared against all 256 centroids. Keeping the query in registers
/// across iterations avoids repeated loads.
pub fn l2_distance_squared_batch(query: &[f32], targets: &[&[f32]]) -> Vec<f32> {
    targets
        .iter()
        .map(|t| l2_distance_squared(query, t))
        .collect()
}

/// L2 normalize a vector in place
///
/// After normalization, ||v|| = 1
pub fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = dot_product(v, v).sqrt();
    if norm > f32::EPSILON {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// L2 normalize a vector, returning a new vector
pub fn l2_normalized(v: &[f32]) -> Vec<f32> {
    let norm: f32 = dot_product(v, v).sqrt();
    if norm > f32::EPSILON {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}

/// Check if a vector is L2 normalized (unit length)
#[inline]
pub fn is_normalized(v: &[f32], tolerance: f32) -> bool {
    let norm_sq: f32 = dot_product(v, v);
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

        assert!(
            v.iter().all(|&x| x == 0.0),
            "Zero vector should remain zero"
        );
    }

    #[test]
    fn test_l2_normalized_zero_vector() {
        let v = vec![0.0f32; 64];
        let result = l2_normalized(&v);

        assert!(
            result.iter().all(|&x| x == 0.0),
            "Zero vector should remain zero"
        );
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
