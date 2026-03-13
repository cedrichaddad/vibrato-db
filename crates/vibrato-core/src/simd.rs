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
    let chunks16 = n / 16;
    let chunks4 = (n % 16) / 4;
    let remainder = n % 4;

    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks16 {
        let base = i * 16;
        let va0 = vld1q_f32(a_ptr.add(base));
        let vb0 = vld1q_f32(b_ptr.add(base));
        let va1 = vld1q_f32(a_ptr.add(base + 4));
        let vb1 = vld1q_f32(b_ptr.add(base + 4));
        let va2 = vld1q_f32(a_ptr.add(base + 8));
        let vb2 = vld1q_f32(b_ptr.add(base + 8));
        let va3 = vld1q_f32(a_ptr.add(base + 12));
        let vb3 = vld1q_f32(b_ptr.add(base + 12));
        acc0 = vfmaq_f32(acc0, va0, vb0);
        acc1 = vfmaq_f32(acc1, va1, vb1);
        acc2 = vfmaq_f32(acc2, va2, vb2);
        acc3 = vfmaq_f32(acc3, va3, vb3);
    }

    let mut acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
    let tail16 = chunks16 * 16;
    for i in 0..chunks4 {
        let base = tail16 + i * 4;
        let va = vld1q_f32(a_ptr.add(base));
        let vb = vld1q_f32(b_ptr.add(base));
        acc = vfmaq_f32(acc, va, vb);
    }

    let mut sum = vaddvq_f32(acc);

    // Handle remainder
    let tail_start = tail16 + chunks4 * 4;
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
    let chunks16 = n / 16;
    let chunks4 = (n % 16) / 4;
    let remainder = n % 4;

    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks16 {
        let base = i * 16;

        let va0 = vld1q_f32(a_ptr.add(base));
        let vb0 = vld1q_f32(b_ptr.add(base));
        let d0 = vsubq_f32(va0, vb0);
        acc0 = vfmaq_f32(acc0, d0, d0);

        let va1 = vld1q_f32(a_ptr.add(base + 4));
        let vb1 = vld1q_f32(b_ptr.add(base + 4));
        let d1 = vsubq_f32(va1, vb1);
        acc1 = vfmaq_f32(acc1, d1, d1);

        let va2 = vld1q_f32(a_ptr.add(base + 8));
        let vb2 = vld1q_f32(b_ptr.add(base + 8));
        let d2 = vsubq_f32(va2, vb2);
        acc2 = vfmaq_f32(acc2, d2, d2);

        let va3 = vld1q_f32(a_ptr.add(base + 12));
        let vb3 = vld1q_f32(b_ptr.add(base + 12));
        let d3 = vsubq_f32(va3, vb3);
        acc3 = vfmaq_f32(acc3, d3, d3);
    }

    let mut acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
    let tail16 = chunks16 * 16;
    for i in 0..chunks4 {
        let base = tail16 + i * 4;
        let va = vld1q_f32(a_ptr.add(base));
        let vb = vld1q_f32(b_ptr.add(base));
        let diff = vsubq_f32(va, vb);
        acc = vfmaq_f32(acc, diff, diff);
    }

    let mut sum = vaddvq_f32(acc);

    let tail_start = tail16 + chunks4 * 4;
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
// Batched dot-product scoring for identify hot path
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn dot_product_scores4_neon_const<const CHUNKS: usize>(
    query: &[f32],
    candidates: &[f32],
    out: &mut [f32],
) {
    use std::arch::aarch64::*;

    debug_assert_eq!(query.len(), CHUNKS * 4);
    debug_assert!(candidates.len() >= 4 * query.len());
    debug_assert!(out.len() >= 4);

    let dim = query.len();
    let q_ptr = query.as_ptr();
    let c0 = candidates.as_ptr();
    let c1 = c0.add(dim);
    let c2 = c1.add(dim);
    let c3 = c2.add(dim);

    let mut acc0_a = vdupq_n_f32(0.0);
    let mut acc0_b = vdupq_n_f32(0.0);
    let mut acc1_a = vdupq_n_f32(0.0);
    let mut acc1_b = vdupq_n_f32(0.0);
    let mut acc2_a = vdupq_n_f32(0.0);
    let mut acc2_b = vdupq_n_f32(0.0);
    let mut acc3_a = vdupq_n_f32(0.0);
    let mut acc3_b = vdupq_n_f32(0.0);

    let mut i = 0usize;
    while i + 1 < CHUNKS {
        let base0 = i * 4;
        let base1 = (i + 1) * 4;

        let q0 = vld1q_f32(q_ptr.add(base0));
        let q1 = vld1q_f32(q_ptr.add(base1));

        acc0_a = vfmaq_f32(acc0_a, q0, vld1q_f32(c0.add(base0)));
        acc0_b = vfmaq_f32(acc0_b, q1, vld1q_f32(c0.add(base1)));
        acc1_a = vfmaq_f32(acc1_a, q0, vld1q_f32(c1.add(base0)));
        acc1_b = vfmaq_f32(acc1_b, q1, vld1q_f32(c1.add(base1)));
        acc2_a = vfmaq_f32(acc2_a, q0, vld1q_f32(c2.add(base0)));
        acc2_b = vfmaq_f32(acc2_b, q1, vld1q_f32(c2.add(base1)));
        acc3_a = vfmaq_f32(acc3_a, q0, vld1q_f32(c3.add(base0)));
        acc3_b = vfmaq_f32(acc3_b, q1, vld1q_f32(c3.add(base1)));
        i += 2;
    }

    if i < CHUNKS {
        let base = i * 4;
        let q = vld1q_f32(q_ptr.add(base));
        acc0_a = vfmaq_f32(acc0_a, q, vld1q_f32(c0.add(base)));
        acc1_a = vfmaq_f32(acc1_a, q, vld1q_f32(c1.add(base)));
        acc2_a = vfmaq_f32(acc2_a, q, vld1q_f32(c2.add(base)));
        acc3_a = vfmaq_f32(acc3_a, q, vld1q_f32(c3.add(base)));
    }

    out[0] = vaddvq_f32(vaddq_f32(acc0_a, acc0_b));
    out[1] = vaddvq_f32(vaddq_f32(acc1_a, acc1_b));
    out[2] = vaddvq_f32(vaddq_f32(acc2_a, acc2_b));
    out[3] = vaddvq_f32(vaddq_f32(acc3_a, acc3_b));
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn dot_product_scores4_neon(query: &[f32], candidates: &[f32], dim: usize, out: &mut [f32]) {
    use std::arch::aarch64::*;

    debug_assert_eq!(query.len(), dim);
    debug_assert!(candidates.len() >= 4 * dim);
    debug_assert!(dim.is_multiple_of(4));

    let chunks = dim / 4;
    let q_ptr = query.as_ptr();
    let c0 = candidates.as_ptr();
    let c1 = c0.add(dim);
    let c2 = c1.add(dim);
    let c3 = c2.add(dim);

    let mut acc0_a = vdupq_n_f32(0.0);
    let mut acc0_b = vdupq_n_f32(0.0);
    let mut acc1_a = vdupq_n_f32(0.0);
    let mut acc1_b = vdupq_n_f32(0.0);
    let mut acc2_a = vdupq_n_f32(0.0);
    let mut acc2_b = vdupq_n_f32(0.0);
    let mut acc3_a = vdupq_n_f32(0.0);
    let mut acc3_b = vdupq_n_f32(0.0);

    let mut i = 0usize;
    while i + 1 < chunks {
        let base0 = i * 4;
        let base1 = (i + 1) * 4;

        let q0 = vld1q_f32(q_ptr.add(base0));
        let q1 = vld1q_f32(q_ptr.add(base1));

        acc0_a = vfmaq_f32(acc0_a, q0, vld1q_f32(c0.add(base0)));
        acc0_b = vfmaq_f32(acc0_b, q1, vld1q_f32(c0.add(base1)));
        acc1_a = vfmaq_f32(acc1_a, q0, vld1q_f32(c1.add(base0)));
        acc1_b = vfmaq_f32(acc1_b, q1, vld1q_f32(c1.add(base1)));
        acc2_a = vfmaq_f32(acc2_a, q0, vld1q_f32(c2.add(base0)));
        acc2_b = vfmaq_f32(acc2_b, q1, vld1q_f32(c2.add(base1)));
        acc3_a = vfmaq_f32(acc3_a, q0, vld1q_f32(c3.add(base0)));
        acc3_b = vfmaq_f32(acc3_b, q1, vld1q_f32(c3.add(base1)));
        i += 2;
    }

    if i < chunks {
        let base = i * 4;
        let q = vld1q_f32(q_ptr.add(base));
        acc0_a = vfmaq_f32(acc0_a, q, vld1q_f32(c0.add(base)));
        acc1_a = vfmaq_f32(acc1_a, q, vld1q_f32(c1.add(base)));
        acc2_a = vfmaq_f32(acc2_a, q, vld1q_f32(c2.add(base)));
        acc3_a = vfmaq_f32(acc3_a, q, vld1q_f32(c3.add(base)));
    }

    out[0] = vaddvq_f32(vaddq_f32(acc0_a, acc0_b));
    out[1] = vaddvq_f32(vaddq_f32(acc1_a, acc1_b));
    out[2] = vaddvq_f32(vaddq_f32(acc2_a, acc2_b));
    out[3] = vaddvq_f32(vaddq_f32(acc3_a, acc3_b));
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline(always)]
unsafe fn hsum256_ps(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;

    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let result = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(result)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline(always)]
unsafe fn dot_product_scores4_avx2_const<const CHUNKS: usize>(
    query: &[f32],
    candidates: &[f32],
    out: &mut [f32],
) {
    use std::arch::x86_64::*;

    debug_assert_eq!(query.len(), CHUNKS * 8);
    debug_assert!(candidates.len() >= 4 * query.len());
    debug_assert!(out.len() >= 4);

    let dim = query.len();
    let q_ptr = query.as_ptr();
    let c0 = candidates.as_ptr();
    let c1 = c0.add(dim);
    let c2 = c1.add(dim);
    let c3 = c2.add(dim);

    let mut acc0_a = _mm256_setzero_ps();
    let mut acc0_b = _mm256_setzero_ps();
    let mut acc1_a = _mm256_setzero_ps();
    let mut acc1_b = _mm256_setzero_ps();
    let mut acc2_a = _mm256_setzero_ps();
    let mut acc2_b = _mm256_setzero_ps();
    let mut acc3_a = _mm256_setzero_ps();
    let mut acc3_b = _mm256_setzero_ps();

    let mut i = 0usize;
    while i + 1 < CHUNKS {
        let base0 = i * 8;
        let base1 = (i + 1) * 8;

        let q0 = _mm256_loadu_ps(q_ptr.add(base0));
        let q1 = _mm256_loadu_ps(q_ptr.add(base1));

        acc0_a = _mm256_fmadd_ps(q0, _mm256_loadu_ps(c0.add(base0)), acc0_a);
        acc0_b = _mm256_fmadd_ps(q1, _mm256_loadu_ps(c0.add(base1)), acc0_b);
        acc1_a = _mm256_fmadd_ps(q0, _mm256_loadu_ps(c1.add(base0)), acc1_a);
        acc1_b = _mm256_fmadd_ps(q1, _mm256_loadu_ps(c1.add(base1)), acc1_b);
        acc2_a = _mm256_fmadd_ps(q0, _mm256_loadu_ps(c2.add(base0)), acc2_a);
        acc2_b = _mm256_fmadd_ps(q1, _mm256_loadu_ps(c2.add(base1)), acc2_b);
        acc3_a = _mm256_fmadd_ps(q0, _mm256_loadu_ps(c3.add(base0)), acc3_a);
        acc3_b = _mm256_fmadd_ps(q1, _mm256_loadu_ps(c3.add(base1)), acc3_b);
        i += 2;
    }

    if i < CHUNKS {
        let base = i * 8;
        let q = _mm256_loadu_ps(q_ptr.add(base));
        acc0_a = _mm256_fmadd_ps(q, _mm256_loadu_ps(c0.add(base)), acc0_a);
        acc1_a = _mm256_fmadd_ps(q, _mm256_loadu_ps(c1.add(base)), acc1_a);
        acc2_a = _mm256_fmadd_ps(q, _mm256_loadu_ps(c2.add(base)), acc2_a);
        acc3_a = _mm256_fmadd_ps(q, _mm256_loadu_ps(c3.add(base)), acc3_a);
    }

    out[0] = hsum256_ps(_mm256_add_ps(acc0_a, acc0_b));
    out[1] = hsum256_ps(_mm256_add_ps(acc1_a, acc1_b));
    out[2] = hsum256_ps(_mm256_add_ps(acc2_a, acc2_b));
    out[3] = hsum256_ps(_mm256_add_ps(acc3_a, acc3_b));
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline(always)]
unsafe fn dot_product_scores4_avx2(query: &[f32], candidates: &[f32], dim: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;

    debug_assert_eq!(query.len(), dim);
    debug_assert!(candidates.len() >= 4 * dim);
    debug_assert!(dim.is_multiple_of(8));

    let chunks = dim / 8;
    let q_ptr = query.as_ptr();
    let c0 = candidates.as_ptr();
    let c1 = c0.add(dim);
    let c2 = c1.add(dim);
    let c3 = c2.add(dim);

    let mut acc0_a = _mm256_setzero_ps();
    let mut acc0_b = _mm256_setzero_ps();
    let mut acc1_a = _mm256_setzero_ps();
    let mut acc1_b = _mm256_setzero_ps();
    let mut acc2_a = _mm256_setzero_ps();
    let mut acc2_b = _mm256_setzero_ps();
    let mut acc3_a = _mm256_setzero_ps();
    let mut acc3_b = _mm256_setzero_ps();

    let mut i = 0usize;
    while i + 1 < chunks {
        let base0 = i * 8;
        let base1 = (i + 1) * 8;

        let q0 = _mm256_loadu_ps(q_ptr.add(base0));
        let q1 = _mm256_loadu_ps(q_ptr.add(base1));

        acc0_a = _mm256_fmadd_ps(q0, _mm256_loadu_ps(c0.add(base0)), acc0_a);
        acc0_b = _mm256_fmadd_ps(q1, _mm256_loadu_ps(c0.add(base1)), acc0_b);
        acc1_a = _mm256_fmadd_ps(q0, _mm256_loadu_ps(c1.add(base0)), acc1_a);
        acc1_b = _mm256_fmadd_ps(q1, _mm256_loadu_ps(c1.add(base1)), acc1_b);
        acc2_a = _mm256_fmadd_ps(q0, _mm256_loadu_ps(c2.add(base0)), acc2_a);
        acc2_b = _mm256_fmadd_ps(q1, _mm256_loadu_ps(c2.add(base1)), acc2_b);
        acc3_a = _mm256_fmadd_ps(q0, _mm256_loadu_ps(c3.add(base0)), acc3_a);
        acc3_b = _mm256_fmadd_ps(q1, _mm256_loadu_ps(c3.add(base1)), acc3_b);
        i += 2;
    }

    if i < chunks {
        let base = i * 8;
        let q = _mm256_loadu_ps(q_ptr.add(base));
        acc0_a = _mm256_fmadd_ps(q, _mm256_loadu_ps(c0.add(base)), acc0_a);
        acc1_a = _mm256_fmadd_ps(q, _mm256_loadu_ps(c1.add(base)), acc1_a);
        acc2_a = _mm256_fmadd_ps(q, _mm256_loadu_ps(c2.add(base)), acc2_a);
        acc3_a = _mm256_fmadd_ps(q, _mm256_loadu_ps(c3.add(base)), acc3_a);
    }

    out[0] = hsum256_ps(_mm256_add_ps(acc0_a, acc0_b));
    out[1] = hsum256_ps(_mm256_add_ps(acc1_a, acc1_b));
    out[2] = hsum256_ps(_mm256_add_ps(acc2_a, acc2_b));
    out[3] = hsum256_ps(_mm256_add_ps(acc3_a, acc3_b));
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
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
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

/// Compute dot products from one query against a contiguous slab of candidate vectors.
///
/// `candidates` must contain `out.len()` back-to-back vectors of length `dim`.
pub fn dot_product_scores(query: &[f32], candidates: &[f32], dim: usize, out: &mut [f32]) {
    assert_eq!(query.len(), dim, "Query length mismatch");
    assert_eq!(
        candidates.len(),
        out.len().saturating_mul(dim),
        "Candidate slab length mismatch"
    );
    if out.is_empty() {
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        let groups = out.len() / 4;
        let remainder = out.len() % 4;

        if dim == 128 {
            for group_idx in 0..groups {
                let cand_base = group_idx * 4 * dim;
                unsafe {
                    dot_product_scores4_neon_const::<32>(
                        query,
                        &candidates[cand_base..cand_base + 4 * dim],
                        &mut out[group_idx * 4..group_idx * 4 + 4],
                    );
                }
            }
        } else if dim == 256 {
            for group_idx in 0..groups {
                let cand_base = group_idx * 4 * dim;
                unsafe {
                    dot_product_scores4_neon_const::<64>(
                        query,
                        &candidates[cand_base..cand_base + 4 * dim],
                        &mut out[group_idx * 4..group_idx * 4 + 4],
                    );
                }
            }
        } else if dim.is_multiple_of(4) {
            for group_idx in 0..groups {
                let cand_base = group_idx * 4 * dim;
                unsafe {
                    dot_product_scores4_neon(
                        query,
                        &candidates[cand_base..cand_base + 4 * dim],
                        dim,
                        &mut out[group_idx * 4..group_idx * 4 + 4],
                    );
                }
            }
        } else {
            for (idx, candidate) in candidates.chunks_exact(dim).enumerate() {
                out[idx] = dot_product(query, candidate);
            }
            return;
        }

        let tail_base = groups * 4;
        for tail_idx in 0..remainder {
            let idx = tail_base + tail_idx;
            let start = idx * dim;
            out[idx] = dot_product(query, &candidates[start..start + dim]);
        }
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            let groups = out.len() / 4;
            let remainder = out.len() % 4;
            if dim == 128 {
                for group_idx in 0..groups {
                    let cand_base = group_idx * 4 * dim;
                    if group_idx + 1 < groups {
                        unsafe {
                            use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
                            _mm_prefetch(
                                candidates.as_ptr().add((group_idx + 1) * 4 * dim) as *const i8,
                                _MM_HINT_T0,
                            );
                        }
                    }
                    unsafe {
                        dot_product_scores4_avx2_const::<16>(
                            query,
                            &candidates[cand_base..cand_base + 4 * dim],
                            &mut out[group_idx * 4..group_idx * 4 + 4],
                        );
                    }
                }
            } else if dim == 256 {
                for group_idx in 0..groups {
                    let cand_base = group_idx * 4 * dim;
                    if group_idx + 1 < groups {
                        unsafe {
                            use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
                            _mm_prefetch(
                                candidates.as_ptr().add((group_idx + 1) * 4 * dim) as *const i8,
                                _MM_HINT_T0,
                            );
                        }
                    }
                    unsafe {
                        dot_product_scores4_avx2_const::<32>(
                            query,
                            &candidates[cand_base..cand_base + 4 * dim],
                            &mut out[group_idx * 4..group_idx * 4 + 4],
                        );
                    }
                }
            } else if dim.is_multiple_of(8) {
                for group_idx in 0..groups {
                    let cand_base = group_idx * 4 * dim;
                    if group_idx + 1 < groups && dim >= 128 {
                        unsafe {
                            use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
                            _mm_prefetch(
                                candidates.as_ptr().add((group_idx + 1) * 4 * dim) as *const i8,
                                _MM_HINT_T0,
                            );
                        }
                    }
                    unsafe {
                        dot_product_scores4_avx2(
                            query,
                            &candidates[cand_base..cand_base + 4 * dim],
                            dim,
                            &mut out[group_idx * 4..group_idx * 4 + 4],
                        );
                    }
                }
            } else {
                for (idx, candidate) in candidates.chunks_exact(dim).enumerate() {
                    out[idx] = dot_product(query, candidate);
                }
                return;
            }

            let tail_base = groups * 4;
            for tail_idx in 0..remainder {
                let idx = tail_base + tail_idx;
                let start = idx * dim;
                out[idx] = dot_product(query, &candidates[start..start + dim]);
            }
            return;
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    for (idx, candidate) in candidates.chunks_exact(dim).enumerate() {
        out[idx] = dot_product(query, candidate);
    }
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

    fn assert_batched_scores_match_scalar(dim: usize, candidate_count: usize) {
        let query = (0..dim)
            .map(|idx| ((idx % 17) as f32 - 8.0) / 17.0)
            .collect::<Vec<_>>();
        let mut candidates = Vec::with_capacity(dim * candidate_count);
        for candidate_idx in 0..candidate_count {
            for lane_idx in 0..dim {
                candidates.push((((candidate_idx * 13 + lane_idx * 7) % 31) as f32 - 15.0) / 31.0);
            }
        }
        let mut scores = vec![0.0f32; candidate_count];
        dot_product_scores(&query, &candidates, dim, &mut scores);

        for (idx, candidate) in candidates.chunks_exact(dim).enumerate() {
            let expected = dot_product(&query, candidate);
            assert!(
                (scores[idx] - expected).abs() < 1e-4,
                "dim={dim} candidate_count={candidate_count} idx={idx} expected={expected} got={}",
                scores[idx]
            );
        }
    }

    #[test]
    fn test_dot_product_scores_matches_scalar_specialized_128() {
        for candidate_count in [3usize, 4, 7] {
            assert_batched_scores_match_scalar(128, candidate_count);
        }
    }

    #[test]
    fn test_dot_product_scores_matches_scalar_specialized_256() {
        for candidate_count in [3usize, 4, 7] {
            assert_batched_scores_match_scalar(256, candidate_count);
        }
    }

    #[test]
    fn test_dot_product_scores_matches_scalar_generic_dim() {
        for candidate_count in [3usize, 4, 7] {
            assert_batched_scores_match_scalar(96, candidate_count);
        }
    }
}
