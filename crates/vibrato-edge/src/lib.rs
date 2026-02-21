//! `vibrato-edge`: C ABI wrapper for embedded/edge deployments.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::slice;
use std::sync::Arc;

use vibrato_core::hnsw::HNSW;
use vibrato_core::simd::l2_normalized;

pub struct EdgeDb {
    dim: usize,
    #[allow(dead_code)]
    vectors: Arc<Vec<Vec<f32>>>,
    hnsw: HNSW,
}

#[inline]
fn checked_len(a: usize, b: usize) -> Option<usize> {
    a.checked_mul(b)
}

/// Build an in-memory HNSW index from a contiguous row-major vector matrix.
///
/// Returns null on invalid arguments or internal failure.
#[no_mangle]
pub extern "C" fn vibrato_edge_build(
    dim: usize,
    vectors_ptr: *const f32,
    num_vectors: usize,
) -> *mut EdgeDb {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if dim == 0 || num_vectors == 0 {
            return std::ptr::null_mut();
        }
        let total = match checked_len(dim, num_vectors) {
            Some(v) => v,
            None => return std::ptr::null_mut(),
        };
        if vectors_ptr.is_null() {
            return std::ptr::null_mut();
        }

        let raw = unsafe { slice::from_raw_parts(vectors_ptr, total) };
        let mut rows = Vec::with_capacity(num_vectors);
        for row in raw.chunks_exact(dim) {
            rows.push(l2_normalized(row));
        }

        let vectors = Arc::new(rows);
        let vectors_for_hnsw = vectors.clone();
        let mut hnsw = HNSW::new_with_accessor_and_seed(
            16,
            100,
            move |id, sink| sink(&vectors_for_hnsw[id]),
            42,
        );
        for id in 0..num_vectors {
            hnsw.insert(id);
        }

        Box::into_raw(Box::new(EdgeDb { dim, vectors, hnsw }))
    }));
    result.unwrap_or(std::ptr::null_mut())
}

/// Free a database pointer allocated by `vibrato_edge_build`.
#[no_mangle]
pub extern "C" fn vibrato_edge_free(db_ptr: *mut EdgeDb) {
    if db_ptr.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
        drop(Box::from_raw(db_ptr));
    }));
}

/// Search a batch of queries.
///
/// # Parameters
/// - `db_ptr`: Pointer returned by `vibrato_edge_build`
/// - `queries_ptr`: Row-major query matrix (`num_queries * dim`)
/// - `num_queries`: Number of query rows
/// - `k`: Results per query
/// - `ef`: HNSW search depth
/// - `out_ids_ptr`: Output IDs buffer (`num_queries * k`)
/// - `out_scores_ptr`: Output scores buffer (`num_queries * k`)
///
/// Returns `0` on success and `-1` on validation/runtime error.
#[no_mangle]
pub extern "C" fn vibrato_search_batch(
    db_ptr: *mut EdgeDb,
    queries_ptr: *const f32,
    num_queries: usize,
    k: usize,
    ef: usize,
    out_ids_ptr: *mut usize,
    out_scores_ptr: *mut f32,
) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if db_ptr.is_null() {
            return -1;
        }
        if num_queries == 0 || k == 0 {
            return 0;
        }
        if queries_ptr.is_null() || out_ids_ptr.is_null() || out_scores_ptr.is_null() {
            return -1;
        }

        let db = unsafe { &mut *db_ptr };
        let query_floats = match checked_len(num_queries, db.dim) {
            Some(v) => v,
            None => return -1,
        };
        let out_len = match checked_len(num_queries, k) {
            Some(v) => v,
            None => return -1,
        };

        let queries = unsafe { slice::from_raw_parts(queries_ptr, query_floats) };
        let out_ids = unsafe { slice::from_raw_parts_mut(out_ids_ptr, out_len) };
        let out_scores = unsafe { slice::from_raw_parts_mut(out_scores_ptr, out_len) };

        for qi in 0..num_queries {
            let query_start = qi * db.dim;
            let query = &queries[query_start..query_start + db.dim];
            let normalized = l2_normalized(query);
            let results = db.hnsw.search(&normalized, k, ef.max(k));

            let out_start = qi * k;
            let mut ri = 0usize;
            for (id, score) in &results {
                out_ids[out_start + ri] = *id;
                out_scores[out_start + ri] = *score;
                ri += 1;
                if ri >= k {
                    break;
                }
            }
            while ri < k {
                out_ids[out_start + ri] = usize::MAX;
                out_scores[out_start + ri] = f32::NEG_INFINITY;
                ri += 1;
            }
        }
        0
    }));
    result.unwrap_or(-1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edge_build_and_search_batch_roundtrip() {
        let dim = 4usize;
        let vectors = vec![
            1.0f32, 0.0, 0.0, 0.0, // id 0
            0.0f32, 1.0, 0.0, 0.0, // id 1
            0.0f32, 0.0, 1.0, 0.0, // id 2
        ];

        let db = vibrato_edge_build(dim, vectors.as_ptr(), 3);
        assert!(!db.is_null());

        let queries = vec![1.0f32, 0.0, 0.0, 0.0];
        let mut out_ids = vec![usize::MAX; 2];
        let mut out_scores = vec![f32::NEG_INFINITY; 2];
        let rc = vibrato_search_batch(
            db,
            queries.as_ptr(),
            1,
            2,
            20,
            out_ids.as_mut_ptr(),
            out_scores.as_mut_ptr(),
        );
        assert_eq!(rc, 0);
        assert_eq!(out_ids[0], 0);
        assert!(out_scores[0].is_finite());

        vibrato_edge_free(db);
    }

    #[test]
    fn edge_rejects_null_inputs() {
        let rc = vibrato_search_batch(
            std::ptr::null_mut(),
            std::ptr::null(),
            1,
            1,
            20,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );
        assert_eq!(rc, -1);
    }
}
