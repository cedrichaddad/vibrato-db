//! `vibrato-edge`: C ABI wrapper for embedded/edge deployments.

use std::collections::VecDeque;
use std::ffi::{c_char, CStr};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::slice;
use std::sync::Arc;

use vibrato_core::hnsw::HNSW;
use vibrato_core::simd::{dot_product, l2_normalized};
use vibrato_core::store::VectorStore;

pub struct EdgeDb {
    dim: usize,
    #[allow(dead_code)]
    vectors: Arc<Vec<Vec<f32>>>,
    hnsw: HNSW,
}

#[derive(Debug, Clone)]
struct OverlayVector {
    id: usize,
    vec: Vec<f32>,
}

#[derive(Debug)]
pub enum EdgeRuntimeError {
    Io(std::io::Error),
    Store(vibrato_core::store::StoreError),
    DimensionMismatch { expected: usize, actual: usize },
}

impl std::fmt::Display for EdgeRuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io error: {e}"),
            Self::Store(e) => write!(f, "store error: {e}"),
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "dimension mismatch: expected {expected}, got {actual}")
            }
        }
    }
}

impl std::error::Error for EdgeRuntimeError {}

impl From<std::io::Error> for EdgeRuntimeError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<vibrato_core::store::StoreError> for EdgeRuntimeError {
    fn from(value: vibrato_core::store::StoreError) -> Self {
        Self::Store(value)
    }
}

/// Immutable mmap base + bounded mutable overlay runtime for edge deployments.
pub struct EdgeRuntime {
    base_store: Arc<VectorStore>,
    base_hnsw: HNSW,
    overlay: VecDeque<OverlayVector>,
    overlay_capacity: usize,
    next_overlay_id: usize,
}

impl EdgeRuntime {
    /// Open an immutable mmap base from a `.vdb` file and create a bounded overlay.
    pub fn open<P: AsRef<Path>>(
        vdb_path: P,
        overlay_capacity: usize,
    ) -> Result<Self, EdgeRuntimeError> {
        let base_store = Arc::new(VectorStore::open(vdb_path)?);
        Ok(Self::from_store(base_store, overlay_capacity))
    }

    /// Build runtime from an already-open mmap store (test/helper path).
    pub fn from_store(base_store: Arc<VectorStore>, overlay_capacity: usize) -> Self {
        let base_for_hnsw = base_store.clone();
        let mut base_hnsw = HNSW::new_with_accessor_and_seed(
            16,
            100,
            move |id, sink| sink(base_for_hnsw.get(id)),
            42,
        );
        for id in 0..base_store.count {
            base_hnsw.insert(id);
        }

        Self {
            next_overlay_id: base_store.count,
            base_store,
            base_hnsw,
            overlay: VecDeque::with_capacity(overlay_capacity.max(1)),
            overlay_capacity: overlay_capacity.max(1),
        }
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.base_store.dim
    }

    #[inline]
    pub fn overlay_len(&self) -> usize {
        self.overlay.len()
    }

    /// Insert a vector into the bounded mutable overlay.
    pub fn insert_overlay(&mut self, vector: &[f32]) -> Result<usize, EdgeRuntimeError> {
        if vector.len() != self.base_store.dim {
            return Err(EdgeRuntimeError::DimensionMismatch {
                expected: self.base_store.dim,
                actual: vector.len(),
            });
        }
        if self.overlay.len() >= self.overlay_capacity {
            self.overlay.pop_front();
        }
        let id = self.next_overlay_id;
        self.next_overlay_id = self.next_overlay_id.saturating_add(1);
        self.overlay.push_back(OverlayVector {
            id,
            vec: l2_normalized(vector),
        });
        Ok(id)
    }

    /// Query both immutable mmap base and mutable overlay, then merge top-k.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> Result<Vec<(usize, f32)>, EdgeRuntimeError> {
        if query.len() != self.base_store.dim {
            return Err(EdgeRuntimeError::DimensionMismatch {
                expected: self.base_store.dim,
                actual: query.len(),
            });
        }
        if k == 0 {
            return Ok(Vec::new());
        }

        let normalized = l2_normalized(query);
        let mut merged = self.base_hnsw.search(&normalized, k.max(32), ef.max(k));
        for ov in &self.overlay {
            merged.push((ov.id, dot_product(&normalized, &ov.vec)));
        }
        merged.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        if merged.len() > k {
            merged.truncate(k);
        }
        Ok(merged)
    }
}

#[inline]
fn checked_len(a: usize, b: usize) -> Option<usize> {
    a.checked_mul(b)
}

/// Best-effort MADV_RANDOM hint for mmap-backed regions.
///
/// Returns `0` on success and also on unsupported platforms/errors (soft-fail by design).
#[no_mangle]
pub extern "C" fn vibrato_edge_advise_random(ptr: *const u8, len: usize) -> i32 {
    if ptr.is_null() || len == 0 {
        return 0;
    }
    #[cfg(unix)]
    {
        let rc = unsafe { libc::madvise(ptr.cast_mut().cast(), len, libc::MADV_RANDOM) };
        if rc != 0 {
            return 0;
        }
    }
    #[cfg(not(unix))]
    {
        let _ = (ptr, len);
    }
    0
}

/// Open an edge runtime from a mmap-backed `.vdb` file and bounded overlay capacity.
///
/// Returns NULL on invalid inputs or open failure.
///
/// # Safety
/// `path_ptr` must be a valid null-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn vibrato_edge_open_runtime(
    path_ptr: *const c_char,
    overlay_capacity: usize,
) -> *mut EdgeRuntime {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if path_ptr.is_null() {
            return std::ptr::null_mut();
        }
        let c_str = unsafe { CStr::from_ptr(path_ptr) };
        let Ok(path) = c_str.to_str() else {
            return std::ptr::null_mut();
        };
        match EdgeRuntime::open(path, overlay_capacity) {
            Ok(runtime) => Box::into_raw(Box::new(runtime)),
            Err(_) => std::ptr::null_mut(),
        }
    }));
    result.unwrap_or(std::ptr::null_mut())
}

/// Free an `EdgeRuntime` pointer returned by `vibrato_edge_open_runtime`.
#[no_mangle]
pub extern "C" fn vibrato_edge_runtime_free(runtime_ptr: *mut EdgeRuntime) {
    if runtime_ptr.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
        drop(Box::from_raw(runtime_ptr));
    }));
}

/// Insert one vector into the bounded overlay.
///
/// Returns `0` on success and `-1` on error.
///
/// # Safety
/// - `runtime_ptr` must be a valid runtime pointer.
/// - `vector_ptr` must point to `dim` contiguous `f32`s.
/// - `out_id_ptr` must be valid for writing one `usize`.
#[no_mangle]
pub unsafe extern "C" fn vibrato_edge_overlay_insert(
    runtime_ptr: *mut EdgeRuntime,
    vector_ptr: *const f32,
    dim: usize,
    out_id_ptr: *mut usize,
) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if runtime_ptr.is_null() || vector_ptr.is_null() || out_id_ptr.is_null() {
            return -1;
        }
        let runtime = unsafe { &mut *runtime_ptr };
        let vec = unsafe { slice::from_raw_parts(vector_ptr, dim) };
        match runtime.insert_overlay(vec) {
            Ok(id) => {
                unsafe {
                    *out_id_ptr = id;
                }
                0
            }
            Err(_) => -1,
        }
    }));
    result.unwrap_or(-1)
}

/// Search a batch of queries against mmap base + overlay runtime.
///
/// Returns `0` on success and `-1` on validation/runtime error.
///
/// # Safety
/// - `runtime_ptr` must be valid.
/// - `queries_ptr` must point to `num_queries * dim` contiguous `f32`s.
/// - `out_ids_ptr` and `out_scores_ptr` must point to `num_queries * k` writable slots.
#[no_mangle]
pub unsafe extern "C" fn vibrato_edge_runtime_search_batch(
    runtime_ptr: *mut EdgeRuntime,
    queries_ptr: *const f32,
    num_queries: usize,
    k: usize,
    ef: usize,
    out_ids_ptr: *mut usize,
    out_scores_ptr: *mut f32,
) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if runtime_ptr.is_null() {
            return -1;
        }
        if num_queries == 0 || k == 0 {
            return 0;
        }
        if queries_ptr.is_null() || out_ids_ptr.is_null() || out_scores_ptr.is_null() {
            return -1;
        }

        let runtime = unsafe { &mut *runtime_ptr };
        let query_floats = match checked_len(num_queries, runtime.dim()) {
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
            let query_start = qi * runtime.dim();
            let query = &queries[query_start..query_start + runtime.dim()];
            let results = match runtime.search(query, k, ef.max(k)) {
                Ok(r) => r,
                Err(_) => return -1,
            };

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
    use std::ffi::CString;
    use vibrato_core::format::VdbWriter;

    fn create_test_vdb(path: &std::path::Path, vectors: &[Vec<f32>]) {
        let dim = vectors.first().map(|v| v.len()).unwrap_or(0);
        let mut writer = VdbWriter::new(path, dim).expect("vdb writer");
        for v in vectors {
            writer.write_vector(v).expect("write vector");
        }
        writer.finish().expect("finish writer");
    }

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

    #[test]
    fn edge_runtime_overlay_insert_and_search() {
        let dir = tempfile::tempdir().unwrap();
        let vdb_path = dir.path().join("edge.vdb");
        create_test_vdb(
            &vdb_path,
            &[
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0, 0.0],
                vec![0.0, 0.0, 1.0, 0.0],
            ],
        );

        let mut runtime = EdgeRuntime::open(&vdb_path, 2).expect("open runtime");
        assert_eq!(runtime.overlay_len(), 0);

        let overlay_id = runtime
            .insert_overlay(&[0.0, 0.0, 0.0, 1.0])
            .expect("insert overlay");
        assert!(overlay_id >= 3);
        assert_eq!(runtime.overlay_len(), 1);

        let results = runtime
            .search(&[0.0, 0.0, 0.0, 1.0], 3, 32)
            .expect("search");
        assert!(!results.is_empty());
        assert_eq!(results[0].0, overlay_id);
        assert!(results[0].1 > 0.99);
    }

    #[test]
    fn edge_runtime_overlay_is_bounded() {
        let dir = tempfile::tempdir().unwrap();
        let vdb_path = dir.path().join("edge_cap.vdb");
        create_test_vdb(&vdb_path, &[vec![1.0, 0.0], vec![0.0, 1.0]]);

        let mut runtime = EdgeRuntime::open(&vdb_path, 2).expect("open runtime");
        runtime.insert_overlay(&[1.0, 0.0]).unwrap();
        runtime.insert_overlay(&[0.0, 1.0]).unwrap();
        runtime.insert_overlay(&[0.7, 0.7]).unwrap();
        assert_eq!(runtime.overlay_len(), 2);
    }

    #[test]
    fn edge_runtime_c_api_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let vdb_path = dir.path().join("edge_runtime.vdb");
        create_test_vdb(
            &vdb_path,
            &[
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ],
        );

        let path = CString::new(vdb_path.to_str().unwrap()).unwrap();
        let runtime = unsafe { vibrato_edge_open_runtime(path.as_ptr(), 2) };
        assert!(!runtime.is_null());

        let overlay_vec = [0.0f32, 0.0, 1.0];
        let mut overlay_id = usize::MAX;
        let rc = unsafe {
            vibrato_edge_overlay_insert(
                runtime,
                overlay_vec.as_ptr(),
                overlay_vec.len(),
                &mut overlay_id as *mut usize,
            )
        };
        assert_eq!(rc, 0);
        assert!(overlay_id >= 3);

        let queries = [0.0f32, 0.0, 1.0];
        let mut out_ids = [usize::MAX; 2];
        let mut out_scores = [f32::NEG_INFINITY; 2];
        let rc = unsafe {
            vibrato_edge_runtime_search_batch(
                runtime,
                queries.as_ptr(),
                1,
                2,
                32,
                out_ids.as_mut_ptr(),
                out_scores.as_mut_ptr(),
            )
        };
        assert_eq!(rc, 0);
        assert!(out_scores[0].is_finite());

        vibrato_edge_runtime_free(runtime);
    }
}
