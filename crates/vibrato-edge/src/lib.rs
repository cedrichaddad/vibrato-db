//! `vibrato-edge`: C ABI wrapper for embedded/edge deployments.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};
use std::ffi::{c_char, CStr, CString};
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::slice;
use std::sync::Arc;

use vibrato_core::format_v2::VdbHeaderV2;
use vibrato_core::hnsw::HNSW;
use vibrato_core::simd::{dot_product, l2_normalized};
use vibrato_core::store::VectorStore;

const EDGE_DEFAULT_OVERLAY_CAPACITY: usize = 1024;
const EDGE_MAX_OVERLAY_CAPACITY: usize = 65_536;

pub struct EdgeDb {
    dim: usize,
    #[allow(dead_code)]
    vectors: Arc<Vec<Vec<f32>>>,
    hnsw: HNSW,
}

thread_local! {
    static LAST_ERROR: std::cell::RefCell<Option<CString>> = const { std::cell::RefCell::new(None) };
}

fn set_last_error(msg: impl AsRef<str>) {
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = CString::new(msg.as_ref()).ok();
    });
}

fn clear_last_error() {
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = None;
    });
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

#[derive(Debug, Clone, Copy)]
struct TopKItem {
    id: usize,
    score: f32,
}

impl Eq for TopKItem {}

impl PartialEq for TopKItem {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.score.to_bits() == other.score.to_bits()
    }
}

impl Ord for TopKItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| self.id.cmp(&other.id))
    }
}

impl PartialOrd for TopKItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl EdgeRuntime {
    /// Open an immutable mmap base from a `.vdb` file and create a bounded overlay.
    pub fn open<P: AsRef<Path>>(
        vdb_path: P,
        overlay_capacity: usize,
    ) -> Result<Self, EdgeRuntimeError> {
        let base_path = vdb_path.as_ref().to_path_buf();
        let base_store = Arc::new(VectorStore::open(&base_path)?);
        let base_hnsw = Self::load_embedded_graph_or_rebuild(&base_path, base_store.clone());
        Ok(Self::from_parts(base_store, base_hnsw, overlay_capacity))
    }

    /// Build runtime from an already-open mmap store (test/helper path).
    pub fn from_store(base_store: Arc<VectorStore>, overlay_capacity: usize) -> Self {
        let base_hnsw = Self::build_hnsw_from_store(base_store.clone());
        Self::from_parts(base_store, base_hnsw, overlay_capacity)
    }

    fn from_parts(base_store: Arc<VectorStore>, base_hnsw: HNSW, overlay_capacity: usize) -> Self {
        advise_random(base_store.vector_bytes());
        let overlay_capacity = sanitize_overlay_capacity(overlay_capacity);
        Self {
            next_overlay_id: base_store.count,
            base_store,
            base_hnsw,
            overlay: VecDeque::with_capacity(overlay_capacity),
            overlay_capacity,
        }
    }

    fn build_hnsw_from_store(base_store: Arc<VectorStore>) -> HNSW {
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
        base_hnsw
    }

    fn load_embedded_graph_or_rebuild(base_path: &Path, base_store: Arc<VectorStore>) -> HNSW {
        let mut header_bytes = [0u8; 64];
        let Ok(mut file) = File::open(base_path) else {
            return Self::build_hnsw_from_store(base_store);
        };
        if file.read_exact(&mut header_bytes).is_err() {
            return Self::build_hnsw_from_store(base_store);
        }
        let Ok(header) = VdbHeaderV2::from_bytes(&header_bytes) else {
            return Self::build_hnsw_from_store(base_store);
        };
        if !header.has_graph() || header.graph_offset == 0 {
            return Self::build_hnsw_from_store(base_store);
        }
        if file.seek(SeekFrom::Start(header.graph_offset)).is_err() {
            return Self::build_hnsw_from_store(base_store);
        }
        let mut reader = BufReader::new(file);
        let base_for_hnsw = base_store.clone();
        match HNSW::load_from_reader_with_accessor(&mut reader, move |id, sink| {
            sink(base_for_hnsw.get(id))
        }) {
            Ok(index) => index,
            Err(_) => Self::build_hnsw_from_store(base_store),
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
        let mut top_k: BinaryHeap<Reverse<TopKItem>> = BinaryHeap::with_capacity(k + 1);
        let mut consider = |id: usize, score: f32| {
            let item = TopKItem { id, score };
            if top_k.len() < k {
                top_k.push(Reverse(item));
                return;
            }
            let should_insert = top_k.peek().map(|worst| item > worst.0).unwrap_or(true);
            if should_insert {
                let _ = top_k.pop();
                top_k.push(Reverse(item));
            }
        };

        for (id, score) in self.base_hnsw.search(&normalized, k.max(32), ef.max(k)) {
            consider(id, score);
        }
        for ov in &self.overlay {
            consider(ov.id, dot_product(&normalized, &ov.vec));
        }

        let mut merged = top_k
            .into_iter()
            .map(|Reverse(item)| (item.id, item.score))
            .collect::<Vec<_>>();
        merged.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        Ok(merged)
    }
}

#[inline]
fn checked_len(a: usize, b: usize) -> Option<usize> {
    a.checked_mul(b)
}

#[inline]
fn sanitize_overlay_capacity(requested: usize) -> usize {
    let requested = if requested == 0 {
        EDGE_DEFAULT_OVERLAY_CAPACITY
    } else {
        requested
    };
    requested.min(EDGE_MAX_OVERLAY_CAPACITY)
}

#[inline]
fn advise_random(bytes: &[u8]) {
    if bytes.is_empty() {
        return;
    }
    #[cfg(unix)]
    unsafe {
        let _ = libc::madvise(
            bytes.as_ptr().cast::<std::ffi::c_void>().cast_mut(),
            bytes.len(),
            libc::MADV_RANDOM,
        );
    }
}

/// Return the most recent edge C-ABI error for the current thread.
///
/// # Safety
/// Returned pointer is owned by this library and must not be freed by caller.
#[no_mangle]
pub extern "C" fn vibrato_edge_last_error() -> *const c_char {
    LAST_ERROR.with(|slot| {
        slot.borrow()
            .as_ref()
            .map(|s| s.as_ptr())
            .unwrap_or(std::ptr::null())
    })
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
            set_last_error("path pointer is null");
            return std::ptr::null_mut();
        }
        let c_str = unsafe { CStr::from_ptr(path_ptr) };
        let Ok(path) = c_str.to_str() else {
            set_last_error("path is not valid UTF-8");
            return std::ptr::null_mut();
        };
        match EdgeRuntime::open(path, overlay_capacity) {
            Ok(runtime) => {
                clear_last_error();
                Box::into_raw(Box::new(runtime))
            }
            Err(err) => {
                set_last_error(format!("failed to open edge runtime: {err}"));
                std::ptr::null_mut()
            }
        }
    }));
    result.unwrap_or_else(|_| {
        set_last_error("panic in vibrato_edge_open_runtime");
        std::ptr::null_mut()
    })
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
            set_last_error("runtime/vector/output pointer must not be null");
            return -1;
        }
        let runtime = unsafe { &mut *runtime_ptr };
        let vec = unsafe { slice::from_raw_parts(vector_ptr, dim) };
        match runtime.insert_overlay(vec) {
            Ok(id) => {
                unsafe {
                    *out_id_ptr = id;
                }
                clear_last_error();
                0
            }
            Err(err) => {
                set_last_error(format!("overlay insert failed: {err}"));
                -1
            }
        }
    }));
    result.unwrap_or_else(|_| {
        set_last_error("panic in vibrato_edge_overlay_insert");
        -1
    })
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
            set_last_error("runtime pointer is null");
            return -1;
        }
        if num_queries == 0 || k == 0 {
            clear_last_error();
            return 0;
        }
        if queries_ptr.is_null() || out_ids_ptr.is_null() || out_scores_ptr.is_null() {
            set_last_error("query/output pointers must not be null");
            return -1;
        }

        let runtime = unsafe { &mut *runtime_ptr };
        let query_floats = match checked_len(num_queries, runtime.dim()) {
            Some(v) => v,
            None => {
                set_last_error("query length overflow");
                return -1;
            }
        };
        let out_len = match checked_len(num_queries, k) {
            Some(v) => v,
            None => {
                set_last_error("output length overflow");
                return -1;
            }
        };

        let queries = unsafe { slice::from_raw_parts(queries_ptr, query_floats) };
        let out_ids = unsafe { slice::from_raw_parts_mut(out_ids_ptr, out_len) };
        let out_scores = unsafe { slice::from_raw_parts_mut(out_scores_ptr, out_len) };

        for qi in 0..num_queries {
            let query_start = qi * runtime.dim();
            let query = &queries[query_start..query_start + runtime.dim()];
            let results = match runtime.search(query, k, ef.max(k)) {
                Ok(r) => r,
                Err(err) => {
                    set_last_error(format!("runtime search failed: {err}"));
                    return -1;
                }
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
        clear_last_error();
        0
    }));
    result.unwrap_or_else(|_| {
        set_last_error("panic in vibrato_edge_runtime_search_batch");
        -1
    })
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
            set_last_error("dim and num_vectors must be > 0");
            return std::ptr::null_mut();
        }
        let total = match checked_len(dim, num_vectors) {
            Some(v) => v,
            None => {
                set_last_error("input length overflow");
                return std::ptr::null_mut();
            }
        };
        if vectors_ptr.is_null() {
            set_last_error("vectors pointer is null");
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

        clear_last_error();
        Box::into_raw(Box::new(EdgeDb { dim, vectors, hnsw }))
    }));
    result.unwrap_or_else(|_| {
        set_last_error("panic in vibrato_edge_build");
        std::ptr::null_mut()
    })
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
            set_last_error("database pointer is null");
            return -1;
        }
        if num_queries == 0 || k == 0 {
            clear_last_error();
            return 0;
        }
        if queries_ptr.is_null() || out_ids_ptr.is_null() || out_scores_ptr.is_null() {
            set_last_error("query/output pointers must not be null");
            return -1;
        }

        let db = unsafe { &mut *db_ptr };
        let query_floats = match checked_len(num_queries, db.dim) {
            Some(v) => v,
            None => {
                set_last_error("query length overflow");
                return -1;
            }
        };
        let out_len = match checked_len(num_queries, k) {
            Some(v) => v,
            None => {
                set_last_error("output length overflow");
                return -1;
            }
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
        clear_last_error();
        0
    }));
    result.unwrap_or_else(|_| {
        set_last_error("panic in vibrato_search_batch");
        -1
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::{CStr, CString};
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
        let err_ptr = vibrato_edge_last_error();
        assert!(!err_ptr.is_null());
        let err = unsafe { CStr::from_ptr(err_ptr) }.to_string_lossy();
        assert!(err.contains("database pointer is null"));
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
    fn edge_runtime_uses_default_overlay_capacity_when_zero_requested() {
        let dir = tempfile::tempdir().unwrap();
        let vdb_path = dir.path().join("edge_default_cap.vdb");
        create_test_vdb(&vdb_path, &[vec![1.0, 0.0], vec![0.0, 1.0]]);

        let mut runtime = EdgeRuntime::open(&vdb_path, 0).expect("open runtime");
        for _ in 0..(EDGE_DEFAULT_OVERLAY_CAPACITY + 10) {
            runtime.insert_overlay(&[1.0, 0.0]).unwrap();
        }
        assert_eq!(runtime.overlay_len(), EDGE_DEFAULT_OVERLAY_CAPACITY);
    }

    #[test]
    fn edge_runtime_caps_overlay_capacity() {
        let dir = tempfile::tempdir().unwrap();
        let vdb_path = dir.path().join("edge_max_cap.vdb");
        create_test_vdb(&vdb_path, &[vec![1.0, 0.0], vec![0.0, 1.0]]);

        let mut runtime =
            EdgeRuntime::open(&vdb_path, EDGE_MAX_OVERLAY_CAPACITY + 1000).expect("open runtime");
        for _ in 0..(EDGE_MAX_OVERLAY_CAPACITY + 10) {
            runtime.insert_overlay(&[1.0, 0.0]).unwrap();
        }
        assert_eq!(runtime.overlay_len(), EDGE_MAX_OVERLAY_CAPACITY);
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
