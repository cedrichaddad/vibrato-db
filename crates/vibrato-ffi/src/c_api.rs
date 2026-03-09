//! C-compatible API for DAW plugin integration
//!
//! All functions use `std::panic::catch_unwind` to prevent Rust panics
//! from unwinding into C/C++ code, which would cause undefined behavior
//! (and crash the host DAW).
//!
//! Return values:
//! - 0 = success
//! - -1 = error (call `vibrato_last_error` for details)

use std::ffi::{c_char, CStr, CString};
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;
use std::slice;
use std::sync::Arc;

use parking_lot::RwLock;
use vibrato_core::format_v2::VdbHeaderV2;
use vibrato_core::hnsw::HNSW;
use vibrato_core::simd::l2_normalized;
use vibrato_core::store::VectorStore;

/// Opaque handle to a Vibrato database
pub struct VibratoHandle {
    store: Arc<VectorStore>,
    index: RwLock<HNSW>,
}

// Thread-local last error message
thread_local! {
    static LAST_ERROR: std::cell::RefCell<Option<CString>> = std::cell::RefCell::new(None);
}

fn set_last_error(msg: &str) {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = CString::new(msg).ok();
    });
}

#[inline]
fn checked_len(a: usize, b: usize) -> Option<usize> {
    a.checked_mul(b)
}

fn load_hnsw_from_vdb_or_rebuild(path: &Path, store: Arc<VectorStore>) -> HNSW {
    let mut header_bytes = [0u8; 64];
    let Ok(mut file) = File::open(path) else {
        return rebuild_hnsw(store);
    };
    if file.read_exact(&mut header_bytes).is_err() {
        return rebuild_hnsw(store);
    }
    let Ok(header) = VdbHeaderV2::from_bytes(&header_bytes) else {
        return rebuild_hnsw(store);
    };
    if !header.has_graph() || header.graph_offset == 0 {
        return rebuild_hnsw(store);
    }
    if file.seek(SeekFrom::Start(header.graph_offset)).is_err() {
        return rebuild_hnsw(store);
    }
    let mut reader = BufReader::new(file);
    let store_for_hnsw = store.clone();
    match HNSW::load_from_reader_with_accessor(&mut reader, move |id, sink| {
        sink(store_for_hnsw.get(id))
    }) {
        Ok(index) => index,
        Err(_) => rebuild_hnsw(store),
    }
}

fn rebuild_hnsw(store: Arc<VectorStore>) -> HNSW {
    let store_for_hnsw = store.clone();
    let mut hnsw =
        HNSW::new_with_accessor_and_seed(16, 100, move |id, sink| sink(store_for_hnsw.get(id)), 42);
    for id in 0..store.count {
        hnsw.insert(id as u64, store.get(id));
    }
    hnsw
}

/// Open a Vibrato database
///
/// Returns a handle on success, NULL on error.
/// The caller must call `vibrato_close` when done.
///
/// # Safety
/// `path` must be a valid null-terminated UTF-8 C string.
#[no_mangle]
pub unsafe extern "C" fn vibrato_open(path: *const c_char) -> *mut VibratoHandle {
    std::panic::catch_unwind(|| {
        if path.is_null() {
            set_last_error("Path is null");
            return std::ptr::null_mut();
        }

        let c_str = unsafe { CStr::from_ptr(path) };
        let path_str = match c_str.to_str() {
            Ok(s) => s,
            Err(e) => {
                set_last_error(&format!("Invalid UTF-8 path: {}", e));
                return std::ptr::null_mut();
            }
        };

        if !Path::new(path_str).exists() {
            set_last_error(&format!("File not found: {}", path_str));
            return std::ptr::null_mut();
        }

        let store = match VectorStore::open(path_str) {
            Ok(store) => Arc::new(store),
            Err(e) => {
                set_last_error(&format!("Failed to open vector store: {}", e));
                return std::ptr::null_mut();
            }
        };

        let hnsw = load_hnsw_from_vdb_or_rebuild(Path::new(path_str), store.clone());

        let handle = Box::new(VibratoHandle {
            store,
            index: RwLock::new(hnsw),
        });

        Box::into_raw(handle)
    })
    .unwrap_or_else(|_| {
        set_last_error("Panic in vibrato_open");
        std::ptr::null_mut()
    })
}

/// Close a Vibrato database handle and free memory
///
/// # Safety
/// `handle` must be a valid pointer returned by `vibrato_open`,
/// or NULL (in which case this is a no-op).
#[no_mangle]
pub unsafe extern "C" fn vibrato_close(handle: *mut VibratoHandle) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if !handle.is_null() {
            unsafe {
                drop(Box::from_raw(handle));
            }
        }
    }));
}

/// Get the last error message
///
/// Returns a pointer to a null-terminated C string, or NULL if no error.
/// The returned string is valid until the next API call from the same thread.
///
/// # Safety
/// The returned pointer must not be freed by the caller.
#[no_mangle]
pub extern "C" fn vibrato_last_error() -> *const c_char {
    LAST_ERROR.with(|e| {
        e.borrow()
            .as_ref()
            .map(|s| s.as_ptr())
            .unwrap_or(std::ptr::null())
    })
}

/// Search the index.
///
/// Returns 0 on success, -1 on error.
///
/// # Safety
/// - `handle` must be a valid pointer returned by `vibrato_open`.
/// - `query_ptr` must point to `dim` contiguous `f32`s.
/// - `out_ids_ptr` and `out_scores_ptr` must each point to writable buffers of at least `k`.
#[no_mangle]
pub unsafe extern "C" fn vibrato_search(
    handle: *mut VibratoHandle,
    query_ptr: *const f32,
    dim: usize,
    k: usize,
    ef: usize,
    out_ids_ptr: *mut usize,
    out_scores_ptr: *mut f32,
) -> i32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if handle.is_null() {
            set_last_error("Handle is null");
            return -1;
        }
        if query_ptr.is_null() {
            set_last_error("Query pointer is null");
            return -1;
        }
        if out_ids_ptr.is_null() || out_scores_ptr.is_null() {
            set_last_error("Output pointers must not be null");
            return -1;
        }
        if k == 0 {
            return 0;
        }

        let handle = unsafe { &mut *handle };
        if dim != handle.store.dim {
            set_last_error(&format!(
                "Dimension mismatch: expected {}, got {}",
                handle.store.dim, dim
            ));
            return -1;
        }

        let Some(_out_len) = checked_len(k, 1) else {
            set_last_error("Output buffer length overflow");
            return -1;
        };

        let query = unsafe { slice::from_raw_parts(query_ptr, dim) };
        let out_ids = unsafe { slice::from_raw_parts_mut(out_ids_ptr, k) };
        let out_scores = unsafe { slice::from_raw_parts_mut(out_scores_ptr, k) };

        let normalized = l2_normalized(query);
        let results = handle.index.read().search(&normalized, k, ef.max(k));
        let mut i = 0usize;
        for (id, score) in &results {
            if i >= k {
                break;
            }
            out_ids[i] = usize::try_from(*id)
                .unwrap_or_else(|_| panic!("data integrity fault: ffi id overflow id={id}"));
            out_scores[i] = *score;
            i += 1;
        }
        while i < k {
            out_ids[i] = usize::MAX;
            out_scores[i] = f32::NEG_INFINITY;
            i += 1;
        }
        0
    }))
    .unwrap_or_else(|_| {
        set_last_error("Panic in vibrato_search");
        -1
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use vibrato_core::format::VdbWriter;

    fn create_test_vdb(path: &std::path::Path) {
        let mut writer = VdbWriter::new(path, 4).expect("vdb writer");
        writer.write_vector(&[1.0, 0.0, 0.0, 0.0]).unwrap();
        writer.write_vector(&[0.0, 1.0, 0.0, 0.0]).unwrap();
        writer.write_vector(&[0.0, 0.0, 1.0, 0.0]).unwrap();
        writer.finish().unwrap();
    }

    #[test]
    fn test_open_null_path() {
        let handle = unsafe { vibrato_open(std::ptr::null()) };
        assert!(handle.is_null());
    }

    #[test]
    fn test_open_nonexistent_file() {
        let path = CString::new("/nonexistent/file.vdb").unwrap();
        let handle = unsafe { vibrato_open(path.as_ptr()) };
        assert!(handle.is_null());
    }

    #[test]
    fn test_close_null_handle() {
        // Should not crash
        unsafe {
            vibrato_close(std::ptr::null_mut());
        }
    }

    #[test]
    fn test_open_close_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let vdb_path = dir.path().join("test.vdb");
        create_test_vdb(&vdb_path);
        let path = CString::new(vdb_path.to_str().unwrap()).unwrap();
        let handle = unsafe { vibrato_open(path.as_ptr()) };
        assert!(!handle.is_null());
        unsafe {
            vibrato_close(handle);
        }
    }

    #[test]
    fn test_search_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let vdb_path = dir.path().join("search_test.vdb");
        create_test_vdb(&vdb_path);
        let path = CString::new(vdb_path.to_str().unwrap()).unwrap();
        let handle = unsafe { vibrato_open(path.as_ptr()) };
        assert!(!handle.is_null());

        let query = [1.0f32, 0.0, 0.0, 0.0];
        let mut out_ids = [usize::MAX; 2];
        let mut out_scores = [f32::NEG_INFINITY; 2];
        let rc = unsafe {
            vibrato_search(
                handle,
                query.as_ptr(),
                query.len(),
                2,
                32,
                out_ids.as_mut_ptr(),
                out_scores.as_mut_ptr(),
            )
        };
        assert_eq!(rc, 0);
        assert_eq!(out_ids[0], 0);
        assert!(out_scores[0].is_finite());

        unsafe {
            vibrato_close(handle);
        }
    }
}
