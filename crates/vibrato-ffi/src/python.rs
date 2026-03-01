use std::path::PathBuf;
use std::sync::Arc;

use parking_lot::RwLock;
use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use vibrato_core::hnsw::HNSW;
use vibrato_core::store::VectorStore;

/// Vibrato-DB Index (Python Wrapper)
#[pyclass]
pub struct VibratoIndex {
    // We hold the HNSW index wrapped in a lock for potential thread safety
    // if we ever support concurrent Python threads accessing the same object.
    // Also, HNSW needs to own the vector data via closure, so we keep
    // a reference to the store alive here too.
    index: Arc<RwLock<HNSW>>,
    store: Arc<VectorStore>,
}

struct ReadOnlyF32Buffer {
    buffer: PyBuffer<f32>,
}

impl ReadOnlyF32Buffer {
    fn try_from_object(query: &Bound<'_, PyAny>) -> PyResult<Option<Self>> {
        let Ok(buffer) = PyBuffer::<f32>::get_bound(query) else {
            return Ok(None);
        };
        if !buffer.readonly() || buffer.dimensions() != 1 || !buffer.is_c_contiguous() {
            return Ok(None);
        }
        if buffer.buf_ptr().is_null() {
            return Ok(None);
        }
        Ok(Some(Self { buffer }))
    }

    #[inline]
    fn as_slice(&self) -> &[f32] {
        unsafe {
            std::slice::from_raw_parts(
                self.buffer.buf_ptr() as *const f32,
                self.buffer.item_count(),
            )
        }
    }
}

#[pymethods]
impl VibratoIndex {
    /// Open a Vibrato-DB index
    ///
    /// Args:
    ///     index_path: Path to the .idx file
    ///     data_path: Path to the .vdb file
    #[new]
    pub fn new(index_path: String, data_path: String) -> PyResult<Self> {
        let index_path = PathBuf::from(index_path);
        let data_path = PathBuf::from(data_path);

        // Load store (mmap)
        let store = VectorStore::open(&data_path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to open data file: {}", e))
        })?;
        let store = Arc::new(store);

        // Load index
        // Capture store in closure
        let store_clone = store.clone();
        let index =
            HNSW::load_with_accessor(&index_path, move |id, sink| sink(store_clone.get(id)))
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                        "Failed to load index: {}",
                        e
                    ))
                })?;

        Ok(VibratoIndex {
            index: Arc::new(RwLock::new(index)),
            store,
        })
    }

    /// Search for nearest neighbors
    ///
    /// Args:
    ///     query: Query vector (list of floats)
    ///     k: Number of results to return (default: 10)
    ///     ef: Search depth (default: 50)
    ///
    /// Returns:
    ///     List of (id, score) tuples
    #[pyo3(signature = (query, k=10, ef=50))]
    pub fn search(
        &self,
        py: Python<'_>,
        query: &Bound<'_, PyAny>,
        k: usize,
        ef: usize,
    ) -> PyResult<Vec<(usize, f32)>> {
        let index = self.index.clone();

        // Prefer zero-copy for read-only contiguous f32 buffers. Writable buffers
        // are copied before releasing the GIL to avoid data races with Python threads.
        if let Some(buf) = ReadOnlyF32Buffer::try_from_object(query)? {
            if buf.as_slice().len() != self.store.dim {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Dimension mismatch: expected {}, got {}",
                    self.store.dim,
                    buf.as_slice().len()
                )));
            }
            let owner = query.clone().unbind();
            let slice = buf.as_slice();
            let results = py.allow_threads(|| {
                let index_guard = index.read();
                index_guard.search(slice, k, ef)
            });
            // Drop Python-owned resources only after allow_threads() has returned,
            // which guarantees the GIL is re-acquired.
            drop(buf);
            drop(owner);
            return Ok(results);
        }

        let query_vec: Vec<f32> = query.extract::<Vec<f32>>()?;
        if query_vec.len() != self.store.dim {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Dimension mismatch: expected {}, got {}",
                self.store.dim,
                query_vec.len()
            )));
        }

        // Release GIL for potentially long-running search.
        let results = py.allow_threads(move || {
            let index_guard = index.read();
            index_guard.search(&query_vec, k, ef)
        });

        Ok(results)
    }

    /// Get index health/stats
    pub fn health(&self) -> PyResult<std::collections::HashMap<String, usize>> {
        let stats = {
            let index = self.index.read();
            index.stats()
        };

        let mut map = std::collections::HashMap::new();
        map.insert("vectors".to_string(), self.store.count);
        map.insert("dimensions".to_string(), self.store.dim);
        map.insert("nodes".to_string(), stats.num_nodes);
        map.insert("max_layer".to_string(), stats.max_layer);
        map.insert("total_edges".to_string(), stats.total_edges);
        map.insert("m".to_string(), stats.m);
        map.insert("ef_construction".to_string(), stats.ef_construction);

        Ok(map)
    }
}

/// Vibrato-DB Python Module
#[pymodule]
fn vibrato_ffi(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<VibratoIndex>()?;
    Ok(())
}

#[cfg(all(test, feature = "python"))]
mod tests {
    use super::*;
    use pyo3::types::PyDict;

    #[test]
    fn readonly_f32_buffer_is_detected() {
        Python::with_gil(|py| {
            let locals = PyDict::new_bound(py);
            py.run_bound(
                "import array\n\
                 a = array.array('f', [1.0, 2.0, 3.0])\n\
                 m = memoryview(a).toreadonly()",
                None,
                Some(&locals),
            )?;
            let m = locals
                .get_item("m")?
                .expect("memoryview object should exist");
            let maybe = ReadOnlyF32Buffer::try_from_object(&m)?;
            let ro = maybe.expect("readonly contiguous f32 buffer should be detected");
            assert_eq!(ro.as_slice().len(), 3);
            assert!((ro.as_slice()[0] - 1.0).abs() < f32::EPSILON);
            Ok::<(), PyErr>(())
        })
        .expect("python readonly buffer test");
    }

    #[test]
    fn writable_f32_buffer_is_rejected_for_zero_copy() {
        Python::with_gil(|py| {
            let locals = PyDict::new_bound(py);
            py.run_bound(
                "import array\n\
                 a = array.array('f', [1.0, 2.0, 3.0])\n\
                 m = memoryview(a)",
                None,
                Some(&locals),
            )?;
            let m = locals
                .get_item("m")?
                .expect("memoryview object should exist");
            let maybe = ReadOnlyF32Buffer::try_from_object(&m)?;
            assert!(
                maybe.is_none(),
                "writable buffer must fall back to copy path"
            );
            Ok::<(), PyErr>(())
        })
        .expect("python writable buffer test");
    }

    #[test]
    fn non_buffer_input_does_not_use_zero_copy() {
        Python::with_gil(|py| {
            let list_obj = pyo3::types::PyList::new_bound(py, [1.0f32, 2.0, 3.0]);
            let maybe = ReadOnlyF32Buffer::try_from_object(list_obj.as_any())?;
            assert!(maybe.is_none(), "non-buffer input should not be accepted");
            Ok::<(), PyErr>(())
        })
        .expect("python non-buffer test");
    }
}
