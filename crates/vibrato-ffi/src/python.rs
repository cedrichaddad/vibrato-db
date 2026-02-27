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
        // Keep the query object alive for the full duration of search to prevent
        // Python GC from reclaiming exporter-owned memory while Rust is reading it.
        let _query_owner: Py<PyAny> = query.clone().unbind();
        let query_buf = PyBuffer::<f32>::get(query)?;

        if query_buf.dimensions() != 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "query must be a 1D float32 buffer (e.g. numpy.ndarray)",
            ));
        }
        if !query_buf.is_c_contiguous() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "query buffer must be C-contiguous",
            ));
        }

        let query_len = query_buf.item_count();
        if query_len != self.store.dim {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Dimension mismatch: expected {}, got {}",
                self.store.dim, query_len
            )));
        }

        let query_ptr = query_buf.buf_ptr() as *const f32;
        let index = self.index.clone();

        // Release GIL for potentially long-running search.
        // SAFETY:
        // - `query_ptr` points to a C-contiguous float32 exporter buffer.
        // - `query_buf` and `_query_owner` are kept alive across this call scope.
        // - We only read `query_len` elements.
        let results = py.allow_threads(move || {
            let query_slice = unsafe { std::slice::from_raw_parts(query_ptr, query_len) };
            let index_guard = index.read();
            index_guard.search(query_slice, k, ef)
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
