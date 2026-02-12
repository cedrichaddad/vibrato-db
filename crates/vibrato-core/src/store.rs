//! Memory-mapped vector storage
//!
//! Provides zero-copy access to vectors stored in .vdb files using OS page cache.
//!
//! # Safety
//!
//! The `VectorStore` uses `bytemuck::cast_slice` to safely convert `&[u8]` to `&[f32]`
//! with proper alignment checks. This avoids crashes on ARM/M1 from unaligned access.

use std::fs::File;
use std::path::Path;

use memmap2::Mmap;
use thiserror::Error;

use crate::format::{FormatError, VdbHeader, HEADER_SIZE};

#[derive(Error, Debug)]
pub enum StoreError {
    #[error("Format error: {0}")]
    Format(#[from] FormatError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Index out of bounds: {index} >= {count}")]
    IndexOutOfBounds { index: usize, count: usize },

    #[error("Alignment error: byte slice not aligned to f32 (4 bytes)")]
    AlignmentError,
}

/// Memory-mapped vector store providing zero-copy access to .vdb files
pub struct VectorStore {
    /// The memory-mapped file
    mmap: Mmap,
    /// Number of vectors in the store
    pub count: usize,
    /// Dimensionality of each vector
    pub dim: usize,
}

impl VectorStore {
    /// Open a .vdb file for reading
    ///
    /// # Safety
    ///
    /// This uses memory mapping which is inherently unsafe:
    /// - If the file is truncated while mapped, reads may cause SIGBUS
    /// - The file should not be modified while the store is open
    ///
    /// We use `bytemuck` for safe byte-to-float conversion with alignment checks.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, StoreError> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        // Parse header
        let header = VdbHeader::from_bytes(&mmap)?;

        // Validate file size
        let expected_size = header.file_size();
        if mmap.len() < expected_size {
            return Err(StoreError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!(
                    "File truncated: expected {} bytes, got {}",
                    expected_size,
                    mmap.len()
                ),
            )));
        }

        Ok(Self {
            mmap,
            count: header.count as usize,
            dim: header.dimensions as usize,
        })
    }

    /// Get a vector by index with zero-copy access
    ///
    /// Returns a slice directly into the memory-mapped file.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds or if alignment is incorrect.
    /// Use `try_get` for a non-panicking version.
    #[inline]
    pub fn get(&self, index: usize) -> &[f32] {
        self.try_get(index).expect("Vector access failed")
    }

    /// Try to get a vector by index
    ///
    /// Returns `None` if the index is out of bounds.
    /// Returns an error if alignment is incorrect (should not happen with valid .vdb files).
    pub fn try_get(&self, index: usize) -> Result<&[f32], StoreError> {
        if index >= self.count {
            return Err(StoreError::IndexOutOfBounds {
                index,
                count: self.count,
            });
        }

        let start = HEADER_SIZE + (index * self.dim * std::mem::size_of::<f32>());
        let end = start + (self.dim * std::mem::size_of::<f32>());
        let bytes = &self.mmap[start..end];

        // Use bytemuck for safe, alignment-checked conversion
        bytemuck::try_cast_slice(bytes).map_err(|_| StoreError::AlignmentError)
    }

    /// Get the total memory footprint of the mapped file
    pub fn memory_bytes(&self) -> usize {
        self.mmap.len()
    }

    /// Get an iterator over all vectors
    pub fn iter(&self) -> VectorIter<'_> {
        VectorIter {
            store: self,
            index: 0,
        }
    }
}

/// Iterator over vectors in the store
pub struct VectorIter<'a> {
    store: &'a VectorStore,
    index: usize,
}

impl<'a> Iterator for VectorIter<'a> {
    type Item = &'a [f32];

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.store.count {
            return None;
        }
        let vec = self.store.get(self.index);
        self.index += 1;
        Some(vec)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.store.count - self.index;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for VectorIter<'a> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::VdbWriter;
    use tempfile::tempdir;

    fn create_test_vdb(vectors: &[Vec<f32>]) -> tempfile::TempDir {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.vdb");

        let dim = vectors.first().map(|v| v.len()).unwrap_or(0);
        let mut writer = VdbWriter::new(&path, dim).unwrap();
        for v in vectors {
            writer.write_vector(v).unwrap();
        }
        writer.finish().unwrap();

        dir
    }

    #[test]
    fn test_open_and_read() {
        let vectors = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
            vec![9.0, 10.0, 11.0, 12.0],
        ];
        let dir = create_test_vdb(&vectors);
        let path = dir.path().join("test.vdb");

        let store = VectorStore::open(&path).unwrap();

        assert_eq!(store.count, 3);
        assert_eq!(store.dim, 4);
        assert_eq!(store.get(0), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(store.get(1), &[5.0, 6.0, 7.0, 8.0]);
        assert_eq!(store.get(2), &[9.0, 10.0, 11.0, 12.0]);
    }

    #[test]
    fn test_index_out_of_bounds() {
        let vectors = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let dir = create_test_vdb(&vectors);
        let path = dir.path().join("test.vdb");

        let store = VectorStore::open(&path).unwrap();
        let result = store.try_get(5);

        assert!(matches!(result, Err(StoreError::IndexOutOfBounds { .. })));
    }

    #[test]
    fn test_iterator() {
        let vectors = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
        ];
        let dir = create_test_vdb(&vectors);
        let path = dir.path().join("test.vdb");

        let store = VectorStore::open(&path).unwrap();
        let collected: Vec<_> = store.iter().collect();

        assert_eq!(collected.len(), 3);
        assert_eq!(collected[0], &[1.0, 2.0]);
        assert_eq!(collected[1], &[3.0, 4.0]);
        assert_eq!(collected[2], &[5.0, 6.0]);
    }
}
