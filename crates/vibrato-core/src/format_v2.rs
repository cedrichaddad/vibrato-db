//! V2 .vdb Binary File Format
//!
//! # File Structure (Hybrid Columnar)
//!
//! ```text
//! Offset   Size    Type        Description
//! ─────────────────────────────────────────────────
//! 0x00     8       [u8; 8]     Magic: "VIBDB002"
//! 0x08     4       u32 LE      Version (2)
//! 0x0C     4       u32 LE      Flags (bit 0 = PQ enabled)
//! 0x10     4       u32 LE      N: Number of vectors
//! 0x14     4       u32 LE      D: Dimensions
//! 0x18     4       u32 LE      PQ subspaces (0 if raw)
//! 0x1C     8       u64 LE      Vectors section offset
//! 0x24     8       u64 LE      Codebook section offset (0 if no PQ)
//! 0x2C     8       u64 LE      Metadata section offset (0 if none)
//! 0x34     8       u64 LE      Graph section offset (0 if none)
//! 0x3C     4       [u8; 4]     Reserved / padding
//! ─────────────────────────────────────────────────
//! TOTAL: 64 bytes (cache-line aligned)
//! ```
//!
//! The 64-byte header guarantees that the Vectors section starts at a cache-line
//! boundary, critical for SIMD alignment on both x86_64 (AVX2) and ARM (NEON).

use std::fs::File;
use std::io::{self, BufWriter, Seek, Write};
use std::path::Path;

use crate::store::VectorStore;
use crate::hnsw::HNSW;

use thiserror::Error;

/// Magic bytes identifying a V2 .vdb file: "VIBDB002"
pub const MAGIC_V2: [u8; 8] = *b"VIBDB002";

/// V2 header size in bytes (cache-line aligned)
pub const HEADER_V2_SIZE: usize = 64;

/// Format flags
pub mod flags {
    /// Product Quantization is enabled (vectors section contains PQ codes, not raw f32)
    pub const PQ_ENABLED: u32 = 1 << 0;
    /// Metadata section is present
    pub const HAS_METADATA: u32 = 1 << 1;
    /// Graph (HNSW index) section is present
    pub const HAS_GRAPH: u32 = 1 << 2;
}

#[derive(Error, Debug)]
pub enum FormatV2Error {
    #[error("Invalid V2 magic bytes: expected VIBDB002")]
    InvalidMagic,

    #[error("Unsupported format version: {0}")]
    UnsupportedVersion(u32),

    #[error("Section misaligned: {section} at offset {offset} (expected alignment {alignment})")]
    Misaligned {
        section: &'static str,
        offset: u64,
        alignment: u64,
    },

    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("IO error: {0}")]
    Io(#[from] io::Error),
}

/// Parsed V2 .vdb file header
#[derive(Debug, Clone, Copy)]
pub struct VdbHeaderV2 {
    pub version: u32,
    pub flags: u32,
    pub count: u32,
    pub dimensions: u32,
    pub pq_subspaces: u32,
    pub vectors_offset: u64,
    pub codebook_offset: u64,
    pub metadata_offset: u64,
    pub graph_offset: u64,
}

impl VdbHeaderV2 {
    /// Parse header from raw bytes (first 64 bytes of file)
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, FormatV2Error> {
        if bytes.len() < HEADER_V2_SIZE {
            return Err(FormatV2Error::Io(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("File too small for V2 header: {} < {}", bytes.len(), HEADER_V2_SIZE),
            )));
        }

        // Validate magic
        if &bytes[0..8] != &MAGIC_V2 {
            return Err(FormatV2Error::InvalidMagic);
        }

        let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        if version != 2 {
            return Err(FormatV2Error::UnsupportedVersion(version));
        }

        let flags = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        let count = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
        let dimensions = u32::from_le_bytes(bytes[20..24].try_into().unwrap());
        let pq_subspaces = u32::from_le_bytes(bytes[24..28].try_into().unwrap());
        let vectors_offset = u64::from_le_bytes(bytes[28..36].try_into().unwrap());
        let codebook_offset = u64::from_le_bytes(bytes[36..44].try_into().unwrap());
        let metadata_offset = u64::from_le_bytes(bytes[44..52].try_into().unwrap());
        let graph_offset = u64::from_le_bytes(bytes[52..60].try_into().unwrap());

        let header = Self {
            version,
            flags,
            count,
            dimensions,
            pq_subspaces,
            vectors_offset,
            codebook_offset,
            metadata_offset,
            graph_offset,
        };

        // Validate alignment of vectors section
        if header.vectors_offset % 64 != 0 {
            return Err(FormatV2Error::Misaligned {
                section: "vectors",
                offset: header.vectors_offset,
                alignment: 64,
            });
        }

        Ok(header)
    }

    /// Write header to bytes (exactly 64 bytes, cache-line aligned)
    pub fn to_bytes(&self) -> [u8; HEADER_V2_SIZE] {
        let mut buf = [0u8; HEADER_V2_SIZE];
        buf[0..8].copy_from_slice(&MAGIC_V2);
        buf[8..12].copy_from_slice(&self.version.to_le_bytes());
        buf[12..16].copy_from_slice(&self.flags.to_le_bytes());
        buf[16..20].copy_from_slice(&self.count.to_le_bytes());
        buf[20..24].copy_from_slice(&self.dimensions.to_le_bytes());
        buf[24..28].copy_from_slice(&self.pq_subspaces.to_le_bytes());
        buf[28..36].copy_from_slice(&self.vectors_offset.to_le_bytes());
        buf[36..44].copy_from_slice(&self.codebook_offset.to_le_bytes());
        buf[44..52].copy_from_slice(&self.metadata_offset.to_le_bytes());
        buf[52..60].copy_from_slice(&self.graph_offset.to_le_bytes());
        // bytes 60..64 are reserved (zero)
        buf
    }

    /// Check if PQ is enabled
    #[inline]
    pub fn is_pq_enabled(&self) -> bool {
        self.flags & flags::PQ_ENABLED != 0
    }

    /// Check if metadata section is present
    #[inline]
    pub fn has_metadata(&self) -> bool {
        self.flags & flags::HAS_METADATA != 0
    }

    /// Check if graph section is present
    #[inline]
    pub fn has_graph(&self) -> bool {
        self.flags & flags::HAS_GRAPH != 0
    }

    /// Calculate byte size of the vectors section
    pub fn vectors_section_size(&self) -> usize {
        if self.is_pq_enabled() {
            // PQ codes: pq_subspaces bytes per vector
            self.count as usize * self.pq_subspaces as usize
        } else {
            // Raw f32: dimensions * 4 bytes per vector
            self.count as usize * self.dimensions as usize * std::mem::size_of::<f32>()
        }
    }

    /// Get byte offset for a raw f32 vector by index
    #[inline(always)]
    pub fn vector_offset(&self, index: usize) -> usize {
        if self.is_pq_enabled() {
            self.vectors_offset as usize + index * self.pq_subspaces as usize
        } else {
            self.vectors_offset as usize + index * self.dimensions as usize * std::mem::size_of::<f32>()
        }
    }
}

/// Detect whether a file is V1 or V2 format based on magic bytes
pub fn detect_format_version(bytes: &[u8]) -> Option<u32> {
    if bytes.len() < 8 {
        return None;
    }
    if &bytes[0..8] == b"VIBDB001" {
        Some(1)
    } else if &bytes[0..8] == b"VIBDB002" {
        Some(2)
    } else {
        None
    }
}

/// Writer for creating V2 .vdb files
pub struct VdbWriterV2 {
    writer: BufWriter<File>,
    dimensions: usize,
    count: u32,
    pq_enabled: bool,
    pq_subspaces: u32,
}

impl VdbWriterV2 {
    /// Create a new V2 .vdb file writer for raw f32 vectors
    pub fn new_raw<P: AsRef<Path>>(path: P, dimensions: usize) -> Result<Self, FormatV2Error> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Write placeholder header (64 bytes, will be updated in finish())
        let header = VdbHeaderV2 {
            version: 2,
            flags: 0,
            count: 0,
            dimensions: dimensions as u32,
            pq_subspaces: 0,
            vectors_offset: HEADER_V2_SIZE as u64, // Vectors start right after header
            codebook_offset: 0,
            metadata_offset: 0,
            graph_offset: 0,
        };
        writer.write_all(&header.to_bytes())?;

        Ok(Self {
            writer,
            dimensions,
            count: 0,
            pq_enabled: false,
            pq_subspaces: 0,
        })
    }

    /// Create a new V2 .vdb file writer for PQ-encoded vectors
    pub fn new_pq<P: AsRef<Path>>(
        path: P,
        dimensions: usize,
        pq_subspaces: usize,
    ) -> Result<Self, FormatV2Error> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        let header = VdbHeaderV2 {
            version: 2,
            flags: flags::PQ_ENABLED,
            count: 0,
            dimensions: dimensions as u32,
            pq_subspaces: pq_subspaces as u32,
            vectors_offset: HEADER_V2_SIZE as u64,
            codebook_offset: 0, // Set in finish()
            metadata_offset: 0,
            graph_offset: 0,
        };
        writer.write_all(&header.to_bytes())?;

        Ok(Self {
            writer,
            dimensions,
            count: 0,
            pq_enabled: true,
            pq_subspaces: pq_subspaces as u32,
        })
    }

    /// Write a single raw f32 vector
    pub fn write_vector(&mut self, vector: &[f32]) -> Result<(), FormatV2Error> {
        if self.pq_enabled {
            return Err(FormatV2Error::DimensionMismatch {
                expected: self.pq_subspaces as usize,
                actual: vector.len(),
            });
        }
        if vector.len() != self.dimensions {
            return Err(FormatV2Error::DimensionMismatch {
                expected: self.dimensions,
                actual: vector.len(),
            });
        }

        for &val in vector {
            self.writer.write_all(&val.to_le_bytes())?;
        }
        self.count += 1;
        Ok(())
    }

    /// Write a single PQ-encoded vector (array of centroid indices)
    pub fn write_pq_codes(&mut self, codes: &[u8]) -> Result<(), FormatV2Error> {
        if !self.pq_enabled {
            return Err(FormatV2Error::DimensionMismatch {
                expected: self.dimensions,
                actual: codes.len(),
            });
        }
        if codes.len() != self.pq_subspaces as usize {
            return Err(FormatV2Error::DimensionMismatch {
                expected: self.pq_subspaces as usize,
                actual: codes.len(),
            });
        }

        self.writer.write_all(codes)?;
        self.count += 1;
        Ok(())
    }

    /// Finalize the file, updating the header with actual counts
    pub fn finish(mut self) -> Result<u32, FormatV2Error> {
        self.writer.flush()?;

        // Seek back and rewrite header with final count
        let file = self.writer.get_mut();
        file.seek(io::SeekFrom::Start(0))?;

        let header = VdbHeaderV2 {
            version: 2,
            flags: if self.pq_enabled { flags::PQ_ENABLED } else { 0 },
            count: self.count,
            dimensions: self.dimensions as u32,
            pq_subspaces: self.pq_subspaces,
            vectors_offset: HEADER_V2_SIZE as u64,
            codebook_offset: 0,
            metadata_offset: 0,
            graph_offset: 0,
        };
        file.write_all(&header.to_bytes())?;
        file.sync_all()?;

        Ok(self.count)
    }

    /// Finalize the file with a graph section
    pub fn finish_with_graph(mut self, graph_offset: u64) -> Result<u32, FormatV2Error> {
        self.writer.flush()?;

        // Seek back and rewrite header with final count
        let file = self.writer.get_mut();
        file.seek(io::SeekFrom::Start(0))?;

        let header = VdbHeaderV2 {
            version: 2,
            flags: (if self.pq_enabled { flags::PQ_ENABLED } else { 0 }) | flags::HAS_GRAPH,
            count: self.count,
            dimensions: self.dimensions as u32,
            pq_subspaces: self.pq_subspaces,
            vectors_offset: HEADER_V2_SIZE as u64,
            codebook_offset: 0,
            metadata_offset: 0,
            graph_offset,
        };
        file.write_all(&header.to_bytes())?;
        file.sync_all()?;

        Ok(self.count)
    }

    /// Merge an existing Valid V2 store with new vectors and a graph
    ///
    /// This performs a ZERO-COPY blit of the old vectors.
    pub fn merge(
        base_store: &VectorStore,
        new_vectors: &[Vec<f32>],
        graph: &HNSW,
        path: &Path,
    ) -> Result<(), FormatV2Error> {
        let mut writer = VdbWriterV2::new_raw(path, base_store.dim)?;

        // --- OPTIMIZATION: ZERO-COPY BLIT ---
        // 1. Copy old vectors directly from raw mmap bytes
        // We trust the base_store is valid.
        let old_vector_bytes_len = base_store.count * base_store.dim * std::mem::size_of::<f32>();
        let old_bytes_slice = &base_store.mmap[
            base_store.data_offset .. base_store.data_offset + old_vector_bytes_len
        ];
        writer.writer.write_all(old_bytes_slice)?;
        writer.count += base_store.count as u32;

        // 2. Append new vectors
        for vec in new_vectors {
            writer.write_vector(vec)?;
        }

        // 3. Serialize Graph
        let graph_offset = writer.writer.stream_position()?;
        graph.serialize(&mut writer.writer)?;

        // 4. Finish (Updates Header with offsets)
        writer.finish_with_graph(graph_offset)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_header_v2_roundtrip() {
        let header = VdbHeaderV2 {
            version: 2,
            flags: flags::PQ_ENABLED | flags::HAS_METADATA,
            count: 50000,
            dimensions: 128,
            pq_subspaces: 16,
            vectors_offset: 64,
            codebook_offset: 850000,
            metadata_offset: 900000,
            graph_offset: 1000000,
        };

        let bytes = header.to_bytes();
        assert_eq!(bytes.len(), 64);

        let parsed = VdbHeaderV2::from_bytes(&bytes).unwrap();
        assert_eq!(parsed.version, 2);
        assert_eq!(parsed.count, 50000);
        assert_eq!(parsed.dimensions, 128);
        assert_eq!(parsed.pq_subspaces, 16);
        assert!(parsed.is_pq_enabled());
        assert!(parsed.has_metadata());
        assert!(!parsed.has_graph());
        assert_eq!(parsed.vectors_offset, 64);
        assert_eq!(parsed.codebook_offset, 850000);
    }

    #[test]
    fn test_header_v2_alignment_validation() {
        let mut bytes = [0u8; 64];
        bytes[0..8].copy_from_slice(b"VIBDB002");
        bytes[8..12].copy_from_slice(&2u32.to_le_bytes()); // version
        // Set vectors_offset to 17 (misaligned)
        bytes[28..36].copy_from_slice(&17u64.to_le_bytes());

        let result = VdbHeaderV2::from_bytes(&bytes);
        assert!(matches!(result, Err(FormatV2Error::Misaligned { .. })));
    }

    #[test]
    fn test_detect_format_version() {
        assert_eq!(detect_format_version(b"VIBDB001"), Some(1));
        assert_eq!(detect_format_version(b"VIBDB002"), Some(2));
        assert_eq!(detect_format_version(b"INVALID!"), None);
        assert_eq!(detect_format_version(b"SHORT"), None);
    }

    #[test]
    fn test_writer_v2_raw() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_v2.vdb");

        let mut writer = VdbWriterV2::new_raw(&path, 4).unwrap();
        writer.write_vector(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        writer.write_vector(&[5.0, 6.0, 7.0, 8.0]).unwrap();
        let count = writer.finish().unwrap();
        assert_eq!(count, 2);

        // Verify file structure
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[0..8], b"VIBDB002");
        let header = VdbHeaderV2::from_bytes(&bytes).unwrap();
        assert_eq!(header.count, 2);
        assert_eq!(header.dimensions, 4);
        assert!(!header.is_pq_enabled());
        assert_eq!(header.vectors_offset, 64);

        // Verify vector data at offset 64
        let v0_start = 64;
        let v0_bytes = &bytes[v0_start..v0_start + 16];
        let v0: Vec<f32> = v0_bytes.chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(v0, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_writer_v2_pq() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_v2_pq.vdb");

        let mut writer = VdbWriterV2::new_pq(&path, 128, 16).unwrap();
        writer.write_pq_codes(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]).unwrap();
        let count = writer.finish().unwrap();
        assert_eq!(count, 1);

        let bytes = std::fs::read(&path).unwrap();
        let header = VdbHeaderV2::from_bytes(&bytes).unwrap();
        assert!(header.is_pq_enabled());
        assert_eq!(header.pq_subspaces, 16);

        // PQ codes start at offset 64
        assert_eq!(bytes[64], 0);
        assert_eq!(bytes[79], 15);
    }

    #[test]
    fn test_writer_v2_dimension_mismatch() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_mismatch.vdb");

        let mut writer = VdbWriterV2::new_raw(&path, 4).unwrap();
        let result = writer.write_vector(&[1.0, 2.0, 3.0]); // Wrong dim
        assert!(matches!(result, Err(FormatV2Error::DimensionMismatch { .. })));
    }

    #[test]
    fn test_vectors_section_size() {
        let raw_header = VdbHeaderV2 {
            version: 2, flags: 0, count: 1000, dimensions: 128,
            pq_subspaces: 0, vectors_offset: 64,
            codebook_offset: 0, metadata_offset: 0, graph_offset: 0,
        };
        assert_eq!(raw_header.vectors_section_size(), 1000 * 128 * 4); // 512KB

        let pq_header = VdbHeaderV2 {
            version: 2, flags: flags::PQ_ENABLED, count: 1000, dimensions: 128,
            pq_subspaces: 16, vectors_offset: 64,
            codebook_offset: 0, metadata_offset: 0, graph_offset: 0,
        };
        assert_eq!(pq_header.vectors_section_size(), 1000 * 16); // 16KB (32x compression)
    }
}
