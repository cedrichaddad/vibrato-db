//! .vdb Binary File Format
//!
//! # File Structure
//!
//! ```text
//! Offset   Size    Type        Description
//! ─────────────────────────────────────────────
//! 0x00     8       [u8; 8]     Magic: "VIBDB001"
//! 0x08     4       u32 LE      N: Number of vectors
//! 0x0C     4       u32 LE      D: Dimensions
//! 0x10     N*D*4   [f32]       Vector data (Little Endian)
//! ```
//!
//! # Example
//!
//! ```ignore
//! let mut writer = VdbWriter::new("data.vdb", 128)?;
//! writer.write_vector(&vec![0.1f32; 128])?;
//! writer.finish()?;
//! ```

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use thiserror::Error;

/// Magic bytes identifying a .vdb file: "VIBDB001"
pub const MAGIC: [u8; 8] = *b"VIBDB001";

/// Header size in bytes: 8 (magic) + 4 (count) + 4 (dims) = 16
pub const HEADER_SIZE: usize = 16;

#[derive(Error, Debug)]
pub enum FormatError {
    #[error("Invalid magic bytes: expected VIBDB001")]
    InvalidMagic,

    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("IO error: {0}")]
    Io(#[from] io::Error),
}

/// Parsed .vdb file header
#[derive(Debug, Clone, Copy)]
pub struct VdbHeader {
    pub count: u32,
    pub dimensions: u32,
}

impl VdbHeader {
    /// Parse header from raw bytes (first 16 bytes of file)
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, FormatError> {
        if bytes.len() < HEADER_SIZE {
            return Err(FormatError::Io(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "File too small for header",
            )));
        }

        // Validate magic bytes
        if &bytes[0..8] != &MAGIC {
            return Err(FormatError::InvalidMagic);
        }

        let count = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let dimensions = u32::from_le_bytes(bytes[12..16].try_into().unwrap());

        Ok(Self { count, dimensions })
    }

    /// Write header to bytes
    pub fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut buf = [0u8; HEADER_SIZE];
        buf[0..8].copy_from_slice(&MAGIC);
        buf[8..12].copy_from_slice(&self.count.to_le_bytes());
        buf[12..16].copy_from_slice(&self.dimensions.to_le_bytes());
        buf
    }

    /// Calculate byte offset for a vector by index
    #[inline(always)]
    pub fn offset(&self, index: usize) -> usize {
        HEADER_SIZE + (index * self.dimensions as usize * std::mem::size_of::<f32>())
    }

    /// Calculate total file size
    pub fn file_size(&self) -> usize {
        HEADER_SIZE + (self.count as usize * self.dimensions as usize * std::mem::size_of::<f32>())
    }
}

/// Writer for creating .vdb files
pub struct VdbWriter {
    writer: BufWriter<File>,
    dimensions: usize,
    count: u32,
}

impl VdbWriter {
    /// Create a new .vdb file writer
    pub fn new<P: AsRef<Path>>(path: P, dimensions: usize) -> Result<Self, FormatError> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Write placeholder header (will be updated in finish())
        let header = VdbHeader {
            count: 0,
            dimensions: dimensions as u32,
        };
        writer.write_all(&header.to_bytes())?;

        Ok(Self {
            writer,
            dimensions,
            count: 0,
        })
    }

    /// Write a single vector to the file
    pub fn write_vector(&mut self, vector: &[f32]) -> Result<(), FormatError> {
        if vector.len() != self.dimensions {
            return Err(FormatError::DimensionMismatch {
                expected: self.dimensions,
                actual: vector.len(),
            });
        }

        // Write floats as little-endian bytes
        for &val in vector {
            self.writer.write_all(&val.to_le_bytes())?;
        }

        self.count += 1;
        Ok(())
    }

    /// Finalize the file, updating the header with the actual count
    pub fn finish(mut self) -> Result<u32, FormatError> {
        use std::io::Seek;

        self.writer.flush()?;

        // Seek back to header and update count
        let file = self.writer.get_mut();
        file.seek(io::SeekFrom::Start(8))?;
        file.write_all(&self.count.to_le_bytes())?;
        file.sync_all()?;

        Ok(self.count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_header_roundtrip() {
        let header = VdbHeader {
            count: 1000,
            dimensions: 128,
        };
        let bytes = header.to_bytes();
        let parsed = VdbHeader::from_bytes(&bytes).unwrap();

        assert_eq!(parsed.count, 1000);
        assert_eq!(parsed.dimensions, 128);
    }

    #[test]
    fn test_offset_calculation() {
        let header = VdbHeader {
            count: 100,
            dimensions: 128,
        };

        assert_eq!(header.offset(0), 16);
        assert_eq!(header.offset(1), 16 + 128 * 4);
        assert_eq!(header.offset(10), 16 + 10 * 128 * 4);
    }

    #[test]
    fn test_writer_basic() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.vdb");

        let mut writer = VdbWriter::new(&path, 4).unwrap();
        writer.write_vector(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        writer.write_vector(&[5.0, 6.0, 7.0, 8.0]).unwrap();
        let count = writer.finish().unwrap();

        assert_eq!(count, 2);

        // Verify file contents
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[0..8], b"VIBDB001");
        assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()), 2);
        assert_eq!(u32::from_le_bytes(bytes[12..16].try_into().unwrap()), 4);
    }

    #[test]
    fn test_writer_dimension_mismatch() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.vdb");

        let mut writer = VdbWriter::new(&path, 4).unwrap();
        let result = writer.write_vector(&[1.0, 2.0, 3.0]); // Wrong size

        assert!(matches!(result, Err(FormatError::DimensionMismatch { .. })));
    }
}
