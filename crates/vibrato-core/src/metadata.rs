//! Zero-copy metadata for vectors
//!
//! Stores per-vector metadata (source file, timestamps, BPM, tags)
//! in a compact format that can be memory-mapped and read without parsing.
//!
//! Uses a simple binary format (not FlatBuffers) to avoid the codegen dependency
//! while maintaining zero-copy properties via careful alignment.
//!
//! # Layout
//!
//! ```text
//! [entry_count: u32]
//! [string_pool_offset: u32]
//! [entries: MetadataEntry × entry_count]
//! [string_pool: packed UTF-8 strings]
//! ```

use std::io::{self, Write};

use thiserror::Error;

#[derive(Error, Debug)]
pub enum MetadataError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    #[error("Invalid metadata: {0}")]
    Invalid(String),

    #[error("Index out of bounds: {index} >= {count}")]
    IndexOutOfBounds { index: usize, count: usize },
}

/// Per-vector metadata entry (serialized as 24 bytes on wire)
///
/// NOTE: This struct is NOT `#[repr(C, packed)]` because packed structs
/// cause UB when taking field references (which `&entry.bpm` would do).
/// Instead, fields are serialized/deserialized manually in the builder/reader.
#[derive(Debug, Clone, Copy)]
pub struct MetadataEntry {
    /// Offset into UTF-8 string pool for source filename
    pub source_file_offset: u32,
    /// Length of source filename string
    pub source_file_len: u16,
    /// Start time in milliseconds within the source file
    pub start_time_ms: u32,
    /// Duration in milliseconds
    pub duration_ms: u16,
    /// BPM (0.0 if unknown)
    pub bpm: f32,
    /// Offset into string pool for comma-separated tags
    pub tags_offset: u32,
    /// Length of tags string
    pub tags_len: u16,
    /// Reserved for future use
    _reserved: u16,
}

/// Builder for creating metadata sections
pub struct MetadataBuilder {
    entries: Vec<MetadataEntry>,
    string_pool: Vec<u8>,
}

impl MetadataBuilder {
    /// Create a new metadata builder
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            string_pool: Vec::new(),
        }
    }

    /// Add a string to the string pool, returning (offset, length)
    fn add_string(&mut self, s: &str) -> (u32, u16) {
        let offset = self.string_pool.len() as u32;
        let len = s.len().min(u16::MAX as usize) as u16;
        self.string_pool.extend_from_slice(&s.as_bytes()[..len as usize]);
        (offset, len)
    }

    /// Add a metadata entry for a vector
    pub fn add_entry(
        &mut self,
        source_file: &str,
        start_time_ms: u32,
        duration_ms: u16,
        bpm: f32,
        tags: &[&str],
    ) {
        let (file_offset, file_len) = self.add_string(source_file);
        let tags_str = tags.join(",");
        let (tags_offset, tags_len) = self.add_string(&tags_str);

        self.entries.push(MetadataEntry {
            source_file_offset: file_offset,
            source_file_len: file_len,
            start_time_ms,
            duration_ms,
            bpm,
            tags_offset,
            tags_len,
            _reserved: 0,
        });
    }

    /// Serialize the metadata section to bytes
    pub fn build(&self) -> Vec<u8> {
        let entry_size = std::mem::size_of::<MetadataEntry>();
        let header_size = 8; // entry_count (4) + string_pool_offset (4)
        let entries_size = self.entries.len() * entry_size;
        let string_pool_offset = (header_size + entries_size) as u32;

        let total_size = header_size + entries_size + self.string_pool.len();
        let mut buf = Vec::with_capacity(total_size);

        // Header
        buf.extend_from_slice(&(self.entries.len() as u32).to_le_bytes());
        buf.extend_from_slice(&string_pool_offset.to_le_bytes());

        // Entries
        for entry in &self.entries {
            // Write each field explicitly to avoid alignment issues
            buf.extend_from_slice(&entry.source_file_offset.to_le_bytes());
            buf.extend_from_slice(&entry.source_file_len.to_le_bytes());
            buf.extend_from_slice(&entry.start_time_ms.to_le_bytes());
            buf.extend_from_slice(&entry.duration_ms.to_le_bytes());
            buf.extend_from_slice(&entry.bpm.to_le_bytes());
            buf.extend_from_slice(&entry.tags_offset.to_le_bytes());
            buf.extend_from_slice(&entry.tags_len.to_le_bytes());
            buf.extend_from_slice(&entry._reserved.to_le_bytes());
        }

        // String pool
        buf.extend_from_slice(&self.string_pool);

        buf
    }

    /// Write the metadata section to a writer
    pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<usize, MetadataError> {
        let data = self.build();
        writer.write_all(&data)?;
        Ok(data.len())
    }

    /// Number of entries added
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for MetadataBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Zero-copy metadata reader over a byte slice (mmap-friendly)
pub struct MetadataReader<'a> {
    data: &'a [u8],
    entry_count: usize,
    string_pool_offset: usize,
}

/// Parsed metadata for a single vector
#[derive(Debug, Clone)]
pub struct VectorMetadata {
    pub source_file: String,
    pub start_time_ms: u32,
    pub duration_ms: u16,
    pub bpm: f32,
    pub tags: Vec<String>,
}

const ENTRY_WIRE_SIZE: usize = 24; // 4+2+4+2+4+4+2+2 = 24 bytes per entry

impl<'a> MetadataReader<'a> {
    /// Create a reader over a metadata byte slice
    pub fn new(data: &'a [u8]) -> Result<Self, MetadataError> {
        if data.len() < 8 {
            return Err(MetadataError::Invalid("Data too short for header".into()));
        }

        let entry_count = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
        let string_pool_offset = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;

        let expected_min = 8 + entry_count * ENTRY_WIRE_SIZE;
        if data.len() < expected_min || string_pool_offset > data.len() {
            return Err(MetadataError::Invalid(format!(
                "Data too short: {} < {}",
                data.len(),
                expected_min
            )));
        }

        Ok(Self {
            data,
            entry_count,
            string_pool_offset,
        })
    }

    /// Number of entries
    pub fn count(&self) -> usize {
        self.entry_count
    }

    /// Read metadata for a specific vector index
    pub fn get(&self, index: usize) -> Result<VectorMetadata, MetadataError> {
        if index >= self.entry_count {
            return Err(MetadataError::IndexOutOfBounds {
                index,
                count: self.entry_count,
            });
        }

        let offset = 8 + index * ENTRY_WIRE_SIZE;
        let entry_bytes = &self.data[offset..offset + ENTRY_WIRE_SIZE];

        let source_file_offset = u32::from_le_bytes(entry_bytes[0..4].try_into().unwrap()) as usize;
        let source_file_len = u16::from_le_bytes(entry_bytes[4..6].try_into().unwrap()) as usize;
        let start_time_ms = u32::from_le_bytes(entry_bytes[6..10].try_into().unwrap());
        let duration_ms = u16::from_le_bytes(entry_bytes[10..12].try_into().unwrap());
        let bpm = f32::from_le_bytes(entry_bytes[12..16].try_into().unwrap());
        let tags_offset = u32::from_le_bytes(entry_bytes[16..20].try_into().unwrap()) as usize;
        let tags_len = u16::from_le_bytes(entry_bytes[20..22].try_into().unwrap()) as usize;

        let pool = &self.data[self.string_pool_offset..];

        // Bounds-check string pool accesses to prevent panics on corrupted data
        if source_file_offset + source_file_len > pool.len() {
            return Err(MetadataError::Invalid(format!(
                "source_file string out of bounds: offset={} len={} pool_size={}",
                source_file_offset, source_file_len, pool.len()
            )));
        }
        let source_file = std::str::from_utf8(&pool[source_file_offset..source_file_offset + source_file_len])
            .map_err(|e| MetadataError::Invalid(format!("Invalid UTF-8 in source_file: {}", e)))?
            .to_string();

        let tags = if tags_len > 0 {
            if tags_offset + tags_len > pool.len() {
                return Err(MetadataError::Invalid(format!(
                    "tags string out of bounds: offset={} len={} pool_size={}",
                    tags_offset, tags_len, pool.len()
                )));
            }
            let tags_str = std::str::from_utf8(&pool[tags_offset..tags_offset + tags_len])
                .map_err(|e| MetadataError::Invalid(format!("Invalid UTF-8 in tags: {}", e)))?;
            tags_str.split(',').map(String::from).collect()
        } else {
            Vec::new()
        };

        Ok(VectorMetadata {
            source_file,
            start_time_ms,
            duration_ms,
            bpm,
            tags,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_roundtrip() {
        let mut builder = MetadataBuilder::new();
        builder.add_entry("drums/kick_01.wav", 0, 960, 120.0, &["drums", "kick"]);
        builder.add_entry("synth/pad_warm.flac", 5000, 960, 0.0, &["synth"]);
        builder.add_entry("bass/sub_bass.mp3", 10000, 1920, 140.5, &[]);

        assert_eq!(builder.len(), 3);

        let data = builder.build();
        let reader = MetadataReader::new(&data).unwrap();

        assert_eq!(reader.count(), 3);

        let m0 = reader.get(0).unwrap();
        assert_eq!(m0.source_file, "drums/kick_01.wav");
        assert_eq!(m0.start_time_ms, 0);
        assert_eq!(m0.duration_ms, 960);
        assert!((m0.bpm - 120.0).abs() < 0.01);
        assert_eq!(m0.tags, vec!["drums", "kick"]);

        let m1 = reader.get(1).unwrap();
        assert_eq!(m1.source_file, "synth/pad_warm.flac");
        assert_eq!(m1.start_time_ms, 5000);
        assert!((m1.bpm - 0.0).abs() < 0.01);
        assert_eq!(m1.tags, vec!["synth"]);

        let m2 = reader.get(2).unwrap();
        assert_eq!(m2.source_file, "bass/sub_bass.mp3");
        assert_eq!(m2.tags.is_empty(), true);
        assert!((m2.bpm - 140.5).abs() < 0.01);
    }

    #[test]
    fn test_metadata_out_of_bounds() {
        let mut builder = MetadataBuilder::new();
        builder.add_entry("test.wav", 0, 960, 0.0, &[]);
        let data = builder.build();
        let reader = MetadataReader::new(&data).unwrap();

        let result = reader.get(5);
        assert!(matches!(result, Err(MetadataError::IndexOutOfBounds { .. })));
    }

    #[test]
    fn test_metadata_empty() {
        let builder = MetadataBuilder::new();
        assert!(builder.is_empty());

        let data = builder.build();
        let reader = MetadataReader::new(&data).unwrap();
        assert_eq!(reader.count(), 0);
    }
}
