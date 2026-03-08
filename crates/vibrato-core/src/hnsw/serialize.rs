use super::HNSW;
use std::io::{self, Write};

impl HNSW {
    /// Serialize the HNSW graph to a writer.
    ///
    /// Format (VIBGRPH2):
    /// - Magic: "VIBGRPH2" (8 bytes)
    /// - Header:
    ///   - NumNodes: u32
    ///   - EntryPointIdx: u32 (u32::MAX if None)
    ///   - MaxLayer: u8
    ///   - m: u32
    ///   - m0: u32
    ///   - ef_construction: u32
    /// - Global IDs table:
    ///   - Count: u32
    ///   - IDs: [u64; Count]
    /// - Body (Sequence of Nodes by dense index):
    ///   - NodeMaxLayer: u8
    ///   - Per Layer 0..NodeMaxLayer:
    ///     - NeighborCount: u32
    ///     - Neighbors: [u32; NeighborCount] (dense node_idx)
    pub fn serialize<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        // 1. Magic & Header
        writer.write_all(b"VIBGRPH2")?;
        let node_count = u32::try_from(self.nodes.len()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "node count exceeds u32::MAX for VIBGRPH2 format",
            )
        })?;
        writer.write_all(&node_count.to_le_bytes())?;

        let entry_idx = self
            .entry_point
            .map(|idx| {
                u32::try_from(idx).map_err(|_| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "entry point index exceeds u32::MAX for VIBGRPH2 format",
                    )
                })
            })
            .transpose()?
            .unwrap_or(u32::MAX);
        writer.write_all(&entry_idx.to_le_bytes())?;
        writer.write_all(&(self.max_layer as u8).to_le_bytes())?;

        // Metadata
        writer.write_all(&(self.m as u32).to_le_bytes())?;
        writer.write_all(&(self.m0 as u32).to_le_bytes())?;
        writer.write_all(&(self.ef_construction as u32).to_le_bytes())?;

        // 2. Global IDs table
        writer.write_all(&node_count.to_le_bytes())?;
        for node in &self.nodes {
            writer.write_all(&node.id.to_le_bytes())?;
        }

        // 2. Body (Nodes)
        for node in &self.nodes {
            // Allow up to 255 layers (HNSW typically < 16)
            let node_max_layer = node.layers.len().saturating_sub(1) as u8;
            writer.write_all(&node_max_layer.to_le_bytes())?;

            for layer_neighbors in &node.layers {
                writer.write_all(&(layer_neighbors.len() as u32).to_le_bytes())?;
                for &neighbor in layer_neighbors {
                    writer.write_all(&neighbor.to_le_bytes())?;
                }
            }
        }
        Ok(())
    }

    /// Save the graph to a file
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> io::Result<()> {
        let file = std::fs::File::create(path)?;
        let mut writer = std::io::BufWriter::new(file);
        self.serialize(&mut writer)
    }

    /// Load the graph from a file
    pub fn load<P, F>(path: P, vector_fn: F) -> io::Result<Self>
    where
        P: AsRef<std::path::Path>,
        F: Fn(usize) -> Vec<f32> + Send + Sync + 'static,
    {
        Self::load_with_accessor(path, move |id, sink| {
            let v = vector_fn(id);
            sink(&v);
        })
    }

    /// Load the graph from a file with a zero-copy accessor callback.
    pub fn load_with_accessor<P, F>(path: P, vector_fn: F) -> io::Result<Self>
    where
        P: AsRef<std::path::Path>,
        F: Fn(usize, &mut dyn FnMut(&[f32])) + Send + Sync + 'static,
    {
        let file = std::fs::File::open(path)?;
        let mut reader = std::io::BufReader::new(file);
        Self::load_from_reader_with_accessor(&mut reader, vector_fn)
    }

    /// Load the graph from a reader
    pub fn load_from_reader<R, F>(reader: &mut R, vector_fn: F) -> io::Result<Self>
    where
        R: std::io::Read,
        F: Fn(usize) -> Vec<f32> + Send + Sync + 'static,
    {
        Self::load_from_reader_with_accessor(reader, move |id, sink| {
            let v = vector_fn(id);
            sink(&v);
        })
    }

    /// Load the graph from a reader with a zero-copy accessor callback.
    pub fn load_from_reader_with_accessor<R, F>(reader: &mut R, vector_fn: F) -> io::Result<Self>
    where
        R: std::io::Read,
        F: Fn(usize, &mut dyn FnMut(&[f32])) + Send + Sync + 'static,
    {
        use super::node::Node;

        // 1. Magic
        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic)?;
        let is_v2 = &magic == b"VIBGRPH2";
        let is_v1 = &magic == b"VIBGRAPH";
        if !is_v1 && !is_v2 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid HNSW magic"));
        }

        // 2. Header
        let mut buf_u32 = [0u8; 4];

        reader.read_exact(&mut buf_u32)?;
        let num_nodes = u32::from_le_bytes(buf_u32) as usize;

        reader.read_exact(&mut buf_u32)?;
        let entry_point_raw = u32::from_le_bytes(buf_u32);
        let entry_point_hint = if entry_point_raw == u32::MAX {
            None
        } else {
            Some(entry_point_raw as usize)
        };

        let mut buf_u8 = [0u8; 1];
        reader.read_exact(&mut buf_u8)?;
        let max_layer = buf_u8[0] as usize;

        reader.read_exact(&mut buf_u32)?;
        let m = u32::from_le_bytes(buf_u32) as usize;

        reader.read_exact(&mut buf_u32)?;
        let m0 = u32::from_le_bytes(buf_u32) as usize;

        reader.read_exact(&mut buf_u32)?;
        let ef_construction = u32::from_le_bytes(buf_u32) as usize;

        let (nodes, entry_point) = if is_v2 {
            // 3a. VIBGRPH2 global IDs table
            reader.read_exact(&mut buf_u32)?;
            let id_count = u32::from_le_bytes(buf_u32) as usize;
            if id_count != num_nodes {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "global id table length mismatch: ids={} nodes={}",
                        id_count, num_nodes
                    ),
                ));
            }
            let mut ids = Vec::with_capacity(id_count);
            let mut buf_u64 = [0u8; 8];
            for _ in 0..id_count {
                reader.read_exact(&mut buf_u64)?;
                ids.push(u64::from_le_bytes(buf_u64));
            }

            // 3b. Nodes with dense u32 neighbor indices
            let mut nodes = Vec::with_capacity(num_nodes);
            for idx in 0..num_nodes {
                reader.read_exact(&mut buf_u8)?;
                let node_max_layer = buf_u8[0] as usize;

                let mut layers = Vec::with_capacity(node_max_layer + 1);
                for _ in 0..=node_max_layer {
                    reader.read_exact(&mut buf_u32)?;
                    let count = u32::from_le_bytes(buf_u32) as usize;

                    let mut neighbors = Vec::with_capacity(count);
                    for _ in 0..count {
                        reader.read_exact(&mut buf_u32)?;
                        neighbors.push(u32::from_le_bytes(buf_u32));
                    }
                    layers.push(neighbors);
                }

                nodes.push(Node { id: ids[idx], layers });
            }

            let entry = entry_point_hint.map(|idx| {
                if idx >= num_nodes {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("entry point idx {} out of bounds {}", idx, num_nodes),
                    ));
                }
                Ok(idx)
            });
            let entry = match entry {
                Some(v) => Some(v?),
                None => None,
            };
            (nodes, entry)
        } else {
            // Legacy VIBGRAPH format:
            // node ids and neighbors are serialized as external u32 IDs.
            let mut legacy_ids = Vec::with_capacity(num_nodes);
            let mut legacy_layers: Vec<Vec<Vec<u32>>> = Vec::with_capacity(num_nodes);

            for _ in 0..num_nodes {
                reader.read_exact(&mut buf_u32)?;
                let id = u32::from_le_bytes(buf_u32);

                reader.read_exact(&mut buf_u8)?;
                let node_max_layer = buf_u8[0] as usize;

                let mut layers = Vec::with_capacity(node_max_layer + 1);
                for _ in 0..=node_max_layer {
                    reader.read_exact(&mut buf_u32)?;
                    let count = u32::from_le_bytes(buf_u32) as usize;

                    let mut neighbors = Vec::with_capacity(count);
                    for _ in 0..count {
                        reader.read_exact(&mut buf_u32)?;
                        neighbors.push(u32::from_le_bytes(buf_u32));
                    }
                    layers.push(neighbors);
                }
                legacy_ids.push(id);
                legacy_layers.push(layers);
            }

            let id_to_index: std::collections::HashMap<u32, u32> = legacy_ids
                .iter()
                .enumerate()
                .map(|(idx, id)| (*id, idx as u32))
                .collect();
            let mut nodes = Vec::with_capacity(num_nodes);
            for (idx, layers) in legacy_layers.into_iter().enumerate() {
                let mut dense_layers = Vec::with_capacity(layers.len());
                for layer in layers {
                    let mut dense_neighbors = Vec::with_capacity(layer.len());
                    for n in layer {
                        let mapped = id_to_index.get(&n).copied().ok_or_else(|| {
                            io::Error::new(
                                io::ErrorKind::InvalidData,
                                format!("legacy graph neighbor id {} missing from id map", n),
                            )
                        })?;
                        dense_neighbors.push(mapped);
                    }
                    dense_layers.push(dense_neighbors);
                }
                nodes.push(Node {
                    id: legacy_ids[idx] as u64,
                    layers: dense_layers,
                });
            }

            let entry = entry_point_hint.map(|legacy_id| {
                let legacy_id_u32 = u32::try_from(legacy_id).map_err(|_| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("legacy entry point {} does not fit u32", legacy_id),
                    )
                })?;
                id_to_index.get(&legacy_id_u32).copied().map(|v| v as usize).ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("legacy entry point id {} missing from graph ids", legacy_id),
                    )
                })
            });
            let entry = match entry {
                Some(v) => Some(v?),
                None => None,
            };
            (nodes, entry)
        };

        Ok(HNSW::from_parts_with_accessor(
            nodes,
            entry_point,
            max_layer,
            m,
            m0,
            1.0 / (m as f64).ln(),
            ef_construction,
            vector_fn,
        ))
    }
}

#[cfg(test)]
mod tests {
    use crate::hnsw::HNSW;
    use std::collections::HashMap;
    use std::io::Cursor;

    fn make_vec(seed: f32) -> Vec<f32> {
        vec![seed, seed * 0.5, 1.0 - seed]
    }

    #[test]
    fn vibgrph2_roundtrip_preserves_ids_and_edges() {
        let vectors: HashMap<u64, Vec<f32>> =
            [(10u64, make_vec(0.2)), (42u64, make_vec(0.8))]
                .into_iter()
                .collect();
        let ordered_ids = [10u64, 42u64];
        let vectors_for_build = vectors.clone();
        let mut hnsw = HNSW::new_with_accessor(8, 64, move |node_idx, sink| {
            let id = ordered_ids[node_idx];
            sink(vectors_for_build.get(&id).expect("missing vector"))
        });
        hnsw.insert(10, vectors.get(&10).expect("missing vector 10"));
        hnsw.insert(42, vectors.get(&42).expect("missing vector 42"));

        let mut bytes = Vec::new();
        hnsw.serialize(&mut bytes).expect("serialize vibgrph2");

        let ordered_ids = [10u64, 42u64];
        let vectors_for_load = vectors.clone();
        let mut cursor = Cursor::new(bytes);
        let loaded = HNSW::load_from_reader_with_accessor(&mut cursor, move |node_idx, sink| {
            let id = ordered_ids[node_idx];
            sink(vectors_for_load.get(&id).expect("missing vector"))
        })
        .expect("load vibgrph2");

        assert_eq!(loaded.len(), 2);
        assert!(loaded.contains_id(10));
        assert!(loaded.contains_id(42));
        let results = loaded.search(vectors.get(&10).unwrap(), 1, 16);
        assert_eq!(results[0].0, 10);
    }

    #[test]
    fn vibgraph_v1_loader_maps_legacy_external_ids_to_dense_neighbors() {
        // Legacy format with external IDs [10, 42], mutual neighbor links on layer 0.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"VIBGRAPH");
        bytes.extend_from_slice(&(2u32).to_le_bytes()); // num_nodes
        bytes.extend_from_slice(&(10u32).to_le_bytes()); // entry point external id
        bytes.extend_from_slice(&(0u8).to_le_bytes()); // max layer
        bytes.extend_from_slice(&(8u32).to_le_bytes()); // m
        bytes.extend_from_slice(&(16u32).to_le_bytes()); // m0
        bytes.extend_from_slice(&(64u32).to_le_bytes()); // ef_construction

        // Node 10
        bytes.extend_from_slice(&(10u32).to_le_bytes()); // id
        bytes.extend_from_slice(&(0u8).to_le_bytes()); // node max layer
        bytes.extend_from_slice(&(1u32).to_le_bytes()); // layer 0 neighbor count
        bytes.extend_from_slice(&(42u32).to_le_bytes()); // neighbor id

        // Node 42
        bytes.extend_from_slice(&(42u32).to_le_bytes()); // id
        bytes.extend_from_slice(&(0u8).to_le_bytes()); // node max layer
        bytes.extend_from_slice(&(1u32).to_le_bytes()); // layer 0 neighbor count
        bytes.extend_from_slice(&(10u32).to_le_bytes()); // neighbor id

        let vectors: HashMap<u64, Vec<f32>> =
            [(10u64, make_vec(0.2)), (42u64, make_vec(0.8))]
                .into_iter()
                .collect();
        let ordered_ids = [10u64, 42u64];
        let vectors_for_load = vectors.clone();
        let mut cursor = Cursor::new(bytes);
        let loaded = HNSW::load_from_reader_with_accessor(&mut cursor, move |node_idx, sink| {
            let id = ordered_ids[node_idx];
            sink(vectors_for_load.get(&id).expect("missing vector"))
        })
        .expect("load legacy vibgraph");

        assert_eq!(loaded.len(), 2);
        assert!(loaded.contains_id(10));
        assert!(loaded.contains_id(42));
        let results = loaded.search(vectors.get(&42).unwrap(), 1, 16);
        assert_eq!(results[0].0, 42);
    }
}
