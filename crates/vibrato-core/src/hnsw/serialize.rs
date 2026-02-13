use std::io::{self, Write};
use super::HNSW;

impl HNSW {
    /// Serialize the HNSW graph to a writer.
    ///
    /// Format:
    /// - Magic: "VIBGRAPH" (8 bytes)
    /// - Header:
    ///   - NumNodes: u32
    ///   - EntryPoint: u32 (u32::MAX if None)
    ///   - MaxLayer: u8
    /// - Body (Sequence of Nodes):
    ///   - NodeID: u32
    ///   - NodeMaxLayer: u8
    ///   - Per Layer 0..NodeMaxLayer:
    ///     - NeighborCount: u32
    ///     - Neighbors: [u32; NeighborCount]
    pub fn serialize<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        // 1. Magic & Header
        writer.write_all(b"VIBGRAPH")?;
        writer.write_all(&(self.nodes.len() as u32).to_le_bytes())?;
        
        let entry_id = self.entry_point.unwrap_or(u32::MAX as usize) as u32;
        writer.write_all(&entry_id.to_le_bytes())?;
        writer.write_all(&(self.max_layer as u8).to_le_bytes())?;
        
        // Metadata
        writer.write_all(&(self.m as u32).to_le_bytes())?;
        writer.write_all(&(self.m0 as u32).to_le_bytes())?;
        writer.write_all(&(self.ef_construction as u32).to_le_bytes())?;

        // 2. Body (Nodes)
        for node in &self.nodes {
            writer.write_all(&(node.id as u32).to_le_bytes())?;
            
            // Allow up to 255 layers (HNSW typically < 16)
            let node_max_layer = node.layers.len().saturating_sub(1) as u8; 
            writer.write_all(&node_max_layer.to_le_bytes())?;

            for layer_neighbors in &node.layers {
                writer.write_all(&(layer_neighbors.len() as u32).to_le_bytes())?;
                for &neighbor in layer_neighbors {
                    writer.write_all(&(neighbor as u32).to_le_bytes())?;
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
        let file = std::fs::File::open(path)?;
        let mut reader = std::io::BufReader::new(file);
        Self::load_from_reader(&mut reader, vector_fn)
    }

    /// Load the graph from a reader
    pub fn load_from_reader<R, F>(reader: &mut R, vector_fn: F) -> io::Result<Self>
    where
        R: std::io::Read,
        F: Fn(usize) -> Vec<f32> + Send + Sync + 'static,
    {
        use std::io::Read;
        use super::node::Node;

        // 1. Magic
        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic)?;
        if &magic != b"VIBGRAPH" {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid HNSW magic"));
        }

        // 2. Header
        let mut buf_u32 = [0u8; 4];
        
        reader.read_exact(&mut buf_u32)?;
        let num_nodes = u32::from_le_bytes(buf_u32) as usize;

        reader.read_exact(&mut buf_u32)?;
        let entry_point_raw = u32::from_le_bytes(buf_u32);
        let entry_point = if entry_point_raw == u32::MAX { None } else { Some(entry_point_raw as usize) };

        let mut buf_u8 = [0u8; 1];
        reader.read_exact(&mut buf_u8)?;
        let max_layer = buf_u8[0] as usize;

        reader.read_exact(&mut buf_u32)?;
        let m = u32::from_le_bytes(buf_u32) as usize;
        
        reader.read_exact(&mut buf_u32)?;
        let m0 = u32::from_le_bytes(buf_u32) as usize;

        reader.read_exact(&mut buf_u32)?;
        let ef_construction = u32::from_le_bytes(buf_u32) as usize;

        // 3. Nodes
        let mut nodes = Vec::with_capacity(num_nodes);
        
        for _ in 0..num_nodes {
            reader.read_exact(&mut buf_u32)?;
            let id = u32::from_le_bytes(buf_u32) as usize;

            reader.read_exact(&mut buf_u8)?;
            let node_max_layer = buf_u8[0] as usize;

            let mut layers = Vec::with_capacity(node_max_layer + 1);
            for _ in 0..=node_max_layer {
                reader.read_exact(&mut buf_u32)?;
                let count = u32::from_le_bytes(buf_u32) as usize;
                
                let mut neighbors = Vec::with_capacity(count);
                for _ in 0..count {
                    reader.read_exact(&mut buf_u32)?;
                    neighbors.push(u32::from_le_bytes(buf_u32) as usize);
                }
                layers.push(neighbors);
            }
            
            // Reconstruct Node struct
            // Note: Node struct definition in node.rs might need adjusting if fields are missing?
            // Node has: id, layers.
            nodes.push(Node { id, layers });
        }
        
        // HNSW struct has: m, m0, ml, ef_construction.
        
        Ok(HNSW::from_parts(
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
