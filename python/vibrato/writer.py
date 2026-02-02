"""VDB binary file writer.

Creates .vdb files compatible with the Rust VectorStore.

File Structure:
    Offset   Size    Type        Description
    ─────────────────────────────────────────
    0x00     8       bytes       Magic: "VIBDB001"
    0x08     4       u32 LE      N: Number of vectors
    0x0C     4       u32 LE      D: Dimensions
    0x10     N*D*4   f32 LE      Vector data
"""

import struct
from pathlib import Path
from typing import Union, Sequence
import numpy as np

MAGIC = b"VIBDB001"
HEADER_SIZE = 16


class VDBWriter:
    """Writer for .vdb binary vector files."""

    def __init__(self, path: Union[str, Path], dimensions: int):
        """
        Create a new VDB file writer.

        Args:
            path: Output file path
            dimensions: Vector dimensionality (e.g., 128 for VGGish)
        """
        self.path = Path(path)
        self.dimensions = dimensions
        self.count = 0
        self._file = open(self.path, "wb")
        
        # Write placeholder header (will be updated in close)
        self._file.write(MAGIC)
        self._file.write(struct.pack("<I", 0))  # count placeholder
        self._file.write(struct.pack("<I", dimensions))

    def write(self, vector: Union[Sequence[float], np.ndarray]) -> None:
        """
        Write a single vector to the file.

        Args:
            vector: 1D array of floats matching the specified dimensions

        Raises:
            ValueError: If vector dimensions don't match
        """
        vec = np.asarray(vector, dtype=np.float32)
        
        if vec.shape != (self.dimensions,):
            raise ValueError(
                f"Dimension mismatch: expected ({self.dimensions},), got {vec.shape}"
            )
        
        # Write as little-endian float32
        self._file.write(vec.tobytes())
        self.count += 1

    def write_batch(self, vectors: np.ndarray) -> None:
        """
        Write multiple vectors at once.

        Args:
            vectors: 2D array of shape (N, dimensions)

        Raises:
            ValueError: If dimensions don't match
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        
        if vectors.ndim != 2 or vectors.shape[1] != self.dimensions:
            raise ValueError(
                f"Expected shape (N, {self.dimensions}), got {vectors.shape}"
            )
        
        self._file.write(vectors.tobytes())
        self.count += vectors.shape[0]

    def close(self) -> int:
        """
        Finalize the file and update the header.

        Returns:
            Total number of vectors written
        """
        # Seek back to count position and update
        self._file.seek(8)
        self._file.write(struct.pack("<I", self.count))
        self._file.close()
        return self.count

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def read_vdb_header(path: Union[str, Path]) -> dict:
    """
    Read the header of a .vdb file.

    Args:
        path: Path to .vdb file

    Returns:
        Dict with 'count' and 'dimensions' keys

    Raises:
        ValueError: If file is not a valid .vdb file
    """
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != MAGIC:
            raise ValueError(f"Invalid magic bytes: expected {MAGIC!r}, got {magic!r}")
        
        count = struct.unpack("<I", f.read(4))[0]
        dimensions = struct.unpack("<I", f.read(4))[0]
        
        return {"count": count, "dimensions": dimensions}


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """
    L2 normalize vectors so ||v|| = 1.

    Args:
        vectors: Array of shape (N, D) or (D,)

    Returns:
        Normalized vectors with same shape
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    
    if vectors.ndim == 1:
        norm = np.linalg.norm(vectors)
        if norm > 1e-10:
            return vectors / norm
        return vectors
    
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)  # Avoid division by zero
    return vectors / norms
