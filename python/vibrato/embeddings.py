"""Audio embedding extraction using VGGish or random fallback.

VGGish produces 128-dimensional embeddings from audio.
If TensorFlow is not available, falls back to random embeddings for testing.
"""

from typing import List, Optional, Union
import numpy as np

from .writer import l2_normalize

# Try to import TensorFlow for VGGish
_tf_hub = None
_vggish_model = None


def _load_vggish():
    """Load VGGish model lazily."""
    global _tf_hub, _vggish_model
    
    if _vggish_model is not None:
        return _vggish_model
    
    try:
        import tensorflow_hub as hub
        _tf_hub = hub
        _vggish_model = hub.load("https://tfhub.dev/google/vggish/1")
        return _vggish_model
    except ImportError:
        return None


class VGGishEncoder:
    """
    Audio embedding encoder using VGGish.

    Falls back to random embeddings if TensorFlow is not available.
    """

    def __init__(self, use_vggish: bool = True, embedding_dim: int = 128):
        """
        Initialize the encoder.

        Args:
            use_vggish: Try to use VGGish model (default: True)
            embedding_dim: Output embedding dimension (default: 128)
        """
        self.embedding_dim = embedding_dim
        self._model = None
        self._use_random = False

        if use_vggish:
            self._model = _load_vggish()
            if self._model is None:
                print(
                    "WARNING: TensorFlow/TensorFlow Hub not available. "
                    "Using random embeddings for testing."
                )
                self._use_random = True
        else:
            self._use_random = True

    def encode(
        self,
        audio: np.ndarray,
        sr: int = 16000,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode audio to embedding vector.

        Args:
            audio: 1D audio array (0.96s frame at 16kHz)
            sr: Sample rate
            normalize: L2 normalize output (default: True)

        Returns:
            Embedding vector of shape (embedding_dim,)
        """
        if self._use_random:
            # Random embedding for testing
            embedding = np.random.randn(self.embedding_dim).astype(np.float32)
        else:
            # Real VGGish embedding
            import tensorflow as tf
            audio_tensor = tf.constant(audio, dtype=tf.float32)
            embeddings = self._model(audio_tensor)
            embedding = embeddings.numpy().flatten()[:self.embedding_dim]

        if normalize:
            embedding = l2_normalize(embedding)

        return embedding

    def encode_batch(
        self,
        audio_frames: List[np.ndarray],
        sr: int = 16000,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode multiple audio frames to embeddings.

        Args:
            audio_frames: List of 1D audio arrays
            sr: Sample rate
            normalize: L2 normalize outputs

        Returns:
            Embeddings array of shape (N, embedding_dim)
        """
        embeddings = np.array(
            [self.encode(frame, sr=sr, normalize=False) for frame in audio_frames],
            dtype=np.float32,
        )

        if normalize:
            embeddings = l2_normalize(embeddings)

        return embeddings


class RandomEncoder:
    """
    Random embedding encoder for testing without TensorFlow.
    """

    def __init__(self, embedding_dim: int = 128, seed: Optional[int] = None):
        """
        Initialize random encoder.

        Args:
            embedding_dim: Output dimension
            seed: Random seed for reproducibility
        """
        self.embedding_dim = embedding_dim
        self.rng = np.random.default_rng(seed)

    def encode(self, audio: np.ndarray, normalize: bool = True) -> np.ndarray:
        """Generate random embedding."""
        embedding = self.rng.standard_normal(self.embedding_dim).astype(np.float32)
        if normalize:
            embedding = l2_normalize(embedding)
        return embedding

    def encode_batch(
        self, audio_frames: List[np.ndarray], normalize: bool = True
    ) -> np.ndarray:
        """Generate random embeddings for batch."""
        embeddings = self.rng.standard_normal(
            (len(audio_frames), self.embedding_dim)
        ).astype(np.float32)
        if normalize:
            embeddings = l2_normalize(embeddings)
        return embeddings
