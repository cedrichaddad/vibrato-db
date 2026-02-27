"""Vibrato-DB Python ingestion package."""

from .client import IngestResult, VibratoClient
from .writer import VDBWriter
from .audio import load_audio, extract_frames, compute_mel_spectrogram

__version__ = "0.2.0"
__all__ = [
    "VibratoClient",
    "IngestResult",
    "VDBWriter",
    "load_audio",
    "extract_frames", 
    "compute_mel_spectrogram",
]
