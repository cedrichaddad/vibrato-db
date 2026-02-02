"""Vibrato-DB Python ingestion package."""

from .writer import VDBWriter
from .audio import load_audio, extract_frames, compute_mel_spectrogram

__version__ = "0.1.0"
__all__ = [
    "VDBWriter",
    "load_audio",
    "extract_frames", 
    "compute_mel_spectrogram",
]
