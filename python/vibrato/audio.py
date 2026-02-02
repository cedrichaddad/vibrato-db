"""Audio processing utilities for Vibrato-DB.

Provides functions for:
- Loading and resampling audio to 16kHz mono
- Extracting non-overlapping frames (0.96s)
- Computing 64-bin mel spectrograms
"""

from typing import Tuple, List, Optional
import numpy as np

# Lazy import to make librosa optional
_librosa = None


def _get_librosa():
    """Lazy load librosa to avoid import errors if not installed."""
    global _librosa
    if _librosa is None:
        try:
            import librosa
            _librosa = librosa
        except ImportError:
            raise ImportError(
                "librosa is required for audio processing. "
                "Install with: pip install librosa"
            )
    return _librosa


def load_audio(
    path: str,
    sr: int = 16000,
    mono: bool = True,
    duration: Optional[float] = None,
    offset: float = 0.0,
) -> Tuple[np.ndarray, int]:
    """
    Load and resample audio file to 16kHz mono.

    This matches VGGish requirements for audio input.

    Args:
        path: Path to audio file (MP3, WAV, FLAC, etc.)
        sr: Target sample rate (default: 16000 for VGGish)
        mono: Convert to mono (default: True)
        duration: Maximum duration in seconds (default: None = full file)
        offset: Start time in seconds (default: 0.0)

    Returns:
        Tuple of (audio_array, sample_rate)
    """
    librosa = _get_librosa()
    
    audio, sample_rate = librosa.load(
        path,
        sr=sr,
        mono=mono,
        duration=duration,
        offset=offset,
    )
    
    return audio, sample_rate


def extract_frames(
    audio: np.ndarray,
    sr: int = 16000,
    frame_duration: float = 0.96,
    hop_duration: Optional[float] = None,
) -> List[np.ndarray]:
    """
    Split audio into non-overlapping frames.

    Args:
        audio: 1D audio array
        sr: Sample rate
        frame_duration: Duration of each frame in seconds (default: 0.96s)
        hop_duration: Hop between frames (default: same as frame_duration)

    Returns:
        List of frame arrays
    """
    frame_samples = int(frame_duration * sr)
    hop_samples = int((hop_duration or frame_duration) * sr)
    
    frames = []
    for start in range(0, len(audio) - frame_samples + 1, hop_samples):
        frame = audio[start : start + frame_samples]
        frames.append(frame)
    
    return frames


def compute_mel_spectrogram(
    audio: np.ndarray,
    sr: int = 16000,
    n_fft: int = 400,
    hop_length: int = 160,
    n_mels: int = 64,
    fmin: float = 125.0,
    fmax: float = 7500.0,
) -> np.ndarray:
    """
    Compute mel spectrogram for audio frame.

    Default parameters match VGGish preprocessing.

    Args:
        audio: 1D audio array
        sr: Sample rate
        n_fft: FFT window size (25ms at 16kHz)
        hop_length: Hop length (10ms at 16kHz)
        n_mels: Number of mel bands (64 for VGGish)
        fmin: Minimum frequency
        fmax: Maximum frequency

    Returns:
        Mel spectrogram of shape (n_mels, time_frames)
    """
    librosa = _get_librosa()
    
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )
    
    # Convert to log scale (dB-like)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    
    return log_mel


def process_audio_file(
    path: str,
    frame_duration: float = 0.96,
    sr: int = 16000,
) -> List[np.ndarray]:
    """
    Full pipeline: load audio and compute mel spectrograms for each frame.

    Args:
        path: Path to audio file
        frame_duration: Duration of each frame in seconds
        sr: Target sample rate

    Returns:
        List of mel spectrogram arrays, one per frame
    """
    audio, _ = load_audio(path, sr=sr)
    frames = extract_frames(audio, sr=sr, frame_duration=frame_duration)
    
    spectrograms = []
    for frame in frames:
        mel = compute_mel_spectrogram(frame, sr=sr)
        spectrograms.append(mel)
    
    return spectrograms
