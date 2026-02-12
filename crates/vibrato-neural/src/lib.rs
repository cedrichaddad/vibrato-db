//! Vibrato Neural – Audio ingestion, DSP, and ONNX inference
//!
//! # Architecture
//!
//! ```text
//! Audio File (.mp3/.wav/.flac)
//!     │
//!     ▼
//! ┌─────────┐    ┌───────────┐    ┌──────────────┐    ┌───────────┐
//! │ Decoder  │───▶│ Resampler │───▶│ Spectrogram  │───▶│ Inference │
//! │Symphonia │    │  Rubato   │    │   rustfft    │    │  ORT/ONNX │
//! └─────────┘    └───────────┘    └──────────────┘    └───────────┘
//!                         Ring Buffer Pool
//! ```
//!
//! The entire pipeline uses pre-allocated ring buffers to avoid
//! per-packet heap allocations. Deterministic fixed-stride windowing
//! is the default; transient-aware mode is behind a feature flag.

pub mod decoder;
pub mod inference;
pub mod resampler;
pub mod spectrogram;
pub mod windowing;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum NeuralError {
    #[error("Decoder error: {0}")]
    Decoder(String),

    #[error("Resampler error: {0}")]
    Resampler(String),

    #[error("Spectrogram error: {0}")]
    Spectrogram(String),

    #[error("Inference error: {0}")]
    Inference(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
