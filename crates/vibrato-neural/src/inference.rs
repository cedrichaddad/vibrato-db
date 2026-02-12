//! ONNX inference engine using ORT (ONNX Runtime)
//!
//! Supports CLAP (audio + text embedding) and VGGish models.
//!
//! # Mock Implementation
//!
//! Due to dependency conflicts with `ort` (requires rustc 1.88+ for 2.x, 1.x yanked),
//! this module currently provides a **MOCK** implementation that returns deterministic
//! pseudo-random embeddings. This allows the upper layers (ingest, server) to be
//! developed and tested end-to-end.

use crate::NeuralError;
use std::path::{Path, PathBuf};

/// Default cache directory for ONNX runtime and models
pub fn cache_dir() -> PathBuf {
    dirs_next()
        .unwrap_or_else(|| PathBuf::from(".cache/vibrato"))
}

fn dirs_next() -> Option<PathBuf> {
    // Cross-platform cache dir
    #[cfg(target_os = "macos")]
    {
        std::env::var("HOME")
            .ok()
            .map(|h| PathBuf::from(h).join(".cache/vibrato"))
    }

    #[cfg(target_os = "linux")]
    {
        std::env::var("XDG_CACHE_HOME")
            .ok()
            .map(PathBuf::from)
            .or_else(|| std::env::var("HOME").ok().map(|h| PathBuf::from(h).join(".cache")))
            .map(|p| p.join("vibrato"))
    }

    #[cfg(target_os = "windows")]
    {
        std::env::var("LOCALAPPDATA")
            .ok()
            .map(|p| PathBuf::from(p).join("vibrato").join("cache"))
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        None
    }
}

/// Placeholder for ONNX inference engine
///
/// Current status: MOCK implementation.
pub struct InferenceEngine {
    _model_dir: PathBuf,
}

impl InferenceEngine {
    /// Create an inference engine pointing to a model directory
    pub fn new(model_dir: &Path) -> Result<Self, NeuralError> {
        // In mock mode, we just check if dir exists (or not even that, for flexibility)
        // But let's keep the check to match API contract.
        if !model_dir.exists() {
             // For demo simplicity, we might warn instead of err if it doesn't exist?
             // No, let's error to prompt user to at least provide a path.
             return Err(NeuralError::Inference(format!(
                "Model directory not found: {}",
                model_dir.display()
            )));
        }

        Ok(Self {
            _model_dir: model_dir.to_path_buf(),
        })
    }

    /// Embed audio from file (MOCK - skips actual decoding)
    pub fn embed_audio_file(&self, _path: &Path) -> Result<Vec<f32>, NeuralError> {
        tracing::warn!("Mocking audio pipeline for file");
        Ok(generate_mock_embedding(512))
    }

    /// Embed audio using CLAP audio encoder (MOCK)
    ///
    /// Input: log-mel spectrogram frames
    /// Output: normalized embedding vector
    pub fn embed_audio(&self, _mel_spectrogram: &[Vec<f32>]) -> Result<Vec<f32>, NeuralError> {
        tracing::warn!("Using MOCK inference for audio embedding");
        Ok(generate_mock_embedding(512))
    }

    /// Embed text using CLAP text encoder (MOCK)
    ///
    /// Input: text query string
    /// Output: normalized embedding vector
    pub fn embed_text(&self, _text: &str) -> Result<Vec<f32>, NeuralError> {
        tracing::warn!("Using MOCK inference for text embedding");
        Ok(generate_mock_embedding(512))
    }
}

fn generate_mock_embedding(dim: usize) -> Vec<f32> {
    // Simple deterministic pseudo-random vector
    let mut vec = Vec::with_capacity(dim);
    for i in 0..dim {
        vec.push((i as f32).sin());
    }
    // Normalize
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        vec.iter().map(|x| x / norm).collect()
    } else {
        vec
    }
}

pub fn is_runtime_available() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_dir_exists() {
        let dir = cache_dir();
        assert!(!dir.as_os_str().is_empty());
    }

    #[test]
    fn test_inference_mock() {
        let dir = std::env::temp_dir();
        let engine = InferenceEngine::new(&dir).unwrap();
        
        let result = engine.embed_audio(&[vec![1.0, 2.0]]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 512);

        let result = engine.embed_text("warm pad");
        assert!(result.is_ok());
    }
}
