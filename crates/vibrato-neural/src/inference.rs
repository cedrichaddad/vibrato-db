//! ONNX inference engine using ORT (ONNX Runtime)
//!
//! Supports CLAP (audio + text embedding) and VGGish models.
//! Uses `load-dynamic` feature to avoid bundling libonnxruntime.
//!
//! # Runtime Download
//!
//! On first use, the engine checks `~/.cache/vibrato/ort/` for the
//! ONNX Runtime shared library. If not found, it can be downloaded
//! automatically via `ensure_runtime()`.

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
/// Full implementation requires the ORT runtime to be available.
/// See `ensure_runtime()` for automatic setup.
pub struct InferenceEngine {
    _model_dir: PathBuf,
}

impl InferenceEngine {
    /// Create an inference engine pointing to a model directory
    ///
    /// The model directory should contain ONNX model files:
    /// - `clap_audio.onnx` - CLAP audio encoder
    /// - `clap_text.onnx` - CLAP text encoder  
    /// - `vggish.onnx` - VGGish audio fingerprint
    pub fn new(model_dir: &Path) -> Result<Self, NeuralError> {
        if !model_dir.exists() {
            return Err(NeuralError::Inference(format!(
                "Model directory not found: {}",
                model_dir.display()
            )));
        }

        Ok(Self {
            _model_dir: model_dir.to_path_buf(),
        })
    }

    /// Embed audio using CLAP audio encoder
    ///
    /// Input: log-mel spectrogram frames
    /// Output: normalized embedding vector
    pub fn embed_audio(&self, _mel_spectrogram: &[Vec<f32>]) -> Result<Vec<f32>, NeuralError> {
        // TODO: Implement when ORT sessions are wired up
        Err(NeuralError::Inference(
            "ONNX inference not yet implemented. Requires ORT runtime.".into(),
        ))
    }

    /// Embed text using CLAP text encoder
    ///
    /// Input: text query string
    /// Output: normalized embedding vector in same space as audio embeddings
    pub fn embed_text(&self, _text: &str) -> Result<Vec<f32>, NeuralError> {
        // TODO: Implement when ORT sessions are wired up
        Err(NeuralError::Inference(
            "ONNX inference not yet implemented. Requires ORT runtime.".into(),
        ))
    }
}

/// Check if ONNX Runtime is available at the expected location
pub fn is_runtime_available() -> bool {
    let cache = cache_dir();
    let lib_name = if cfg!(target_os = "macos") {
        "libonnxruntime.dylib"
    } else if cfg!(target_os = "linux") {
        "libonnxruntime.so"
    } else if cfg!(target_os = "windows") {
        "onnxruntime.dll"
    } else {
        return false;
    };

    cache.join("ort").join(lib_name).exists()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_dir_exists() {
        let dir = cache_dir();
        // Should return some path (may not exist on disk)
        assert!(!dir.as_os_str().is_empty());
    }

    #[test]
    fn test_inference_engine_missing_dir() {
        let result = InferenceEngine::new(Path::new("/nonexistent/model/dir"));
        assert!(result.is_err());
    }

    #[test]
    fn test_inference_engine_not_implemented() {
        let dir = std::env::temp_dir();
        let engine = InferenceEngine::new(&dir).unwrap();
        
        let result = engine.embed_audio(&[vec![1.0, 2.0]]);
        assert!(result.is_err());

        let result = engine.embed_text("warm pad");
        assert!(result.is_err());
    }
}
