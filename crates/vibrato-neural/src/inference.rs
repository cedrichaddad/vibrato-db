//! ONNX inference engine using ORT (ONNX Runtime)
//!
//! Supports CLAP (audio + text embedding).

use std::path::Path;
use std::sync::{Arc, Mutex};

use ort::session::{Session, builder::SessionBuilder};
use ort::value::Value;
use ort::inputs;
use thiserror::Error;
use tokio::task;

use crate::spectrogram::Spectrogram;
use crate::models::ModelManager;
use crate::decoder;
use crate::resampler;

#[derive(Error, Debug)]
pub enum InferenceError {
    #[error("ORT error: {0}")]
    Ort(#[from] ort::Error),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Model error: {0}")]
    Model(#[from] crate::models::ModelError),
    #[error("Join error: {0}")]
    Join(#[from] task::JoinError),
    #[error("Other error: {0}")]
    Other(String),
}

/// Neural Inference Engine
///
/// Wraps ONNX Runtime sessions for audio and text embedding.
#[derive(Clone)]
pub struct InferenceEngine {
    audio_session: Arc<Mutex<Session>>,
    // text_session: Arc<Mutex<Session>>, // TODO: Add text model later
    spectrogram: Arc<Spectrogram>,
}

impl InferenceEngine {
    /// Create a new inference engine
    ///
    /// Downloads models if missing.
    pub fn new(_model_dir: &Path) -> Result<Self, InferenceError> {
        let manager = ModelManager::new();
        
        // Ensure audio model is available
        let audio_model_path = manager.get_clap_audio()?;
        
        // Initialize ORT environment (global)
        // We ignore error if already initialized
        let _ = ort::init()
            .with_name("vibrato")
            .commit();

        // Load models
        // Optimizations: intra_threads=1 to avoid oversubscription in async context
        let audio_session = SessionBuilder::new()?
            .with_intra_threads(1)?
            .commit_from_file(audio_model_path)?;

        Ok(Self {
            audio_session: Arc::new(Mutex::new(audio_session)),
            spectrogram: Arc::new(Spectrogram::new()),
        })
    }

    /// Embed an audio buffer
    ///
    /// 1. Resample to 48kHz (caller responsibility, widely assumed)
    /// 2. Compute Mel-Spectrogram
    /// 3. Run ONNX inference
    /// 4. L2 Normalize
    pub async fn embed_audio(&self, audio: Vec<f32>) -> Result<Vec<f32>, InferenceError> {
        let session = self.audio_session.clone();
        let spectrogram = self.spectrogram.clone();

        // Offload CPU-intensive inference to blocking thread
        task::spawn_blocking(move || {
            // 1. Compute Mel Spectrogram
            // Matches Librosa: [n_mels, time]
            let mel_spec = spectrogram.compute(&audio);
            
            // 2. Prepare Input Tensor
            let (n_mels, time) = mel_spec.dim();
            
            // CLAP expects [Batch, 1, Freq, Time] or similar.
            // Let's assume [1, 1, 64, T] based on standard image-like input.
            let input_shape = vec![1, 1, n_mels, time];
            
            // ndarray is row-major. mel_spec is [n_mels, time].
            // To flatten correctly to [1, 1, 64, T], the memory layout must match.
            // default into_raw_vec() iterates rows then cols, which matches
            // [row0_col0, row0_col1, ... row1_col0 ...]
            // So this should be correct.
            let input_value = Value::from_array((input_shape, mel_spec.into_raw_vec()))?;
            
            // 3. Run Inference
            let mut session = session.lock().unwrap();
            let outputs = session.run(inputs![input_value])?;
            
            // 4. Extract and Normalize
            // Output is usually [Batch, Dim] e.g. [1, 512]
            let embedding_tensor = outputs[0].try_extract_tensor::<f32>()?;
            let embedding: Vec<f32> = embedding_tensor.1.to_vec();
            
            Ok(normalize(&embedding))
        }).await?
    }

    /// Embed audio from file
    ///
    /// 1. Decode file (mp3/wav/flac/etc) -> Mono f32
    /// 2. Resample to 48kHz
    /// 3. Call embed_audio()
    pub async fn embed_audio_file(&self, path: &Path) -> Result<Vec<f32>, InferenceError> {
        let path = path.to_path_buf();
        let engine = self.clone();
        
        // Offload decoding (blocking I/O) to thread pool
        let samples = task::spawn_blocking(move || {
            // 1. Decode
            let buffer = crate::decoder::decode_file(&path)
                .map_err(|e| InferenceError::Other(e.to_string()))?;
            
            // 2. Resample if needed
            if buffer.sample_rate != 48000 {
                 crate::resampler::resample(&buffer.samples, buffer.sample_rate, 48000)
                    .map_err(|e| InferenceError::Other(e.to_string()))
            } else {
                Ok(buffer.samples)
            }
        }).await??;

        // 3. Embed
        engine.embed_audio(samples).await
    }

    /// Mock method for demo compatibility until text model is added
    pub async fn embed_text(&self, _text: &str) -> Result<Vec<f32>, InferenceError> {
        // TODO: Implement real text embedding
        Ok(vec![0.0; 512]) 
    }
}

fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-6 {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}
