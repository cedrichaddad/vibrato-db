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
use tokenizers::Tokenizer;

use crate::spectrogram::Spectrogram;
use crate::models::ModelManager;

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
    text_session: Arc<Mutex<Session>>,
    tokenizer: Arc<Tokenizer>,
    spectrogram: Arc<Spectrogram>,
}

impl InferenceEngine {
    /// Create a new inference engine from pre-downloaded local model files.
    ///
    /// This method is intentionally offline-only. Use `setup_models()` first.
    pub fn new(model_dir: &Path) -> Result<Self, InferenceError> {
        let manager = ModelManager::from_dir(model_dir.to_path_buf());

        // Resolve artifacts without network side effects.
        let audio_model_path = manager.get_clap_audio_offline()?;
        let text_model_path = manager.get_clap_text_offline()?;
        let tokenizer_path = manager.get_tokenizer_offline()?;
        
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

        let text_session = SessionBuilder::new()?
            .with_intra_threads(1)?
            .commit_from_file(text_model_path)?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| InferenceError::Other(e.to_string()))?;

        Ok(Self {
            audio_session: Arc::new(Mutex::new(audio_session)),
            text_session: Arc::new(Mutex::new(text_session)),
            tokenizer: Arc::new(tokenizer),
            spectrogram: Arc::new(Spectrogram::new()),
        })
    }

    /// Explicitly download and stage all model artifacts in `model_dir`.
    pub fn setup_models(model_dir: &Path) -> Result<(), InferenceError> {
        let manager = ModelManager::from_dir(model_dir.to_path_buf());
        manager.setup_models()?;
        Ok(())
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
            
            // 3. Inference with Panic Safety
            // If the model crashes (e.g. shape mismatch), we don't want to kill the worker thread.
            let embedding = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut session = session.lock().unwrap();
                let outputs = session.run(inputs![input_value])?;
                let embedding_tensor = outputs[0].try_extract_tensor::<f32>()?;
                Ok::<Vec<f32>, InferenceError>(embedding_tensor.1.to_vec())
            }))
            .map_err(|_| InferenceError::Other("Inference panicked".to_string()))??;
            
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

    /// Embed text query
    pub async fn embed_text(&self, text: &str) -> Result<Vec<f32>, InferenceError> {
        let tokenizer = self.tokenizer.clone();
        let session = self.text_session.clone();
        let text = text.to_string();

        task::spawn_blocking(move || {
            // 1. Tokenize
            let encoding = tokenizer.encode(text, true)
                .map_err(|e| InferenceError::Other(e.to_string()))?;

            let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
            let attention_mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&x| x as i64).collect();
            
            let seq_len = input_ids.len();
            let shape = vec![1, seq_len];
            
            let input_ids_val = Value::from_array((shape.clone(), input_ids))?;
            let attention_mask_val = Value::from_array((shape, attention_mask))?;

            // 2. Inference with Panic Safety
            let embedding = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut session = session.lock().unwrap();
                let outputs = session.run(inputs![input_ids_val, attention_mask_val])?;
                let embedding_tensor = outputs[0].try_extract_tensor::<f32>()?;
                Ok::<Vec<f32>, InferenceError>(embedding_tensor.1.to_vec())
            }))
            .map_err(|_| InferenceError::Other("Inference panicked".to_string()))??;
            
            Ok(normalize(&embedding))
        }).await?
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
