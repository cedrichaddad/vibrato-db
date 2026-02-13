use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

const CLAP_AUDIO_URL: &str =
    "https://huggingface.co/laion/clap-htsat-unfused/resolve/main/clap_audio.onnx";
const CLAP_TEXT_URL: &str =
    "https://huggingface.co/Xenova/laion-clap-htsat-unfused/resolve/main/onnx/text_model.onnx";
const TOKENIZER_URL: &str =
    "https://huggingface.co/Xenova/laion-clap-htsat-unfused/resolve/main/tokenizer.json";
const MANIFEST_FILE: &str = "model-manifest.json";

#[derive(Error, Debug)]
pub enum ModelError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),
    #[error("Checksum mismatch for {path:?}: expected {expected}, got {actual}")]
    ChecksumMismatch {
        path: PathBuf,
        expected: String,
        actual: String,
    },
    #[error("Model directory not found")]
    DirectoryNotFound,
    #[error("Required model file missing: {0}")]
    ModelMissing(PathBuf),
    #[error("Model manifest missing: {0}")]
    ManifestMissing(PathBuf),
    #[error("Model manifest invalid at {path:?}: {reason}")]
    ManifestInvalid { path: PathBuf, reason: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelManifest {
    version: u32,
    clap_audio_sha256: String,
    clap_text_sha256: String,
    tokenizer_sha256: String,
}

pub struct ModelManager {
    root: PathBuf,
}

impl ModelManager {
    pub fn from_dir(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        fs::create_dir_all(&root).ok();
        Self { root }
    }

    pub fn new() -> Self {
        // 1. Check Env Var Override (Air-Gap Support)
        if let Ok(p) = std::env::var("VIBRATO_MODEL_DIR") {
            return Self::from_dir(PathBuf::from(p));
        }

        // 2. Default to XDG Cache
        if let Some(dirs) = ProjectDirs::from("com", "vibrato", "vibrato") {
            Self::from_dir(dirs.cache_dir().join("models"))
        } else {
            // Fallback for weird systems
            Self::from_dir(PathBuf::from(".vibrato/models"))
        }
    }

    #[inline]
    fn clap_audio_path(&self) -> PathBuf {
        self.root.join("clap_audio.onnx")
    }

    #[inline]
    fn clap_text_path(&self) -> PathBuf {
        self.root.join("text_model.onnx")
    }

    #[inline]
    fn tokenizer_path(&self) -> PathBuf {
        self.root.join("tokenizer.json")
    }

    #[inline]
    fn manifest_path(&self) -> PathBuf {
        self.root.join(MANIFEST_FILE)
    }

    fn load_manifest(&self) -> Result<ModelManifest, ModelError> {
        let path = self.manifest_path();
        if !path.exists() {
            return Err(ModelError::ManifestMissing(path));
        }
        let bytes = fs::read(&path)?;
        serde_json::from_slice(&bytes).map_err(|e| ModelError::ManifestInvalid {
            path,
            reason: e.to_string(),
        })
    }

    fn write_manifest(&self, manifest: &ModelManifest) -> Result<(), ModelError> {
        let path = self.manifest_path();
        let tmp_path = self.root.join(format!("{}.tmp", MANIFEST_FILE));
        let payload =
            serde_json::to_vec_pretty(manifest).map_err(|e| ModelError::ManifestInvalid {
                path: path.clone(),
                reason: e.to_string(),
            })?;

        let mut file = File::create(&tmp_path)?;
        file.write_all(&payload)?;
        file.sync_all()?;
        fs::rename(&tmp_path, &path)?;
        sync_parent_dir(&path)?;
        Ok(())
    }

    fn download_if_missing(&self, url: &str, dest: &Path) -> Result<(), ModelError> {
        if dest.exists() {
            return Ok(());
        }
        tracing::info!("Downloading model artifact {} -> {:?}", url, dest);
        self.download_file(url, dest)
    }

    fn verify_expected_hash(&self, path: &Path, expected: &str) -> Result<(), ModelError> {
        let actual = compute_sha256(path)?;
        if actual == expected {
            Ok(())
        } else {
            Err(ModelError::ChecksumMismatch {
                path: path.to_path_buf(),
                expected: expected.to_string(),
                actual,
            })
        }
    }

    fn ensure_verified(&self, path: PathBuf, expected: &str) -> Result<PathBuf, ModelError> {
        if !path.exists() {
            return Err(ModelError::ModelMissing(path));
        }
        self.verify_expected_hash(&path, expected)?;
        Ok(path)
    }

    /// Ensure the CLAP Audio model exists and is verified.
    /// Returns the path to the model file.
    pub fn get_clap_audio(&self) -> Result<PathBuf, ModelError> {
        let path = self.clap_audio_path();
        let manifest = self.load_manifest()?;
        if !path.exists() {
            self.download_file(CLAP_AUDIO_URL, &path)?;
        }
        self.ensure_verified(path, &manifest.clap_audio_sha256)
    }

    /// Return CLAP audio model path without network side effects.
    pub fn get_clap_audio_offline(&self) -> Result<PathBuf, ModelError> {
        let manifest = self.load_manifest()?;
        self.ensure_verified(self.clap_audio_path(), &manifest.clap_audio_sha256)
    }

    /// Ensure the CLAP Text model exists.
    pub fn get_clap_text(&self) -> Result<PathBuf, ModelError> {
        let path = self.clap_text_path();
        let manifest = self.load_manifest()?;
        if !path.exists() {
            self.download_file(CLAP_TEXT_URL, &path)?;
        }
        self.ensure_verified(path, &manifest.clap_text_sha256)
    }

    /// Return CLAP text model path without network side effects.
    pub fn get_clap_text_offline(&self) -> Result<PathBuf, ModelError> {
        let manifest = self.load_manifest()?;
        self.ensure_verified(self.clap_text_path(), &manifest.clap_text_sha256)
    }

    /// Ensure the tokenizer exists.
    pub fn get_tokenizer(&self) -> Result<PathBuf, ModelError> {
        let path = self.tokenizer_path();
        let manifest = self.load_manifest()?;
        if !path.exists() {
            self.download_file(TOKENIZER_URL, &path)?;
        }
        self.ensure_verified(path, &manifest.tokenizer_sha256)
    }

    /// Return tokenizer path without network side effects.
    pub fn get_tokenizer_offline(&self) -> Result<PathBuf, ModelError> {
        let manifest = self.load_manifest()?;
        self.ensure_verified(self.tokenizer_path(), &manifest.tokenizer_sha256)
    }

    /// Download and stage all required model artifacts.
    ///
    /// This performs TOFU (trust on first use): hashes are captured into a local
    /// manifest and strictly enforced on every subsequent load.
    pub fn setup_models(&self) -> Result<(), ModelError> {
        let audio_path = self.clap_audio_path();
        let text_path = self.clap_text_path();
        let tokenizer_path = self.tokenizer_path();

        self.download_if_missing(CLAP_AUDIO_URL, &audio_path)?;
        self.download_if_missing(CLAP_TEXT_URL, &text_path)?;
        self.download_if_missing(TOKENIZER_URL, &tokenizer_path)?;

        let manifest = ModelManifest {
            version: 1,
            clap_audio_sha256: compute_sha256(&audio_path)?,
            clap_text_sha256: compute_sha256(&text_path)?,
            tokenizer_sha256: compute_sha256(&tokenizer_path)?,
        };
        self.write_manifest(&manifest)?;

        // Verify immediately to catch any I/O races or corruption before returning.
        self.verify_expected_hash(&audio_path, &manifest.clap_audio_sha256)?;
        self.verify_expected_hash(&text_path, &manifest.clap_text_sha256)?;
        self.verify_expected_hash(&tokenizer_path, &manifest.tokenizer_sha256)?;
        Ok(())
    }

    fn download_file(&self, url: &str, dest: &Path) -> Result<(), ModelError> {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .map_err(reqwest::Error::from)?;

        let tmp_path = dest.with_extension("tmp");
        let mut response = client.get(url).send()?.error_for_status()?;
        let mut file = File::create(&tmp_path)?;
        response.copy_to(&mut file)?;
        file.sync_all()?;
        fs::rename(&tmp_path, dest)?;
        sync_parent_dir(dest)?;
        Ok(())
    }

    pub fn verify_hash(path: &Path, expected: &str) -> Result<bool, ModelError> {
        Ok(compute_sha256(path)? == expected)
    }
}

fn compute_sha256(path: &Path) -> Result<String, ModelError> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher)?;
    Ok(hex::encode(hasher.finalize()))
}

fn sync_parent_dir(path: &Path) -> Result<(), ModelError> {
    #[cfg(unix)]
    {
        if let Some(parent) = path.parent() {
            File::open(parent)?.sync_all()?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn write_dummy_models(root: &Path) {
        fs::write(root.join("clap_audio.onnx"), b"audio-bytes").unwrap();
        fs::write(root.join("text_model.onnx"), b"text-bytes").unwrap();
        fs::write(root.join("tokenizer.json"), b"{\"v\":\"1\"}").unwrap();
    }

    #[test]
    fn offline_load_requires_manifest() {
        let dir = tempdir().unwrap();
        write_dummy_models(dir.path());
        let manager = ModelManager::from_dir(dir.path());

        let err = manager.get_clap_audio_offline().unwrap_err();
        assert!(matches!(err, ModelError::ManifestMissing(_)));
    }

    #[test]
    fn offline_load_validates_checksums() {
        let dir = tempdir().unwrap();
        write_dummy_models(dir.path());
        let manager = ModelManager::from_dir(dir.path());

        let manifest = ModelManifest {
            version: 1,
            clap_audio_sha256: compute_sha256(&dir.path().join("clap_audio.onnx")).unwrap(),
            clap_text_sha256: compute_sha256(&dir.path().join("text_model.onnx")).unwrap(),
            tokenizer_sha256: compute_sha256(&dir.path().join("tokenizer.json")).unwrap(),
        };
        manager.write_manifest(&manifest).unwrap();

        assert!(manager.get_clap_audio_offline().is_ok());
        assert!(manager.get_clap_text_offline().is_ok());
        assert!(manager.get_tokenizer_offline().is_ok());

        fs::write(dir.path().join("clap_audio.onnx"), b"tampered").unwrap();
        let err = manager.get_clap_audio_offline().unwrap_err();
        assert!(matches!(err, ModelError::ChecksumMismatch { .. }));
    }
}
