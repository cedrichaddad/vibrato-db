//! Audio resampler using Rubato
//!
//! Normalizes all audio to a target sample rate (default 48kHz)
//! for consistent spectrogram computation.

use crate::NeuralError;

/// Default target sample rate
pub const DEFAULT_TARGET_SR: u32 = 48000;

/// Resample audio to a target sample rate
///
/// Uses Rubato's SincFixedIn for high-quality resampling.
/// If the source rate matches the target, returns a clone to avoid unnecessary work.
pub fn resample(
    samples: &[f32],
    source_rate: u32,
    target_rate: u32,
) -> Result<Vec<f32>, NeuralError> {
    if source_rate == target_rate {
        return Ok(samples.to_vec());
    }

    use rubato::{SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction, Resampler};

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    let ratio = target_rate as f64 / source_rate as f64;
    let chunk_size = 1024;

    let mut resampler = SincFixedIn::<f32>::new(
        ratio,
        2.0,  // max relative ratio
        params,
        chunk_size,
        1,    // mono
    )
    .map_err(|e| NeuralError::Resampler(format!("Failed to create resampler: {}", e)))?;

    // Process in chunks, reusing buffers
    let mut output = Vec::with_capacity((samples.len() as f64 * ratio * 1.1) as usize);
    let mut pos = 0;

    while pos < samples.len() {
        let end = (pos + chunk_size).min(samples.len());
        let mut chunk = samples[pos..end].to_vec();

        // Pad last chunk if needed
        if chunk.len() < chunk_size {
            chunk.resize(chunk_size, 0.0);
        }

        let input = vec![chunk];
        let resampled = resampler
            .process(&input, None)
            .map_err(|e| NeuralError::Resampler(format!("Resample failed: {}", e)))?;

        if !resampled.is_empty() && !resampled[0].is_empty() {
            output.extend_from_slice(&resampled[0]);
        }

        pos += chunk_size;
    }

    // Trim output to expected length
    let expected_len = (samples.len() as f64 * ratio) as usize;
    output.truncate(expected_len);

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resample_same_rate() {
        let samples = vec![1.0, 2.0, 3.0, 4.0];
        let result = resample(&samples, 44100, 44100).unwrap();
        assert_eq!(result, samples);
    }

    #[test]
    fn test_resample_duration_preserved() {
        // Generate 1 second of 440Hz sine wave at 44100Hz
        let sr = 44100;
        let duration_secs = 1.0f32;
        let num_samples = (sr as f32 * duration_secs) as usize;
        let samples: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin())
            .collect();

        let target_sr = 48000;
        let result = resample(&samples, sr, target_sr).unwrap();

        // Duration should be preserved (allow 5% tolerance due to chunking)
        let expected_samples = (target_sr as f32 * duration_secs) as usize;
        let ratio = result.len() as f32 / expected_samples as f32;
        assert!(
            (ratio - 1.0).abs() < 0.05,
            "Duration should be preserved: got {} samples, expected ~{}",
            result.len(),
            expected_samples
        );
    }
}
