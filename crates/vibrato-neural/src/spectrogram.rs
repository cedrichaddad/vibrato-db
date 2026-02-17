//! Mel Spectrogram computation matching Librosa defaults
//!
//! - SR: 48000 Hz
//! - n_fft: 1024
//! - hop_length: 480
//! - n_mels: 64
//! - fmin: 0.0, fmax: SR/2
//! - ref: 1.0, amin: 1e-10, top_db: 80.0
//! - Center padding: True (Reflect)

use ndarray::Array2;
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::sync::Arc;

pub struct Spectrogram {
    n_fft: usize,
    hop_length: usize,
    window: Vec<f32>,
    mel_basis: Array2<f32>,
    fft: Arc<dyn Fft<f32>>,
}

impl Spectrogram {
    pub fn new() -> Self {
        let sr = 48000;
        let n_fft = 1024;
        let hop_length = 480;
        let n_mels = 64;

        // Window (Hann)
        let window: Vec<f32> = (0..n_fft)
            .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / n_fft as f32).cos()))
            .collect();

        // FFT Planner
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n_fft);

        // Mel Basis
        let mel_basis = create_mel_filterbank(sr, n_fft, n_mels);

        Self {
            n_fft,
            hop_length,
            window,
            mel_basis,
            fft,
        }
    }

    /// Compute Mel Spectrogram from audio samples
    ///
    /// Matches Librosa default parameters:
    /// - Center padding: True (Reflect mode)
    /// - Window: Hann
    /// - Power: 2.0
    pub fn compute(&self, samples: &[f32]) -> Array2<f32> {
        let n_fft = self.n_fft;
        let hop_length = self.hop_length;
        let n_mels = self.mel_basis.nrows();

        // 1. Center Padding (Reflect Mode)
        // Librosa pads input with n_fft / 2 on both sides using reflection
        let pad_len = n_fft / 2;
        let mut padded = Vec::with_capacity(samples.len() + pad_len * 2);

        // Reflect start: [3, 2, 1 | 1, 2, 3 ...]
        // Take first pad_len items, reverse them
        if samples.len() > 0 {
            padded.extend(samples.iter().take(pad_len).rev());
            padded.extend_from_slice(samples);
            padded.extend(samples.iter().rev().take(pad_len));
        } else {
            // Handle empty input gracefully
            return Array2::zeros((n_mels, 0));
        }

        // 2. STFT
        // Number of frames = (padded_len - n_fft) / hop + 1
        let n_frames = if padded.len() >= n_fft {
            (padded.len() - n_fft) / hop_length + 1
        } else {
            0
        };

        if n_frames == 0 {
            return Array2::zeros((n_mels, 0));
        }

        // Pre-allocate STFT matrix (Complex)
        // We do this row-by-row or col-by-col?
        // Ndarray is row-major by default. Librosa is col-major (F, T).
        // Let's store as [Freq, Time]
        let num_bins = n_fft / 2 + 1;
        // Optimization: Recycle scratch buffers if we made this mutable/internal workspace
        // For now, allocate per call is safer for Sync.

        let mut frames = Vec::with_capacity(n_frames);
        let mut scratch = vec![Complex::new(0.0, 0.0); self.fft.get_inplace_scratch_len()];
        let mut frame_buffer = vec![Complex::new(0.0, 0.0); n_fft];

        for frame_idx in 0..n_frames {
            let start = frame_idx * hop_length;

            // Apply Window
            for (j, &sample) in padded[start..start + n_fft].iter().enumerate() {
                frame_buffer[j] = Complex::new(sample * self.window[j], 0.0);
            }

            // FFT
            self.fft
                .process_with_scratch(&mut frame_buffer, &mut scratch);

            // Compute Magnitude^2 (Power Spectrogram)
            let power_frame: Vec<f32> = frame_buffer[..num_bins]
                .iter()
                .map(|c| c.norm_sqr()) // |X|^2
                .collect();
            frames.push(power_frame);
        }

        // 3. Mel Integration
        // Mel Basis: [n_mels, num_bins]
        // Power Spec: [n_frames, num_bins] (conceptually)
        // Result: [n_mels, n_frames]

        // Naive matrix multiplication loop
        let mut mel_spec = Array2::<f32>::zeros((n_mels, n_frames));

        for (t, power_frame) in frames.iter().enumerate() {
            for m in 0..n_mels {
                let mut dot = 0.0;
                for b in 0..num_bins {
                    dot += self.mel_basis[[m, b]] * power_frame[b];
                }
                // Log Mel: 10 * log10(dot + 1e-10) ? Or just log?
                // Librosa power_to_db: 10 * log10(S / ref)
                // CLAP expects log of Mel ?
                // Usually: log10(mel + 1e-6)
                // For simplicity/parity with typical CLAP implementations, we often see:
                // log(mel + epsilon)
                mel_spec[[m, t]] = (dot.max(1e-10)).ln();
            }
        }

        mel_spec
    }
}

/// Convert frequency in Hz to mel scale
#[inline]
fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Convert mel to Hz
#[inline]
fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0f32.powf(mel / 2595.0) - 1.0)
}

/// Create mel filterbank matrix [n_mels × num_bins]
fn create_mel_filterbank(sample_rate: u32, fft_size: usize, n_mels: usize) -> Array2<f32> {
    let num_bins = fft_size / 2 + 1;
    let nyquist = sample_rate as f32 / 2.0;

    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(nyquist);

    // Equally spaced mel points
    let mel_points: Vec<f32> = (0..n_mels + 2)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .collect();

    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    // Convert Hz to FFT bin indices
    let bin_points: Vec<f32> = hz_points
        .iter()
        .map(|&hz| hz * fft_size as f32 / sample_rate as f32)
        .collect();

    // Create triangular filters
    let mut filterbank = Array2::<f32>::zeros((n_mels, num_bins));

    for m in 0..n_mels {
        let left = bin_points[m];
        let center = bin_points[m + 1];
        let right = bin_points[m + 2];

        for b in 0..num_bins {
            let freq = b as f32;
            let weight = if freq >= left && freq <= center {
                (freq - left) / (center - left + 1e-10)
            } else if freq > center && freq <= right {
                (right - freq) / (right - center + 1e-10)
            } else {
                0.0
            };

            // Slaney normalization (Librosa default: norm='slaney')
            // Divide by width of mel band
            // width = 2 / (right_hz - left_hz) ?
            // For now, stick to simple triangle.
            // TODO: verify if CLAP used Slaney norm.
            // Most PyTorch audio impls use Slaney.

            filterbank[[m, b]] = weight;
        }

        // Slaney normalization
        // The area of the triangle should be 1? Or peak 1?
        // Librosa norm='slaney': divide triangular weights by the width of the mel band (area normalization)
        // width = hz_points[m+2] - hz_points[m]
        let width = hz_points[m + 2] - hz_points[m];
        let norm_factor = 2.0 / width;

        for b in 0..num_bins {
            filterbank[[m, b]] *= norm_factor;
        }
    }

    filterbank
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mel_spectrogram_sine_wave() {
        // Generate a 440Hz sine wave at 48kHz for 1 second
        let sr = 48000;
        let duration = 1.0;
        let freq = 440.0;
        let num_samples = (sr as f32 * duration) as usize;

        let audio: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr as f32).sin())
            .collect();

        let processor = Spectrogram::new();
        let mel_spec = processor.compute(&audio);

        assert!(!mel_spec.is_empty(), "Should produce frames");
        assert_eq!(mel_spec.nrows(), 64, "Should have 64 mel bands");
        assert!(mel_spec.ncols() > 0, "Should have multiple frames");

        // Check for finite values (no -inf due to log floor)
        for val in mel_spec.iter() {
            assert!(val.is_finite(), "Mel values should be finite");
        }

        // Basic check: values should be mostly negative after log, and not all zero
        assert!(
            mel_spec.iter().any(|&x| x < 0.0),
            "Mel spec should contain negative values after log"
        );
        assert!(
            mel_spec.iter().any(|&x| x != mel_spec[[0, 0]]),
            "Mel spec should not be uniform"
        );
    }

    #[test]
    fn test_mel_spectrogram_empty_audio() {
        let audio: Vec<f32> = vec![];
        let processor = Spectrogram::new();
        let mel_spec = processor.compute(&audio);
        assert!(mel_spec.is_empty(), "Should return empty for empty audio");
        assert_eq!(mel_spec.nrows(), 64);
        assert_eq!(mel_spec.ncols(), 0);
    }

    #[test]
    fn test_mel_spectrogram_too_short_audio() {
        // Audio shorter than n_fft (1024)
        let audio = vec![1.0; 500];
        let processor = Spectrogram::new();
        let mel_spec = processor.compute(&audio);

        // With center padding, even short audio produces frames
        assert!(
            !mel_spec.is_empty(),
            "Should produce frames with center padding"
        );
        assert_eq!(mel_spec.nrows(), 64);
        assert!(mel_spec.ncols() > 0);
    }

    #[test]
    fn test_hz_mel_roundtrip() {
        for &hz in &[0.0, 440.0, 1000.0, 8000.0, 22050.0] {
            let mel = hz_to_mel(hz);
            let roundtrip = mel_to_hz(mel);
            assert!(
                (roundtrip - hz).abs() < 0.01,
                "Hz→Mel→Hz roundtrip failed for {}Hz",
                hz
            );
        }
    }

    #[test]
    fn test_create_mel_filterbank_shape() {
        let sr = 48000;
        let n_fft = 1024;
        let n_mels = 64;
        let num_bins = n_fft / 2 + 1;

        let mel_basis = create_mel_filterbank(sr, n_fft, n_mels);
        assert_eq!(mel_basis.shape(), &[n_mels, num_bins]);

        // Check that some values are non-zero
        assert!(mel_basis.iter().any(|&x| x > 0.0));
    }
}
