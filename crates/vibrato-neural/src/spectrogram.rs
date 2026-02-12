//! STFT and Mel Spectrogram computation using rustfft
//!
//! Pre-allocates all FFT buffers to avoid per-frame heap allocations.

use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::sync::Arc;


/// STFT Processor with pre-allocated buffers
pub struct StftProcessor {
    fft: Arc<dyn Fft<f32>>,
    fft_size: usize,
    hop_size: usize,
    window: Vec<f32>,
    // Pre-allocated scratch buffers
    fft_input: Vec<Complex<f32>>,
    fft_scratch: Vec<Complex<f32>>,
}

impl StftProcessor {
    /// Create a new STFT processor
    ///
    /// # Arguments
    /// * `fft_size` - FFT window size (typically 2048)
    /// * `hop_size` - Hop size between frames (typically fft_size / 4)
    pub fn new(fft_size: usize, hop_size: usize) -> Self {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(fft_size);
        let scratch_len = fft.get_inplace_scratch_len();

        // Pre-compute periodic Hann window (divides by N, not N-1)
        // The periodic form is correct for STFT because it avoids the
        // discontinuity at window boundaries when frames overlap.
        let window: Vec<f32> = (0..fft_size)
            .map(|i| {
                0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / fft_size as f32).cos())
            })
            .collect();

        Self {
            fft,
            fft_size,
            hop_size,
            window,
            fft_input: vec![Complex::new(0.0, 0.0); fft_size],
            fft_scratch: vec![Complex::new(0.0, 0.0); scratch_len],
        }
    }

    /// Compute magnitude spectrogram
    ///
    /// Returns a Vec of frames, each frame containing fft_size/2 + 1 magnitude bins.
    pub fn magnitude_spectrogram(&mut self, audio: &[f32]) -> Vec<Vec<f32>> {
        if audio.len() < self.fft_size {
            return Vec::new();
        }

        let num_frames = (audio.len() - self.fft_size) / self.hop_size + 1;
        let num_bins = self.fft_size / 2 + 1;
        let mut spectrogram = Vec::with_capacity(num_frames);

        for frame_idx in 0..num_frames {
            let start = frame_idx * self.hop_size;

            // Apply window and fill FFT input buffer (reusing pre-allocated buffer)
            for i in 0..self.fft_size {
                self.fft_input[i] = Complex::new(audio[start + i] * self.window[i], 0.0);
            }

            // In-place FFT (reusing scratch buffer)
            self.fft
                .process_with_scratch(&mut self.fft_input, &mut self.fft_scratch);

            // Extract magnitude (only positive frequencies)
            let magnitudes: Vec<f32> = self.fft_input[..num_bins]
                .iter()
                .map(|c| (c.re * c.re + c.im * c.im).sqrt())
                .collect();

            spectrogram.push(magnitudes);
        }

        spectrogram
    }

    /// Compute log-mel spectrogram
    ///
    /// # Arguments
    /// * `audio` - Mono audio samples
    /// * `sample_rate` - Sample rate of the audio
    /// * `n_mels` - Number of mel bands (typically 128)
    pub fn log_mel_spectrogram(
        &mut self,
        audio: &[f32],
        sample_rate: u32,
        n_mels: usize,
    ) -> Vec<Vec<f32>> {
        let mag_spec = self.magnitude_spectrogram(audio);
        if mag_spec.is_empty() {
            return Vec::new();
        }

        let num_bins = self.fft_size / 2 + 1;
        let mel_filterbank = create_mel_filterbank(sample_rate, self.fft_size, n_mels);

        mag_spec
            .iter()
            .map(|frame| {
                let mut mel_frame = vec![0.0f32; n_mels];
                for m in 0..n_mels {
                    let mut sum = 0.0f32;
                    for b in 0..num_bins {
                        sum += frame[b] * mel_filterbank[m * num_bins + b];
                    }
                    // Log scale with floor to avoid -inf
                    mel_frame[m] = (sum.max(1e-10)).ln();
                }
                mel_frame
            })
            .collect()
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
fn create_mel_filterbank(sample_rate: u32, fft_size: usize, n_mels: usize) -> Vec<f32> {
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
    let mut filterbank = vec![0.0f32; n_mels * num_bins];

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
            filterbank[m * num_bins + b] = weight;
        }
    }

    filterbank
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stft_sine_wave() {
        // Generate a 440Hz sine wave at 48kHz for 0.5 seconds
        let sr = 48000;
        let duration = 0.5;
        let freq = 440.0;
        let num_samples = (sr as f32 * duration) as usize;

        let audio: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr as f32).sin())
            .collect();

        let mut processor = StftProcessor::new(2048, 512);
        let spec = processor.magnitude_spectrogram(&audio);

        assert!(!spec.is_empty(), "Should produce frames");
        assert_eq!(spec[0].len(), 1025, "Should have fft_size/2+1 bins");

        // Find the bin with maximum magnitude
        let peak_bin = spec[spec.len() / 2]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;

        // Expected bin for 440Hz: 440 * 2048 / 48000 ≈ 18.77
        let expected_bin = (freq * 2048.0 / sr as f32).round() as usize;
        assert!(
            (peak_bin as i32 - expected_bin as i32).abs() <= 2,
            "Peak at bin {} should be near expected bin {} for {}Hz",
            peak_bin,
            expected_bin,
            freq
        );
    }

    #[test]
    fn test_mel_spectrogram() {
        let sr = 48000;
        let num_samples = sr; // 1 second
        let audio: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin())
            .collect();

        let mut processor = StftProcessor::new(2048, 512);
        let mel_spec = processor.log_mel_spectrogram(&audio, sr as u32, 128);

        assert!(!mel_spec.is_empty());
        assert_eq!(mel_spec[0].len(), 128, "Should have 128 mel bands");

        // Values should be finite (no -inf due to log floor)
        for frame in &mel_spec {
            for &val in frame {
                assert!(val.is_finite(), "Mel values should be finite");
            }
        }
    }

    #[test]
    fn test_stft_too_short() {
        let audio = vec![1.0; 100]; // Shorter than fft_size
        let mut processor = StftProcessor::new(2048, 512);
        let spec = processor.magnitude_spectrogram(&audio);
        assert!(spec.is_empty(), "Should return empty for too-short audio");
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
}
