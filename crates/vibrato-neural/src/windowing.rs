//! Audio windowing strategies
//!
//! Fixed-stride windowing (deterministic, default) and transient-aware
//! windowing (behind `transient_aware` feature flag).

use std::ops::Range;

/// Fixed-stride windower with configurable overlap
///
/// This is the default, deterministic windowing strategy. Every invocation
/// with the same audio produces identical windows (idempotent).
pub struct FixedStrideWindower {
    /// Window size in samples
    pub window_size: usize,
    /// Hop size in samples (window_size / 2 for 50% overlap)
    pub hop_size: usize,
}

impl FixedStrideWindower {
    /// Create a windower with 50% overlap
    pub fn new(sample_rate: u32, window_duration_ms: u32) -> Self {
        let window_size = (sample_rate as usize * window_duration_ms as usize) / 1000;
        let hop_size = window_size / 2;
        Self {
            window_size,
            hop_size,
        }
    }

    /// Create a windower with custom overlap ratio (0.0 = no overlap, 0.5 = 50%)
    pub fn with_overlap(sample_rate: u32, window_duration_ms: u32, overlap: f32) -> Self {
        let window_size = (sample_rate as usize * window_duration_ms as usize) / 1000;
        let hop_size = ((1.0 - overlap) * window_size as f32) as usize;
        Self {
            window_size,
            hop_size: hop_size.max(1),
        }
    }

    /// Segment audio into windows, returning sample ranges
    pub fn segment(&self, audio_len: usize) -> Vec<Range<usize>> {
        if audio_len < self.window_size {
            if audio_len > 0 {
                return vec![0..audio_len];
            }
            return Vec::new();
        }

        let num_windows = (audio_len - self.window_size) / self.hop_size + 1;
        let mut windows = Vec::with_capacity(num_windows);

        for i in 0..num_windows {
            let start = i * self.hop_size;
            let end = start + self.window_size;
            windows.push(start..end);
        }

        windows
    }
}

/// Transient-aware windower using spectral flux onset detection
///
/// Adjusts window boundaries near transients to avoid splitting them.
/// Only available with the `transient_aware` feature flag.
#[cfg(feature = "transient_aware")]
pub struct TransientAwareWindower {
    /// Base window size in samples
    pub window_size: usize,
    /// Hop size in samples
    pub hop_size: usize,
    /// Spectral flux threshold for transient detection (0.0 - 1.0)
    pub sensitivity: f32,
    /// Maximum shift in samples (10% of window)
    pub max_shift: usize,
}

#[cfg(feature = "transient_aware")]
impl TransientAwareWindower {
    /// Create a transient-aware windower
    pub fn new(sample_rate: u32, window_duration_ms: u32, sensitivity: f32) -> Self {
        let window_size = (sample_rate as usize * window_duration_ms as usize) / 1000;
        let hop_size = window_size / 2;
        let max_shift = window_size / 10;
        Self {
            window_size,
            hop_size,
            sensitivity: sensitivity.clamp(0.0, 1.0),
            max_shift,
        }
    }

    /// Compute spectral flux onset detection function
    fn spectral_flux(spectrogram: &[Vec<f32>]) -> Vec<f32> {
        if spectrogram.len() < 2 {
            return vec![0.0; spectrogram.len()];
        }

        let mut flux = vec![0.0f32; spectrogram.len()];
        for i in 1..spectrogram.len() {
            let mut sum = 0.0f32;
            for (curr, prev) in spectrogram[i].iter().zip(spectrogram[i - 1].iter()) {
                let diff = curr - prev;
                if diff > 0.0 {
                    sum += diff * diff;
                }
            }
            flux[i] = sum.sqrt();
        }
        flux
    }

    /// Segment audio with transient-aware window shifting
    pub fn segment(
        &self,
        audio_len: usize,
        spectrogram: &[Vec<f32>],
        spec_hop_size: usize,
    ) -> Vec<Range<usize>> {
        let flux = Self::spectral_flux(spectrogram);

        // Find threshold as mean + sensitivity * std
        let mean: f32 = flux.iter().sum::<f32>() / flux.len().max(1) as f32;
        let variance: f32 = flux.iter().map(|&f| (f - mean) * (f - mean)).sum::<f32>()
            / flux.len().max(1) as f32;
        let threshold = mean + self.sensitivity * variance.sqrt();

        // Start with fixed-stride windows
        let base = FixedStrideWindower {
            window_size: self.window_size,
            hop_size: self.hop_size,
        };
        let mut windows = base.segment(audio_len);

        // Adjust boundaries near transients
        for window in &mut windows {
            let window_start_frame = window.start / spec_hop_size;
            let window_end_frame = (window.end / spec_hop_size).min(flux.len() - 1);

            // Check if there's a transient near the boundary (within 10% of edges)
            let boundary_zone = (window_end_frame - window_start_frame) / 10;
            let start_zone = window_start_frame..window_start_frame + boundary_zone;
            let end_zone = window_end_frame.saturating_sub(boundary_zone)..window_end_frame;

            // Shift start if transient is near the beginning
            for f in start_zone {
                if f < flux.len() && flux[f] > threshold {
                    let shift = (f * spec_hop_size).saturating_sub(window.start);
                    let shift = shift.min(self.max_shift);
                    window.start = window.start.saturating_sub(shift);
                    break;
                }
            }

            // Shift end if transient is near the end
            for f in end_zone {
                if f < flux.len() && flux[f] > threshold {
                    let shift = window.end.saturating_sub(f * spec_hop_size);
                    let shift = shift.min(self.max_shift);
                    window.end = (window.end + shift).min(audio_len);
                    break;
                }
            }
        }

        windows
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_stride_windows() {
        let windower = FixedStrideWindower::new(48000, 960); // 960ms
        let windows = windower.segment(48000 * 5); // 5 seconds of audio

        assert!(!windows.is_empty());

        // All windows should have the same size (except possibly the last)
        let expected_size = 48000 * 960 / 1000;
        for w in &windows {
            assert_eq!(w.end - w.start, expected_size);
        }

        // Windows should overlap by 50%
        if windows.len() > 1 {
            let overlap = windows[0].end - windows[1].start;
            assert_eq!(overlap, expected_size / 2);
        }
    }

    #[test]
    fn test_fixed_stride_short_audio() {
        let windower = FixedStrideWindower::new(48000, 960);
        let windows = windower.segment(100); // Very short
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0], 0..100);
    }

    #[test]
    fn test_fixed_stride_empty() {
        let windower = FixedStrideWindower::new(48000, 960);
        let windows = windower.segment(0);
        assert!(windows.is_empty());
    }

    #[test]
    fn test_fixed_stride_deterministic() {
        let windower = FixedStrideWindower::new(48000, 500);
        let w1 = windower.segment(48000 * 3);
        let w2 = windower.segment(48000 * 3);
        assert_eq!(w1, w2, "Same input should produce same windows");
    }
}
