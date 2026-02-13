//! Audio decoder using Symphonia
//!
//! Supports MP3, WAV, FLAC, OGG/Vorbis. Uses a ring buffer to avoid
//! per-packet allocations in the decode loop.

use crate::NeuralError;
use std::path::Path;

/// Decoded audio buffer
#[derive(Debug, Clone)]
pub struct AudioBuffer {
    /// Interleaved samples (mono after conversion)
    pub samples: Vec<f32>,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of channels in original file
    pub channels: u16,
}

/// Decode an audio file to mono f32 samples
///
/// Uses Symphonia for format detection and decoding.
/// Multi-channel audio is downmixed to mono.
pub fn decode_file(path: &Path) -> Result<AudioBuffer, NeuralError> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let file = std::fs::File::open(path)
        .map_err(|e| NeuralError::Decoder(format!("Failed to open {}: {}", path.display(), e)))?;

    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    // Probe the format
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .map_err(|e| NeuralError::Decoder(format!("Failed to probe format: {}", e)))?;

    let mut format = probed.format;

    // Get the default audio track
    let track = format
        .default_track()
        .ok_or_else(|| NeuralError::Decoder("No audio track found".into()))?;

    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or_else(|| NeuralError::Decoder("Unknown sample rate".into()))?;

    let channels = track
        .codec_params
        .channels
        .map(|c| c.count() as u16)
        .unwrap_or(1);

    let track_id = track.id;

    // Create decoder
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| NeuralError::Decoder(format!("Failed to create decoder: {}", e)))?;

    // Pre-allocate ring buffer for samples
    // Estimate ~5 minutes of audio at 48kHz
    let estimated_samples = sample_rate as usize * 300;
    let mut all_samples: Vec<f32> = Vec::with_capacity(estimated_samples);
    let mut sample_buf: Option<SampleBuffer<f32>> = None;

    // Decode loop - reuse sample_buf across packets
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(e) => {
                return Err(NeuralError::Decoder(format!("Packet decode error: {}", e)));
            }
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(symphonia::core::errors::Error::DecodeError(e)) => {
                tracing::warn!("Decode error (skipping packet): {}", e);
                continue;
            }
            Err(e) => {
                return Err(NeuralError::Decoder(format!("Fatal decode error: {}", e)));
            }
        };

        // Reuse sample buffer
        if sample_buf.is_none() || sample_buf.as_ref().unwrap().capacity() < decoded.capacity() {
            sample_buf = Some(SampleBuffer::new(decoded.capacity() as u64, *decoded.spec()));
        }

        if let Some(ref mut buf) = sample_buf {
            buf.copy_interleaved_ref(decoded);
            all_samples.extend_from_slice(buf.samples());
        }
    }

    // Downmix to mono if multi-channel
    let mono_samples = if channels > 1 {
        downmix_to_mono(&all_samples, channels as usize)
    } else {
        all_samples
    };

    Ok(AudioBuffer {
        samples: mono_samples,
        sample_rate,
        channels,
    })
}

/// Downmix interleaved multi-channel audio to mono
fn downmix_to_mono(interleaved: &[f32], channels: usize) -> Vec<f32> {
    let num_frames = interleaved.len() / channels;
    let mut mono = Vec::with_capacity(num_frames);
    let scale = 1.0 / channels as f32;

    for frame in 0..num_frames {
        let mut sum = 0.0f32;
        for ch in 0..channels {
            sum += interleaved[frame * channels + ch];
        }
        mono.push(sum * scale);
    }

    mono
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_downmix_to_mono() {
        // Stereo: L=1.0, R=0.0 → mono=0.5
        let stereo = vec![1.0, 0.0, 0.5, 0.5, 0.0, 1.0];
        let mono = downmix_to_mono(&stereo, 2);
        assert_eq!(mono.len(), 3);
        assert!((mono[0] - 0.5).abs() < 1e-6);
        assert!((mono[1] - 0.5).abs() < 1e-6);
        assert!((mono[2] - 0.5).abs() < 1e-6);
    }
}
