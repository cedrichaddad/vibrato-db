#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DspState {
    pub wet_gain: f32,
}

impl DspState {
    pub fn new(wet_gain: f32) -> Self {
        Self {
            wet_gain: wet_gain.clamp(0.0, 1.5),
        }
    }

    pub fn from_score(score: f32) -> Self {
        // Map ANN confidence to a conservative wet gain envelope.
        Self::new((0.35 + score * 0.65).clamp(0.35, 1.0))
    }
}

#[derive(Debug, Clone)]
pub struct RealtimeDspState {
    current: DspState,
    from: DspState,
    to: DspState,
    fade_total_samples: usize,
    fade_remaining_samples: usize,
    fade_step: f32,
}

impl Default for RealtimeDspState {
    fn default() -> Self {
        let s = DspState::new(1.0);
        Self {
            current: s,
            from: s,
            to: s,
            fade_total_samples: 0,
            fade_remaining_samples: 0,
            fade_step: 0.0,
        }
    }
}

impl RealtimeDspState {
    pub fn schedule_transition(
        &mut self,
        target: DspState,
        sample_rate_hz: f32,
        crossfade_ms: f32,
    ) {
        // Retarget from the instantaneous gain, not the last fully committed state.
        // This prevents gain discontinuities when updates arrive mid-crossfade.
        let start_gain = self.current_gain();
        if (start_gain - target.wet_gain).abs() < 1e-6 {
            self.current = target;
            self.from = target;
            self.to = target;
            self.fade_total_samples = 0;
            self.fade_remaining_samples = 0;
            self.fade_step = 0.0;
            return;
        }

        let clamped_ms = crossfade_ms.clamp(2.0, 5.0);
        let fade_samples = ((sample_rate_hz.max(1.0) * clamped_ms) / 1000.0)
            .round()
            .max(1.0) as usize;

        let start = DspState::new(start_gain);
        self.current = start;
        self.from = start;
        self.to = target;
        self.fade_total_samples = fade_samples;
        self.fade_remaining_samples = fade_samples;
        self.fade_step = 1.0 / fade_samples as f32;
    }

    #[inline(always)]
    fn current_gain(&self) -> f32 {
        if self.fade_remaining_samples == 0 || self.fade_total_samples == 0 {
            return self.current.wet_gain;
        }

        let progressed = self.fade_total_samples - self.fade_remaining_samples;
        let t = (progressed as f32 * self.fade_step).min(1.0);
        self.from.wet_gain + (self.to.wet_gain - self.from.wet_gain) * t
    }

    #[inline(always)]
    pub fn process_sample(&mut self, input: f32) -> f32 {
        let gain = self.current_gain();
        let out = input * gain;

        if self.fade_remaining_samples > 0 {
            self.fade_remaining_samples -= 1;
            if self.fade_remaining_samples == 0 {
                self.current = self.to;
            }
        }
        out
    }

    #[inline(always)]
    pub fn current_state(&self) -> DspState {
        self.current
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transition_crossfades_without_snap() {
        let mut rt = RealtimeDspState::default();
        rt.schedule_transition(DspState::new(0.5), 48_000.0, 3.0);

        let mut prev = rt.process_sample(1.0);
        // Should begin from the last-known-good state and then crossfade.
        assert!(prev > 0.5);
        assert!(prev <= 1.0);

        for _ in 0..256 {
            let now = rt.process_sample(1.0);
            // Linear fade should be monotonic for this transition.
            assert!(now <= prev + 1e-4);
            prev = now;
        }

        assert!((rt.current_state().wet_gain - 0.5).abs() < 1e-3);
    }

    #[test]
    fn holds_last_known_good_state_without_updates() {
        let mut rt = RealtimeDspState::default();
        rt.schedule_transition(DspState::new(0.8), 48_000.0, 3.0);
        for _ in 0..400 {
            let _ = rt.process_sample(1.0);
        }
        // No new transition: value must remain at the last completed state.
        let out = rt.process_sample(1.0);
        assert!((out - 0.8).abs() < 1e-3);
    }

    #[test]
    fn retarget_mid_fade_stays_continuous() {
        let mut rt = RealtimeDspState::default();
        rt.schedule_transition(DspState::new(0.5), 48_000.0, 3.0);
        for _ in 0..50 {
            let _ = rt.process_sample(1.0);
        }

        let before = rt.process_sample(1.0);
        rt.schedule_transition(DspState::new(0.9), 48_000.0, 3.0);
        let after = rt.process_sample(1.0);

        // No gain jump/pop at retarget boundary.
        assert!(
            (after - before).abs() < 0.02,
            "expected continuous retarget; before={before} after={after}"
        );
    }
}
